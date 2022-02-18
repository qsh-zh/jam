import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from subprocess import Popen

import psutil

from jammy.comm import get_local_addr, is_port_used
from jammy.comm.cs import ClientPipe, ServerPipe
from jammy.logging import get_logger
from jammy.utils.env import jam_getenv
from jammy.utils.printing import kvformat, stformat

logger = get_logger(
    os.path.expanduser("~/jsh.log"),
    level="DEBUG",
    retention="10 MB",
    enqueue=True,
    filter=lambda record: "jsh" in record["extra"],
)
ERROR_FORMAT = (
    "<green>{time:MM-DD HH:mm:ss.SSS}</green> |"
    "<red>[{process.name}]</red>|"
    "<level>{level: <8}</level> |"
    "<level>{message}</level>"
)
logger = get_logger(
    os.path.expanduser("~/jsh_error.log"),
    level="DEBUG",
    retention="10 MB",
    enqueue=True,
    filter=lambda record: "jsherror" in record["extra"],
    format=ERROR_FORMAT,
)

# pylint: disable=global-variable-not-assigned, global-statement


@dataclass
class PCState:
    mem_avail: float = 0.0
    cpu_usage: float = 1.0

    def ready(self, cpu_usage_thres, mem_avail_thres):
        cpu_ready = cpu_usage_thres > self.cpu_usage
        mem_ready = mem_avail_thres < self.mem_avail
        return cpu_ready and mem_ready


def get_pc_state(interval_cpu_check: int = 2):
    mem_avail = psutil.virtual_memory().available / 1024 / 1024 / 1024
    cpu_usage = psutil.cpu_percent(interval=interval_cpu_check)
    return PCState(mem_avail, cpu_usage)


class ThreadSafeDict(dict):
    def __init__(self, *p_arg, **n_arg):
        dict.__init__(self, *p_arg, **n_arg)
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, tp, value, traceback):  # pylint: disable=invalid-name
        del tp, value, traceback
        self._lock.release()


class ThreadSafeList(list):
    def __init__(self, *p_arg, **n_arg):
        list.__init__(self, *p_arg, **n_arg)
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, type_p, value, traceback):
        del type_p, value, traceback
        self._lock.release()


class CmdExecutor:
    def __init__(self):
        self.active_tasks = ThreadSafeDict()
        self.run = False
        self.thread = None
        self._last_msg = []
        self._last_msg_lock = threading.Lock()

    def append(self, item):
        with Popen(item, shell=True) as process:
            with self.active_tasks as active_tasks:
                active_tasks[item] = process
                return process

    def __len__(self):
        with self.active_tasks as active_tasks:
            return len(active_tasks)

    @property
    def last_msg(self):
        with self._last_msg_lock:
            if len(self._last_msg) == 0:
                return None
            temp = self._last_msg
            self._last_msg = []
            return temp

    def start(self):
        self.run = True
        self.thread = threading.Thread(target=self._update_fn)
        self.thread.start()
        logger.bind(jsh=True).debug("Starting CmdExecutor")

    def stop(self):
        if self.run:
            self.run = False
            logger.bind(jsh=True).info("STOP CmdExecutor")
            with self.active_tasks as active_task:
                for cmd, task_process in active_task.items():
                    task_process.terminate()
                    logger.bind(jsh=True).info(
                        f"Work Term PID:{task_process.pid}\t {cmd}"
                    )
                active_task.clear()
            self.thread.join()
            self.thread = None
            with self.active_tasks as active_task:
                for cmd, task_process in active_task.items():
                    task_process.terminate()
                    logger.bind(jsh=True).info(
                        f"Work Term PID:{task_process.pid}\t {cmd}"
                    )
                active_task.clear()

    def join(self):
        if self.run:
            while len(self) > 0:
                time.sleep(3)
            self.stop()

    def _update_fn(self):
        while self.run:
            self._check_cmds()
            time.sleep(2)

    def _check_one_cmd(self, str_cmd, task_process):
        poll_state = task_process.poll()
        if poll_state is None:
            return False
        with self._last_msg_lock:
            self._last_msg.append((str_cmd, task_process, len(self.active_tasks) - 1))
        return True

    def _check_cmds(self):
        with self.active_tasks as active_task:
            del_tasks = []
            for cmd, task_process in active_task.items():
                if self._check_one_cmd(cmd, task_process):
                    del_tasks.append(cmd)
            for key in del_tasks:
                del self.active_tasks[key]


class Scheduler:  # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self._cmd_executor = CmdExecutor()
        self._task_buffer = ThreadSafeList()
        self.error_cmds = []
        self.num_total_task = 0
        self.num_finished_task = 0

        self.thread = None
        self.run = False

        self.interval_cpu_check = 2
        self.seconds_resouece_wait = 2
        self.cpu_usage_thres = 92
        self.mem_avail_thres = 0.9
        self.num_process = 4

    def reset(self):
        self._cmd_executor = CmdExecutor()
        self._task_buffer = ThreadSafeList()
        self.error_cmds = []
        self.num_total_task = 0
        self.num_finished_task = 0

        self.thread = None
        self.run = False

    def start(self):
        self.run = True
        self._cmd_executor.start()
        self.thread = threading.Thread(target=self._update_fn)
        self.thread.start()

    def stop(self):
        if self.run:
            logger.bind(jsh=True).info("STOP Scheduler")
            self.run = False
            self.thread.join()
            self.thread = None
            self._cmd_executor.stop()

    def join(self):
        if self.run:
            while len(self._task_buffer) > 0:
                time.sleep(3)
            self._cmd_executor.join()
        self.stop()

    def append(self, cmd: str):
        with self._task_buffer as buffer:
            buffer.append(cmd)
            logger.bind(jsh=True).debug(f"Scheduler recieves {cmd}")

    def _update_fn(self):
        logger.bind(jsh=True).debug("Entering Scheduler worker")
        while self.run:
            runenr_msg = self._cmd_executor.last_msg
            self._process_msg(runenr_msg)

            if len(self._task_buffer) == 0:
                time.sleep(3)
                continue

            if len(self._cmd_executor) == self.num_process:
                time.sleep(1)
                continue

            with self._task_buffer as buffer:
                str_cmd = buffer.pop()
            while not self.is_resource_ready():
                time.sleep(self.seconds_resouece_wait)
            process = self._cmd_executor.append(str_cmd)
            self.post_start_process(str_cmd, process)
        logger.bind(jsh=True).debug("Exiting Scheduler worker")

    def proc_event(self, cur_process):
        return f"\nPID:{cur_process.pid}. Active:{len(self._cmd_executor):>5d}\
             Finished/TODO/Error: {self.num_finished_task:>5d}/{len(self._task_buffer):>5d}/{len(self.error_cmds):>5d}"  # pylint: disable=line-too-long

    def log_state(self):
        logger.bind(jsh=True).debug(
            f"Active:{len(self._cmd_executor)}. Finished/TODO/Error: {self.num_finished_task:>5d}/{len(self._task_buffer):>5d}/{len(self.error_cmds):>5d}"  # pylint: disable=line-too-long
        )

    def _process_msg(self, msgs):
        if msgs:
            for msg in msgs:
                str_cmd, process, _ = msg
                poll_state = process.poll()
                if poll_state == 0:
                    self.post_work_cmd(str_cmd, process)
                else:
                    self.post_broken_cmd(str_cmd, process)

    def is_resource_ready(self):
        pc_state = get_pc_state(self.interval_cpu_check)
        state = pc_state.ready(self.cpu_usage_thres, self.mem_avail_thres)
        return state

    def post_start_process(self, str_cmd, process):
        self.num_total_task += 1
        logger.bind(jsh=True).info(f"\nFire {str_cmd}! {self.proc_event(process)}")

    def post_work_cmd(self, str_cmd, process):
        self.num_finished_task += 1
        logger.bind(jsh=True).info(f"\nFinish {str_cmd}! {self.proc_event(process)}")

    def post_broken_cmd(self, str_cmd, process):
        self.num_finished_task += 1
        poll_state = process.poll()
        logger.bind(jsh=True).critical(
            f"\nError:\t{poll_state:03d}!\n{str_cmd}{self.proc_event(process)}"
        )
        logger.bind(jsherror=True).debug(f"PID:{process.pid:>06d} {str_cmd}")

    def __exit__(self, *args, **kwargs):
        self.stop()


worker = None


def set_scheduler(pipe, identifier, inp=None):
    global worker
    idx = identifier.decode("ascii")
    logger.info(f"REQ SET from {idx}\n {kvformat(inp)}")
    for key, value in inp.items():
        try:
            setattr(worker, key, value)
        except Exception as error:  # pylint: disable=broad-except
            print(error)
    logger.info("SET Done")
    pipe.send(identifier, None)


def add_job(pipe, identifier, inp=None):
    global worker
    idx = identifier.decode("ascii")
    if isinstance(inp, str):
        inp = [inp]
    logger.info(f"REQ job from {idx}\n {stformat(inp)}")
    for job in inp:
        worker.append(job)
    logger.info("SET Done")
    pipe.send(identifier, None)


def kill_all(pipe, identifier, inp=None):
    del inp
    global worker
    idx = identifier.decode("ascii")
    logger.info(f"REQ kill_all from {idx}\n")
    worker.stop()
    worker.reset()
    worker.start()
    pipe.send(identifier, None)


def start_sever():
    global worker
    worker = Scheduler()
    worker.start()
    try:
        server = ServerPipe("server", mode="tcp")
        server.dispatcher.register("set", set_scheduler)
        server.dispatcher.register("job", add_job)
        server.dispatcher.register("killf", kill_all)
        p_router = jam_getenv("ShExecutor", default=1089, type=int)
        assert not is_port_used(p_router)
        assert not is_port_used(p_router + 1)
        with server.activate(tcp_port=[str(p_router), str(p_router + 1)]):
            logger.critical("GPU ROUTER and PULL IP:")
            logger.critical(server.conn_info)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()


def instantiate_client(flag="_"):
    p_dealer = jam_getenv("ShExecutor", default=1089, type=int)
    tcp_dealer = f"tcp://{get_local_addr()}:{p_dealer}"
    tcp_push = f"tcp://{get_local_addr()}:{p_dealer+1}"
    client = ClientPipe(
        f"gpu_client{flag}" + uuid.uuid4().hex[:8], conn_info=[tcp_dealer, tcp_push]
    )
    return client


def echo_hello():
    client = instantiate_client()
    logger.info("Identity: {}.".format(client.identity))
    with client.activate():
        client.query("job", f"echo Hello from {socket.gethostname()} && pwd")


if __name__ == "__main__":
    worker = Scheduler()
    worker.start()
    for i in range(7):
        worker.append(f"sleep {i*2}")
    time.sleep(1)
    worker.stop()

import argparse
import os
import signal
import socket
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from subprocess import Popen
from typing import List, Optional, Union

import gpustat
import psutil

from jammy.comm import get_local_addr, is_port_used
from jammy.comm.cs import ClientPipe, ServerPipe
from jammy.logging import get_logger
from jammy.utils.env import jam_getenv
from jammy.utils.printing import kvformat, stformat

logger = get_logger(
    os.path.expanduser("~/jsh.log"),
    level="DEBUG",
    rotation="10 MB",
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
    rotation="10 MB",
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


@dataclass
class GPUState:
    query: gpustat.core.GPUStatCollection

    def ready(self, i_th: int, gpu_usage_thres, mem_avail_thres):
        gpu_info = self.query[i_th]
        gpu_util_ready = gpu_usage_thres > gpu_info["utilization.gpu"]
        gpu_mem_ready = (
            mem_avail_thres
            < (gpu_info["memory.total"] - gpu_info["memory.used"]) / 1024.0
        )
        return gpu_util_ready and gpu_mem_ready


@dataclass
class ProcTask:
    cmd: str
    proc: Popen
    gpus: list
    pid: int = field(init=False)

    def __post_init__(self):
        self.pid = self.proc.pid

    def poll(self):
        return self.proc.poll()


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

    def append(self, item, gpus: Optional[Union[None, int, list]] = None):
        # https://jorgenmodin.net/index_html/Link---unix---Killing-a-subprocess-including-its-children-from-python---Stack-Overflow
        # os.setsid assign a anew process group
        gpu_prefix = ""
        if gpus:
            gpu_prefix = "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, gpus)) + " "
        str_cmd = gpu_prefix + item
        process = (
            Popen(  # pylint: disable=consider-using-with, subprocess-popen-preexec-fn
                str_cmd, shell=True, preexec_fn=os.setsid
            )
        )
        cur_task = ProcTask(str_cmd, process, gpus=gpus)
        # with Popen(item, shell=True) as process:
        with self.active_tasks as active_tasks:
            active_tasks[str_cmd] = cur_task
            return cur_task

    def __len__(self):
        return len(self.active_tasks)

    @property
    def last_msg(self) -> List[ProcTask]:
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
                    os.killpg(os.getpgid(task_process.pid), signal.SIGKILL)
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

    def _check_one_cmd(self, cmd: str, proc_task: ProcTask):
        # TODO: we actually do not need dict for task
        del cmd
        poll_state = proc_task.proc.poll()
        if poll_state is None:
            return False
        with self._last_msg_lock:
            self._last_msg.append(proc_task)
        return True

    def _check_cmds(self):
        del_tasks = []
        for cmd, proc_task in self.active_tasks.items():
            if self._check_one_cmd(cmd, proc_task):
                del_tasks.append(cmd)
        with self.active_tasks as active_tasks:
            for key in del_tasks:
                del active_tasks[key]

    def __str__(self):
        return "Active tasks\n" + stformat(
            {proc.pid: proc.cmd for proc in self.active_tasks.values()}
        )


@dataclass
class SchedulerSetting:  # pylint: disable=too-many-instance-attributes
    seconds_resouece_wait: float = 2.0
    interval_cpu_check: int = 1
    cpu_usage_thres: float = 92
    mem_avail_thres: float = 0.9
    cpu_proc_upper: int = 4

    gpu_proc_upper: int = 2
    gpu_usage_thres: float = 80
    gpu_mem_avail_thres: float = 0.4


class Scheduler:  # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self._cmd_executor = CmdExecutor()
        self._task_buffer = ThreadSafeList()

        self.num_total_task = 0
        self.num_finished_task = 0
        self.num_error_task = 0

        self.thread = None
        self.run = False

        self.cfg = SchedulerSetting()

        # gpus
        self.gpus_num_proc = defaultdict(lambda: 0)

    def reset(self):
        self._cmd_executor = CmdExecutor()
        self._task_buffer = ThreadSafeList()

        self.num_total_task = 0
        self.num_finished_task = 0
        self.num_error_task = 0

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

    def append(self, cmd: str, gpus: Optional[Union[None, int, list]] = None):
        with self._task_buffer as buffer:
            if isinstance(gpus, int):
                gpus = [gpus]
            buffer.append((cmd, gpus))
            logger.bind(jsh=True).debug(f"Scheduler recieves {cmd} on GPUs: {gpus}")

    def _update_fn(self):
        logger.bind(jsh=True).debug("Entering Scheduler worker")
        while self.run:
            runner_msgs = self._cmd_executor.last_msg
            self._process_msg(runner_msgs)

            if len(self._task_buffer) == 0:
                time.sleep(3)
                continue

            if not self.is_cpu_ready():
                time.sleep(self.cfg.seconds_resouece_wait)
                continue

            gpu_ready = False
            cur_i = 0
            for cur_i, cur_task in enumerate(self._task_buffer):
                str_cmd, req_gpus = cur_task
                req_gpus = self.check_gpu_input(req_gpus)  # check if need auto tune gpu
                if self.is_gpu_ready(req_gpus):
                    gpu_ready = True
                    break
            if gpu_ready:
                with self._task_buffer as buffer:
                    del buffer[cur_i]

                proc_task = self._cmd_executor.append(str_cmd, req_gpus)
                self.post_start_process(proc_task)
            else:
                time.sleep(1)
        logger.bind(jsh=True).debug("Exiting Scheduler worker")

    def event_info(self, proc_task: Union[ProcTask, None] = None):
        if proc_task:
            return f"\nPID:{proc_task.pid}. Active:{len(self._cmd_executor):>5d}\
                Finished/TODO/Error: {self.num_finished_task:>5d}/{len(self._task_buffer):>5d}/{self.num_error_task:>5d}"  # pylint: disable=line-too-long
        return f"Finished/TODO/Error: {self.num_finished_task:>5d}/{len(self._task_buffer):>5d}/{self.num_error_task:>5d}"  # pylint: disable=line-too-long

    def log_state(self):
        logger.bind(jsh=True).debug(
            f"Active:{len(self._cmd_executor)}. Finished/TODO/Error: {self.num_finished_task:>5d}/{len(self._task_buffer):>5d}/{self.num_error_task:>5d}"  # pylint: disable=line-too-long
        )

    def _process_msg(self, msgs: List[ProcTask]) -> None:
        if msgs:
            for proc_task in msgs:
                poll_state = proc_task.poll()
                if poll_state == 0:
                    self.post_work_cmd(proc_task)
                else:
                    self.post_broken_cmd(proc_task)

    def is_cpu_ready(self):
        pc_state = get_pc_state(self.cfg.interval_cpu_check)
        state = pc_state.ready(self.cfg.cpu_usage_thres, self.cfg.mem_avail_thres)
        is_lower_upper_proc = len(self._cmd_executor) < self.cfg.cpu_proc_upper
        return state and is_lower_upper_proc

    def is_gpu_ready(self, gpus: List[int] = None):
        if gpus:
            gpu_state = GPUState(gpustat.new_query())
            for cur_id in gpus:
                if self.gpus_num_proc[cur_id] >= self.cfg.gpu_proc_upper:
                    return False

                if not gpu_state.ready(
                    cur_id, self.cfg.gpu_usage_thres, self.cfg.gpu_mem_avail_thres
                ):
                    return False

        return True

    def check_gpu_input(self, gpus: List[int] = None):
        if gpus:
            if len(gpus) == 1 and gpus[-1] < 0:
                rtn_gpu = self.select_gpu(abs(gpus[-1]))
                if rtn_gpu:
                    return rtn_gpu
        return gpus

    def select_gpu(self, num_gpu: int = 2):
        rtn = []
        gpu_state = GPUState(gpustat.new_query())
        for cur_id in range(len(gpu_state.query)):
            if self.gpus_num_proc[cur_id] >= self.cfg.gpu_proc_upper:
                continue

            if not gpu_state.ready(
                cur_id, self.cfg.gpu_usage_thres, self.cfg.gpu_mem_avail_thres
            ):
                continue
            rtn.append(cur_id)
        if len(rtn) < num_gpu:
            return None
        logger.bind(jsh=True).debug(f"Select GPUs : {rtn[:num_gpu]}")
        return rtn[:num_gpu]

    def post_start_process(self, proc_task: ProcTask):
        self.num_total_task += 1
        if proc_task.gpus:
            for item in proc_task.gpus:
                self.gpus_num_proc[item] += 1
        logger.bind(jsh=True).info(
            f"\nFire {proc_task.cmd}! {self.event_info(proc_task)}"
        )

    def post_work_cmd(self, proc_task: ProcTask):
        self.num_finished_task += 1
        if proc_task.gpus:
            for item in proc_task.gpus:
                self.gpus_num_proc[item] -= 1
        logger.bind(jsh=True).info(
            f"\nFinish {proc_task.cmd}! {self.event_info(proc_task)}"
        )

    def post_broken_cmd(self, proc_task: ProcTask):
        self.num_finished_task += 1
        if proc_task.gpus:
            for item in proc_task.gpus:
                self.gpus_num_proc[item] -= 1

        poll_state = proc_task.poll()
        logger.bind(jsh=True).critical(
            f"\nError:\t{poll_state:03d}!\n{proc_task.cmd}{self.event_info(proc_task)}"
        )
        logger.bind(jsherror=True).debug(f"PID:{proc_task.pid:>06d} {proc_task.cmd}")

    def __exit__(self, *args, **kwargs):
        self.stop()

    def __str__(self):
        return (
            "jsh setting\n"
            + stformat(asdict(self.cfg))
            + "\n\n TODO\n"
            + stformat(self._task_buffer)
            + self.event_info()
            + "\n\n"
            + str(self._cmd_executor)
            + "GPUs\n"
            + stformat(dict(self.gpus_num_proc))
        )


worker = None


def set_scheduler(pipe, identifier, inp=None):
    global worker
    idx = identifier.decode("ascii")
    logger.info(f"REQ SET from {idx}\n {kvformat(inp)}")
    for key, value in inp.items():
        try:
            setattr(worker.cfg, key, value)
        except Exception as error:  # pylint: disable=broad-except
            print(error)
    logger.debug("SET Done")
    pipe.send(identifier, None)


def add_job(pipe, identifier, inp=None):
    global worker
    idx = identifier.decode("ascii")
    if isinstance(inp, (str, tuple)):
        inp = [inp]
    logger.info(f"REQ job from {idx}\n {stformat(inp)}")
    for job in inp:
        if isinstance(job, str):
            worker.append(job)
        else:
            worker.append(job[0], job[1])
    logger.debug("Add work done")
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


def response_state(pipe, identifier, inp=None):
    del inp
    global worker
    idx = identifier.decode("ascii")
    logger.info(f"REQ state from {idx}\n")
    pipe.send(identifier, str(worker))


def start_sever():
    parser = argparse.ArgumentParser(prog="jshs: start a shell executor server")
    parser.add_argument("-p", "--port", type=int, default=-1, help="set default port")
    args = parser.parse_args()

    global worker
    worker = Scheduler()
    worker.start()
    try:
        server = ServerPipe("server", mode="tcp")
        server.dispatcher.register("set", set_scheduler)
        server.dispatcher.register("job", add_job)
        server.dispatcher.register("killf", kill_all)
        server.dispatcher.register("state", response_state)
        if args.port < -1:
            p_router = jam_getenv("ShExecutor", default=1089, type=int)
            logger.warning(f"Use default port {p_router}")
        else:
            p_router = args.port
            logger.warning(f"Use passed Port {p_router}")
        assert not is_port_used(p_router)
        assert not is_port_used(p_router + 1)
        with server.activate(tcp_port=[str(p_router), str(p_router + 1)]):
            logger.critical("GPU ROUTER and PULL IP:")
            logger.critical(server.conn_info)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()


def instantiate_client(flag="_", p_dealer=-1):
    if p_dealer < 0:
        p_dealer = jam_getenv("ShExecutor", default=1089, type=int)
        logger.warning(f"Use default port {p_dealer}")
    tcp_dealer = f"tcp://{get_local_addr()}:{p_dealer}"
    tcp_push = f"tcp://{get_local_addr()}:{p_dealer+1}"
    client = ClientPipe(
        f"gpu_client{flag}" + uuid.uuid4().hex[:8], conn_info=[tcp_dealer, tcp_push]
    )
    return client


def _shell_instantiate_client(port=-1):
    parser = argparse.ArgumentParser(prog="jsh client")
    parser.add_argument("-p", "--port", type=int, default=-1, help="set default port")
    args = parser.parse_args()
    if args.port > 0:
        port = args.port
        logger.warning(f"Use passed Port {args.port}")
    return instantiate_client(p_dealer=port)


def echo_hello():
    client = _shell_instantiate_client()
    with client.activate():
        client.query("job", f"echo Hello from {socket.gethostname()} && pwd")


def echo_state():
    client = _shell_instantiate_client()
    with client.activate():
        state_info = client.query("state")
        print(state_info)


def client_kill_all():
    client = _shell_instantiate_client()
    logger.info("Identity: {}.".format(client.identity))
    with client.activate():
        client.query("killf")


# TODO: adding state output and fzf kill

if __name__ == "__main__":
    worker = Scheduler()
    worker.start()
    for i in range(7):
        worker.append(f"sleep {i*2}")
    time.sleep(1)
    worker.stop()

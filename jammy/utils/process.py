import threading
import time
from subprocess import PIPE, Popen

import psutil

from jammy.logging import get_logger

__all__ = ["run_simple_command", "BashRunner"]


def run_simple_command(cmd: str):
    cmd = cmd.split(" ")
    with Popen(cmd, stdout=PIPE, stderr=PIPE) as prc:
        stdout, stderr = prc.communicate()
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")
        prc.terminate()
    return stdout, stderr


class ThreadSafeDict(dict):
    def __init__(self, *p_arg, **n_arg):
        dict.__init__(self, *p_arg, **n_arg)
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, type_p, value, traceback):
        del type_p, value, traceback
        self._lock.release()


def _get_mem():
    return psutil.virtual_memory().available / 1024 / 1024 / 1024


logger = get_logger()


class BashRunner:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        notifier=None,
        interval_cpu_check: float = 1.0,
        cpu_usage_thres: float = 92,
        seconds_numthread_wait: int = 10,
        mem_avail_thres: float = 0.9,
        seconds_resouece_wait: int = 2,
    ):
        self.notifier = notifier
        self.interval_cpu_check = interval_cpu_check
        self.cpu_usage_thres = cpu_usage_thres
        self.mem_avail_thres = mem_avail_thres
        self.seconds_numthread_wait = seconds_numthread_wait
        self.seconds_resouece_wait = seconds_resouece_wait

        self.complete_record = []
        self.active_task = ThreadSafeDict()
        self.error_task = {}
        self.num_total_task = 0

    def proc_event(self, cur_process):
        finished_num = len(self.complete_record) + len(self.error_task)
        return f"\nPID:{cur_process.pid}. Active:{len(self.active_task)}.\
             Finished: {finished_num}/{self.num_total_task}"

    def _execute_single_task(self, str_cmd):
        logger.debug("Checking states")
        while not (self._check_cpu() and self._check_mem()):
            time.sleep(self.seconds_resouece_wait)
        with Popen(str_cmd, shell=True) as process:  # , stdout=PIPE, stderr=PIPE)
            return process

    def _check_single_task(self, cmd, task_process):
        poll_state = task_process.poll()
        if poll_state is None:
            return False
        elif poll_state == 0:
            if cmd not in self.complete_record:
                self._print_finish(cmd, task_process)
        else:
            if cmd not in self.error_task:
                self._print_error(cmd, task_process)
        return True

    def reset(self, batch_cmd):
        self.num_total_task = len(batch_cmd)
        self.complete_record.clear()
        self.active_task.clear()
        self.error_task.clear()

    def execute_tasks(self, batch_cmd, num_thread=0, msg=None):
        self.reset(batch_cmd)
        if num_thread == 1:
            self.queue_tasks(batch_cmd, msg)
        else:
            thread_check = threading.Thread(target=self._check_process)
            thread_check.start()
            for task in batch_cmd:
                self.check_num_thread(num_thread)
                process = self._execute_single_task(task)
                self._print_start(task, process)
                with self.active_task as active_task:
                    active_task[task] = process
            thread_check.join()
            self._send_msg(msg)

    def queue_tasks(self, batch_cmd, msg=None):
        for task in batch_cmd:
            with Popen(task, shell=True) as process:
                self._print_start(task, process)
                process.communicate()
                if process.poll() == 0:
                    self._print_finish(task, process)
                else:
                    self._print_error(task, process)
        self._print_complete()
        self._send_msg(msg)

    def _send_msg(self, msg=None):
        if self.notifier is None:
            return
        if msg == 0:
            return
        else:
            if len(self.error_task) == 0:
                msg += "\n All tasks success!"
            else:
                for cmd, error_code in self.error_task.items():
                    msg += f"\nError: {error_code:03d}\t{cmd}"
            self.notifier.notify(msg)

    def check_num_thread(self, num_thread):
        if num_thread > 0:
            while len(self.active_task) >= num_thread:
                time.sleep(self.seconds_numthread_wait)

    def kill_all(self):
        for cmd, process in self.active_task.items():
            process.kill()
            logger.info(f"Kill {cmd}")

    def _check_process(self):
        logger.debug("Start monitor thread")
        is_exit = False
        while not is_exit:
            delete_active = []
            with self.active_task as active_task:
                for cmd, task_process in active_task.items():
                    if self._check_single_task(cmd, task_process):
                        delete_active.append(cmd)

            for key in delete_active:
                del self.active_task[key]

            time.sleep(5)
            if len(self.complete_record) + len(self.error_task) == self.num_total_task:
                self._print_complete()
                is_exit = True
        logger.debug("Exit check thread")

    def _check_cpu(self):
        logger.debug(f"cpu {psutil.cpu_percent(interval=self.interval_cpu_check)}")
        return self.cpu_usage_thres > psutil.cpu_percent(
            interval=self.interval_cpu_check
        )

    def _check_mem(self):
        logger.debug(f"mem {_get_mem()}")
        return self.mem_avail_thres < _get_mem()

    def _print_finish(self, cmd, task_process):
        self.complete_record.append(cmd)
        logger.info(f"Finish {cmd}! {self.proc_event(task_process)}")

    def _print_error(self, cmd, task_process):
        poll_state = task_process.poll()
        self.error_task[cmd] = poll_state
        logger.critical(
            f"Error:\t{poll_state:03d}! {self.proc_event(task_process)}\n{cmd}"
        )

    def _print_start(self, cmd, task_process):
        logger.info(f"Fire {cmd}.{self.proc_event(task_process)}")

    def _print_complete(self):
        logger.success(
            f"Complete tasks: {len(self.complete_record)}\tError tasks: {len(self.error_task)}"
        )

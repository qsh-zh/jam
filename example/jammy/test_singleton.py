from jammy.utils.meta import Singleton
import threading
import time
import random
import os
import multiprocessing as mp


class Earth(metaclass=Singleton):
    def __init__(self, name):
        self.name = name

    def __call__(self):
        print(threading.get_ident(), self.name)


def test_global():
    this_earth = Earth("abc")
    that_earth = Earth("def")
    this_earth()
    that_earth()


def create(name):
    earth = Earth(name)
    t_time = random.randint(1, 5)
    print(os.getpid(), threading.get_ident(), "sleep: ", t_time)
    time.sleep(t_time)
    earth()


def test_thread():
    N = 10
    thread_l = []
    for i in range(N):
        thread_l.append(threading.Thread(target=create, args=[f"THD Naming {i}"]))
    for i in range(N):
        thread_l[i].start()
    for i in range(N):
        thread_l[i].join()


def test_mp():
    N = 10
    mp_l = []
    for i in range(N):
        mp_l.append(mp.Process(target=create, args=[f"MP Naming {i}"]))
    for i in range(N):
        mp_l[i].start()
    for i in range(N):
        mp_l[i].join()


if __name__ == "__main__":
    test_global()
    Singleton._instances.clear()
    print("*" * 100)
    test_thread()
    Singleton._instances.clear()
    print("*" * 100)
    test_mp()

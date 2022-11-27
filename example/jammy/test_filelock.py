# pylint: skip-file
import time
from multiprocessing import Pool

from filelock import FileLock

w_file = "writing.log"


def work(pid):
    global w_file
    try:
        with FileLock(f"{w_file}.lock", 20) as flock:
            if flock.is_locked:
                with open(w_file, "a") as f:
                    f.write(f"{pid}\n")
                print(f"{pid} start sleep")
                time.sleep(5)
                print(f"{pid} end sleep")
            else:
                print(f"{pid} timeout")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    with Pool(5) as p:
        print(p.map(work, [1, 2, 3, 4, 5]))

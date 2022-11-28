# pylint: skip-file
import time
from multiprocessing import Pool

from filelock import FileLock

from jammy.utils.filelock import get_filelock

w_file = "writing.log"


def work(pid):
    global w_file
    with get_filelock(w_file, 10) as fl:
        if fl:
            with open(w_file, "a") as f:
                f.write(f"{pid}\n")
            print(f"{pid} start sleep")
            time.sleep(5)
            print(f"{pid} end sleep")


if __name__ == "__main__":
    with Pool(5) as p:
        print(p.map(work, [1, 2, 3, 4, 5]))

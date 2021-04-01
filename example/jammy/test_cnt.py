from jammy.utils.cnt import CBCnt, ExitCnt


def test_cbcnt():
    N = 10
    cnter = CBCnt(lambda: print("Callback starts running"), N)
    for i in range(3 * N):
        cnter(i % 2)
    for _ in range(3 * N):
        cnter(True)


def test_exitcnt():
    N = 10
    cnter = ExitCnt("Quit", N)
    for _ in range(3 * N):
        cnter(True)


if __name__ == "__main__":
    test_cbcnt()
    test_exitcnt()

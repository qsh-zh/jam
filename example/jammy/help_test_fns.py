import jammy.utils.meta as meta


def fart(*args, **kwargs):
    print("empty")


meta.run_once = fart

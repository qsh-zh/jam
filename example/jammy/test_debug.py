from jammy.utils.debug import decorate_exception_hook


# pylint: disable=invalid-name
@decorate_exception_hook
def divide_zero():
    a = 10
    b = a / 0
    return b


if __name__ == "__main__":
    divide_zero()

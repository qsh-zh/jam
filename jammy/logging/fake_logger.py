__all__ = ["fake_logger"]

class FakeLogger:
    def info(*args, **kwargs):
        pass

    def debug(*args, **kwargs):
        pass

    def trace(*args, **kwargs):
        pass

    def parse(*args, **kwargs):
        pass

    def warning(*args, **kwargs):
        pass

    def critical(*args, **kwargs):
        pass

    def log(*args, **kwargs):
        pass

    def error(*args, **kwargs):
        pass

    def success(*args, **kwargs):
        pass

fake_logger = FakeLogger()
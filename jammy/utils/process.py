from subprocess import PIPE, Popen

__all__ = ["run_simple_command"]


def run_simple_command(cmd: str):
    cmd = cmd.split(" ")
    with Popen(cmd, stdout=PIPE, stderr=PIPE) as prc:
        stdout, stderr = prc.communicate()
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")
        prc.terminate()
    return stdout, stderr

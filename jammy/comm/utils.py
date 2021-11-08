import socket

__all__ = ["find_free_port", "get_local_addr", "is_port_used"]


def find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


# http://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
def get_local_addr():
    try:
        resolve = [
            ip
            for ip in socket.gethostbyname_ex(socket.gethostname())[2]
            if not ip.startswith("127.")
        ][:1]
        if len(resolve):
            return resolve[0]
        # AF_INET: Address family, ipv4. SOCK_DGRAM: connections, unreliable, datagrams UDP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        addr = s.getsockname()[0]
        s.close()
        return addr
    except Exception:  # pylint: disable=broad-except
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:  # pylint: disable=broad-except
            return "127.0.0.1"


def is_port_used(port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

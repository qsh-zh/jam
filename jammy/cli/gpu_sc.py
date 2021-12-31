import os
import time
import uuid

import gpustat

from jammy.comm import get_local_addr, is_port_used
from jammy.comm.cs import ClientPipe, ServerPipe
from jammy.logging import get_logger
from jammy.utils import gpu
from jammy.utils.env import jam_getenv
from jammy.utils.printing import kvformat

logger = get_logger()


def req_util(pipe, identifier, inp=None):
    num_gpus = inp.get("num_gpus", 1)
    sleep_sec = inp.get("sleep_sec", 3)
    idx = identifier.decode("ascii")
    logger.info(f"REQ from {idx}\n {kvformat(inp)}")
    gpu_list = gpu.gpu_by_util()
    msg = None
    if len(gpu_list) < num_gpus:
        msg = gpu_list
    else:
        msg = gpu_list[:num_gpus]
    pipe.send(identifier, msg)
    logger.info(f"SEND: {idx} {msg}")
    time.sleep(sleep_sec)
    logger.info(f"CLOSE: {idx}")


def len_available_gpu(pipe, identifier, inp=None):
    del inp
    pipe.send(identifier, len(gpustat.new_query()))


def start_sever():
    server = ServerPipe("server", mode="tcp")
    server.dispatcher.register("req_util", req_util)
    server.dispatcher.register("total", len_available_gpu)
    p_router = jam_getenv("GPUPort", default=1080, type=int)
    assert not is_port_used(p_router)
    assert not is_port_used(p_router + 1)
    with server.activate(tcp_port=[str(p_router), str(p_router + 1)]):
        logger.critical("GPU ROUTER and PULL IP:")
        logger.critical(server.conn_info)
        while True:
            time.sleep(1)


def instantiate_client(flag="_"):
    p_dealer = jam_getenv("GPUPort", default=1080, type=int)
    tcp_dealer = f"tcp://{get_local_addr()}:{p_dealer}"
    tcp_push = f"tcp://{get_local_addr()}:{p_dealer+1}"
    client = ClientPipe(
        f"gpu_client{flag}" + uuid.uuid4().hex[:8], conn_info=[tcp_dealer, tcp_push]
    )
    return client


def start_client():
    client = instantiate_client()
    logger.info("Identity: {}.".format(client.identity))
    with client.activate():
        echo = client.query("total")
        logger.info(f"Contains {echo} gpus")
        query_gpu_ids = client.query("req_util", dict(num_gpus=1))
        logger.info(f"req get {query_gpu_ids}")


def get_gpu_by_utils(num_gpus: int = 1, sleep_sec: int = 3):
    client = instantiate_client(os.getpid())
    with client.activate():
        inp = {
            "num_gpus": num_gpus,
            "sleep_sec": sleep_sec,
        }
        query_gpu_ids = client.query("req_util", inp)
    return query_gpu_ids

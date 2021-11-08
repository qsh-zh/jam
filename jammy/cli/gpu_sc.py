import os
import time
import uuid

import gpustat

from jammy.comm import get_local_addr, is_port_used
from jammy.comm.cs import ClientPipe, ServerPipe
from jammy.utils import gpu
from jammy.utils.env import jam_getenv


def req_util(pipe, identifier, num_gpu: int = 1):
    gpu_list = gpu.gpu_by_util()
    msg = None
    if len(gpu_list) < num_gpu:
        msg = gpu_list
    else:
        msg = gpu_list[:num_gpu]
    pipe.send(identifier, msg)
    time.sleep(3)


def len_available_gpu(pipe, identifier, *args):
    del args
    pipe.send(identifier, len(gpustat.new_query()))


def start_sever():
    server = ServerPipe("server", mode="tcp")
    server.dispatcher.register("req_util", req_util)
    server.dispatcher.register("total", len_available_gpu)
    p_router = jam_getenv("GPUPort", default=1080, type=int)
    assert not is_port_used(p_router)
    assert not is_port_used(p_router + 1)
    with server.activate(tcp_port=[str(p_router), str(p_router + 1)]):
        print("GPU ROUTER and PULL IP:")
        print(server.conn_info)
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
    print("Identity: {}.".format(client.identity))
    with client.activate():
        echo = client.query("total")
        print(f"Contains {echo} gpus")
        query_gpu_ids = client.query("req_util", 1)
        print(f"req get {query_gpu_ids}")


def get_gpu_by_utils(num_gpus: int = 1):
    client = instantiate_client(os.getpid())
    with client.activate():
        query_gpu_ids = client.query("req_util", num_gpus)
    return query_gpu_ids

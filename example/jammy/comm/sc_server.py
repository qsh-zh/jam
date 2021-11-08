#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : sc_server.py
# Author : Jiayuan Mao, Qsh.Zh
# Email  : qsh.zh27@gmail.com
# Date   : 11/17/2021
#
# Qinsheng modifies based on Jacinle.
# Distributed under terms of the MIT license.

import time

from jammy.comm.cs import ServerPipe


def answer(pipe, identifier, inp):
    out = inp["a"] + inp["b"]
    pipe.send(identifier, dict(out=out))


def main():
    server = ServerPipe("server", mode="ipc")
    server.dispatcher.register("calc", answer)
    with server.activate():
        print("Client command:")
        print(
            "jam-run sc_client.py", *server.conn_info  # pylint: disable=not-an-iterable
        )
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()

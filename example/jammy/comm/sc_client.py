#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : sc_client.py
# Author : Jiayuan Mao, Qsh.Zh
# Email  : qsh.zh27@gmail.com
# Date   : 11/17/2021
#
# Qinsheng modifies based on Jacinle.
# Distributed under terms of the MIT license.

import sys
import time
import uuid

from jammy.comm.cs import ClientPipe


def main():
    client = ClientPipe("client" + uuid.uuid4().hex[:8], conn_info=sys.argv[1:3])
    print("Identity: {}.".format(client.identity))
    with client.activate():
        in_ = dict(a=1, b=2)
        out = client.query("calc", in_)
        print("Success: input={}, output={}".format(in_, out))
        time.sleep(1)


if __name__ == "__main__":
    main()

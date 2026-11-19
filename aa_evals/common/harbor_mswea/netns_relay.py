#!/usr/bin/env python3
"""Docker compose `network_mode: "service:<X>"` for harbor_ns: localhost ports of service X reachable from a
service that should share X's network namespace (TB 4.0: ctr-optimization `main` -> `api`, nextjs-performance
`main` -> `warehouse-api`).

  netns_relay.py LISTEN_PID TARGET_PID      (run inside the pod's user namespace; ns_env.py starts it)

harbor_ns gives every compose service its own network namespace, so `localhost:5000` in main does not reach X.
This relay listens on 127.0.0.1:P and [::1]:P in LISTEN_PID's network namespace for every TCP port P that the
target namespace listens on (rediscovered every 2 s), and forwards each connection to 127.0.0.1:P inside the
target namespace (sockets stay in the namespace they were created in; os.setns switches between them).
Not emulated: the reverse direction (X reaching main's localhost ports) and a shared IP address.
"""
import asyncio
import os
import socket
import sys

LISTEN = '0A'


def listening_ports(pid: int) -> set[int]:
    ports = set()
    for name in ('tcp', 'tcp6'):
        try:
            lines = open(f'/proc/{pid}/net/{name}').read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            if len(f) > 3 and f[3] == LISTEN:
                ports.add(int(f[1].rsplit(':', 1)[1], 16))
    return ports


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            writer.close()
        except OSError:
            pass


async def main() -> None:
    listen_pid, target_pid = int(sys.argv[1]), int(sys.argv[2])
    ns_listen = os.open(f'/proc/{listen_pid}/ns/net', os.O_RDONLY)
    ns_target = os.open(f'/proc/{target_pid}/ns/net', os.O_RDONLY)
    os.setns(ns_target, os.CLONE_NEWNET)   # resting namespace: outgoing sockets are created in the target's
    served: set[int] = set()

    async def handle(port: int, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            tr, tw = await asyncio.open_connection('127.0.0.1', port)   # created in the target namespace
        except OSError:
            w.close()
            return
        await asyncio.gather(pipe(r, tw), pipe(tr, w))

    while True:
        for port in sorted(listening_ports(target_pid) - served - listening_ports(listen_pid)):
            socks = []
            os.setns(ns_listen, os.CLONE_NEWNET)
            try:
                for fam, addr in ((socket.AF_INET, '127.0.0.1'), (socket.AF_INET6, '::1')):
                    s = socket.socket(fam, socket.SOCK_STREAM)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        s.bind((addr, port))
                        s.listen(128)
                        socks.append(s)
                    except OSError:
                        s.close()
            finally:
                os.setns(ns_target, os.CLONE_NEWNET)
            for s in socks:
                await asyncio.start_server(lambda r, w, p=port: handle(p, r, w), sock=s)
            served.add(port)
            print(f'relay: localhost:{port} -> target namespace ({len(socks)} listeners)', flush=True)
        await asyncio.sleep(2)


if __name__ == '__main__':
    asyncio.run(main())

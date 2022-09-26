"""
file: service.py
socket service
"""

import socket
import threading
import sys
import struct


def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # s.bind(('192.168.0.126', 8888))
        s.bind(('127.0.0.1', 8888))
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print('Waiting connection...')

    conn, addr = s.accept()
    t = threading.Thread(target=deal_data, args=(conn, addr))
    t.start()


def deal_data(conn, addr):
    print('Accept new connection from {0}'.format(addr))
    print(len(create_data()))

    conn.send(create_data())
    conn.close()


def create_data():
    pitch = struct.pack('<L', 13)
    yaw = struct.pack('<L', 23)
    pull = struct.pack('<L', 33)

    pitch += struct.pack('<L', 44)
    yaw += struct.pack('<L', 56)
    pull += struct.pack('<L', 155)

    pitch += struct.pack('<L', 65)
    yaw += struct.pack('<L', 19)
    pull += struct.pack('<L', 76)

    pitch += struct.pack('<L', 365)
    yaw += struct.pack('<L', 23)
    pull += struct.pack('<L', 33)
    print(len(pitch))
    print(len(yaw))
    print(len(pull))
    for i in range(int(len(pitch) / 4), 100):
        pitch += struct.pack('<L', 0)
    for i in range(int(len(yaw) / 4), 100):
        yaw += struct.pack('<L', 0)
    for i in range(int(len(pull) / 4), 100):
        pull += struct.pack('<L', 0)

    return pitch + yaw + pull


if __name__ == '__main__':
    # socket_service()
    create_data()

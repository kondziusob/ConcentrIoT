#!/usr/bin/env python

import socket
import _thread as thread

class TCPRequest(object):
    def __init__(self, remote_socket, remote_addr, local_socket, local_addr, buffersize, content=b""):
        self.remote_addr = remote_addr
        self.remote_socket = remote_socket
        self.local_socket = local_socket
        self.local_addr = local_addr
        self.buffersize = buffersize

        self.content = content

    def more(self):
        self.content += self.remote_socket.recv(self.buffersize)

        return self.content

    def receive(self):
        self.content = b""

        return self.more()

    def close(self):
        self.remote_socket.close()

class TCPResponse(object):
    def __init__(self, remote_socket, remote_addr, local_socket, local_addr):
        self.remote_socket = remote_socket
        self.remote_addr = remote_addr
        self.local_socket = local_socket
        self.local_addr = local_addr

    def write(self, data):
        self.remote_socket.sendall(data)

    def write_string(self, data, encoding="utf-8"):
        self.remote_socket.sendall(bytes(data, encoding))

    def close(self):
        self.remote_socket.close()

class TCPServer(object):
    DEFAULT_TCP_IP = '127.0.0.1'
    DEFAULT_TCP_PORT = 44542
    DEFAULT_BUFFERSIZE = 1024
    DEFAULT_MAX_REFUSED = 2

    def __init__(self, ip=DEFAULT_TCP_IP, 
        port=DEFAULT_TCP_PORT, buffersize=DEFAULT_BUFFERSIZE, 
        max_refused=DEFAULT_MAX_REFUSED):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((ip, port))
        s.listen(max_refused)

        self._socket = s
        self.addr = (ip, port)
        self.buffersize = buffersize
        self.max_refused = max_refused

    def listen(self, handler, connection_handler=None):
        def _request_handler(conn, addr):
            req = TCPRequest(conn, addr, self._socket, self.addr, self.buffersize)
            res = TCPResponse(conn, addr, self._socket, self.addr)

            if callable(connection_handler):
                connection_handler(conn, addr)
            elif connection_handler is not None:
                raise ValueError("Connection handler must be a function or None.")

            if not callable(handler):
                raise ValueError("Handler must be a function.")

            finished = False
            while not finished:
                finished = handler(req, res)

            conn.close()

        print("Server listening on: {}:{}".format(self.addr[0], self.addr[1]))
        
        while True:
            conn, addr = self._socket.accept()
            thread.start_new_thread(_request_handler, (conn, addr))

    def close(self):
        self._socket.close()
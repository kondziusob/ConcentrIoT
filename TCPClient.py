#!/usr/bin/env python
import socket

class TCPClient(object):
    DEFAULT_BUFFERSIZE = 1024

    def __init__(self, ip, port, buffersize=DEFAULT_BUFFERSIZE):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((ip, port))

        self._socket = s
        self.addr = (ip, port)
        self.buffersize = buffersize

    def write_string(self, content, encoding='utf-8'):
        self._socket.sendall(bytes(content, encoding))

    def write(self, content):
        self._socket.sendall(content)

    def receive(self):
        self.content = ""

        return self.more()

    def more(self):
        self.content += self._socket.recv(self.buffersize)

        return self.content
     
    def close(self):
        self._socket.close()
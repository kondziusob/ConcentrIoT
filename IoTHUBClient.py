from TCPClient import *
from IoTHUBMessage import *
from setInterval import *
import socket

class IoTHUBClient(object):
	VERSION = '1.0'

	def __init__(self, ip='127.0.0.1', port=69420, uri=None):
		self.ip = ip
		self.port = port

		self.uri = uri if uri is not None else "{}@{}".format('root', ip)

		hub_found = False
		while not hub_found:
			hub_found = self.lookup_hub()

		self._client = TCPClient(self.hub[0], self.hub[1])

		self.intervals = []

		self.set_interval(self.keepalive, 3)

	def write(self, rs):
		rs = IoTHUBMessage.from_string(rs)

		self._client.write_string(rs)

	def read(self):
		response_content = self.parse_request(self._client.receive())

		while not response_content.params_received():
			response_content = self.parse_request(self._client.more())


	def interval(self, interval):
		def inner(callback):
			self.set_interval(interval, callback)

		return interval

	def set_interval(self, interval, callback):
		self.intervals.append(setInterval(callback, interval))

	def lookup_hub(self):
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

		s.sendto(bytes('DISCOVER {} {}'.format(self.uri, self.VERSION), 'utf-8'), ('255.255.255.255', 44045))

		s.settimeout(.02)
		try:
			(msg, _) = s.recvfrom(1024)

			print(msg.decode('utf-8'))
			msg = msg.decode('utf-8').split('@')[-1].split(':')

			self.hub = (msg[0], int(msg[1]))

			return True
		except:
                        self.lookup_hub()

	def keepalive(self):
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

		s.sendto(bytes('DISCOVER {} {}'.format(self.uri, self.VERSION), 'utf-8'), ('255.255.255.255', 44045))

		s.settimeout(.02)
		try:
			(msg, _) = s.recvfrom(1024)

			print(msg.decode('utf-8'))
			msg = msg.decode('utf-8').split('@')[-1].split(':')

			self.hub = (msg[0], int(msg[1]))

			return True
		except:
			raise Exception("HUB got disconnected while at work.")

	def parse_response(self, response):
		content = IoTHUBMessage.from_string(content.decode('utf-8'))

		return content

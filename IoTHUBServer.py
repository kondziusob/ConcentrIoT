from TCPServer import TCPServer, TCPRequest, TCPResponse
from IoTHUBMessage import IoTHUBMessage
import typing

IoTHUBMessage = IoTHUBMessage

class IoTHUBServer:
	VERSION = '1.0'

	def __init__(self, ip : str, port : int, uri : str = None):
		self._server = TCPServer(ip, port)

		self.uri = uri if uri is not None else "{}@{}".format(ip, ip)

		self.callbacks = []

	def connection_handler(self, conn, addr):
		pass

	def request_handler(self, request : TCPRequest, response : TCPResponse):
		try: 
			request_content = self.parse_request(request.receive())

			while not request_content.params_received():
				request_content = self.parse_request(request.more())

			callbacks = [value for key, value in self.callbacks if key == request_content.type]

			if len(callbacks) > 0:
				for cb in callbacks:
					cb(request, response)

				return True # handling has been done
			else:
				raise Exception("Operation {} not supported".format(request_content.type))

		except Exception as e:
			res = IoTHUBMessage()
			res.type = "ERROR"
			res.uri = self.uri
			res.protocol_version = self.VERSION
			res.sender = IoTHUBMessage.parse_address(response.local_addr)
			res.receiver = IoTHUBMessage.parse_address(request.remote_addr)
			res.ctype = "text/plain"
			res.parameters = {
				"message" : str(e)
			}

			response.write_string(str(res))

			return True #close the connection

	def on(self, operation):
		def inner(callback):
			self.callbacks.append((operation, callback))

		return inner

	def parse_request(self, content : bytes):
		content = IoTHUBMessage.from_string(content.decode('utf-8'))

		return content

	def listen(self):
		self._server.listen(self.request_handler, self.connection_handler)


if __name__ == "__main__":
	IoTHUBServer('127.0.0.1', 44044).listen()
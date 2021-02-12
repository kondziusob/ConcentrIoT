import typing

class IoTHUBMessage:
	def __init__(self, **kwargs):
		self.type = None
		self.uri = None
		self.protocol_version = None
		self.sender = None
		self.receiver = None
		self.ctype = None
		self.length = 0

		self.parameters = {}
		self._parameters = ""

		for key, value in kwargs.items():
			settattr(self, key, value)

	def params_received(self):
		return len(self._parameters) == self.length

	@classmethod
	def from_string(cls, source : str):
		try:
			msg = IoTHUBMessage()

			source = [s.strip() for s in source.split('\r\n')]

			msg.type = source[0].split(' ')[0]
			msg.uri = source[0].split(' ')[1]
			msg.protocol_version = source[0].split(' ')[2]
			msg.sender = source[1].split('Sender: ')[1]
			msg.receiver = source[2].split('Receiver: ')[1]
			msg.ctype = source[3].split('Content-Type: ')[1]
			msg.length = int(source[4].split('Content-Length: ')[1])
			#in case the incoming message has params created not according to
			#the protocol definition
			if len(source) >= 6:
				msg._parameters = ', '.join(source[5:])
				msg.parameters = dict([(s.split('=')[0], s.split('=')[1][1:-1]) 
					for s in (msg._parameters).split(', ')])

			return msg

		except:
			raise Exception("Incorrect message format.")

	@classmethod
	def parse_address(cls, addr):
		return ':'.join((addr[0], str(addr[1])))

	def __str__(self) -> str:
		params = ', '.join('{}=\'{}\''.format(k, v) for k, v 
				in self.parameters.items())

		return '\r\n'.join(
			[' '.join([self.type, self.uri, self.protocol_version]),
			' '.join(['Sender:', self.sender]),
			' '.join(['Receiver:', self.receiver]),
			' '.join(['Content-Type:', self.ctype]),
			' '.join(['Content-Length', str(len(params))]),
			params])

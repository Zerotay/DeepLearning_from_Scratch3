import numpy as np

class Variable:
	def __init__(self, data):
		if data is not None and not isinstance(data, np.ndarray):
			raise TypeError(f'{type(data)} not supported, only ndarray')

		self.data = data
		self.grad = None
		self.creator = None

	def set_creator(self, func):
		self.creator = func #자신의 창조자를 저장함

	def backward(self):
		if self.grad is None:
			self.grad = np.ones_like(self.data) # 맨 마지막 미분값만 1이 됨

		funcs = [self.creator]
		while funcs:
			f = funcs.pop()
			x, y = f.input, f.output
			x.grad = f.backward(y.grad)

			if x.creator is not None:
				funcs.append(x.creator)

def as_array(x):
	if np.isscalar(x): return np.array(x)
	return x

class Function:
	def __call__(self, input):
		x = input.data
		y = self.forward(x)
		output = Variable(as_array(y))
		output.set_creator(self)
		self.input = input
		self.output = output
		return output

	def forward(self, x):
		raise NotImplementedError()

	def backward(self, gy):
		raise NotImplementedError()

class Square(Function):
	def forward(self, x):
		return x ** 2

	def backward(self, gy):
		x = self.input.data
		gx = 2 * x * gy
		return gx

class Exp(Function):
	def forward(self, x):
		return np.exp(x)

	def backward(self, gy):
		x = self.input.data
		gx = np.exp(x) * gy
		return gx

def square(x):
	return Square()(x) #함수화 시켜버림
def exp(x):
	return Exp()(x)


x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)

x = Variable(np.array(1.0))
x = Variable(None)
# x = Variable(1.0)
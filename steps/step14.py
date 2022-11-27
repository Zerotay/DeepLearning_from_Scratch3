import numpy as np

class Variable:
	def __init__(self, data):
		if data is not None and not isinstance(data, np.ndarray):
			raise TypeError(f'{type(data)} not supported, only ndarray')

		self.data = data
		self.grad = None
		self.creator = None

	def set_creator(self, func):
		self.creator = func

	def backward(self):
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = [self.creator]
		while funcs:
			f = funcs.pop()
			gys = [output.grad for output in f.outputs]
			gxs = f.backward(*gys)
			if not isinstance(gxs, tuple): gxs = (gxs, )

			for x, gx in zip(f.inputs, gxs):
				if x.grad is None:
					x.grad = gx
				else:
					# x.grad = x.grad + gx
					x.grad += gx # 절못된 코드.

				if x.creator is not None:
					funcs.append(x.creator)

	def cleargrad(self):
		self.grad = None

def as_array(x):
	if np.isscalar(x): return np.array(x)
	return x

class Function:
	def __call__(self, *inputs):
		xs = [x.data for x in inputs]
		ys = self.forward(*xs)
		if not isinstance(ys, tuple): ys = (ys, )
		outputs = [Variable(as_array(y)) for y in ys]

		for output in outputs: output.set_creator(self)
		self.inputs = inputs
		self.outputs = outputs
		return outputs if len(outputs) > 1 else outputs[0]

	def forward(self, x):
		raise NotImplementedError()

	def backward(self, gy):
		raise NotImplementedError()

class Add(Function):
	def forward(self, x0, x1):
		y = x0 + x1
		return y
	def backward(self, gy):
		return gy, gy

class Square(Function):
	def forward(self, x):
		return x ** 2

	def backward(self, gy):
		x = self.inputs[0].data
		gx = 2 * x * gy
		return gx

class Exp(Function):
	def forward(self, x):
		return np.exp(x)

	def backward(self, gy):
		x = self.inputs[0].data
		gx = np.exp(x) * gy
		return gx

class Triple(Function):
	def forward(self, x):
		return x ** 3

	def backward(self, gy):
		x = self.inputs[0].data
		gx = 3 * (x ** 2) * gy
		return gx

def square(x): return Square()(x)
def exp(x): return Exp()(x)
def triple(x): return Triple()(x)
def add(x0, x1): return Add()(x0, x1)

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))


y = add(add(x0, x0), add(x1,x1))
y.backward()

print(y.data)
print(x0.grad, x1.grad, y.grad)

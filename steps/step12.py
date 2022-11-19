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
			x, y = f.input, f.output
			x.grad = f.backward(y.grad)

			if x.creator is not None:
				funcs.append(x.creator)

def as_array(x):
	if np.isscalar(x): return np.array(x)
	return x

class Function:
	def __call__(self, *inputs): #위치가변인자로 받겠다
		xs = [x.data for x in inputs]
		ys = self.forward(*xs) #언팩하겠다
		if not isinstance(ys, tuple): ys = (ys, )
		outputs = [Variable(as_array(y)) for y in ys]
		for output in outputs: output.set_creator(self)
		self.input = input
		self.outputs = outputs
		return outputs if len(outputs) > 1 else outputs[0]
		#출력값이 하나일 거라면, 그냥 그째로 받고 싶다!

	def forward(self, x):
		raise NotImplementedError()

	def backward(self, gy):
		raise NotImplementedError()

#간단하게 두 개를 받는 함수 구현
class Add(Function):
	def forward(self, x0, x1):
		y = x0 + x1
		return y

def add(x0, x1): return Add()(x0, x1)

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0,x1)
print(y.data)
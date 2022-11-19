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

		funcs = [self.creator] # 드디어 리스트로 쓰이는 이유가 나온다.
		while funcs:
			f = funcs.pop()
			gys = [output.grad for output in f.outputs]
			#원래는 y를 꺼내고, y.grad를 통해 역전파를 함.
			#원래는 바로 x.grad에 역전파한 값을 저장함. 근데 지금은 그리 안함
			gxs = f.backward(*gys)
			if not isinstance(gxs, tuple): gxs = (gxs, )
			#경우에 따라서는 묶어줘야 하기 때문!

			for x, gx in zip(f.inputs, gxs): #이 둘이 매칭이 된다는 건 사실 당연하다
				x.grad = gx #여기에서 비로소 값을 저장해준다.
				if x.creator is not None:
					funcs.append(x.creator) # 각 분기에 대해 역전파를 지속할 수 있도록!

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
		return gy, gy #각각 그대로 흘려보내주면 된다.

class Square(Function):
	def forward(self, x):
		return x ** 2

	def backward(self, gy):
		x = self.inputs[0].data #이제는 인덱싱을 해준다. 튜플로 묶여있으니
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

y = add(square(x0), triple(x1))
y.backward()
print(y.data)
print(x0.grad)
print(x1.grad)
import numpy as np
import heapq as hq
import weakref
import contextlib

class Config:
	enable_backprop = True

class Variable:
	def __init__(self, data, name=None):
		if data is not None and not isinstance(data, np.ndarray):
			raise TypeError(f'{type(data)} not supported, only ndarray')

		self.data = data
		self.name = name
		self.grad = None
		self.creator = None
		self.generation = 0

	def set_creator(self, func):
		self.creator = func
		self.generation = func.generation + 1

	def backward(self, retain_grad=False):
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = []
		seen_set = set()

		def add_func(f):
			if f not in seen_set:
				hq.heappush(funcs, (-f.generation, str(f), f))
				seen_set.add(f)

		add_func(self.creator)

		while funcs:
			_, _, f = hq.heappop(funcs)

			gys = [output().grad for output in f.outputs]
			gxs = f.backward(*gys)
			if not isinstance(gxs, tuple): gxs = (gxs, )

			for x, gx in zip(f.inputs, gxs):
				if x.grad is None: x.grad = gx
				else: x.grad = x.grad + gx

				if x.creator is not None:
					add_func(x.creator)

			if not retain_grad:
				for y in f.outputs:
					y().grad = None

	def cleargrad(self):
		self.grad = None

	@property
	def shape(self):
		return self.data.shape
	@property
	def ndim(self):
		return self.data.ndim
	@property
	def size(self):
		return self.data.size
	@property
	def dtype(self):
		return self.data.dtype

	def __len__(self):
		return len(self.data)

	def __repr__(self):
		if self.data is None:
			return 'variable(None)'
		print(str(123456))
		p = str(self.data).replace('\n', '\n' + ' ' * 9)
		return f'variable({p})'

	def __str__(self):
		return (self.name)

	def __mul__(self, other):
		return mul(self, other)

	def __add__(self, other):
		return add(self, other)


def as_array(x):
	if np.isscalar(x): return np.array(x)
	return x

class Function:
	def __call__(self, *inputs):
		xs = [x.data for x in inputs]
		ys = self.forward(*xs)
		if not isinstance(ys, tuple): ys = (ys, )
		outputs = [Variable(as_array(y)) for y in ys]

		if Config.enable_backprop:
			self.generation = max([x.generation for x in inputs])
			for output in outputs: output.set_creator(self)
			self.inputs = inputs
			self.outputs = [weakref.ref(output) for output in outputs]
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

class Mul(Function):
	def forward(self, x0, x1):
		y = x0 * x1
		return y

	def backward(self, gy):
		x0, x1 = self.inputs[0].data, self.inputs[1].data
		return gy * x1, gy * x0



def square(x): return Square()(x)
def exp(x): return Exp()(x)
def triple(x): return Triple()(x)
def add(x0, x1): return Add()(x0, x1)
def mul(x0, x1): return Mul()(x0, x1)

Variable.__add__ = add
Variable.__mul__ = mul

@contextlib.contextmanager
def using_config(name, value):
	old_value = getattr(Config, name)
	setattr(Config, name, value)
	try: yield
	finally: setattr(Config, name, old_value)
def no_grad(): return using_config('enable_backprop', False)


### 예제 ###

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = (a * b) + c

y.backward()
print(a.grad)
print(b.grad)

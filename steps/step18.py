import numpy as np
import heapq as hq
import weakref
import contextlib

class Config:
	enable_backprop = True

class Variable:
	def __init__(self, data):
		if data is not None and not isinstance(data, np.ndarray):
			raise TypeError(f'{type(data)} not supported, only ndarray')

		self.data = data
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
		# if tmp:
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

def square(x): return Square()(x)
def exp(x): return Exp()(x)
def triple(x): return Triple()(x)
def add(x0, x1): return Add()(x0, x1)

### 예제 ###

# Config.enable_backprop = False
tmp = True
# 왜 굳이 클래스로 만들어야 함? 그냥 변수는 안 됨?

# @contextlib.contextmanager
# def make_false():
# 	Config.enable_backprop = False
# 	print(Config.enable_backprop)
# 	try:
# 		yield
# 	finally:
# 		Config.enable_backprop = True
# 		print(Config.enable_backprop)

# with make_false():
# 	print(Config.enable_backprop)
# 	x0 = Variable(np.array(2.0))
# 	x1 = Variable(np.array(1.0))
# 	t = add(x0, x1)
# 	y= add(x0, t)
# 	y.backward()



#단순 tmp로 할 수 있는지? 왜 클래스 쓰는지
# @contextlib.contextmanager
# def make_false(tmp):
# 	tmp = False
# 	print(tmp)
# 	try:
# 		yield
# 	finally:
# 		tmp = True
# 		print('good!')

# with make_false(tmp):
# 	print(tmp)
# 	x0 = Variable(np.array(2.0))
# 	x1 = Variable(np.array(1.0))
# 	t = add(x0, x1)
# 	y= add(x0, t)
# 	y.backward()



# with구문 연습
# @contextlib.contextmanager
# def config_test():
# 	print('start')
# 	yield
# 	print('good')
# 	print('done')

# with config_test():
# 	print('process')


@contextlib.contextmanager
def using_config(name, value):
	old_value = getattr(Config, name)
	setattr(Config, name, value)
	try: yield
	finally: setattr(Config, name, old_value)


x0 = Variable(np.array(2.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y= add(x0, t)
y.backward()
print('good')
def no_grad(): return using_config('enable_backprop', False)

with no_grad():
	x0 = Variable(np.array(2.0))
	x1 = Variable(np.array(1.0))
	t = add(x0, x1)
	y= add(x0, t)
	print('not good')
	y.backward()

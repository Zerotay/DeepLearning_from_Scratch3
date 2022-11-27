import numpy as np
import heapq as hq

class Variable:
	def __init__(self, data):
		if data is not None and not isinstance(data, np.ndarray):
			raise TypeError(f'{type(data)} not supported, only ndarray')

		self.data = data
		self.grad = None
		self.creator = None
		self.generation = 0 # 세대를 기록하는 변수

	def set_creator(self, func):
		self.creator = func
		self.generation = func.generation + 1
		# 함수에 기록된 세대에 +1을 한다.
		# 즉 자신을 창조한 함수보다 아래 세대라는 것.

	def backward(self):
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = []
		seen_set = set()

		#기존 sort활용한 함수
		# def add_func(f):
		# 	if f not in seen_set:
		# 		funcs.append(f)
		# 		seen_set.add(f)
		# 		funcs.sort(key=lambda x: x.generation)
		# 		# print(funcs) #seen_set이 하는 일을 보기 위해

		#우선순위 큐 활용한 함수
		def add_func(f):
			if f not in seen_set:
				hq.heappush(funcs, (-f.generation, str(f), f))
				seen_set.add(f)

		add_func(self.creator)

		while funcs:
			#우선순위큐 사용해보자.
			# f = funcs.pop()
			_, _, f = hq.heappop(funcs)

			gys = [output.grad for output in f.outputs]
			gxs = f.backward(*gys)
			if not isinstance(gxs, tuple): gxs = (gxs, )

			for x, gx in zip(f.inputs, gxs):
				if x.grad is None: x.grad = gx
				else: x.grad = x.grad + gx

				if x.creator is not None:
					add_func(x.creator)

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

		# 자신에게 들어온 입력값의 세대를 확인하고 가장 늦은 세대에 맞춘다.
		self.generation = max([x.generation for x in inputs])
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

### 예제 ###

x = Variable(np.array(2.0))
a = square(x)
y = add(triple(a), square(a))
y.backward()

print(y.data)
print(x.grad)

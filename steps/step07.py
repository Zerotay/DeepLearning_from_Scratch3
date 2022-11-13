import numpy as np

class Variable:
	def __init__(self, data):
		self.data = data
		self.grad = None
		self.creator = None

	def set_creator(self, func):
		self.creator = func #자신의 창조자를 저장함

	def backward(self):
		f = self.creator
		if f is not None: # 창조한 함수가 없을 때까지
			x = f.input
			x.grad = f.backward(self.grad)
			x.backward()


class Function:
	def __call__(self, input):
		x = input.data
		y = self.forward(x)
		output = Variable(y) # 함수로 만들어진 출력이 변수에 담김
		output.set_creator(self) # 그 변수는 자신을 만든 함수를 저장함
		self.input = input # 입력변수 보관
		self.output = output # 출력변수도 보관
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

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

assert y.creator == C
assert y.creator.output == y
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)
print(b.grad)
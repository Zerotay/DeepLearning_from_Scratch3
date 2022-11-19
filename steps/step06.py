import numpy as np

class Variable:
	def __init__(self, data):
		self.data = data
		self.grad = None # 앞에서부터의 미분값.

class Function:
	def __call__(self, input):
		x = input.data
		y = self.forward(x)
		output = Variable(y)
		self.input = input # 변수 보관
		return output

	def forward(self, x):
		raise NotImplementedError()

	def backward(self, gy): #엄밀하게 back이 아님. 이것도 forward
		raise NotImplementedError()


class Square(Function):
	def forward(self, x):
		return x ** 2

	def backward(self, gy): #gy도 앞에서 전달되는 값
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

def numerical_diff(f, x, eps=1e-4):
	x0 = Variable(x.data - eps)
	x1 = Variable(x.data + eps)

	y0 = f(x0)
	y1 = f(x1)
	return (y1.data - y0.data) / (2 * eps)

a = Square()
b = Exp()
c = Square()

x = Variable(np.array(0.5))
p = a(x)
q = b(p)
y = c(q)


#앞에서부터 미분을 진행한다면?
x.grad = np.array(1.0)
p.grad = a.backward(x.grad)
q.grad = b.backward(p.grad)
y.grad = c.backward(q.grad)
print(x.grad)
print(p.grad)
print(q.grad)
print(y.grad)
#결과적으로 grad의 의미가 바뀌었고, 저장되는 위치가 바뀌게 되었다.


# y.grad = np.array(1.0)
# q.grad = c.backward(y.grad)
# p.grad = b.backward(q.grad)
# x.grad = a.backward(p.grad)
# print(y.grad)
# print(q.grad)
# print(p.grad)
# print(x.grad)


# import numpy as np

# class Variable:
# 	def __init__(self, data):
# 		self.data = data
# 		self.grad = None

# class Function:
# 	def __call__(self, input):
# 		x = input.data
# 		y = self.forward(x)
# 		output = Variable(y)
# 		self.input = input # 변수 보관
# 		return output

# 	def forward(self, x):
# 		raise NotImplementedError()

# 	def backward(self, gy):
# 		raise NotImplementedError()


# class Square(Function):
# 	def forward(self, x):
# 		return x ** 2

# 	def backward(self, gy):
# 		x = self.input.data
# 		gx = 2 * x * gy
# 		return gx

# class Exp(Function):
# 	def forward(self, x):
# 		return np.exp(x)

# 	def backward(self, gy):
# 		x = self.input.data
# 		gx = np.exp(x) * gy
# 		return gx

# def numerical_diff(f, x, eps=1e-4):
# 	x0 = Variable(x.data - eps)
# 	x1 = Variable(x.data + eps)

# 	y0 = f(x0)
# 	y1 = f(x1)
# 	return (y1.data - y0.data) / (2 * eps)

# a = Square()
# b = Exp()

# x = Variable(np.array(0.5))
# p = a(x)
# q = b(p)
# y = a(q)

# y.grad = np.array(1.0)
# q.grad = a.backward(y.grad)
# p.grad = b.backward(q.grad)
# x.grad = a.backward(p.grad)
# print(x.grad)
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
	def __call__(self, inputs):
		xs = [x.data for x in inputs] # 입력온 것들을 리스트로 저장
		ys = self.forward(xs) #당연히 각 함수의 forward에 수정이 가해진다.
		outputs = [Variable(as_array(y)) for y in ys] # 출력된 것들도 리스트로
		for output in outputs: output.set_creator(self) #각각 창조함수 지정
		self.input = input
		self.outputs = outputs
		return outputs

	def forward(self, x):
		raise NotImplementedError()

	def backward(self, gy):
		raise NotImplementedError()

#간단하게 두 개를 받는 함수 구현
class Add(Function):
	def forward(self, xs): #리스트 형태로 들어오게 될 것이다.
		x0, x1 = xs
		y = x0 + x1
		return (y, )
	#__call__에서 outputs도 리스트 컴프리헨션을 쓰기에, 한번 묶여야 함


xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add() #함수 인스턴스
ys = f(xs) # 2개 원소 가진 리스트 넣기
y = ys[0] # 출력은 set이다.
print(ys)
print(y.data)



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

# class Triple(Function):
# 	def forward(self, x):
# 		return x ** 3

# 	def backward(self, gy):
# 		x = self.input.data
# 		gx = 3 * (x ** 2) * gy
# 		return gx

# def square(x): return Square()(x)
# def exp(x): return Exp()(x)
# def triple(x): return Triple()(x)
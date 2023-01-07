if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def f(x):
	y = x ** 4 - 2 * x ** 2
	return y


x = Variable(np.array(2.0))
epochs = 10

for i in range(epochs):
	print(i, x)

	y = f(x)
	x.cleargrad()
	y.backward(create_graph=True)

	gx = x.grad #역전파해서 생긴 값을 저장하기
	x.cleargrad() # x.grad에 값이 중첩되지 않도록 초기화
	gx.backward()
	gx2 = x.grad # 2차 역전파를 통해 생긴 값

	x.data -= gx.data / gx2.data

# 2차 미분 도함수를 직접 만들어서 쓰던 시절의 코드가 움직일 수 있도록
# def gx2(x):
# 	return 12 * x ** 2 - 4

# x = Variable(np.array(2.0))
# epochs = 10

# for i in range(epochs):
# 	print(i, x)

# 	y = f(x)
# 	x.cleargrad()
# 	y.backward(create_graph=True)

# 	x.data -= x.grad.data / gx2(x.data) # x.grad.data를 넣어줘야 x.data에 ndarray가 담기게 된다.
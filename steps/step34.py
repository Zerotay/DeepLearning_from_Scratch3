if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import matplotlib.pyplot as plt
import dezero.functions as F

x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
	logs.append(x.grad.data)
	gx = x.grad
	x.cleargrad()
	gx.backward(create_graph=True)

labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
	plt.plot(x.data, logs[i], label=labels[i]) # x값, y값, 라벨
plt.legend(loc='lower right')
plt.show()
# plt.savefig('tmp.png') # 서버에서는 제대로 보이지 않기에 파일로 저장
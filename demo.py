from varbvs import varbvs
import numpy as np

x = np.random.randn(100,1000)
beta = np.zeros(1000)
beta[:5] = 1
y = x.dot(beta) + np.random.randn(100)
Z = np.random.randn(100,10)

model = varbvs(x,y,Z)
pip=model.fit(verbose=True)
print('posterior inclusion probabilities: \n',sorted(pip)[-10:])



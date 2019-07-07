import primitives.tree as tcpp
import primitives.tree_py as tpy
import random
import numpy as np
import time

num = 1000000

def compPy(a,b):
    if a[0] < b[0]: return -1
    if a[0] > b[0]: return 1
    return 0

def compCpp(a,b):
    return a[0] < b[0]

treePy = tpy.Tree(compPy)
treeCpp = tcpp.Tree(compCpp)

rands = [random.randint(0,100000000) for x in range(num)]

start = time.time()

for i in rands:
    treePy.insert(np.array([i]))

print('Duration fill py: {}'.format(time.time() - start))

start = time.time()

for i in rands:
    treeCpp.insert(np.array([i]))

print('Duration fill cpp: {}'.format(time.time() - start))
num_ops = 1000

durSumPy = 0
durSumCpp = 0
for i in range(num_ops):
    r = random.randint(0,100000000)
    start = time.time()
    treePy.insert(np.array([r]))
    durSumPy += time.time() - start

    start = time.time()
    treeCpp.insert(np.array([r]))
    durSumCpp += time.time() - start

print('average dur insert py: {}'.format(durSumPy/num_ops))
print('average dur insert cpp: {}'.format(durSumCpp/num_ops))

durSumPy = 0
durSumCpp = 0
for i in range(num_ops):
    r = random.randint(0,100000000)
    start = time.time()
    treePy.find(np.array([r]))
    durSumPy += time.time() - start

    start = time.time()
    treeCpp.find(np.array([r]))
    durSumCpp += time.time() - start

print('average dur find py: {}'.format(durSumPy/num_ops))
print('average dur find cpp: {}'.format(durSumCpp/num_ops))

print('sizes {} {}'.format(treePy.size, treeCpp.size()))

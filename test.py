import primitives.tree as tree
import primitives.grid_tools as gt 
import numpy as np
import random


tree = tree.Tree(gt.lexicographicalOrderingOnRegions)

def divideRegion(r):
    rows = r.shape[0]
    numDims = np.random.randint(1, rows)
    dims = list(range(rows))
    np.random.shuffle(dims)
    ret1 = np.empty_like(r)
    ret2 = np.empty_like(r)
    for i in range(rows):
        curRow = dims[i]
        if i < numDims:
            low = r[curRow,0]
            high = r[curRow,1]
            ret1[curRow,0] = low
            ret2[curRow,1] = high
            division = (high + low)/2.0
            ret1[curRow,1] = division
            ret2[curRow,0] = division
        else:
            ret1[curRow,0] = r[curRow,0]
            ret1[curRow,1] = r[curRow,1]
            ret2[curRow,0] = r[curRow,0]
            ret2[curRow,1] = r[curRow,1]

    return ret1, ret2

r = np.zeros((10,2))
r[:,1] = 100

regions = [r]
for i in range(100):
    idx = random.choice(range(len(regions)))
    region = regions[idx]
    del regions[idx]
    r1, r2 = divideRegion(region)
    regions += [r1, r2]

for region in regions:
    tree.insert(region)

print('sizes {} {}'.format(len(regions), tree.size()))

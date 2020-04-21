import numpy as np
import math

def snapGrid(pt, ref, granularity):
    ret = np.empty_like(pt)
    absgran = np.abs(granularity)
    for i in range(np.prod(pt.shape)):
        index = np.unravel_index(i, pt.shape)
        multiplier = math.floor((pt[index] - ref[index])/absgran[index])
        ret[index] = ref[index] + multiplier*absgran[index]
    return ret


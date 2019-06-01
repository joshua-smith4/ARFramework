import numpy as np

def lexicalOrdering(a,b,predicate,getShapeWithoutChannels=lambda x,y: x.shape):
    '''
    Returns the lexicographical ordering of a and b based on a predicate
    0: a and b are same
    -1: a is less than b
    1: a is greater than b
    '''
    shape = getShapeWithoutChannels(a, b)
    numElements = np.prod(shape)
    i = 0
    while i < numElements:
        shapedIndex = np.unravel_index(i, shape)
        i += 1
        if predicate(a[shapedIndex], b[shapedIndex]) != 0:
            break
    return predicate(a[shapedIndex], b[shapedIndex])

# if comparing points to regions, the point should be argument 1 (r1)
# and the region should be argument 2 (r2)
# if not, an error will be thrown
def lexicographicalOrderingOnRegions(r1, r2):
    '''
    orders regions and/or points based on lexicographical ordering
    0: r1 equal to/in r2
    -1: r1 is less than/outside left r2
    1: r1 is greater than/outside right r2
    '''
    def predicate(a_elem, b_elem):
        if type(a_elem) == np.ndarray or type(b_elem) == np.ndarray:
            # case both inputs are regions
            if type(a_elem) == np.ndarray and type(b_elem) == np.ndarray:
                # range a_elem is equal to or inside range b_elem
                if a_elem[0] >= b_elem[0] and a_elem[1] <= b_elem[1]:
                    return 0
                # range a_elem is entirely less than range b_elem
                elif a_elem[1] < b_elem[0]:
                    return -1
                # range a_elem is greater than range b_elem
                return 1
            # case one input is a region and the other is a point
            else:
                tmp_range = a_elem if type(a_elem) == np.ndarray else b_elem
                tmp_point = b_elem if type(a_elem) == np.ndarray else a_elem
                if tmp_point >= tmp_range[0] and tmp_point < tmp_range[1]:
                    return 0
                elif tmp_point < tmp_range[0]:
                    return -1
                return 1 
        if a_elem == b_elem:
            return 0
        elif a_elem < b_elem:
            return -1
        return 1
    return lexicalOrdering(r1, r2, predicate)


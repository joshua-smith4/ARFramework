import numpy as np

class GridHashAndTreeStorageStrategy:
    class Node:
        def __init__(self, data):
            self.less = None
            self.greater = None
            self.data = data


    def __init__(self, compFunc):
        '''
        tree - a pointer to a node, implemenation of binary tree
        hash - hash table used to quickly re-reference searched nodes
        compFunc - function returning 0, -1, 1, for equal, lt, gt
        hashIdx - idx used as index for hash table
        '''
        self.tree = None
        self.hash = {}
        self.compFunc = compFunc
        self.hashIdx = 0

    def removeRegionContainingPoint(self, point):
        cur = self.tree
        if cur is None: return None
        res = self.compFunc(point, cur.data)
        while cur is not None and res != 0:
            if res == -1:
                cur = cur.less
            else:
                cur = cur.greater
            if cur is not None:
                res = self.compFunc(point, cur.data)
        if cur is None: return None
        ret = cur.data
        self.deleteNode(cur)
        return ret

    def deleteNode(self, n):
        if n.less is not None and n.greater is not None:
            # has two children
            tmp = n
            cur = n.less
            parcur = n
            while cur.greater is not None:
                parcur = cur
                cur = cur.greater
            n = cur
            cur = None
            n.less = tmp.less
            n.greater = tmp.greater
            return
        if n.less is not None:
            n = n.less
            return
        if n.greater is not None:
            n = n.greater
            return
        n = None

    def getRegionContainingPoint(self, point):
        '''
        This function returns None if region is not found
        Otherwise, the found region and an index is returned
        into the hash table accessed using the below function
        getRegionByIdx.
        The hash table is a quick way to re-reference found
        regions.
        '''
        cur = self.tree
        if cur is None: return None
        res = self.compFunc(point, cur.data)
        while cur is not None and res != 0:
            if res == -1:
                cur = cur.less
            else:
                cur = cur.greater
            if cur is not None:
                res = self.compFunc(point, cur.data)
        if cur is None: return None
        self.hash[self.hashIdx] = cur.data
        tmp = self.hashIdx
        self.hashIdx += 1
        return cur.data, tmp

    def traverse(self, func):
        self.traverse_helper(func, self.tree)

    def traverse_helper(self, func, cur):
        if cur is None: return
        self.traverse_helper(func, cur.less)
        func(cur.data)
        self.traverse_helper(func, cur.greater)

    def __repr__(self):
        ret = []
        def compileResults(x, ret=ret):
            ret += [str(x)]
        self.traverse(compileResults)
        return '\n'.join(str(x) for x in ret)

    def getRegionByIdx(self, hashIdx, keepEntry=False):
        try:
            ret = self.hash[hashIdx]
        except KeyError:
            return None
        if not keepEntry:
            del self.hash[hashIdx]
        return ret

    def insertRegion(self, region):
        cur = self.tree
        if cur is None: 
            self.tree = GridHashAndTreeStorageStrategy.Node(region)
            print('inserting region empty tree',region)
            return
        res = self.compFunc(region, cur.data)
        while True:
            print('res: ',res)
            if res == -1:
                if cur.less is None:
                    print('inserting into less',region)
                    cur.less = GridHashAndTreeStorageStrategy.Node(region)
                    return
                print('recursing into less')
                cur = cur.less
            elif res == 1:
                if cur.greater is None:
                    print('inserting into greater',region)
                    cur.greater = GridHashAndTreeStorageStrategy.Node(region)
                    return
                print('recursing into greater')
                cur = cur.greater
            else:
                assert False, 'Encountered a duplicate in tree, not allowed'
            res = self.compFunc(region, cur.data)



class Grid:
    def __init__(self, storageStrategy):
        self.db = storageStrategy

    def findAndRemoveRegion(self, point):
        return self.db.findAndRemoveRegion(point)

    def refineRegion(self, refinementStrategy):
        pass

def compInt(a,b):
    if a==b: return 0
    if a<b: return -1
    return 1

a = GridHashAndTreeStorageStrategy(compInt)
a.insertRegion(5)
a.insertRegion(-1)
a.insertRegion(9)
a.insertRegion(12)
a.insertRegion(4)
a.insertRegion(3)
a.insertRegion(6)


import collections
# Tries is from the word re-trie-val
# this is basic R-nary Tries. there is a more space efficient Ternary Tries. 
# assume input is string and value is integer
class Node(object): 
    def __init__(self, val):
        self.R = 256
        self.val = val
        self.next = [None] * self.R


class Tries(object):
    def __init__(self): 
        # null root
        self.root = Node()
    
    def is_empty(self):
        pass 


    def size(self):
        pass 


    def constains(self, key):
         return self.get(key) != None


    def get(self, key):
        n = self.get_helper(self.root, key, 0)
        if not n: return None
        return n.val

    # return value node 
    def get_helper(self, r, key, d):
        if not r:
            return None
        if d == len(key):
            return r
        next_node = r.next[ord[d]]
        return self.get_helper(next_node, key, d+1)


    def put(self, key, val):
        def put_helper(r, k, d, v):
            if not r: r = Node()
            # last node is for value 
            if d == len(key): 
                r.val = v
                return r
            n = r.next[ord(k[d])]
            r.next[ord(k[d])] = put_helper(n, k, d+1, v)
            return r
        self.root = put_helper(self.root, key, 0, val)


    # return all the stored keys
    def keys(self):
        return self.keys_with_prefix('')


    def keys_with_prefix(self, prefix):
        queue = collections.deque()
        # get value node with given prefix
        n = self.get_helper(self.root, prefix, 0)
        self.collect(n, prefix, queue)
        return queue

    # collecting all keys with value and store in queue
    def collect(self, r, prefix, queue): 
        if not r: return
        if r.val != None: queue.append(prefix)
        for i in range(self.R): 
            collect(r.next[i], prefix + chr(i), queue)


    def keysThatMatch(self, st):
        pass 


    def longestPrefixOf(self, st):
        pass


    def delete(self, key): 
        pass


    

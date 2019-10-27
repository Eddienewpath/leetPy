def binaryTreePaths(root):
    res = []
    if not root: return res 
    def build_path(r, res, st):
        if not r: return
        if not r.left and not r.right: 
            st += str(r.val) 
            res.append(st)
            return
        
        st +=  str(r.val) + '->'
        
        build_path(r.left, res, st)
        build_path(r.right, res, st)
    build_path(root, res, '')
    return res
##############################################################
### Utilities for Sampling and managing PyTorch Parameters ###
##############################################################

# Author: Linus Groner, 2018

import torch
import numpy as np

def sampleSingle(data_x,data_y,n,replacement):
    if n < data_x.size(0):
        if data_x.is_cuda:
            idx = torch.cuda.LongTensor( np.random.choice(data_x.size(0),replace=replacement,size=n) )
        else:
            idx = torch.LongTensor( np.random.choice(data_x.size(0),replace=replacement,size=n) ) 
        data_res_X = torch.index_select(data_x,0,idx)
        data_res_Y = torch.index_select(data_y,0,idx)
    else: 
        data_res_X=data_x
        data_res_Y=data_y
    return data_res_X,data_res_Y

def independentSampling(data_x,data_y,n1,n2,n3,replacement):
    if n1 < data_x.size(0):
        if data_x.is_cuda:
            idx = torch.cuda.LongTensor( np.random.choice(data_x.size(0),replace=replacement,size=n1) )
        else:
            idx = torch.LongTensor( np.random.choice(data_x.size(0),replace=replacement,size=n1) ) 
        data_res_X1 = torch.index_select(data_x,0,idx)
        data_res_Y1 = torch.index_select(data_y,0,idx)
    else: 
        data_res_X1=data_x
        data_res_Y1=data_y
        
    if n2 < data_x.size(0):
        if data_x.is_cuda:
            idx = torch.cuda.LongTensor( np.random.choice(data_x.size(0),replace=replacement,size=n2) )
        else:
            idx = torch.LongTensor( np.random.choice(data_x.size(0),replace=replacement,size=n2) ) 
        data_res_X2 = torch.index_select(data_x,0,idx)
        data_res_Y2 = torch.index_select(data_y,0,idx)
    else: 
        data_res_X2=data_x
        data_res_Y2=data_y
        
    if n3 < data_x.size(0):
        if data_x.is_cuda:
            idx = torch.cuda.LongTensor( np.random.choice(data_x.size(0),replace=replacement,size=n3) )
        else:
            idx = torch.LongTensor( np.random.choice(data_x.size(0),replace=replacement,size=n3) ) 
        data_res_X3 = torch.index_select(data_x,0,idx)
        data_res_Y3 = torch.index_select(data_y,0,idx)
    else: 
        data_res_X3=data_x
        data_res_Y3=data_y

    return (data_res_X1,data_res_Y1),(data_res_X2,data_res_Y2),(data_res_X3,data_res_Y3)

def nestedSampling(data_x,data_y,n1,n2,n3,replacement=False):
    assert(not replacement), "A combination of nested sub-samples and sampling with replacement is not meaningful. The replacement parameter is only to keep the signature consistent with independentSampling."
    
    ns = np.array([n1,n2,n3])
    ns = np.maximum(1,np.minimum(data_x.size(0),ns))
    idx = np.argsort(ns)
    small_set_size,middle_set_size,large_set_size = ns[idx] 

    large_set_x,large_set_y = sampling(data_x,data_y,large_set_size,False)
    middle_set_x,middle_set_y = sampling(large_set_x,large_set_y,middle_set_size,False) 
    small_set_x,small_set_y = sampling(middle_set_x,middle_set_y,small_set_size,False) 
    sets = ((small_set_x,small_set_y),(middle_set_x,middle_set_y),(large_set_x,large_set_y))
    return [sets[i] for i in np.argsort(idx)]

def flattenList(vec,sizes):
    dim = sum(sizes)
    
    dim_ctr = 0
    res = torch.cat([v.data.contiguous().view(-1) for v in vec])
    return res

def setParameters(net,vec,sizes):
    res = []
    dim_ctr = 0
    for i,p in enumerate(net.parameters()):
        dim = sizes[i]
        p.data = vec[dim_ctr:dim_ctr+dim].view(p.size())
        dim_ctr+=dim

def inflateVector(vec,net,sizes):
    res = []
    dim_ctr = 0
    for i,p in enumerate(net.parameters()):
        dim = sizes[i]
        res.append(vec[dim_ctr:dim_ctr+dim].view(p.size()))
        dim_ctr+=dim
    return res

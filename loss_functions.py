###################################################
### Common Loss Functions and their derivatives ###
###################################################

# Authors: Jonas Kohler and Aurelien Lucchi, 2017
# Linus Groner, 2018:
#     * added class PyTorchWrapper to provide a simple interface to compute Hv products of PyTorch models.


from math import log
from math import sqrt

import random
import time
import os

import numpy as np
from scipy import linalg
from sklearn.utils.extmath import randomized_svd
import torch
import util
# Return the loss as a numpy array

def square_loss(w, X , Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    P = X.dot(w)
    l = 0.5 * np.average([(Y[i] - P[i]) ** 2 for i in range(len(Y))])
    l = l + 0.5 * alpha * (np.linalg.norm(w) ** 2)
    return linalg

def hinge_loss(w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    P = [np.dot(w, X[i]) for i in range(n)]  # prediction <w, x>
    l = np.sum([max(0, 1 - Y[i] * P[i]) for i in range(len(Y))]) / n
    l = l + 0.5 * alpha * (np.linalg.norm(w) ** 2)
    return l

def logistic_loss(w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w) 
    l= - (np.dot(log_phi(z),Y)+np.dot(np.ones(n)-Y,one_minus_log_phi(z)))/n
    l = l + 0.5*  alpha * (np.linalg.norm(w) ** 2)
    return l

def logistic_loss_nonconvex(w,X,Y,alpha=1e-3,beta=1):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w)  # prediction <w, x>
    h = phi(z)
    l= - (np.dot(np.log(h),Y)+np.dot(np.ones(n)-Y,np.log(np.ones(n)-h)))/n
    l= l + alpha*np.dot(beta*w**2,1/(1+beta*w**2))
    return l

def softmax_loss(w,X,ground_truth,alpha=1e-3,n_classes=None):
    assert (n_classes is not None),"Please specify number of classes as n_classes for softmax regression"
    n = X.shape[0]
    d = X.shape[1]
    w=np.matrix(w.reshape(n_classes,d).T)
    z=np.dot(X,w) #activation of each i for class c
    z-=np.max(z,axis=1)  # model is overparametrized. allows to subtract maximum to prevent overflow for each datapoint!
    h = np.exp(z)
    P= h/np.sum(h,axis = 1) #gives matrix with with [P]i,j = probability of sample i to be in class j (n x nC)
    error=np.multiply(ground_truth, np.log(P)) 
    l = -(np.sum(error) / n)
    l += 0.5*alpha*(np.sum(np.multiply(w,w))) #weight decay
    return l 

def monkey_loss(w,X,Y):
    l = w[0] ** 3 - 3 * w[0] * w[1] ** 2
    return l

def rosenbrock_loss(w,X,Y):
    l = (1. - w[0]) ** 2 + 100. * (w[1] - w[0] ** 2) ** 2
    return l

def non_convex_coercive_loss(w,X,Y):
    l = 0.5 * w[0] ** 2 + 0.25 * w[1] ** 4 - 0.5 * w[1] ** 2 # this can be found in Nesterov’s Book: Introductory Lectures on Convex Optimization
    return l

# Return the gradient as a numpy array
def square_loss_gradient(w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    grad = (-X.T.dot(Y)+np.dot(X.T,X.dot(w)))/n
    grad = grad + alpha * w
    return grad

def logistic_loss_gradient(w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w)  
    h = phi(z)
    grad= X.T.dot(h-Y)/n
    grad = grad + alpha * w
    return grad

def logist_loss_nonconvex_gradient(w, X, Y, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w)   # prediction <w, x>
    h = phi(z)
    grad= X.T.dot(h-Y)/n
    grad = grad + alpha*np.multiply(2*beta*w,(1+beta*w**2)**(-2))
    return grad

def softmax_loss_gradient(w, X, ground_truth, alpha=1e-3,n_classes=None):
    assert (n_classes is not None), "Please specify number of classes as n_classes for softmax regression"
    
    n = X.shape[0]
    d = X.shape[1]
    w=np.matrix(w.reshape(n_classes,d).T)
    z=np.dot(X,w) 
    z-=np.max(z,axis=1)  
    h = np.exp(z)
    P= h/np.sum(h,axis = 1) 

    grad = -np.dot(X.T, (ground_truth - P))

    grad = grad / n  + alpha* w
    grad = np.array(grad)
    grad = grad.flatten(('F'))
    return grad
        
def monkey_gradient(w,X,Y):
    grad = np.array([3 * (w[0] ** 2 - w[1] ** 2), -6 * w[0] * w[1]])
    return grad

def rosenbrock_gradient(w,X,Y):
    grad = np.array([-2 + 2. * w[0] - 400 * w[0] * w[1] + 400 * w[0] ** 3, 200. * w[1] - 200 * w[0] ** 2])
    return grad

def non_convex_coercive_gradient(w,X,Y):
    grad = np.array([w[0], w[1] ** 3 - w[1]])
    return grad

# Return the Hessian matrix as a 2d numpy array
def square_loss_hessian( w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    H = np.dot(X.T, X) / n + (alpha * np.eye(d, d))
    return H
def logistic_loss_hessian( w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z= X.dot(w)
    q=phi(z)
    h= np.array(q*(1-phi(z)))
    H = np.dot(np.transpose(X),h[:, np.newaxis]* X) / n
    H = H + alpha * np.eye(d, d) 
    return H 

def logistic_loss_nonconvex_hessian( w, X, Y, alpha=1e-3,beta=1):
    z= X.dot(w)
    q=phi(z)
    h= q*(1-phi(z))
    H = np.dot(np.transpose(X),h[:, np.newaxis]* X) / n  
    H = H + alpha * np.eye(d,d)*np.multiply(2*beta-6*beta**2*w**2,(beta*w**2+1)**(-3))
    return H

def softmax_loss_hessian( w, X, Y, alpha=1e-3,n_classes=None):
    assert (n_classes is not None),"Please specify number of classes as n_classes for softmax regression"

    n = X.shape[0]
    d = X.shape[1]
    w=np.matrix(w.reshape(n_classes,d).T)
    z=np.dot(X,w) 
    z-=np.max(z,axis=1)  
    h = np.exp(z)
    P= h/np.sum(h,axis = 1) 
    H=np.zeros([d*n_classes,d*n_classes])
    for c in range(n_classes):
        for k in range (n_classes):

            if c==k:
                D=np.diag(np.multiply(P[:,c],1-P[:,c]).A1)
                Hcc = np.dot(np.dot(np.transpose(X), D), X) 
            else:
                D=np.diag(-np.multiply(P[:,c],P[:,k]).A1)
                Hck = np.dot(np.dot(np.transpose(X), D), X) 
                H[c*d:(c+1)*d,k*d:(k+1)*d]=Hck
                H[k*d:(k+1)*d,c*d:(c+1)*d,]=Hck

    H = H/n + alpha*np.eye(d*n_classes,d*n_classes) 
    return H
        
def monkey_hessian(w,X,Y):
    H = np.array([[6 * w[0], -6 * w[1]], [-6 * w[1], -6 * w[0]]])
    return H

def rosenbrock_hessian(w,X,Y):
    H = np.array([[2 - 400 * w[1] + 1200 * w[0] ** 2, -400 * w[0]], [-400 * w[0], 200]])
    return H

def non_convex_coercive_hessian(w,X,Y):
    H = np.array([[1, 0], [0, 3 * w[1] ** 2 - 1]])
    return H

# Return the Hessian-vector product as a numpy array

def square_loss_Hv(w,X, Y, v,alpha=1e-3): 
    Xv=np.dot(X,v)
    Hv=np.dot(X.T,Xv)/n + alpha * v
    return Hv

def logistic_loss_Hv(w,X, Y, v,alpha=1e-3): 
    n = X.shape[0]
    d = X.shape[1]
    _z=X.dot(w)
    _z = phi(-_z)
    d_binary = _z * (1 - _z)
    wa = d_binary * X.dot(v)
    Hv = X.T.dot(wa)/n
    out = Hv + alpha * v
    return out

def logistic_loss_nonconvex_Hv(w, X, Y, v,alpha=1e-3,beta=1): 
    n = X.shape[0]
    d = X.shape[1]
    _z=X.dot(w)
    _z = phi(-_z)
    d_binary = _z * (1 - _z)
    wa = d_binary * X.dot(v)
    Hv = X.T.dot(wa)/n
    out = Hv + alpha *np.multiply(np.multiply(2*beta-6*beta**2*w**2,(beta*w**2+1)**(-3)), v)
    return out

def softmax_loss_Hv(w, X, ground_truth, v, alpha=1e-30,n_classes=None):
    assert (n_classes is not None),"Please specify number of classes as n_classes for softmax regression"
    n = X.shape[0]
    d = X.shape[1]
    
    w_multi=np.matrix(w.reshape(n_classes,d).T)
    z_multi=np.dot(X,w_multi) #activation of each i for class c
    z_multi-=np.max(z_multi,axis=1)  # model is overparametrized. allows to subtract maximum to prevent overflow. 
    h_multi = np.exp(z_multi)
    P_multi= np.array(h_multi/np.sum(h_multi,axis = 1)) #gives matrix with with [P]i,j = probability of sample i to be in class j (n x nC)

    v = v.reshape(n_classes, -1)

    r_yhat = np.dot(X, v.T)
    r_yhat += (-P_multi * r_yhat).sum(axis=1)[:, np.newaxis]
    r_yhat *= P_multi
    hessProd = np.zeros((n_classes, d))
    hessProd[:, :d] = np.dot(r_yhat.T, X)/n
    hessProd[:, :d] += v * alpha
    return hessProd.ravel()

def monkey_Hv(w,X,Y,v):
    H=monkey_hessian(w,X,Y)
    return np.dot(H,v)

def rosenbrock_Hv(w,X,Y,v):
    H=rosenbrock_hessian(w,X,Y)
    return np.dot(H,v)

def non_convex_coercive_Hv(w,X,Y,v):
    H=non_convex_coercive_hessian(w,X,Y)
    return np.dot(H,v)

class PyTorchWrapper:
    """
    Provides an interface to get exact loss, gradients and Hessian-vector products of a given pytorch module and loss.

    """
    def __init__(self,net, loss, l2_reg = 0.0, on_gpu=False):
        """
        Args:
            net (torch.nn.Module): The model.
            loss (callable): The loss function.
            l2_reg (float): l2 regularization penalty parameter.
            on_gpu (bool): True if the net is to be copied to gpu.
        """
        self.on_gpu = on_gpu
        self.l2_reg = l2_reg
        self.net = net
        self.loss = loss
        
        self.d = 0
        self.sizes=[]
        for p in self.net.parameters():
            dim = int(np.prod(p.size()))
            self.sizes.append(dim)
            self.d += dim
        

        # collect the parameters of net as a numpy vector
        dim_ctr = 0
        self.w_init = np.empty(self.d)
        for i,p in enumerate(self.net.parameters()):
            dim = self.sizes[i]
            self.w_init[dim_ctr:dim_ctr+dim] = p.data.view(-1).cpu().numpy()
            dim_ctr+=dim
        
        #invalidate the gradient for hv computations to ensure that
        #Hv_store_gradient computes it the first time.
        self.invalidate_gradient_for_hv()
        
    def getInitialization(self):
        return self.w_init
        
    def __call__(self,w,X,Y):
        return self.forward(w,X,Y)
            
    def forward(self,w,X,Y,**kwargs):
        """
        Computes the forward pass of self.net and self.loss as a float with the model parametrized with w
        for the batch (X,Y)

        Args:
            w (torch.Tensor): The weight vector as a 1D-torch.Tensor
            X (torch.Tensor): The input batch.
            Y (torch.Tensor): The output label batch.
        """
        if self.on_gpu=='batch': # if True, then we copy each batch to gpu
            X_device = X.cuda()
            if id(X)!=id(Y): # if input==labels, e.g. for autoencoders, no need to copy Y again.
                Y_device = Y.cuda()
            else:
                Y_device = X_device
        else: # either cpu or batch already on gpu
            Y_device = Y
            X_device = X
        
        # volatile makes the forward pass more memory efficient if no subsequent backprop follows
        # presumably by not storing forward maps
        Y_torch = torch.autograd.Variable(Y_device,requires_grad=False,volatile=True) 
        
        #the actual forward computation
        out = self.evaluate(w,X_device,volatile=True)
        res = self.loss(out,Y_torch)
        
        # adding the l2 penalty if required
        if self.l2_reg:
            res = float((res.data + self.l2_reg*torch.dot(w,w)).cpu())
        else:
            res = float(res.data.cpu())
        return res

    def evaluate(self,w,X,volatile=False,**kwargs):
        """
        Computes the forward pass of self.net and self.loss and returns the resulting
        torch.autograd.Variable

        Args:
            w (torch.Tensor): The weight vector as a 1D-torch.Tensor
            X (torch.Tensor): The input batch.
            Y (torch.Tensor): The output label batch.
        """
        X_torch = torch.autograd.Variable(X,requires_grad=False,volatile=volatile)

        # fill self.net.parameters() with the data provided through w
        util.setParameters(self.net,w,self.sizes)
        res = self.net(X_torch)
        
        return res
    
    def gradient(self,w,X,Y,**kwargs):
        """
        Computes the forward and backward pass of self.net and returns the gradient as 1D-torch.Tensor
        
        Args:
            w (torch.Tensor): The weight vector as a 1D-torch.Tensor
            X (torch.Tensor): The input batch.
            Y (torch.Tensor): The output label batch.
        """
        
        # Hacky way to determine when the gradient for Hv_store_gradient needs to be recomputed.
        # In both TR and ARC, the gradient for Hv_store_gradient must be recomputed coincidentally
        # whenever the gradient is computed on the gradient batch. For other algorithms to come,
        # a better alternative may be to actively call invalidate_gradient_for_hv() from the 
        # optimizer, yet then the optimizer would no longer be oblivious of  this implementation
        # of forward/gradient/Hv
        self.invalidate_gradient_for_hv()
        
        
        if self.on_gpu=='batch': # if True, then we copy each batch to gpu
            X_device = X.cuda()
            if id(X)!=id(Y): # if input==labels, e.g. for autoencoders, no need to copy Y again.
                Y_device = Y.cuda()
            else:
                Y_device = X_device
        else: # either cpu or batch already on gpu
            Y_device = Y
            X_device = X
        
        Y_torch = torch.autograd.Variable(Y_device,requires_grad=False)
        
        # compute the forward pass
        out = self.evaluate(w,X_device)
        E = self.loss(out,Y_torch)

        # compute the backward pass
        self.net.zero_grad()
        grad_list = torch.autograd.grad(E,self.net.parameters())

        # flatten the gradients returned by pytorch to a single 1D-torch.Tensor and adding the l2 penalty if required
        res = util.flattenList(grad_list,self.sizes)
        if self.l2_reg:
            return res+(2*self.l2_reg)*w
        else:
            return res 
    
    def Hv(self,w,X,Y,v,**kwargs):
        """
        Computes the Hessian-vector product of the model using the double backprop method and returns it as as 1D-torch.Tensor
        
        Args:
            w (torch.Tensor): The weight vector as a 1D-torch.Tensor
            X (torch.Tensor): The input batch.
            Y (torch.Tensor): The output label batch.
        """
        self.invalidate_gradient_for_hv()
        return self.Hv_store_gradient(w,X,Y,v) 
    
    def invalidate_gradient_for_hv(self):
        self.evaluate_gradient_for_hv = True


    def Hv_store_gradient(self,w,X,Y,v,**kwargs):
        """
        Computes the Hessian-vector product of the model using the double backprop method and returns it as as 1D-torch.Tensor.
        If the gradient is known from a previous computation, (indicated by self.evaluate_gradient_for_hv )it is not recomputed.
        
        Args:
            w (torch.Tensor): The weight vector as a 1D-torch.Tensor
            X (torch.Tensor): The input batch.
            Y (torch.Tensor): The output label batch.
            v (torch.Tensor): the vector to be multiplied
        """
        if self.on_gpu=='batch': # if True, then we copy each batch to gpu
            X_device = X.cuda()
            if id(X)!=id(Y): # if input==labels, e.g. for autoencoders, no need to copy Y again.
                Y_device = Y.cuda()
            else:
                Y_device = X_device
        else: # either cpu or batch already on gpu
            Y_device = Y
            X_device = X
        
        if self.evaluate_gradient_for_hv:
            Y_torch = torch.autograd.Variable(Y_device,requires_grad=False)
            out = self.evaluate(w,X_device)
            E = self.loss(out,Y_torch)

            self.net.zero_grad()
            self.gradient_for_hv = list(torch.autograd.grad(E,self.net.parameters(),retain_graph=True,create_graph=True))
            self.evaluate_gradient_for_hv = False
            
        self.net.zero_grad()

        # reshape 1D-torch.Tensor v such that it has the same shape as the net.parameters()
        v_list = util.inflateVector(v,self.net,self.sizes)
        v_torch = []
        for vv in v_list:
            v_torch.append(torch.autograd.Variable(vv,requires_grad=False))
        
        # I don't really understand why grad_outputs=v works, credit goes to user ezyang on https://github.com/pytorch/pytorch/issues/5112
        # who uses this to emulate Rop in a thread on an unrelated issue. This is only a one-liner compared to what follows below 
        # and behaves correctly when veryfied against finite differences.
        hv = list(torch.autograd.grad(self.gradient_for_hv,self.net.parameters(),grad_outputs=v_torch,retain_graph=True,create_graph=False))
        # a less concise version would look like this:
        #dw_v_sum = 0.0
        #for i in range(len(self.gradient_for_hv)):
        #    dw_v_sum += torch.sum(self.gradient_for_hv[i] * v_torch[i])
        #hv = list(torch.autograd.grad(dw_v_sum,self.net.parameters(),retain_graph=True,create_graph=False))
       
        # flatten the Hv-product returned by pytorch to a single 1D-torch.Tensor and adding the l2 penalty if required
        res = util.flattenList(hv,self.sizes)
        if self.l2_reg:
            return res+self.l2_reg*2*v
        else:
            return res
        #return res

    def hessian(self,w,X,Y,**kwargs):
        """
        Computes the Hessian of the model by filling each column by a Hessian-vector 
        product with the unit vectors of the standard basis.
        
        Args:
            w (torch.Tensor): The weight vector as a 1D-torch.Tensor
            X (torch.Tensor): The input batch.
            Y (torch.Tensor): The output label batch.
        """
        #self.invalidate_gradient_for_hv()
        H = w.new((self.d,self.d))#np.empty((self.d,self.d))
        v = torch.zeros_like(w)
        for i in range(len(v)):
            v[i] = 1
            H[i] = self.Hv_store_gradient(w,X,Y,v)
            v[i] = 0
        return H


######## Auxiliary Functions: robust Sigmoid, log(sigmoid) and 1-log(sigmoid) computations ########

def phi(t): #Author: Fabian Pedregosa
    # logistic function returns 1 / (1 + exp(-t))
    idx = t > 0
    out = np.empty(t.size, dtype=np.float)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out

def log_phi(t):
    # log(Sigmoid): log(1 / (1 + exp(-t)))
    idx = t>0
    out = np.empty(t.size, dtype=np.float)
    out[idx]=-np.log(1+np.exp(-t[idx]))
    out[~idx]= t[~idx]-np.log(1+np.exp(t[~idx]))
    return out

def one_minus_log_phi(t):
    # log(1-Sigmoid): log(1-1 / (1 + exp(-t)))
    idx = t>0
    out = np.empty(t.size, dtype=np.float)
    out[idx]= -t[idx]-np.log(1+np.exp(-t[idx]))
    out[~idx]=-np.log(1+np.exp(t[~idx]))
    return out

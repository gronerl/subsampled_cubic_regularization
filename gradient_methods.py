#################################
### Adaptive Gradient Methods ###
#################################

# This SGD implementation assumes a constant stepsize that can be specified as opt['learning_rate']= ...

# Authors: Jonas Kohler and Aurelien Lucchi, 2017
# Linus Groner, 2018:
#     * repurposed the sgd.py file as gradient_methods to cover all adaptive methods
#     * moved performance critical parts from NumPy to PyTorch,
#     * added adaptive gradient methods: Adagrad, RMSprop,AdaDelta, GGT
#     * increased number of parameters for which statistical data is collected
#     * other minor additions and changes,

from datetime import datetime
import numpy as np
import torch
from util import sampleSingle
import signal

def Gradient_Method(w, loss,gradient, X=None, Y=None, opt=None, statistics_callback=None, **kwargs):
    method_name = opt.get('method_name','SGD')
    if method_name not in ['SGD','Adagrad','RMSprop','AdaDelta','Adam','GGT']:
        raise NotImplementedError('Gradient method "'+method_name+'" unknown.')
        
    print ('---',method_name,'---')
    
    machine_precision = np.nextafter(1,2,dtype=w.new().cpu().numpy().dtype)-1
    
    n = X.shape[0]
    d = len(w)
   
    ## Reading parameters from opt.
    print('\n\n'+ method_name +' configuration:')
    
    
    # convergence criteria
    print('\n* termination criteria ')
    
    max_iterations = opt.get('max_iterations',float('inf'))
    if max_iterations != float('inf'):
        print('   - max_iterations:', max_iterations)
        
    max_epochs = opt.get('max_epochs',float('inf'))
    if max_epochs != float('inf'):
        print('   - max_epochs:', max_epochs)
        
    max_time = opt.get('max_time',float('inf'))
    if max_time != float('inf'):
        print('   - max_time:', max_time)
        
    grad_tol = opt.get('grad_tol',0.0)
    if grad_tol != 0.0:
        print('   - grad_tol:', grad_tol)
        
    if min(max_iterations,max_epochs,max_time)==float('inf') and grad_tol<=0.0:
        print('   - No termination criteria specified. Abort by sending SIGINT (e.g. CTRL-C) or SIGUSR2 (e.g. LSF timelimit).')

    #general gradient method parameters: learning rate and sampling parameters
    print('\n* general parameters ')
    eta = opt.get('learning_rate',1e-1)
    print('   - learning_rate:', eta)
    batch_size =int(opt.get('batch_size',0.01*n))
    print('   - batch_size:', batch_size)
    sample_loss =opt.get('sample_loss',False)#only necessary for tracking progress on very large models
    print('   - sample_loss:', sample_loss)
    replacement = opt.get('replacement', False)
    print('   - replacement:', replacement)

    
    #general gradient method parameters: learning rate and sampling parameters
    if method_name != 'SGD':
        print('\n* scaling parameters ')
        
    if method_name == 'SGD':
        pass
    elif method_name == 'Adagrad':
        epsilon = opt.get('epsilon',machine_precision)
        print('   - epsilon:', epsilon)
    elif method_name == 'RMSprop':
        beta = opt.get('beta',0.9)
        print('   - beta:', beta)
        epsilon = opt.get('epsilon',machine_precision)
        print('   - epsilon:', epsilon)
    elif method_name=='AdaDelta':
        beta_s = opt.get('beta_s',0.999)
        print('   - beta_s:', beta_s)
        beta_g = opt.get('beta_g',0.8)
        print('   - beta_g:', beta_g)
        epsilon = opt.get('epsilon',machine_precision)
        print('   - epsilon:', epsilon)
    elif method_name == 'Adam':
        beta_1 = opt.get('beta_1',0.9)
        print('   - beta_1:', beta_1)
        beta_2 = opt.get('beta_2',0.999)
        print('   - beta_2:', beta_2)
        epsilon = opt.get('epsilon',1e-8)
        print('   - epsilon:', epsilon)
    elif method_name == 'GGT':
        history_size = opt.get('history_size',50)
        print('   - history_size:', history_size)
        beta = opt.get('beta',1.0)
        print('   - beta:', beta)
        epsilon = opt.get('epsilon',np.sqrt(machine_precision))
        print('   - epsilon:', epsilon)
        
     # custom statistics collection
    print('\n* custom statistics collection:')
    if statistics_callback is not None:
        print('   - custom statistics callback specified')
    else:
        print('   - no custom statistics callback specified')
    print()
    
    
    ## initialize data recordings
    _X,_Y = sampleSingle(X, Y, batch_size, replacement)
    if sample_loss:
        _loss = loss(w, _X,_Y, **kwargs)
    else:
        _loss = loss(w, X, Y, **kwargs)
    grad = gradient(w, _X, _Y,**kwargs)
    grad_norm = torch.norm(grad)
    print ('Epoch ' + str(0) + ': loss={:.20f}'.format(_loss) + ' ||g||={:.3e}'.format(grad_norm),'time={:3e}'.format(0))
    
    #initialize data recordings
    stats_collector={
        'initial_guess': w.cpu().numpy(),
        'parameters_dict': opt,
        
        'time': [0],
        'samples': [0],
        'iterations': [0],
        'loss': [_loss],
        'grad_norm': [grad_norm],
        
        'step_norm': [],
        'travel_distance': [0],
    }
    
    if statistics_callback is not None:
        statistics_callback(0,w,stats_collector)
    
    ## initialize variables
    timing=0
    n_samples_seen = batch_size  # number of samples processed so far
    w0 = w
    k = 0
    i = 0 
    #set up termination in case of e.g. KeyBoardInterrupt or LSF timelimit
    class AbortException(Exception):
        pass
    def raiseAbortException(signum,frame):
        raise AbortException()
    signal.signal(signal.SIGUSR2, raiseAbortException)
    signal.signal(signal.SIGINT, raiseAbortException)

    start = datetime.now()
    try:
        while True:

            #### I: Update Scaling Matrix and calculate step#####
            if method_name == 'SGD':
                s = - eta * grad
            elif method_name == 'Adagrad':
                if i==0:
                    g_tau = torch.zeros_like(w)
                g_tau += grad * grad
                s = - eta * (1/(torch.sqrt(g_tau)+epsilon)) * grad
            elif method_name == 'RMSprop':
                if i==0:
                    g_tau = torch.zeros_like(w)
                g_tau = beta * g_tau + (1-beta) * grad*grad
                s = - eta * (1/(torch.sqrt(g_tau)+epsilon))*grad
            elif method_name == 'AdaDelta':
                if i==0:
                    g_tau = grad*grad
                    s_tau = torch.ones_like(grad)
                else:
                    g_tau = beta_g*g_tau + (1-beta_g)*grad*grad
                    s_tau = beta_s*s_tau + (1-beta_s)*s*s
                s = -eta*((torch.sqrt(s_tau)+epsilon)/(torch.sqrt(g_tau)+epsilon))*grad
            elif method_name == 'Adam':
                if i==0:
                    m = torch.zeros_like(w)
                    v = torch.zeros_like(w)
                m = beta_1*m+(1-beta_1)*grad
                v = beta_2*v+(1-beta_2)*grad * grad
                mhat = m/(1-beta_1**(k+1))
                vhat = v/(1-beta_2**(k+1))
                s =  - eta * (mhat/(torch.sqrt(vhat)+epsilon))
            elif method_name == 'GGT':
                if i==0:
                    Gt_buffer = w.new(history_size,d)
                Gt_buffer *= beta #columns residing in Gt_buffer for k iterations are weighted with beta**k
                Gt_buffer[k%history_size] = grad #k modulo history_size gives the index in a circular buffer
                if k+1<history_size: #history buffer is not yet full
                    Gt = Gt_buffer[:k+1]
                    wsize = k+1
                else:
                    Gt = Gt_buffer
                    wsize = history_size
                G = Gt.t() # we filled the transpose due to efficiency considerations
                GTG = torch.matmul(G.t(),G) 
                sig_r_squared,V = GTG.eig(eigenvectors=True)
                sig_r_squared = sig_r_squared[:,0]
                sig_r = torch.sqrt(sig_r_squared)
                sig_r[sig_r!=sig_r] = 0.0 # remove nans (occur in sqrt if 0 eigenvalues are returned as slightly negative by GTG.eig)
                sig_r_inv = 1/sig_r
                sig_r_inv[sig_r_inv==float('inf')] = 0.0 # pseudoinverse: use 0.0 where sig_r==0
                sig_r_inv[sig_r_inv==float('-inf')] = 0.0
                sig_r_inv = torch.diag(sig_r_inv) # technically not necessary to use dense matrix here, but nicer to read 
                Ur = torch.matmul(G,torch.matmul(V,sig_r_inv))
                def MinvV(v):# compute M^{-1}v with separate terms with epsilon and Sigma for better numerical properties 
                    Urv = torch.matmul(Ur.view(d,-1).t(),v)
                    epsterm = v - torch.matmul(Ur.view(d,-1),Urv)
                    tmp2 = torch.matmul(torch.diag(epsilon/(sig_r+epsilon)),Urv)
                    sigterm = torch.matmul(Ur.view(d,-1),tmp2)
                    return (sigterm+epsterm)/epsilon
                s = - eta*MinvV(grad)

            w = w+s
            
            #### II: Computing Gradient #####
            _X,_Y = sampleSingle(X, Y, batch_size, replacement)
            grad = gradient(w, _X, _Y,**kwargs)
            n_samples_seen += batch_size
            
            ### III: Save Iteration Information  ###
            if n_samples_seen >= n*(k+1) or i>=max_iterations:
                k += 1
                if w.is_cuda:# make sure all asynchronous computations complete inside the timed region
                    torch.cuda.synchronize()
                stop=datetime.now()
                timing_epoch = (stop - start).total_seconds()
                timing += timing_epoch
                if sample_loss:
                    _loss = loss(w, _X, _Y, **kwargs)
                else:
                    _loss = loss(w, X, Y, **kwargs)
                grad_norm = np.linalg.norm(grad)
                sn=torch.norm(s)
                if i>=max_iterations and n_samples_seen<n*k:
                    print ('Termination at {:.2f} epochs'.format(n_samples_seen/n) + ': loss={:.20f}'.format(_loss) + ' ||g||={:.3e}'.format(grad_norm),'time={:3e}'.format(timing),'dt={:.3e}'.format(timing_epoch), '||s||={:.3e}'.format(sn))
                else:
                    print ('Epoch ' + str(k) + ': loss={:.20f}'.format(_loss) + ' ||g||={:.3e}'.format(grad_norm),'time={:3e}'.format(timing),'dt={:.3e}'.format(timing_epoch), '||s||={:.3e}'.format(sn))
                if statistics_callback is not None:
                    statistics_callback(k,w,stats_collector)
                    if w.is_cuda:#time spent in the callback is not a property of the algorithm, and is hence excluded from the time measurements.
                         torch.cuda.synchronize()
                
                # record statistical data of the step
                stats_collector['time'].append(timing)
                stats_collector['samples'].append(n_samples_seen)
                stats_collector['loss'].append(_loss)
                stats_collector['travel_distance'].append(torch.norm(w-w0))
                stats_collector['iterations'].append(n_samples_seen/batch_size)
                stats_collector['grad_norm'].append(grad_norm) 
                stats_collector['step_norm'].append(sn)

                #check for termination
                if i >= max_iterations:
                    print('Terminating due to iteration limit.')
                    break
                if n_samples_seen/n >= max_epochs:
                    print('Terminating due to epoch limit')
                    break
                if timing>max_time:
                    print('Terminating due to time limit')
                    break
                if grad_norm < grad_tol:
                    print('Terminating due to gradient tolerance ( grad_norm =',grad_norm,'<',grad_tol,')')
                    break
                start=datetime.now()
            i += 1
    except AbortException:
        print('Caught SIGINT (e.g. CTRL-C) or SIGUSR2 (e.g. LSF Runtime limit) -- Breaking out.')
    return w.cpu().numpy(), stats_collector

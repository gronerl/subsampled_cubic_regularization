###########################
### Trust Region Method ###
###########################

# Authors: Jonas Kohler and Aurelien Lucchi, 2017
# Linus Groner, 2018:
#     * moved performance critical parts from NumPy to PyTorch,
#     * added support for scaled trust regions
#     * added exact subsolver exploiting tridiagonal subproblem structure
#     * increased number of parameters for which statistical data is collected
#     * implemented the propositions by Conn et al. Section 17.4 
#     * other minor additions and changes,

from termcolor import colored
from math import sqrt, ceil, log
import time
from datetime import datetime
import numpy as np
import torch
import scipy
from util import independentSampling, nestedSampling
import signal

def Trust_Region(w, loss, gradient, Hv=None, hessian=None, X=None, Y={}, opt=None, statistics_callback=None, **kwargs):
    """
    Minimize a continous, unconstrained function using the Trust Region method.

    References
    ----------
    Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Trust region methods. Society for Industrial and Applied Mathematics.

    Parameters
    ----------
    loss : callable f(w,X,Y,**kwargs)
        Objective function to be minimized.
    gradient : callable f'(w,X,Y,**kwargs), optional
        Gradient of f.
    Hv : callable Hv(w,X,Y,v,**kwargs)
        Matrix-vector-product of Hessian of f and arbitrary vector v
    hessian : callable hessian(w,X,Y,**kwargs), optional
        Hessian matrix of f 
    opt : dictionary, optional
        optional arguments passed to ARC
    X : 
        The input training set.
    Y : 
        The labels of the training set.
    statistics_callback: callable, optional
        This callback is called after each iteration, can be used to e.g. calculate validation loss
        or accuracy and other quantities. Must accept as input arguments:
        i : int, iteration counter
        w : torch.Tensor, current weight vector
        s : dict, current stats_dict
    **kwargs : dict, optional
        Extra arguments passed to loss, gradient and Hessian-vector-product computation,
        e.g. regularization constant or number of classes for softmax regression.
    """

    print ('--- Trust Region ---\n')
    
    machine_precision = np.nextafter(1,2,dtype=w.new().cpu().numpy().dtype)-1
    
    if X is None:
        n=1 
        d=len(w)
    else:
        n = X.shape[0]
        d = len(w)
    print('cardinality of dataset n =',n)
    print('dimension of parameter space d =',d)

    
    ## Reading parameters from opt.
    print('\n\nTR configuration:')
    
    
    # (outer) convergence criteria
    print('\n* (outer) termination criteria ')
    
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

        
    # adaption parameters
    print('\n* trust region adaption parameters:')
    
    tr_radius = opt.get('initial_trust_radius',1.)  # intial tr radius
    print('   - initial_trust_radius:', tr_radius)
    
    max_tr_radius = opt.get('max_trust_radius',float('inf'))  # max tr radius
    print('   - max_trust_radius:', max_tr_radius)
    
    assert (tr_radius > 0 and max_tr_radius > 0), "Trust region radius must be positive"
    
    eta_1 = opt.get('successful_threshold',0.1)
    print('   - successful_threshold:', eta_1)
    
    eta_2 = opt.get('very_successful_threshold',0.9)
    print('   - very_successful_threshold:', eta_2)
    
    gamma_1 = opt.get('radius_decrease_multiplier',2.)
    print('   - radius_decrease_multiplier:', gamma_1)
    
    gamma_2= opt.get('radius_increase_multiplier',2.)
    print('   - radius_increase_multiplier:', gamma_2)
    
    accept_all_decreasing = opt.get('accept_all_decreasing_steps',True)
    print('   - accept_all_decreasing_steps:', accept_all_decreasing)
    
    assert (gamma_1 >= 1 and gamma_2 >= 1), "Trust radius update parameters must be greater or equal to 1."
    
    # subsolver and related parameters
    print('\n* subsolver and related parameters:')
    subproblem_solver= opt.get('subproblem_solver','GLTR')
    print('   - subproblem_solver:', subproblem_solver)
    
    assert (( 
        not isinstance(subproblem_solver,DogLegTrSubproblemSolver) and not isinstance(subproblem_solver,ExactTrSubproblemSolver))
        or not w.is_cuda),"GPU support for dog_leg and exact subsolver is not implemented."
        
    if subproblem_solver=='GLTR':
        krylov_tol=opt.get('krylov_tol',1e-2)
        print('   - krylov_tol:', krylov_tol)
        exact_tol=opt.get('exact_tol',machine_precision)
        print('   - exact_tol:', exact_tol)
        subproblem_solver = GltrTrSubproblemSolver(exact_tol,krylov_tol)
    elif subproblem_solver=='CG':
        krylov_tol=opt.get('krylov_tol',1e-2)
        print('   - krylov_tol:', krylov_tol)
        subproblem_solver = CgTrSubproblemSolver(krylov_tol)
    elif subproblem_solver=='cauchy':
        subproblem_solver = CauchyTrSubproblemSolver()
    elif subproblem_solver=='dog_leg':
        subproblem_solver = DogLegTrSubproblemSolver()
    elif subproblem_solver=='exact':
        exact_tol=opt.get('exact_tol',machine_precision)
        print('   - exact_tol:', exact_tol)
        subproblem_solver = ExactTrSubproblemSolver(exact_tol)
    else:
        raise NotImplementedError('Subproblem solver "'+subproblem_solver+'" unknown.')
        
    # trust region shape and related parameters
    print('\n* trust region shape and related parameters:')
    
    scaling_matrix = opt.get('scaling_matrix','uniform')
    print('   - scaling_matrix: ', scaling_matrix)
    if scaling_matrix=='uniform':
        pass
    elif scaling_matrix=='Adagrad':
        epsilon = opt.get('epsilon',machine_precision)
        print('   - epsilon:', epsilon)
    elif scaling_matrix=='RMSprop':
        beta = opt.get('beta',0.8)
        print('   - beta:', beta)
        epsilon = opt.get('epsilon',machine_precision)
        print('   - epsilon:', epsilon)
    elif scaling_matrix=='AdaDelta':
        beta_s = opt.get('beta_s',0.999)
        print('   - beta_s:', beta_s)
        beta_g = opt.get('beta_g',0.8)
        print('   - beta_g:', beta_g)
        epsilon = opt.get('epsilon',machine_precision)
        print('   - epsilon:', epsilon)
    elif scaling_matrix=='approximate_hessian_diagonal':
        n_samples_hdiag = opt.get('n_samples_hdiag',2)
        print('   - n_samples_hdiag:', n_samples_hdiag)
        power = opt.get('power',1/3)
        print('   - power:', power)
        beta = opt.get('beta',0.8)
        print('   - beta:', beta)
        epsilon = opt.get('epsilon',machine_precision)
        print('   - epsilon:', epsilon)
    elif scaling_matrix=='GGT':
        history_size = opt.get('history_size',50)
        print('   - history_size:', history_size)
        beta = opt.get('beta',1.0)
        print('   - beta:', beta)
        epsilon = opt.get('epsilon',sqrt(machine_precision))
        print('   - epsilon:', epsilon)
    
    assert (( 
        not isinstance(subproblem_solver,DogLegTrSubproblemSolver) and not isinstance(subproblem_solver,ExactTrSubproblemSolver))
        or scaling_matrix == 'uniform'), "Scaled trust regions are not supported by dog_leg and exact subsolvers."

    #sampling parameters
    print('\n* sampling parameters:')
    #sampling with replacement
    replacement = opt.get('replacement', False)
    print('   - replacement:', replacement)
    
    #nested sampling means that the sub samples based on which the hessian, gradient and loss are computed, respectively
    #are subsets of each other.
    #independent denotes that these sets are independent of each other, yet does not imply that each sample is drawn i.i.d., 
    #namely not if replacement=True is specified.
    sampling_scheme = opt.get('sampling_scheme', 'independent') #alternatives:'nested'; 'independent'
    print('   - sampling_scheme:', sampling_scheme)
    if sampling_scheme == 'independent':
        sampling=independentSampling
    if sampling_scheme == 'nested':
        sampling=nestedSampling
        assert ((not replacement) or (sampling_scheme != 'nested')), "A combination of nested sub-samples and sampling with replacement is not meaningful."
        
    #determine if the Hessian is sub-sampled and the Hessian sub-sample size
    initial_sample_size_Hessian=int(opt.get('initial_sample_size_Hessian', int(0.05*n) ))
    Hessian_sampling_flag=opt.get('Hessian_sampling', initial_sample_size_Hessian<n )
    print('   - Hessian_sampling_flag:', Hessian_sampling_flag )
    if Hessian_sampling_flag:
        print('   - initial_sample_size_Hessian:', initial_sample_size_Hessian )
    else:
        initial_sample_size_Hessian = n

    #determine if the gradient is sub-sampled and the gradient sub-sample size
    initial_sample_size_gradient=int(opt.get('initial_sample_size_gradient',n))
    gradient_sampling_flag=opt.get('gradient_sampling', initial_sample_size_gradient<n)
    print('   - gradient_sampling_flag:', gradient_sampling_flag)
    if gradient_sampling_flag:
        print('   - initial_sample_size_gradient:', initial_sample_size_gradient)
    else:
        initial_sample_size_gradient = n

    #determine if the loss is sub-sampled and the loss sub-sample size
    initial_sample_size_loss = int(opt.get('initial_sample_size_loss',n))
    loss_sampling_flag=opt.get('loss_sampling', initial_sample_size_loss<n)
    print('   - loss_sampling_flag:', loss_sampling_flag)
    if loss_sampling_flag:
        print('   - initial_sample_size_loss:', initial_sample_size_loss)
    else:
        initial_sample_size_loss = n

    if loss_sampling_flag or gradient_sampling_flag or Hessian_sampling_flag:
        assert (X is not None and Y is not None), "Subsampling is only possible if data is passed, i.e. X and Y may not be None"

    #sampling parameters
    print('\n* sampling size scheme:')
    
    sample_size_scheme=opt.get('sample_size_scheme', 'constant')
    print('   - sample_size_scheme:',sample_size_scheme)
    
    if sample_size_scheme=='constant':
        #keep same sample size across all iterations
        pass

    elif sample_size_scheme=='linear':
        #increase sample size by lin_growth_constant (additive) each iteration
        if  (max_iterations <= 0 or max_iterations >= float('inf') ):
            assert('lin_growth_constant' in opt and opt['lin_growth_constant']>0 and opt['lin_growth_constant']<float('inf')), "Linear sample size scheme only possible if max_iterations or a finite, positive lin_growth_constant is specified."
        lin_growth_constant = opt.get('lin_growth_constant',(n/max_iterations))
        print('   - lin_growth_constant:',lin_growth_constant)
    
    elif sample_size_scheme=='exponential':
        #increase sample size by the factor exp_growth_constant each iteration
        if  (max_iterations <= 0 or max_iterations >= float('inf') ):
            assert('exp_growth_constant' in opt and opt['exp_growth_constant']>1 and opt['exp_growth_constant']<float('inf')), "Linear sample size scheme only possible if max_iterations or a finite exp_growth_constant >1 is specified."
        exp_growth_constant = opt.get('exp_growth_constant',((n-initial_sample_size_Hessian)**(1/max_iterations)).real)
        print('   - exp_growth_constant:',exp_growth_constant)
    
    elif sample_size_scheme=='adaptive':
        #determine sample size based on success and step length of previous step
        unsuccessful_sample_scaling=opt.get('unsuccessful_sample_scaling',1.25)
        print('   - unsuccessful_sample_scaling:',unsuccessful_sample_scaling)
        sample_scaling_Hessian=opt.get('sample_scaling_Hessian',1.25)
        print('   - sample_scaling_Hessian:',sample_scaling_Hessian)
        sample_scaling_gradient=opt.get('sample_scaling_gradient',1)
        print('   - sample_scaling_gradient:',sample_scaling_gradient)

    # custom statistics collection
    print('\n* custom statistics collection:')
    if statistics_callback is not None:
        print('   - custom statistics callback specified')
    else:
        print('   - no custom statistics callback specified')
    print()

    ## initialize data recordings
    #draw a sample to compute the initial loss.
    (_X3,_Y3),(_X2,_Y2),(_X,_Y) = sampling(X, Y, initial_sample_size_loss, initial_sample_size_gradient, initial_sample_size_Hessian, replacement)
    _loss = loss(w,_X3,_Y3,**kwargs)
    
    # initialize data recordings
    stats_collector={
        'initial_guess': w.cpu().numpy(),
        'parameters_dict': opt,
        
        'time': [0],
        'samples': [0],
        'loss': [_loss],
        'grad_norm': [], #the grad_norm at w0 will be added after sampling the gradient again
        
        'step_norm': [],
        'step_norm_scaled': [],
        'travel_distance': [0],
        
        'tr_radius': [tr_radius],
        
        'sample_size_Hessian': [initial_sample_size_Hessian],
        'sample_size_gradient': [initial_sample_size_gradient],
        'sample_size_loss': [initial_sample_size_loss],
        
        'subproblem_stats': []
    }
    
    if statistics_callback is not None:
        statistics_callback(0,w,stats_collector)

    ## initialize variables
    timing=0 # counts total runtime
    n_samples_seen = 0 # counts number of samples
    previous_f = _loss # needed to compute function decrease
    
    accepted_flag=False
    successful_flag=False
    w0 = w # initial guess to record distance traveled
    k = 0 # iteration counter
    
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
            #### I: Subsampling #####
            ## a) determine batchsize ##
            if sample_size_scheme == 'exponential':
                sample_size_Hessian = min(n, int(initial_sample_size_Hessian + exp_growth_constant**(k)-1))
                sample_size_gradient = min(n, int(initial_sample_size_gradient + exp_growth_constant**(k)-1))
                sample_size_loss = min(n, initial_sample_size_loss + exp_growth_constant**(k)-1)
                
            elif sample_size_scheme == 'linear':
                sample_size_Hessian = min(n, max(initial_sample_size_Hessian,int(lin_growth_constant*k)))
                sample_size_gradient= min(n, max(initial_sample_size_gradient,int(lin_growth_constant*k)))
                sample_size_loss= min(n, max(initial_sample_size_loss,int(lin_growth_constant*k)))
                
            elif sample_size_scheme=='adaptive':
                if k==0:
                    sample_size_Hessian=initial_sample_size_Hessian
                    sample_size_gradient=initial_sample_size_gradient
                else:
                    #adjust sampling constant c such that the first step would have given a sample size of initial_sample_size
                    if k==1:
                        c_Hessian=(initial_sample_size_Hessian*sn**2)/log(d)
                        c_gradient=(initial_sample_size_gradient*sn**4)/log(d)
                    sample_size_Hessian_unsucc=min(n,ceil(sample_size_Hessian*unsuccessful_sample_scaling))
                    sample_size_gradient_unsucc=min(n,ceil(sample_size_gradient*unsuccessful_sample_scaling))
                    sample_size_Hessian_adapt=min(n,int(max((c_Hessian*log(d)/(sn**2)*sample_scaling_Hessian),initial_sample_size_Hessian)))    
                    sample_size_gradient_adapt=min(n,int(max((c_gradient*log(d)/(sn**4)*sample_scaling_gradient),initial_sample_size_gradient)))
                    if accepted_flag==False:
                        sample_size_Hessian = sample_size_Hessian_unsucc
                        sample_size_gradient = sample_size_gradient_unsucc
                    elif successful_flag==False:
                        sample_size_Hessian = max(sample_size_Hessian_adapt,sample_size_Hessian_unsucc)
                        sample_size_gradient = max(sample_size_gradient_adapt,sample_size_gradient_unsucc)
                    else:
                        sample_size_Hessian = sample_size_Hessian_adapt
                        sample_size_gradient = sample_size_gradient_adapt
                sample_size_loss = max(sample_size_gradient,sample_size_Hessian,initial_sample_size_loss)
            #end elif sample_size_scheme=='adaptive': 
            elif sample_size_scheme=='constant':
                sample_size_Hessian = int(initial_sample_size_Hessian)
                sample_size_gradient = int(initial_sample_size_gradient)
                sample_size_loss = int(initial_sample_size_loss)
            else:
                raise NotImplementedError('Sampling size scheme "'+method_name+'" unknown.')
            n_samples_per_step = sample_size_Hessian+sample_size_gradient #TODO or max(sample_size_Hessian,sample_size_gradient,sample_size_loss)

            ## b) draw batches ##
            (_X3,_Y3),(_X2,_Y2),(_X,_Y) = sampling(X,Y,sample_size_loss,sample_size_gradient,sample_size_Hessian,replacement)
            
            ## c) recompute gradient either because of accepted step or because of re-sampling ##
            if (gradient_sampling_flag==True or accepted_flag==True or k==0):
                grad = gradient(w, _X2, _Y2, **kwargs)
                grad_norm =torch.norm(grad)
                if grad_norm < grad_tol:
                    print('Terminating due to gradient tolerance ( grad_norm =',grad_norm,'<',grad_tol,')')
                    break
            stats_collector['grad_norm'].append(grad_norm)
            n_samples_seen += n_samples_per_step

            #### II: Update Scaling Matrix #####
            if scaling_matrix == 'uniform':
                if k==0:
                    Mdiag = torch.ones_like(grad)
            elif scaling_matrix == 'Adagrad':
                if k==0:
                    g_tau = grad*grad
                elif accepted_flag:
                    g_tau += grad*grad 
                Mdiag = torch.sqrt(g_tau)+epsilon
                
            elif scaling_matrix == 'RMSprop':
                if k==0:
                    g_tau = grad*grad
                elif accepted_flag:
                    g_tau = beta*g_tau + (1-beta)*grad*grad
                Mdiag = torch.sqrt(g_tau)+epsilon
                
            elif scaling_matrix == 'approximate_hessian_diagonal':
                tmp = w.new(w.size())
                if k==0:
                    hdiag = torch.zeros_like(w)
                    for i in range(10):
                        tmp.normal_()
                        hdiag += Hv(w,_X,_Y,tmp)*tmp
                    h = torch.abs(hdiag/10)
                else:
                    hdiag = torch.zeros_like(w)
                    for i in range(n_samples_hdiag):
                        tmp.normal_()
                        hdiag += Hv(w,_X,_Y,tmp)*tmp
                    h = beta*h + (1-beta)*torch.abs(hdiag/n_samples_hdiag)
                Mdiag = torch.pow(h,power)+epsilon
                
            elif scaling_matrix == 'AdaDelta':
                if k==0:
                    g_tau = grad*grad
                    s_tau = torch.ones_like(grad)
                elif accepted_flag:
                    g_tau = beta_g*g_tau + (1-beta_g)*grad*grad
                    s_tau = beta_s*s_tau + (1-beta_s)*s*s
                Mdiag = (torch.sqrt(g_tau)+epsilon)/(torch.sqrt(s_tau)+epsilon)
                
            elif scaling_matrix == 'GGT':
                if k==0:
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
                
                def MV(v):# compute Mv as Ur^T Sigma_r Ur v + epsilon*v
                    Urv = torch.matmul(Ur.view(d,-1).t(),v)
                    tmp2 = torch.matmul(torch.diag(sig_r),Urv)
                    tmp2 = torch.matmul(Ur.view(d,-1),tmp2)
                    tmp2 += v*epsilon
                    return tmp2
                
                def MinvV(v):# compute M^{-1}v with separate terms with epsilon and Sigma for better numerical properties 
                    Urv = torch.matmul(Ur.view(d,-1).t(),v)
                    epsterm = v - torch.matmul(Ur.view(d,-1),Urv)
                    tmp2 = torch.matmul(torch.diag(epsilon/(sig_r+epsilon)),Urv)
                    sigterm = torch.matmul(Ur.view(d,-1),tmp2)
                    return (sigterm+epsterm)/epsilon
            else:
                raise NotImplementedError('Trust region scaling "'+method_name+'" unknown.')
                
            if not scaling_matrix == 'GGT':#wrap multiplication with Mdiag to fit suitable interface for GGT
                def MV(v):
                    return v*Mdiag
                def MinvV(v):
                    return v/Mdiag
            
            #### III: Compute Step ####
            subproblem_stats_collector={} #filled by subsolver
            s = subproblem_solver(grad, Hv, hessian, tr_radius, _X, _Y, w,MV, MinvV, accepted_flag, subproblem_stats_collector,**kwargs)
            stats_collector['subproblem_stats'].append(subproblem_stats_collector)

            sn=torch.norm(s)
            stats_collector['step_norm'].append(sn)
            sns = np.sqrt(torch.dot(s,MV(s)))
            stats_collector['step_norm_scaled'].append(sns)

            #### IV: Regularization Update ####
            current_f = loss(w+s, _X3, _Y3,**kwargs)
            #if loss is sampled, previous and current f need to be computed on same sample.
            #otherwise, the loss is known from the previous iteration
            if loss_sampling_flag:
                previous_f = loss(w, _X3, _Y3,**kwargs)

            function_decrease = previous_f - current_f
            model_decrease = subproblem_solver.decrease
            
            rho = function_decrease / model_decrease
            
            #handle special/numerical corner cases. I'm not sure which case should take precedence.
            if model_decrease < 0:
                print('Negative model decrease.')
                rho = float('-inf')
                #the problem is, that if both function and model decrease are negative,
                #this results in a positive rho, despite a bad step.
            if model_decrease < machine_precision and function_decrease < machine_precision: # see Conn et al. Section 17.4.2
                print('Small decreases, setting rho=1')
                rho = 1.0
                
            #Update w if step s is successful
            if (accept_all_decreasing and function_decrease>=0) or rho > eta_1:
                w = w+s
                _loss=current_f
                previous_f = current_f
                accepted_flag=True
            else:
                accepted_flag=False
                _loss=previous_f

            # Update trust region radius
            successful_flag=True
            _tr_radius=tr_radius
            if rho < eta_1:
                tr_radius *= 1/gamma_1
                successful_flag=False
            elif rho > eta_2:
                #increase only if boundary point, plus some tolerance for numerical reasons 
                #(falsely increasing doesn't hurt that much if interior point is found, and
                #the problems houldn't reocur if it really was an interior points, since in
                #this case the difference between sns and tr_radius will grow)
                if np.abs(sns - tr_radius) < tr_radius/gamma_2:
                    tr_radius = min(gamma_2 * tr_radius, max_tr_radius)

            ### V: Save Iteration Information  ###
            #we exclude this part from timing to be consistent with gradient methods, where this part
            #is excluded since it also computes the loss, which is not an inherent part of gradient methods,
            #and only for the purpose of tracking progress. 
            if statistics_callback is not None:
                if w.is_cuda:#time spent in the callback is not a property of the algorithm, and is hence excluded from the time measurements.
                     torch.cuda.synchronize()
            timing_iteration=(datetime.now() - start).total_seconds()
            
            if statistics_callback is not None:
                statistics_callback(k+1,w,stats_collector)
            
            timing += timing_iteration
            print ('Iter ' + str(k) + ': loss={:.20f}'.format(_loss) + ' ||g||={:.3e}'.format(grad_norm),'time={:3e}'.format(timing),'dt={:.3e}'.format(timing_iteration), 'tr_radius={:.3e}'.format(_tr_radius))
            print(''.join([' ']*(6+len(str(k)))),'||s||={:.3e}'.format(sn),'||s||_M={:.3e}'.format(sns),'samples Hessian=', int(sample_size_Hessian),'samples gradient=', int(sample_size_gradient),'samples loss=', int(sample_size_loss))
            print(''.join([' ']*(6+len(str(k)))),'epoch={:.3e}'.format(n_samples_seen/n),'rho={:.6e}'.format(rho),'accepted=',colored(str(accepted_flag),('green' if accepted_flag else 'red')),'successful='+colored(str(successful_flag),('green' if successful_flag else 'red')),"\n")
            
            # record statistical data of the step
            stats_collector['time'].append(timing)
            stats_collector['samples'].append(n_samples_seen)
            stats_collector['sample_size_Hessian'].append(sample_size_Hessian)
            stats_collector['sample_size_gradient'].append(sample_size_gradient)
            stats_collector['sample_size_loss'].append(sample_size_loss)
            stats_collector['loss'].append(_loss)
            stats_collector['travel_distance'].append(float(torch.norm(w-w0)))
            stats_collector['tr_radius'].append(tr_radius)
            
            #check for termination
            k += 1
            if k >= max_iterations:
                print('Terminating due to iteration limit')
                break
            if n_samples_seen/n >= max_epochs:
                print('Terminating due to epoch limit')
                break
            if timing>max_time:
                print('Terminating due to time limit')
                break
                
            start = datetime.now()

    except AbortException:
        print('Caught SIGINT (e.g. CTRL-C) or SIGUSR2 (e.g. LSF Runtime limit) -- Breaking out.')
    return w.cpu().numpy(), stats_collector

class CauchyTrSubproblemSolver:
    """
    This solver calculates the generalized Cauchy point of the subproblem,
    which is the minimizer of a linear model s.t. the scaled boundary constraints,
    rescaled as a second order minimizer along that direction. (Note that this is
    not the gradient direction in general, but the direction of an adaptive gradient
    method with the same scaling matrix.)
    See 
    Nocedal, Jorge, and Stephen J. Wright. (2006) Numerical optimization 2nd edition.
    Algorithm 4.4
    """
    
    def __call__(self,grad,Hv, hessian ,tr_radius, X, Y, w, MV, MinvV, accepted_flag,collector,**kwargs):
        Minvg = MinvV(grad)
        HMinvg=Hv(w, X, Y, Minvg,**kwargs)
        gBg = torch.dot(Minvg, HMinvg) # actually MinvgBMinvg
        gMg = torch.dot(grad,Minvg) # actually gMinvg
        gnorm = np.sqrt(gMg)
        tau = 1
        if gBg > 0:  # if model is convex quadratic the unconstrained minimizer may be inside the TR
            tau = min(gnorm ** 3 / (tr_radius * gBg), 1)
        
        dc = - float(tau * tr_radius/gnorm)
        pc = dc * Minvg
        self.decrease = - dc*gMg - (dc)**2*gBg/2 # see Conn et al. Sect. 17.4.1
        return pc


class GltrTrSubproblemSolver:
    """
    Generalized Lanczos Trust Region algorithm, Conn et al. 
    """
    def __init__(self, exact_tol, krylov_tol):
        self.exact_tol = exact_tol
        self.krylov_tol = krylov_tol
        self.lambda_k = 0
        
    def __call__(self,grad,Hv, hessian ,tr_radius, X, Y, w, MV, MinvV, accepted_flag,collector,**kwargs):
        machine_precision = np.nextafter(1,2,dtype=w.new().cpu().numpy().dtype)-1
        from scipy import linalg
        dimensionality = len(grad)
        
        g = grad
        s = torch.zeros_like(g)
        self.dim = len(grad)

        if (g == 0).all():
            try:
                print ('zero gradient encountered')
                exact_subsolver = ExactTrSubproblemSolver(self.exact_tol,self.lambda_k)
                s =  exact_subsolver(grad,Hv, hessian ,tr_radius, X, Y, w, MV, MinvV, accepted_flag,collector,**kwargs)
                self.decrease = exact_subsolver.decrease
                self.lambda_k = exact_subsolver.lambda_k
                return s
            except MemoryError:
                print('MemoryError when handling zero gradient, continuing')
        else: 
            # initialize
            v = MinvV(g)
            p = -v
            gamma_next = gamma = np.sqrt(torch.dot(v,g))
            alpha_k = []
            beta_k = []
            gamma_k = [gamma_next]
            Hp_store = []
            interior_flag = True
            k = 0

            while True:
                gn = float(gamma_k[k])
                if k == 0:
                    sigma = 1.0
                else:
                    sigma = -np.sign(alpha_k[k - 1]) * sigma
                self.growQ(sigma * v / gn,k)

                Hp=Hv(w, X, Y, p, **kwargs)
                pHp = torch.dot(p,Hp)
                alpha = float(gamma_k[k])**2 / pHp

                alpha_k.append(alpha)

                ###Lanczos Step 1: Build up subspace 
                # a) Create g_lanczos = gamma*e_1
                e_1 = np.zeros(k + 1) 
                e_1[0] = 1.0
                g_lanczos = gamma * e_1
                # b) Create T for Lanczos Model
                if k == 0:
                    Tdiag = np.array([1./alpha])
                    Toffdiag = np.array([])
                else:
                    Tdiag = np.append(Tdiag,[1. / alpha + beta/alpha_k[k-1] ])
                    Toffdiag = np.append(Toffdiag,sqrt(beta)/abs(alpha_k[k-1]))

                #if (interior_flag == True and alpha <= 0) or np.sqrt(np.dot(s + alpha * p,d*(s + alpha * p))) >= tr_radius:
                if (interior_flag == True and alpha <= 0) or np.sqrt(torch.dot(s+alpha*p,MV(s+alpha*p)))>=tr_radius:
                    interior_flag = False


                if interior_flag == True:
                    s += alpha*p
                else:
                    ###Lanczos Step 2: solve problem in subspace
                    if k>0:
                        h = self.exact_TR_subproblem_solver_tridiagonal(g_lanczos, Tdiag,Toffdiag, tr_radius, accepted_flag)
                    else:# avoids having to handle the 1d case where the offidagonals are empty lists
                        exact_solver = ExactTrSubproblemSolver(self.exact_tol,self.lambda_k)
                        h = exact_solver.exact_TR_subproblem_solver(g_lanczos, np.array([[Tdiag[0]]]), tr_radius, accepted_flag)
                        self.lambda_k = exact_solver.lambda_k
                g_next = g+alpha*Hp
                v_next = MinvV(g_next)
                # test for convergence
                e_k = np.zeros(k + 1)
                e_k[k] = 1.0
                gamma_next = np.sqrt(torch.dot(g_next,v_next))
                gamma_k.append(gamma_next)
                
                #if interior_flag == True and gamma_next < self.krylov_tol:
                if interior_flag == True and gamma_next < min(np.sqrt(gamma),self.krylov_tol)*gamma:
                    print('Interior point after',k+1,'iterations')
                    break
                #if interior_flag == False and gamma_next * abs(np.dot(h,e_k)) < self.krylov_tol:
                if interior_flag == False and  gamma_next *  abs(np.dot(h, e_k)) < min(np.sqrt(gamma),self.krylov_tol)*gamma:
                    print('Boundary point after',k+1,'iterations')
                    break
                if k==dimensionality-1:
                    print ('Krylov dimensionality reached full space! Breaking out..')
                    break
                beta = float(gamma_k[k+1])**2 / float(gamma_k[k])**2
                beta_k.append(beta)
                p = -v_next + beta*p
                g = g_next
                v = v_next
                k = k + 1
                
            if interior_flag == False:
                tmp = torch.from_numpy(h).type(self.Q.type())
                s = torch.matmul(tmp,self.Q[:k+1,:])
                Th = h*Tdiag
                if k>0:
                    Th[:-1] += h[1:]*Toffdiag
                    Th[1:] += h[:-1]*Toffdiag
                hTh = np.dot(h,Th)
                self.decrease = - gamma * h[0] - 0.5*hTh # See Conn et al. Section 17.4.1
            else:
                self.decrease = 0.5*sum([aa*gg**2 for aa,gg in zip(alpha_k,gamma_k[:-1])]) # See Conn et al. Section 17.4.1
        
        collector['sub_iterations']=k+1
        collector['interior_flag']=interior_flag
        return s

    def exact_TR_subproblem_solver_tridiagonal(self,grad, Hdiag,Hoffdiag, tr_radius, accepted_flag):
        from scipy import linalg
        from scipy.sparse import diags
        d = len(Hdiag)
        
        H_band = np.empty((2,d))
        H_band[1,:] = Hdiag
        H_band[0,1:] = Hoffdiag
        s = np.zeros_like(grad)
        
        ## Step 0: initialize safeguards
        absHdiag = np.abs(Hdiag)
        absHoffdiag = np.abs(Hoffdiag)
        H_ii_min = min(Hdiag)
        H_max_norm = sqrt(len(Hdiag) ** 2) * max(absHoffdiag.max(),absHdiag.max())
        H_fro_norm = np.sqrt(2*np.dot(absHoffdiag,absHoffdiag) + np.dot(absHdiag,absHdiag))

        gerschgorin_l = max([Hdiag[0]+absHoffdiag[0] , Hdiag[d-1]+absHoffdiag[d-2]])
        gerschgorin_l = max([gerschgorin_l]+[Hdiag[i]+absHoffdiag[i]+absHoffdiag[i-1] for i in range(1,d-1)])#see conn2000,sect 7.3.8, \lambda^L
        gerschgorin_u = max([-Hdiag[0]+absHoffdiag[0] , -Hdiag[d-1]+absHoffdiag[d-2]])
        gerschgorin_u = max([gerschgorin_u]+[-Hdiag[i]+absHoffdiag[i]+absHoffdiag[i-1] for i in range(1,d-1)])
        lambda_lower = max(0, -H_ii_min, np.linalg.norm(grad) / tr_radius - min(H_fro_norm, H_max_norm, gerschgorin_l))
        lambda_upper = max(0, np.linalg.norm(grad) / tr_radius + min(H_fro_norm, H_max_norm, gerschgorin_u))
        #print('diag',lambda_lower,lambda_upper,gerschgorin_l,gerschgorin_u,H_ii_min,H_max_norm,H_fro_norm)

        if accepted_flag==False and lambda_lower <= self.lambda_k <= lambda_upper: #reinitialize at previous lambda in case of unscuccesful iterations
            lambda_j=self.lambda_k
        elif lambda_lower == 0:  # allow for fast convergence in case of inner solution
            lambda_j = lambda_lower
        else:
            #lambda_j = (lambda_upper+lambda_lower)/2
            lambda_j=np.random.uniform(lambda_lower, lambda_upper)

        i=0
        # Root Finding
        phi_history = []
        lambda_history = []
        while True:
            i+=1
            lambda_in_N = False
            lambda_plus_in_N = False
            #B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
            B = np.empty((2,d))
            B[1,:] = Hdiag + lambda_j
            B[0,1:] = Hoffdiag
            try:
                # 1 Factorize B
                L = linalg.cholesky_banded(B)
                L = L[[1,0]]
                L[1,:-1] = L[1,1:]
                # 2 Solve LL^Ts=-g
                s = linalg.solveh_banded(B,-grad)
                sn = np.linalg.norm(s)
                
                ## 2.1 Termination: Lambda in F, if q(s(lamda))<eps_opt q(s*) and sn<eps_tr tr_radius -> stop. By Conn: Lemma 7.3.5:
                phi_lambda = 1. / sn - 1. / tr_radius
                
                #if (abs(sn - tr_radius) <= exact_tol * tr_radius):
                if i>1 and phi_lambda in phi_history: #detect if we are caught in a loop due to finite precision
                    lambda_j = lambda_history[np.argmin(phi_history)] #pick best lambda observed so far and exit
                    break
                else:
                    phi_history.append(phi_lambda)
                    lambda_history.append(lambda_j)
                if (abs(phi_lambda)<=self.exact_tol): #
                    break

                # 3 Solve Lw=s
                w = linalg.solve_banded((1,0),L,s)
                wn = np.linalg.norm(w)

                ##Step 1: Lambda in L
                if lambda_j > 0 and (phi_lambda) < 0:
                    #print ('lambda: ',lambda_j, ' in L')
                    lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)
                    lambda_j = lambda_plus


                ##Step 2: Lambda in G    (sn<tr_radius)
                elif (phi_lambda) > 0 and lambda_j > 0 and np.any(grad != 0): #TBD: remove grad
                    lambda_upper = lambda_j
                    lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)

                    ##Step 2a: If factorization succeeds: lambda_plus in L
                    if lambda_plus > 0:
                        try:
                            # 1 Factorize B
                            B_plus = np.empty((2,d))
                            B_plus[1,:] = Hdiag + lambda_plus
                            B_plus[0,1:] = Hoffdiag
                            _ = linalg.cholesky_banded(B_plus) # throws LinAlgError of B_plus is not pos.def.
                            lambda_j = lambda_plus

                        except np.linalg.LinAlgError:
                            lambda_plus_in_N = True

                    ##Step 2b/c: If not: Lambda_plus in N
                    if lambda_plus <= 0 or lambda_plus_in_N == True:
                        # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
                        try:
                            _ = linalg.cholesky_banded(H_band)
                            H_pd = True
                        except np.linalg.LinAlgError:
                            H_pd = False

                        if lambda_lower == 0 and H_pd == True and phi_lambda >= 0: #cannot happen in ARC!
                            lambda_j = 0
                            break
                            
                        # 2. Else, choose a lambda within the safeguard interval
                        else:
                            lambda_lower = max(lambda_lower, lambda_plus)  # reset lower safeguard
                            lambda_j = max(sqrt(lambda_lower * lambda_upper),
                                           lambda_lower + 0.01 * (lambda_upper - lambda_lower))

                            lambda_upper = float(
                                lambda_upper) 
                            #if lambda_lower == lambda_upper:
                            if lambda_upper <= np.nextafter(lambda_lower,lambda_upper,dtype=Hdiag.dtype):
                                lambda_j = lambda_lower
                                ew,ev = linalg.eig_banded(H_band,select_range=(0,0))
                                ew = ew[0]
                                d = ev[:, 0]
                                dn = np.linalg.norm(d)
                                
                                #assert usually doesn't hold in finite precision (only approximately)
                                #assert (ew == -lambda_j), "Ackward: in hard case but lambda_j != -lambda_1"

                                tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-tr_radius**2)
                                s=s + tao_lower * d 
                                print ('hard case resolved inside')
                                
                                self.lambda_k = lambda_j
                                return s

                elif (phi_lambda) == 0: 
                    break
                else:      #TBD:  move into if lambda+ column #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                    lambda_in_N = True
            ##Step 3: Lambda in N
            except np.linalg.LinAlgError:
                lambda_in_N = True
            if lambda_in_N == True:
                try:
                    _ = linalg.cholesky_banded(H_band)
                    H_pd = True
                except np.linalg.LinAlgError:
                    H_pd = False

                # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
                if lambda_lower == 0 and H_pd == True and phi_lambda >= 0: 
                    lambda_j = 0
                    break
                # 2. Else, choose a lambda within the safeguard interval
                else:
                    lambda_lower = max(lambda_lower, lambda_j)  # reset lower safeguard
                    lambda_j = max(sqrt(lambda_lower * lambda_upper),
                                   lambda_lower + 0.01 * (lambda_upper - lambda_lower))  # eq 7.3.14
                    lambda_upper = float(lambda_upper)  
                    
                    # Check for Hard Case:
                    #print('884',i,lambda_upper-lambda_lower,np.nextafter(lambda_lower,lambda_upper)-lambda_lower,np.nextafter(1,2,dtype=Hdiag.dtype)-1,np.nextafter(1,2,dtype=grad.dtype)-1)
                    if lambda_upper <= np.nextafter(lambda_lower,lambda_upper,dtype=Hdiag.dtype):
                    #if lambda_lower == lambda_upper:
                        lambda_j = lambda_lower
                        ew,ev = linalg.eig_banded(H_band,select_range=(0,0))
                        ew = ew[0]
                        d = ev[:, 0]
                        dn = np.linalg.norm(d)
                        
                        #assert usually doesn't hold in finite precision (only approximately)
                        #assert (ew == -lambda_j), "Ackward: in hard case but lambda_j != -lambda_1"

                        tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-tr_radius**2)
                        s=s + tao_lower * d 

                        print ('hard case resolved outside')
                        self.lambda_k = lambda_j
                        return s



        # compute final step
        B = np.empty((2,d))
        B[1,:] = Hdiag + lambda_j
        B[0,1:] = Hoffdiag
        # Solve LL^Ts=-g
        s = linalg.solveh_banded(B,-grad)
        self.lambda_k = lambda_j
        return s
    
    def growQ(self,v,j):
        """
        Manges the matrix Q.
        
        Q is kept allocated across iteration to prevent repeated expensive allocations.
        
        The allocated memory doubles in size each time its capacity is exceeded,
        leading to amortized complexity for growing sizes of Q of O(dj), as oposed to O(dj^2)
        if Q is grown and copied in each iteration.
        """
        
        # lazy initialization
        try:
            self.Q
        except:
            self.Q = v.new(10,self.dim)
            
        # if capacity of Q is exceeded, double it's size, and refill the columns from before.
        if j>=self.Q.size()[0]:
            Qtmp = self.Q
            self.Q = self.Q.new(j*2,self.dim)
            self.Q[:j,:] = Qtmp
            
        # set v 
        self.Q[j,:] = v

class CgTrSubproblemSolver:
    
    def __init__(self,krylov_tol):
        self.krylov_tol = krylov_tol
        
    def __call__(self,grad,Hv, hessian ,tr_radius, X, Y, w, MV, MinvV, accepted_flag,collector,**kwargs): 
            grad_norm = sqrt(torch.dot(grad,MV(grad)))
            p_start = torch.zeros_like(grad)

#             if grad_norm < min(sqrt(grad_norm) * grad_norm,self.krylov_tol):
#                 print('no step!')
#                 self.decrease=0
#                 return p_start

            # initialize
            z = p_start
            r = -grad
            r_tilde = MinvV(r)
            gamma_k = [torch.dot(r,r_tilde)]
            d = r_tilde
            k = 0
            alpha_k=[]
            while True:
                Bd=Hv(w, X, Y, d, **kwargs)
                dBd = torch.dot(d, Bd)
                # terminate when encountering a direction of negative curvature with lowest boundary point along current search direction
                if dBd <= 0:
                    Mz = MV(z)
                    zMz = torch.dot(z,Mz)
                    dMd = torch.dot(d,MV(d))
                    zMd = torch.dot(d,Mz)
                    t_lower, t_upper = mitternachtsformel_torch(dMd,2*zMd,zMz-tr_radius**2) #solve_quadratic_equation_torch(z, d, tr_radius)
                    collector['sub_iterations']=k+1
                    collector['interior_flag']=False
                    
                    self.decrease = 0.5*sum([aa*gg for aa,gg in zip(alpha_k,gamma_k[:-1])])
                    self.decrease += t_upper*gamma_k[-1]-0.5*t_upper**2*dBd
                    print('Boundary point after',k+1,'iterations.')
                    return z + t_upper*d


                alpha = torch.dot(r, r_tilde) / dBd
                alpha_k.append(alpha)
                z_next = z + alpha * d
                # terminate if z_next violates TR bound
                Mz = MV(z_next)
                zMz = torch.dot(z_next,Mz)
                if np.sqrt(zMz) >= tr_radius:
                    # return intersect of current search direction w/ boud
                    Mz = MV(z)
                    Md = MV(d)
                    zMz = torch.dot(z,Mz)
                    dMd = torch.dot(d,Md)
                    zMd = torch.dot(d,Mz)
                    t_lower, t_upper = mitternachtsformel_torch(dMd,2*zMd,zMz-tr_radius**2)#solve_quadratic_equation_torch(z, d, tr_radius)
                    print('Boundary point after',k+1,'iterations.')
                    collector['sub_iterations']=k+1
                    collector['interior_flag']=False
                    self.decrease = 0.5*sum([aa*gg for aa,gg in zip(alpha_k[:-1],gamma_k[:-1])]) # See Conn et al. Section 17.4.1
                    self.decrease += t_upper*gamma_k[-1]-0.5*t_upper**2*dBd  
                    return z + t_upper * d
                r_next = r - alpha * Bd
                #if torch.norm(r_next) < min(sqrt(torch.norm(grad)) * torch.norm(grad),krylov_tol):
                if np.sqrt(torch.dot(r_next,MV(r_next))) < min(sqrt(grad_norm) * grad_norm,self.krylov_tol):
                    print('Interior point after',k+1,'iterations.')
                    collector['sub_iterations']=k+1
                    collector['interior_flag']=True
                    self.decrease = 0.5*sum([aa*gg for aa,gg in zip(alpha_k,gamma_k)]) # See Conn et al. Section 17.4.1
                    return z_next

                r_tilde_next = MinvV(r_next)
                gamma_k.append(torch.dot(r_next,r_tilde_next))
                beta_next = torch.dot(r_next, r_tilde_next) / torch.dot(r, r_tilde)
                d_next = r_tilde_next + beta_next * d
                # update iterates
                z = z_next
                r = r_next
                r_tilde = r_tilde_next
                d = d_next
                k = k + 1
                
                
class DogLegTrSubproblemSolver:
    
    def __call__(self,grad,Hv, hessian ,tr_radius, X, Y, w, MV, MinvV, accepted_flag,collector,**kwargs):
        H = hessian(w,X,Y,**kwargs).numpy()
        s = self.dogLeg_TR_subproblem_solver(grad.numpy(), H, tr_radius,accepted_flag)
        s_pytorch = torch.from_numpy(s).type(w.type())
        if Hv is not None:
            self.decrease = -(np.dot(grad.numpy(), s) + 0.5*torch.dot(s_pytorch, Hv(w,X,Y,s_pytorch,**kwargs)))
        else:
            self.decrease = -(np.dot(grad.numpy(), s) + 0.5*torch.dot(s, np.dot(H,s)))
        return s_pytorch
        
    def dogLeg_TR_subproblem_solver(self,grad, H, tr_radius,accepted_flag):
        from scipy import linalg
        gBg = np.dot(grad, np.dot(H, grad))
        if gBg <= 0:
            raise ValueError(
                'dog_leg requires H to be positive definite in all steps!') 

        ## Compute the Newton Point and return it if inside the TR
        cholesky_B = linalg.cho_factor(H)
        pn = -linalg.cho_solve(cholesky_B, grad)
        if (np.linalg.norm(pn) < tr_radius):
            return pn


        # Compute the 'unconstrained Cauchy Point'
        pc = -(np.dot(grad, grad) / gBg) * grad
        pc_norm = np.linalg.norm(pc)

        # if it is outside the TR, return the point where the path intersects the boundary
        if pc_norm >= tr_radius:
            p_boundary = pc * (tr_radius / pc_norm)
            return p_boundary


        # else, give intersection of path from pc to pn with tr_radius.
        t_lower, t_upper = solve_quadratic_equation(pc, pn, tr_radius)
        p_boundary = pc + t_upper * (pn - pc)
        return p_boundary

class ExactTrSubproblemSolver:
    
    def __init__(self,exact_tol,lambda_k=0):
        self.exact_tol = exact_tol
        self.lambda_k=lambda_k
    def __call__(self,grad,Hv, hessian ,tr_radius, X, Y, w, MV, MinvV, accepted_flag,collector,**kwargs):
        H = hessian(w,X,Y,**kwargs).numpy()
        s = self.exact_TR_subproblem_solver(grad.numpy(), H, tr_radius,accepted_flag)
        s_pytorch = torch.from_numpy(s).type(w.type())
        if Hv is not None:
            self.decrease = -(np.dot(grad.numpy(), s) + 0.5*torch.dot(s_pytorch, Hv(w,X,Y,s_pytorch,**kwargs)))
        else:
            self.decrease = -(np.dot(grad.numpy(), s) + 0.5*torch.dot(s, np.dot(H,s)))
        return s_pytorch
        
    def exact_TR_subproblem_solver(self,grad, H, tr_radius,accepted_flag):
        from scipy import linalg
        s = np.zeros_like(grad)
        ## Step 0: initialize safeguards
        H_ii_min = min(np.diagonal(H))
        H_max_norm = sqrt(H.shape[0] ** 2) * np.absolute(H).max()
        H_fro_norm = np.linalg.norm(H, 'fro')
        gerschgorin_l = max([H[i, i] + (np.sum(np.abs(H[i, :])) - np.abs(H[i, i])) for i in range(len(H))])
        gerschgorin_u = max([-H[i, i] + (np.sum(np.abs(H[i, :])) - np.abs(H[i, i])) for i in range(len(H))])

        lambda_lower = max(0, -H_ii_min, np.linalg.norm(grad) / tr_radius - min(H_fro_norm, H_max_norm, gerschgorin_l))
        lambda_upper = max(0, np.linalg.norm(grad) / tr_radius + min(H_fro_norm, H_max_norm, gerschgorin_u))

        if accepted_flag==False and lambda_lower <= self.lambda_k <= lambda_upper: #reinitialize at previous lambda in case of unscuccesful iterations
            lambda_j=self.lambda_k
        elif lambda_lower == 0:  # allow for fast convergence in case of inner solution
            lambda_j = lambda_lower
        else:
            lambda_j=np.random.uniform(lambda_lower, lambda_upper)
            #lambda_j=(lambda_lower+lambda_upper)/2

        i=0
        # Root Finding
        #while True:
        for i in range(50):
            i+=1
            lambda_in_N = False
            lambda_plus_in_N = False
            B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
            try:
                # 1 Factorize B
                L = np.linalg.cholesky(B)
                # 2 Solve LL^Ts=-g
                Li = np.linalg.inv(L)
                s = - np.dot(np.dot(Li.T, Li), grad)
                sn = np.linalg.norm(s)
                ## 2.1 Termination: Lambda in F, if q(s(lamda))<eps_opt q(s*) and sn<eps_tr tr_radius -> stop. By Conn: Lemma 7.3.5:
                phi_lambda = 1. / sn - 1. / tr_radius
                #if (abs(sn - tr_radius) <= exact_tol * tr_radius):
                if (abs(phi_lambda)<=self.exact_tol): #
                    break;

                # 3 Solve Lw=s
                w = np.dot(Li, s)
                wn = np.linalg.norm(w)


                ##Step 1: Lambda in L
                if lambda_j > 0 and (phi_lambda) < 0:
                    # print ('lambda: ',lambda_j, ' in L')
                    lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)
                    lambda_j = lambda_plus


                ##Step 2: Lambda in G    (sn<tr_radius)
                elif (phi_lambda) > 0 and lambda_j > 0 and np.any(grad != 0): #TBD: remove grad
                    # print ('lambda: ',lambda_j, ' in G')
                    lambda_upper = lambda_j
                    lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)

                    ##Step 2a: If factorization succeeds: lambda_plus in L
                    if lambda_plus > 0:
                        try:
                            # 1 Factorize B
                            B_plus = H + lambda_plus * np.eye(H.shape[0], H.shape[1])
                            L = np.linalg.cholesky(B_plus)
                            lambda_j = lambda_plus
                            # print ('lambda+', lambda_plus, 'in L')


                        except np.linalg.LinAlgError:
                            lambda_plus_in_N = True

                    ##Step 2b/c: If not: Lambda_plus in N
                    if lambda_plus <= 0 or lambda_plus_in_N == True:
                        # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
                        try:
                            U = np.linalg.cholesky(H)
                            H_pd = True
                        except np.linalg.LinAlgError:
                            H_pd = False

                        if lambda_lower == 0 and H_pd == True and phi_lambda >= 0: #cannot happen in ARC!
                            lambda_j = 0
                            #print ('inner solution found')
                            break
                        # 2. Else, choose a lambda within the safeguard interval
                        else:
                            # print ('lambda_plus', lambda_plus, 'in N')
                            lambda_lower = max(lambda_lower, lambda_plus)  # reset lower safeguard
                            lambda_j = max(sqrt(lambda_lower * lambda_upper),
                                           lambda_lower + 0.01 * (lambda_upper - lambda_lower))

                            lambda_upper = float(
                                lambda_upper) 

                            if lambda_lower == lambda_upper:
                                lambda_j = lambda_lower
                                ## Hard case
                                ew, ev = linalg.eigh(H, eigvals=(0, 0))
                                d = ev[:, 0]
                                dn = np.linalg.norm(d)
                                assert (ew == -lambda_j), "Awkward: in hard case but lambda_j != -lambda_1"
                                tao_lower, tao_upper = solve_quadratic_equation(1, 2*np.dot(s,d), np.dot(s,s)-tr_radius**2)
                                s=s + tao_lower * d   
                                print ('hard case resolved inside')
                                self.lambda_k=lambda_j
                                return s

                elif (phi_lambda) == 0: 
                    break
                else:      #TBD:  move into if lambda+ column #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                    lambda_in_N = True
            ##Step 3: Lambda in N
            except np.linalg.LinAlgError:
                lambda_in_N = True
            if lambda_in_N == True:
                # print ('lambda: ',lambda_j, ' in N')
                try:
                    U = np.linalg.cholesky(H)
                    H_pd = True
                except np.linalg.LinAlgError:
                    H_pd = False

                # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
                if lambda_lower == 0 and H_pd == True and phi_lambda >= 0: 
                    lambda_j = 0
                    #print ('inner solution found')
                    break
                # 2. Else, choose a lambda within the safeguard interval
                else:
                    lambda_lower = max(lambda_lower, lambda_j)  # reset lower safeguard
                    lambda_j = max(sqrt(lambda_lower * lambda_upper),
                                   lambda_lower + 0.01 * (lambda_upper - lambda_lower))  # eq 7.3.14
                    lambda_upper = float(lambda_upper)  
                    # Check for Hard Case:
                    if lambda_lower == lambda_upper:
                        lambda_j = lambda_lower
                        ew, ev = linalg.eigh(H, eigvals=(0, 0))
                        d = ev[:, 0]
                        dn = np.linalg.norm(d)
                        try:
                            assert (ew == -lambda_j), "Awkward: in hard case but lambda_j != -lambda_1"
                        except:
                            print("Awkward: in hard case but lambda_j != -lambda_1",ew,'!=',-lambda_j)
                        tao_lower, tao_upper = solve_quadratic_equation(1, 2*np.dot(s,d), np.dot(s,s)-tr_radius**2)
                        s=s + tao_lower * d 

                        print ('hard case resolved outside')
                        self.lambda_k = lambda_j
                        return s




        # compute final step
        B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
        # 1 Factorize B
        L = np.linalg.cholesky(B)
        # 2 Solve LL^Ts=-g
        Li = np.linalg.inv(L)
        s = - np.dot(np.dot(Li.T, Li), grad)
        #print (i,' exact solver iterations')
        self.lambda_k = lambda_j
        return s

def mitternachtsformel_torch(a,b,c):
    sqrt_discriminant = sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper

def mitternachtsformel(a,b,c):
    sqrt_discriminant = sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper

def solve_quadratic_equation_torch(pc, pn, tr_radius):
    # solves ax^2+bx+c=0
    a = torch.dot(pn - pc, pn - pc)
    b = 2 * torch.dot(pc, pn - pc)
    c = torch.dot(pc, pc) - tr_radius ** 2
    sqrt_discriminant = sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper

def solve_quadratic_equation(pc, pn, tr_radius):
    # solves ax^2+bx+c=0
    a = np.dot(pn - pc, pn - pc)
    b = 2 * np.dot(pc, pn - pc)
    c = np.dot(pc, pc) - tr_radius ** 2
    sqrt_discriminant = sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper

#######################################
### Subsampled Cubic Regularization ###
#######################################

# Authors: Jonas Kohler and Aurelien Lucchi, 2017
# Linus Groner, 2018:
#     * moved performance critical parts from NumPy to PyTorch,
#     * added support for M-norm regularization
#     * added exact subsolver exploiting tridiagonal subproblem structure
#     * increased number of parameters for which statistical data is collected
#     * implemented the propositions by Conn et al. Section 17.4 analogously for ARC/SCR
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


def SCR(w, loss, gradient, Hv=None, hessian=None, X=None, Y={}, opt=None, statistics_callback=None, **kwargs):

    """
    Minimize a continous, unconstrained function using the Adaptive Cubic Regularization method.

    References
    ----------
    Cartis, C., Gould, N. I., & Toint, P. L. (2011). Adaptive cubic regularisation methods for unconstrained optimization. Part I: motivation, convergence and numerical results. Mathematical Programming, 127(2), 245-295.
    Chicago 

    Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Trust region methods. Society for Industrial and Applied Mathematics.

    Kohler, J. M., & Lucchi, A. (2017). Sub-sampled Cubic Regularization for Non-convex Optimization. arXiv preprint arXiv:1705.05933.


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
    
    print ('--- Subsampled Cubic Regularization ---\n')
    
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
    print('\n\nSCR configuration:')
    
    
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
    
    sigma = opt.get('initial_penalty',1.)  # intial penalty sigma
    print('   - initial_penalty:', sigma)
    
    min_penalty = opt.get('min_penalty',0.0)  # min penalty
    print('   - min_penalty:', min_penalty)
    
    assert (sigma < float('inf') and min_penalty < float('inf') ), "Penalty parameter must be finite."
    
    eta_1 = opt.get('successful_threshold',0.1)
    print('   - successful_threshold:', eta_1)
    
    eta_2 = opt.get('very_successful_threshold',0.9)
    print('   - very_successful_threshold:', eta_2)
    
    gamma_1 = opt.get('penalty_increase_multiplier',2.)
    print('   - penalty_increase_multiplier:', gamma_1)
    
    gamma_2= opt.get('penalty_decrease_multiplier',2.)
    print('   - penalty_decrease_multiplier:', gamma_2)
    
    accept_all_decreasing = opt.get('accept_all_decreasing_steps',True)
    print('   - accept_all_decreasing_steps:', accept_all_decreasing)
    
    assert (gamma_1 >= 1 and gamma_2 >= 1), "penalty update parameters must be greater or equal to 1"
    
    # subsolver and related parameters
    print('\n* subsolver and related parameters:')
    subproblem_solver= opt.get('subproblem_solver','lanczos')
    print('   - subproblem_solver:', subproblem_solver)
    
    assert (( 
        not isinstance(subproblem_solver,ExactArcSubproblemSolver))
        or not w.is_cuda),"GPU support for exact subsolver is not implemented."
    
    if subproblem_solver=='lanczos':
        krylov_tol=opt.get('krylov_tol',1e-2)
        print('   - krylov_tol:', krylov_tol)
        exact_tol=opt.get('exact_tol',machine_precision)
        print('   - exact_tol:', exact_tol)
        lanczos_termcrit=opt.get('lanczos_termination_criterion','g')
        print('   - lanczos_termination_criterion:', lanczos_termcrit)
        solve_each_i_th_krylov_space=opt.get('solve_each_i_th_krylov_space',1)
        print('   - solve_each_i_th_krylov_space:', solve_each_i_th_krylov_space)
        keep_Q_matrix_in_memory=opt.get('keep_Q_matrix_in_memory',True)
        print('   - keep_Q_matrix_in_memory:', keep_Q_matrix_in_memory)
        subproblem_solver = LanczosArcSubproblemSolver(exact_tol,krylov_tol,
                                                       termcrit=lanczos_termcrit,
                                                       solve_each_i_th_krylov_space=solve_each_i_th_krylov_space,
                                                       keep_Q_matrix_in_memory=keep_Q_matrix_in_memory)
    elif subproblem_solver=='cauchy':
        subproblem_solver = CauchyArcSubproblemSolver()
    elif subproblem_solver=='exact':
        exact_tol=opt.get('exact_tol',machine_precision)
        print('   - exact_tol:', exact_tol)
        subproblem_solver = ExactArcSubproblemSolver(exact_tol)
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
        not isinstance(subproblem_solver,ExactArcSubproblemSolver))
        or scaling_matrix == 'uniform'), "Scaled trust regions are not supported by the exact subsolver."

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
    print ('Iter ' + str(0) + ': loss={:.20f}'.format(_loss), 'time={:3e}'.format(0), 'penalty={:.3e}'.format(sigma))
    print()
    
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
        
        'sigma': [sigma],
        
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
            subproblem_stats_collector={} # filled by subsolver
            s = subproblem_solver(grad, Hv, hessian, sigma, _X, _Y, w,MV, MinvV, accepted_flag, subproblem_stats_collector,**kwargs)
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
            if model_decrease < machine_precision and function_decrease < machine_precision: #  Conn et al. Section 17.4.2
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
    
            #Update penalty parameter
            successful_flag=True
            if rho < eta_1:
                sigma *= gamma_1
                successful_flag=False
            
            elif rho >= eta_2:
                sigma=max(sigma/gamma_2,min_penalty)
                #alternative (Cartis et al. 2011): sigma= max(min(grad_norm,sigma),machine_precision) 
            
            n_samples_seen += n_samples_per_step
    
            ### V: Save Iteration Information  ###
            #we exclude this part from timing to be consistent with gradient methods, where this part
            #is excluded since it also computes the loss, which is not an inherent part of gradient methods,
            #and only for the purpose of tracking progress. 
            if statistics_callback is not None:
                if w.is_cuda:#time spent in the callback is not a property of the algorithm, and is hence excluded from the time measurements.
                     torch.cuda.synchronize()
            timing_iteration=(datetime.now() - start).total_seconds()
            k += 1
            
            if statistics_callback is not None:
                statistics_callback(k,w,stats_collector)
                if w.is_cuda:#time spent in the callback is not a property of the algorithm, and is hence excluded from the time measurements.
                     torch.cuda.synchronize()
            
            timing += timing_iteration
            print ('Iter ' + str(k) + ': loss={:.20f}'.format(_loss) + ' ||g||={:.3e}'.format(grad_norm),'time={:3e}'.format(timing),'dt={:.3e}'.format(timing_iteration), 'penalty={:.3e}'.format(sigma))
            print(''.join([' ']*(6+len(str(k)))),'||s||={:.3e}'.format(sn),'||s||_M={:.3e}'.format(sns),'samples Hessian=', int(sample_size_Hessian),'samples Gradient=', int(sample_size_gradient),'samples loss=', int(sample_size_loss))
            print(''.join([' ']*(6+len(str(k)))),'epoch={:.3e}'.format(n_samples_seen/n),'rho={:.6e}'.format(rho),'||w-w0||={:.3e}'.format(float(torch.norm(w-w0))),'accepted=',colored(str(accepted_flag),('green' if accepted_flag else 'red')),'successful='+colored(str(successful_flag),('green' if successful_flag else 'red')),"\n")

            # record statistical data of the step
            stats_collector['time'].append(timing)
            stats_collector['samples'].append(n_samples_seen)
            stats_collector['sample_size_Hessian'].append(sample_size_Hessian)
            stats_collector['sample_size_gradient'].append(sample_size_gradient)
            stats_collector['sample_size_loss'].append(sample_size_loss)
            stats_collector['loss'].append(_loss)
            stats_collector['travel_distance'].append(float(torch.norm(w-w0)))
            stats_collector['sigma'].append(sigma)
            
            #check for termination
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

class CauchyArcSubproblemSolver:
    """
    This solver calculates the generalized Cauchy point of the subproblem,
    which is the minimizer of a linear model s.t. the scaled boundary constraints,
    rescaled as a second order minimizer along that direction. (Note that this is
    not the gradient direction in general, but the direction of an adaptive gradient
    method with the same scaling matrix.)
    The (non-generalized) cauchy point's adaption analogously to TR:
    Nocedal, Jorge, and Stephen J. Wright. (2006) Numerical optimization 2nd edition.
    Algorithm 4.4
    """
    
    def __call__(self,grad,Hv, hessian ,sigma, X, Y, w, MV, MinvV, accepted_flag,collector,**kwargs):
        Minvg = MinvV(grad)
        Hg=Hv(w, X, Y,Minvg,**kwargs)

        gHg = torch.dot(Minvg,Hg)
        gMg = torch.dot(grad,Minvg)
        gnorm = np.sqrt(gMg)
        a=sigma*float(gnorm)**3
        b=gHg
        c=-gMg
        (alpha_l,alpha_h)=mitternachtsformel(a,b,c)
        alpha=alpha_h
        s=-float(alpha)*grad
        
        self.decrease = -(float(alpha)**2)*gHg/2 + float(alpha)*gMg - (float(alpha)**3)*sigma*(gnorm**3)/3
        return s

class LanczosArcSubproblemSolver:
    def __init__(self,exact_tol,krylov_tol,
                    termcrit='g',
                    solve_each_i_th_krylov_space=1,
                    keep_Q_matrix_in_memory=True):
        self.exact_tol = exact_tol
        self.krylov_tol = krylov_tol
        self.solve_each_i_th_krylov_space = solve_each_i_th_krylov_space
        self.keep_Q_matrix_in_memory = keep_Q_matrix_in_memory
        self.lambda_k = 0
        
        if termcrit == 'g': # 'g-rule' by Cartis et al., Equation (7.1)
            self.terminate = lambda :self.gamma_k_next*self.uk<min(self.krylov_tol,np.sqrt(self.gamma0))*self.gamma0
        if termcrit == 's':# 's-rule' by Cartis et al., Equation (7.2)
            self.terminate = lambda :self.gamma_k_next*self.uk<min(self.krylov_tol,self.un)*self.gamma0
        if termcrit == 's/sigma':# 's/sigma' by Cartis et al., Equation (7.3)
            self.terminate = lambda :self.gamma_k_next*self.uk<min(self.krylov_tol,self.un/max(1,self.sigma))*self.gamma0
        if termcrit == 'TR': #cf. Jonas' TR code
            self.terminate = lambda :self.gamma_k_next*self.uk<min(np.power(self.gamma_k,1.5),self.krylov_tol)
        if termcrit == 'SCR': #cf. SCR/Jonas' Master Thesis
            self.terminate = lambda :self.gamma_k_next*self.uk<self.krylov_tol*min(1,self.un)*self.gamma0
            
    def __call__(self,grad,Hv, hessian, sigma, X, Y, w, MV, MinvV, accepted_flag, collector, **kwargs):
            self.sigma = sigma
            
            t=grad
            y=MinvV(grad)
            self.dim = len(grad)
            grad_norm=torch.norm(grad)
            self.gamma_k_next=float(np.sqrt(torch.dot(t,y)))
            self.gamma0 = self.gamma_k_next
            delta=[] 
            gamma=[] # save for cheaper reconstruction of Q
            q = y/self.gamma_k_next
            dimensionality = len(w)

            k=0

            while True:
                if (grad==0).all(): #From T 7.5.16 u_k was the minimizer of m_k. But it was not accepted. Thus we have to be in the hard case.
                    try:
                        print ('zero gradient encountered')
                        #only spherical
                        exact_subsolver = ExactArcSubproblemSolver(self.exact_tol,self.lambda_k)
                        s =  exact_subsolver(grad,Hv, hessian ,sigma, X, Y, w, MV, MinvV, accepted_flag,collector,**kwargs)
                        self.decrease = exact_subsolver.decrease
                        self.lambda_k = exact_subsolver.lambda_k
                        return s
                    except MemoryError:
                        print('MemoryError when handling zero gradient, continuing')

                #a) create g
                e_1=np.zeros(k+1)
                e_1[0]=1.0
                g_lanczos=self.gamma0*e_1
                #b) generate H
                self.gamma_k = self.gamma_k_next
                gamma.append(self.gamma_k)

                if not k==0:
                    ww_old=ww
                ww=t/self.gamma_k
                q=y/self.gamma_k
                    
                if self.keep_Q_matrix_in_memory:
                    self.growQ(q,k)
                        
                Hq=Hv(w, X, Y, q, **kwargs) #matrix free            
                delta_k=torch.dot(q,Hq)
                delta.append(delta_k)
                if k == 0:
                    Tdiag = np.array([delta_k])
                    Toffdiag = np.array([])
                    t = Hq - delta_k*ww
                else:
                    Tdiag = np.append(Tdiag,delta_k)
                    Toffdiag = np.append(Toffdiag,self.gamma_k)
                    t = Hq - delta_k*ww - self.gamma_k*ww_old
                    
                y=MinvV(t)
                self.gamma_k_next = float(np.sqrt(torch.dot(t,y)))

                #### Solve Subproblem only in each i-th Krylov space 
                if k %(self.solve_each_i_th_krylov_space) ==0 or (k==dimensionality-1) or self.gamma_k_next==0:
                    if k==0:# avoids having to handle the 1d case where the offidagonals are empty lists
                        exact_solver = ExactArcSubproblemSolver(self.exact_tol,self.lambda_k)
                        u = exact_solver.exact_ARC_subproblem_solver(g_lanczos, np.array([[Tdiag[0]]]), sigma, accepted_flag)
                        self.lambda_k = exact_solver.lambda_k
                    else:
                        u = self.exact_ARC_suproblem_solver_tridiagonal(g_lanczos,Tdiag,Toffdiag,sigma,accepted_flag)
                    self.uk = abs(u[k])
                    self.un = np.linalg.norm(u)
#                     e_k=np.zeros(k+1)
#                     e_k[k]=1.0
                    #if self.gamma_k_next*abs(np.dot(u,e_k))< min(krylov_tol,np.linalg.norm(u)/max(1, sigma))*grad_norm:
                    
                    if self.terminate():
                    #if self.terminate(gamma0,self.gamma_k,self.gamma_k_next,abs(u[k]),np.linalg.norm(u),sigma):
                        print('Lanczos converged after',k+1,'iterations')
                        break
                        
                if k==dimensionality-1:
                    print ('Krylov dimensionality reach full space!')
                    break      
                            
                accepted_flag=False     


                k=k+1
            
            # Recover Q to compute s
            #Q=np.zeros((k + 1,n))  #<--------- since numpy is ROW MAJOR its faster to fill the transpose of Q
            collector['sub_iterations']=k+1
            if not self.keep_Q_matrix_in_memory:
                t=grad
                y=MinvV(grad)
                for j in range (0,k+1):
                    if not j==0:
                        ww_re_old=ww_re
                    ww_re=t/gamma[j]
                    q_re=y/gamma[j]
                    #Q[:,j]=q_re
                    Q[j,:]=q_re
                    Hq=Hv(w, X, Y, q_re, **kwargs) #matrix free

                    if j==0:
                        t=Hq-delta[j]*ww_re
                    elif not j==k:
                        t=Hq-delta[j]*ww_re-gamma[j]*ww_re_old
                    y=MinvV(t)
            u_pytorch = torch.from_numpy(u).type(self.Q.type())
            s = torch.matmul(u_pytorch,self.Q[:k+1,:])
            
            Tu = u*Tdiag
            if k>0:
                Tu[:-1] += u[1:]*Toffdiag
                Tu[1:] += u[:-1]*Toffdiag
            uTu = np.dot(u,Tu)
            
            self.decrease = - self.gamma0 * u[0] - 0.5*uTu - self.sigma*self.un**3/3 # Analogous to Conn et al. Section 17.4.1
            return s
        
    def exact_ARC_suproblem_solver_tridiagonal(self,grad,Hdiag,Hoffdiag,sigma,accepted_flag):
        from scipy import linalg
        from scipy.sparse import diags
        s = np.zeros_like(grad)
        d = len(Hdiag)
        H_band = np.empty((2,d))
        H_band[1,:] = Hdiag
        H_band[0,1:] = Hoffdiag
        ## Step 0: initialize safeguards
        absHdiag = np.abs(Hdiag)
        absHoffdiag = np.abs(Hoffdiag)

        #a) EV Bounds
        #gershgorin_l=min([H[i, i] - np.sum(np.abs(H[i, :])) + np.abs(H[i, i]) for i in range(len(H))]) 
        #gershgorin_u=max([H[i, i] + np.sum(np.abs(H[i, :])) - np.abs(H[i, i]) for i in range(len(H))]) 
        gerschgorin_l = min([Hdiag[0]-absHoffdiag[0] , Hdiag[d-1]-absHoffdiag[d-2]])
        gerschgorin_l = min([gerschgorin_l]+[Hdiag[i]-absHoffdiag[i]-absHoffdiag[i-1] for i in range(1,d-1)])
        gerschgorin_u = max([Hdiag[0]+absHoffdiag[0] , Hdiag[d-1]+absHoffdiag[d-2]])
        gerschgorin_u = max([gerschgorin_u]+[Hdiag[i]+absHoffdiag[i]+absHoffdiag[i-1] for i in range(1,d-1)])#see conn2000,sect 7.3.8, \lambda^L
        H_ii_min=min(Hdiag)
        H_max_norm = sqrt(len(Hdiag) ** 2) * max(absHoffdiag.max(),absHdiag.max())
        H_fro_norm = np.sqrt(2*np.dot(absHoffdiag,absHoffdiag) + np.dot(absHdiag,absHdiag))

        #b) solve quadratic equation that comes from combining rayleigh coefficients
        (lambda_l1,lambda_u1)=mitternachtsformel(1,gerschgorin_l,-np.linalg.norm(grad)*sigma)
        (lambda_u2,lambda_l2)=mitternachtsformel(1,gerschgorin_u,-np.linalg.norm(grad)*sigma)

        lambda_lower=max(0,-H_ii_min,lambda_l2)  
        lambda_upper=max(0,lambda_u1)            #0's should not be necessary


        if accepted_flag==False and lambda_lower <= self.lambda_k <= lambda_upper: #reinitialize at previous lambda in case of unscuccesful iterations
            lambda_j=self.lambda_k
        else:
            lambda_j=np.random.uniform(lambda_lower, lambda_upper)
        
        i=0
        # Root Finding
        phi_history = []
        lambda_history = []
        while True:
            i+=1
            
            lambda_plus_in_N=False
            lambda_in_N=False

            #B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
            B = np.empty((2,d))
            B[1,:] = Hdiag + lambda_j
            B[0,1:] = Hoffdiag

            if lambda_lower==lambda_upper==0 or np.any(grad)==0:
                lambda_in_N=True
            else:
                try: # if this succeeds lambda is in L or G.
                    # 1 Factorize B
                    L = linalg.cholesky_banded(B)
                    L = L[[1,0]]
                    L[1,:-1] = L[1,1:]
                    # 2 Solve LL^Ts=-g
                    s = linalg.solveh_banded(B,-grad)
                    sn = np.linalg.norm(s)

                    ## 2.1 Terminate <- maybe more elaborated check possible as Conn L 7.3.5 ??? 
                    phi_lambda=1./sn -sigma/lambda_j
                    
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



                    ## Step 1: Lambda in L and thus lambda+ in L
                    if phi_lambda < 0: 
                        #print ('lambda: ',lambda_j, ' in L')
                        c_lo,c_hi= mitternachtsformel((wn**2/sn**3),1./sn+(wn**2/sn**3)*lambda_j,1./sn*lambda_j-sigma)
                        lambda_plus=lambda_j+c_hi
                        #lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?

                        lambda_j = lambda_plus

                    ## Step 2: Lambda in G, hard case possible
                    elif phi_lambda>0:
                        #print ('lambda: ',lambda_j, ' in G')
                        #lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?
                        lambda_upper=lambda_j
                        _lo,c_hi= mitternachtsformel((wn**2/sn**3),1./sn+(wn**2/sn**3)*lambda_j,1./sn*lambda_j-sigma)
                        lambda_plus=lambda_j+c_hi
                        ##Step 2a: If lambda_plus positive factorization succeeds: lambda+ in L (right of -lambda_1 and phi(lambda+) always <0) -> hard case impossible
                        if lambda_plus >0:
                            try:
                                #1 Factorize B
                                #B_plus = H + lambda_plus * np.eye(H.shape[0], H.shape[1])
                                B_plus = np.empty((2,d))
                                B_plus[1,:] = Hdiag + lambda_plus
                                B_plus[0,1:] = Hoffdiag
                                _ = linalg.cholesky_banded(B_plus)
                                #L = np.linalg.cholesky(B_plus)
                                lambda_j=lambda_plus
                                #print ('lambda+', lambda_plus, 'in L')
                            except np.linalg.LinAlgError: 
                                lambda_plus_in_N=True

                        ##Step 2b/c: else lambda+ in N, hard case possible
                        if lambda_plus <=0 or lambda_plus_in_N==True:
                            #print ('lambda_plus', lambda_plus, 'in N')
                            lambda_lower=max(lambda_lower,lambda_plus) #reset lower safeguard
                            lambda_j=max(sqrt(lambda_lower*lambda_upper),lambda_lower+0.01*(lambda_upper-lambda_lower))  

                            lambda_lower=float(lambda_lower)
                            lambda_upper=float(lambda_upper)
                            
                            if lambda_upper <= np.nextafter(lambda_lower,lambda_upper,dtype=Hdiag.dtype):
                            #if lambda_lower==lambda_upper:
                                    lambda_j = lambda_lower #should be redundant?
                                    ew,ev = linalg.eig_banded(H_band,select='i',select_range=(0,0))
                                    ew = ew[0]
                                    d = ev[:, 0]
                                    dn = np.linalg.norm(d)
                                    #note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk facto may fall. lambda_j-1 should only be digits away!
                                    tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-lambda_j**2/sigma**2)
                                    s = s + tao_lower * d # both, tao_l and tao_up should give a model minimizer!
                                    print ('hard case resolved') 
                                    break
                        #else: #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                         #   lambda_in_N = True
                    ##Step 3: Lambda in N
                except np.linalg.LinAlgError:
                    lambda_in_N = True
            if lambda_in_N == True:
                #print ('lambda: ',lambda_j, ' in N')
                lambda_lower = max(lambda_lower, lambda_j)  # reset lower safeguard
                lambda_j = max(sqrt(lambda_lower * lambda_upper), lambda_lower + 0.01 * (lambda_upper - lambda_lower))  # eq 7.3.1
                #Check Hardcase
                #if (lambda_upper -1e-4 <= lambda_lower <= lambda_upper +1e-4):
                lambda_lower=float(lambda_lower)
                lambda_upper=float(lambda_upper)
                
                # if lambda_lower==lambda_upper:
                if lambda_upper <= np.nextafter(lambda_lower,lambda_upper,dtype=Hdiag.dtype):
                    lambda_j = lambda_lower #should be redundant?
                    #ew, ev = linalg.eigh(H, eigvals=(0, 0))
                    ew,ev = linalg.eig_banded(H_band,select_range=(0,0))
                    ew = ew[0]
                    d = ev[:, 0]
                    dn = np.linalg.norm(d)
                    if ew >=0: #H is pd and lambda_u=lambda_l=lambda_j=0 (as g=(0,..,0)) So we are done. returns s=(0,..,0)
                        break
                    #note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk.fact. may fail. lambda_j-1 should only be digits away!
                    tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-lambda_j**2/sigma**2)
                    s = s + tao_lower * d 
                    print ('hard case resolved')
                    break
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

class ExactArcSubproblemSolver:
    
    def __init__(self,exact_tol,lambda_k=0):
        self.exact_tol = exact_tol
        self.lambda_k=lambda_k
        
    def __call__(self,grad,Hv, hessian ,sigma, X, Y, w, MV, MinvV, accepted_flag,collector,**kwargs):
        H = hessian(w,X,Y,**kwargs).numpy()
        s = self.exact_ARC_subproblem_solver(grad.numpy(), H, sigma,accepted_flag)
        s_pytorch = torch.from_numpy(s).type(w.type())
        if Hv is not None:
            self.decrease = -(np.dot(grad.numpy(), s) + 0.5*torch.dot(s_pytorch, Hv(w,X,Y,s_pytorch,**kwargs)))
        else:
            self.decrease = -(np.dot(grad.numpy(), s) + 0.5*torch.dot(s, np.dot(H,s)))
        return s_pytorch
        
    def exact_ARC_subproblem_solver(self,grad,H,sigma,accepted_flag):
        from scipy import linalg
        s = np.zeros_like(grad)

        #a) EV Bounds
        gershgorin_l=min([H[i, i] - np.sum(np.abs(H[i, :])) + np.abs(H[i, i]) for i in range(len(H))]) 
        gershgorin_u=max([H[i, i] + np.sum(np.abs(H[i, :])) - np.abs(H[i, i]) for i in range(len(H))]) 
        H_ii_min=min(np.diagonal(H))
        H_max_norm=sqrt(H.shape[0]**2)*np.absolute(H).max() 
        H_fro_norm=np.linalg.norm(H,'fro') 

        #b) solve quadratic equation that comes from combining rayleigh coefficients
        (lambda_l1,lambda_u1)=mitternachtsformel(1,gershgorin_l,-np.linalg.norm(grad)*sigma)
        (lambda_u2,lambda_l2)=mitternachtsformel(1,gershgorin_u,-np.linalg.norm(grad)*sigma)

        lambda_lower=max(0,-H_ii_min,lambda_l2)  
        lambda_upper=max(0,lambda_u1)            #0's should not be necessary


        if accepted_flag==False and lambda_lower <= self.lambda_k <= lambda_upper: #reinitialize at previous lambda in case of unscuccesful iterations
            lambda_j=self.lambda_k
        else:
            lambda_j=np.random.uniform(lambda_lower, lambda_upper)
        phi_old = float('inf')
        no_of_calls=0
        max_iter=50
        for v in range(0,max_iter):
            no_of_calls+=1
            lambda_plus_in_N=False
            lambda_in_N=False

            B = H + lambda_j * np.eye(H.shape[0], H.shape[1])

            if lambda_lower==lambda_upper==0 or np.any(grad)==0:
                lambda_in_N=True
            else:
                try: # if this succeeds lambda is in L or G.
                    # 1 Factorize B
                    L = np.linalg.cholesky(B)
                    # 2 Solve LL^Ts=-g
                    Li = np.linalg.inv(L)
                    s = - np.dot(np.dot(Li.T, Li), grad)
                    sn = np.linalg.norm(s)

                    ## 2.1 Terminate <- maybe more elaborated check possible as Conn L 7.3.5 ??? 
                    phi_lambda=1./sn -sigma/lambda_j
                    if v>0 and (abs(phi_lambda)<=self.exact_tol): #
                        #print('exact solver converged after',v,'iterations')
                        break
                    #print(abs(phi_lambda))
                    if v>0 and (phi_old==abs(phi_lambda)):
                        #print('exact solver stalled after',v,'iterations')
                        break
                    phi_old = abs(phi_lambda)
                    # 3 Solve Lw=s
                    w = np.dot(Li, s)
                    wn = np.linalg.norm(w)



                    ## Step 1: Lambda in L and thus lambda+ in L
                    if phi_lambda < 0: 
                        #print ('lambda: ',lambda_j, ' in L')
                        c_lo,c_hi= mitternachtsformel((wn**2/sn**3),1./sn+(wn**2/sn**3)*lambda_j,1./sn*lambda_j-sigma)
                        lambda_plus=lambda_j+c_hi
                        #lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?

                        lambda_j = lambda_plus

                    ## Step 2: Lambda in G, hard case possible
                    elif phi_lambda>0:
                        #print ('lambda: ',lambda_j, ' in G')
                        #lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?
                        lambda_upper=lambda_j
                        _lo,c_hi= mitternachtsformel((wn**2/sn**3),1./sn+(wn**2/sn**3)*lambda_j,1./sn*lambda_j-sigma)
                        lambda_plus=lambda_j+c_hi
                        ##Step 2a: If lambda_plus positive factorization succeeds: lambda+ in L (right of -lambda_1 and phi(lambda+) always <0) -> hard case impossible
                        if lambda_plus >0:
                            try:
                                #1 Factorize B
                                B_plus = H + lambda_plus*np.eye(H.shape[0], H.shape[1])
                                L = np.linalg.cholesky(B_plus)
                                lambda_j=lambda_plus
                                #print ('lambda+', lambda_plus, 'in L')
                            except np.linalg.LinAlgError: 
                                lambda_plus_in_N=True

                        ##Step 2b/c: else lambda+ in N, hard case possible
                        if lambda_plus <=0 or lambda_plus_in_N==True:
                            #print ('lambda_plus', lambda_plus, 'in N')
                            lambda_lower=max(lambda_lower,lambda_plus) #reset lower safeguard
                            lambda_j=max(sqrt(lambda_lower*lambda_upper),lambda_lower+0.01*(lambda_upper-lambda_lower))  

                            lambda_lower=np.float32(lambda_lower)
                            lambda_upper=np.float32(lambda_upper)
                            if lambda_lower==lambda_upper:
                                    lambda_j = lambda_lower #should be redundant?
                                    ew, ev = linalg.eigh(H, eigvals=(0, 0))
                                    d = ev[:, 0]
                                    dn = np.linalg.norm(d)
                                    #note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk facto may fall. lambda_j-1 should only be digits away!
                                    tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-lambda_j**2/sigma**2)
                                    s = s + tao_lower * d # both, tao_l and tao_up should give a model minimizer!
                                    print ('hard case resolved') 
                                    break
                        #else: #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                         #   lambda_in_N = True
                    ##Step 3: Lambda in N
                except np.linalg.LinAlgError:
                    lambda_in_N = True
            if lambda_in_N == True:
                #print ('lambda: ',lambda_j, ' in N')
                lambda_lower = max(lambda_lower, lambda_j)  # reset lower safeguard
                lambda_j = max(sqrt(lambda_lower * lambda_upper), lambda_lower + 0.01 * (lambda_upper - lambda_lower))  # eq 7.3.1
                #Check Hardcase
                #if (lambda_upper -1e-4 <= lambda_lower <= lambda_upper +1e-4):
                lambda_lower=np.float32(lambda_lower)
                lambda_upper=np.float32(lambda_upper)

                if lambda_lower==lambda_upper:
                    lambda_j = lambda_lower #should be redundant?
                    ew, ev = linalg.eigh(H, eigvals=(0, 0))
                    d = ev[:, 0]
                    dn = np.linalg.norm(d)
                    if ew >=0: #H is pd and lambda_u=lambda_l=lambda_j=0 (as g=(0,..,0)) So we are done. returns s=(0,..,0)
                        break
                    #note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk.fact. may fail. lambda_j-1 should only be digits away!
                    sn= np.linalg.norm(s)
                    tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-lambda_j**2/sigma**2)
                    s = s + tao_lower * d 
                    print ('hard case resolved') 
                    break
        self.lambda_k = lambda_j
        return s

############################
### Auxiliary Functions ###
############################
def mitternachtsformel(a,b,c):
    sqrt_discriminant = sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper

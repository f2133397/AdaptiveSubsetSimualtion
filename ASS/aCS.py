import numpy as np
#import scipy as sp
import warnings

def aCS(N, old_lam, b, u_j, G_LSF):
    # %% Initialize variables
    n  = np.size(u_j,axis=1)     # number of uncertain parameters
    Ns = np.size(u_j,axis=0)     # number of seeds
    Na = int(np.ceil(100*Ns/N))  # number of chains after which the proposal is adapted

    # number of samples per chain
    Nchain = np.ones(Ns,dtype=int)*int(np.floor(N/Ns))
    Nchain[:np.mod(N,Ns)] = Nchain[:np.mod(N,Ns)]+1

    # initialization
    u_jp1  = np.zeros((N,n))                       # generated samples
    geval  = np.zeros(N)                           # store lsf evaluations
    acc    = np.zeros(N,dtype=int)                 # store acceptance
    mu_acc = np.zeros(int(np.floor(Ns/Na)+1))      # store acceptance
    hat_a  = np.zeros(int(np.floor(Ns/Na)+1))      # average acceptance rate of the chains
    lam    = np.zeros(int(np.floor(Ns/Na)+1))      # scaling parameter \in (0,1)

    # %% 1. compute the standard deviation
    opc = 'a'
    if opc == 'a': # 1a. sigma = ones(n,1)
        sigma_0 = np.ones(n)
    elif opc == 'b': # 1b. sigma = sigma_hat (sample standard deviations)
        mu_hat  = np.mean(u_j,axis=1)    # sample mean
        var_hat = np.zeros(n)            # sample std
        for i in range(n):  # dimensions
            for k in range(Ns):  # samples
                var_hat[i] = var_hat[i] + (u_j[k,i]-mu_hat[i])**2
            var_hat[i] = var_hat[i]/(Ns-1)

        sigma_0 = np.sqrt(var_hat)
    else:
        raise RuntimeError('Choose a or b')

    # %% 2. iteration
    star_a = 0.44      # optimal acceptance rate
    lam[0] = old_lam   # initial scaling parameter \in (0,1)

    # a. compute correlation parameter
    i         = 0                                        # index for adaptation of lambda
    sigma     = np.minimum(lam[i]*sigma_0, np.ones(n))   # Ref. 1 Eq. 23
    rho       = np.sqrt(1-sigma**2)                      # Ref. 1 Eq. 24
    mu_acc[i] = 0

    # b. apply conditional sampling
    for k in range(1, Ns+1):
        idx          = sum(Nchain[:k-1])     # beginning of each chain index
        #acc[idx]     = 1                    # seed acceptance
        u_jp1[idx,:] = u_j[k-1,:]            # pick a seed at random
        geval[idx]   = G_LSF(u_jp1[idx,:])   # store the lsf evaluation

        for t in range(1, Nchain[k-1]):
            # generate candidate sample
            v = np.random.normal(loc=rho*u_jp1[idx+t-1,:], scale=sigma)
            #v = sp.stats.norm.rvs(loc=rho*u_jp1[:,idx+t-1], scale=sigma)
            #v = sp.stats.multivariate_normal.rvs(mean=rho*u_jp1[:,idx+t-1], cov=np.diag(sigma**2))


            Ge = G_LSF(v)
            if Ge <= b:
                u_jp1[idx+t,:] = v
                geval[idx+t]   = Ge
                acc[idx+t]     = 1
            else:
                u_jp1[idx+t,:] = u_jp1[idx+t-1,:]
                geval[idx+t]   = geval[idx+t-1]
                acc[idx+t]     = 0


        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mu_acc[i] = mu_acc[i] + np.minimum(1, np.mean(acc[idx+1:idx+Nchain[k-1]]))

        if np.mod(k,Na) == 0:
            if Nchain[k-1] > 1:

                hat_a[i] = mu_acc[i]/Na   # Ref. 1 Eq. 25


                zeta     = 1/np.sqrt(i+1)
                lam[i+1] = np.exp(np.log(lam[i]) + zeta*(hat_a[i]-star_a))  # Ref. 1 Eq. 26


                sigma = np.minimum(lam[i+1]*sigma_0, np.ones(n))
                rho   = np.sqrt(1-sigma**2)


                i = i+1


    new_lambda = lam[i]


    if i != 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            accrate = np.mean(hat_a[:i-1])

    else:   # no adaptation
        accrate = sum(acc[:np.mod(N,Ns)])/np.mod(N,Ns)

    return u_jp1, geval, new_lambda, sigma, accrate


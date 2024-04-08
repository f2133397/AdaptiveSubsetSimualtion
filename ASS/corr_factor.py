import numpy as np

def corr_factor(I_Fj, p_j, Ns, Nc):
    ## Initialize variables
    gamma = np.zeros(Ns-1)   # correlation factor
    R     = np.zeros(Ns-1)   # autocovariance sequence
    N     = Nc*Ns           # total number of samples

    ## correlation at lag 0
    sums = 0
    for k in range(Nc):
        for ip in range(Ns):
            sums = sums + (I_Fj[ip,k]*I_Fj[ip,k])

    R_0 = (1/N)*sums - p_j**2

    ## correlation factor calculation
    for i in range(Ns-1):
        sums = 0
        for k in range(Nc):
            for ip in range(Ns-i-1):
                sums = sums + (I_Fj[ip,k]*I_Fj[ip+i+1,k])

        R[i]     = (1/(N-(i+1)*Nc))*sums - p_j**2
        gamma[i] = (1-((i+1)/Ns))*(R[i]/R_0)

    gamma = 2*np.sum(gamma)

    return gamma
##END

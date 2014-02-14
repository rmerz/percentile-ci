#!/usr/bin/env python
"""Calculate confidence intervals of confidence levels for
percentiles. See Jean-Yves Le Boudec, Performance Evaluation of
Computer and Communication Systems, EPFL Press, 2010

"""
import argparse
import numpy as np
from scipy.stats.distributions import binom,norm

def setup_args():
    parser = argparse.ArgumentParser('Calculation of confidence intervals for percentile.')
    parser.add_argument('-p','--percentile', type=float, default=0.5, help='Percentile/100.')
    parser.add_argument('-n','--number_of_samples', type=float, default=31, help='Number of samples.')
    parser.add_argument('-s','--sigma', type=float, default=0.95, help='Expected confidence level.')
    args   = parser.parse_args()
    return args

def _calculate_ci_approx(p,sigma,n):
    """Return index j and k that correspond to confidence interval
    of level sigma for percentile p*100 along with the respective
    confidence level

    Large n approximation
    """
    nu = norm.ppf((1+sigma)/2)*np.sqrt(p*(1-p))
    # print(nu)
    j = np.floor(n*p-nu*np.sqrt(n))
    k = np.ceil(n*p+nu*np.sqrt(n))
    return (j,k,sigma)

def _calculate_ci(p,sigma,n):
    """Return all indices j and k that correspond to confidence interval
    of level sigma for percentile p*100 along with the respective
    confidence levels

    Arguments:
    p    : p-quantile e.g. F(m_p) = P(X < m_p) = p for 0 < p < 1
    sigma: confidence interval level
    n    : number of samples

    Returns:
    (j_selection,k_selection,confidence_levels)

    We need to calculate

    B_{n,p}(k-1)-B_{n,p}(j-1) \leq \sigma

    Therefore, we do an exhaustive search for all values of k and j
    and then filter out

    See Jean-Yves Le Boudec, Performance Evaluation of Computer and
    Communication Systems, EPFL Press, 2010

    """

    j_k_range = np.arange(0,n)  # Already j-1 and k-1
    J = np.tile(j_k_range,(n,1)).T
    K = np.tile(j_k_range,(n,1))
    # print(J)
    # print(K)
    diff_Bk_Bj  = binom.cdf(K,n,p)-binom.cdf(J,n,p)
    j_all,k_all = np.where(diff_Bk_Bj >= sigma)  # We get too many of them
    if len(j_all) == 0:
        return None
    diff_k_j    = k_all-j_all  # Hence, find the minimum interval
    index_min_int = np.where(diff_k_j == diff_k_j.min())  # There might be several of them
    j_selection = j_all[index_min_int]+1  # j and k can range from 1 to n
    k_selection = k_all[index_min_int]+1
    confidence_levels = diff_Bk_Bj[j_selection-1,k_selection-1]
    # All confidence intervals and their confidence level
    return (j_selection,k_selection,confidence_levels)
    

def _filter_ci(p,n,j_selection,k_selection,confidence_levels):
    """Return index j and k that correspond to confidence interval of
    level sigma for percentile p*100 along with its respective
    confidence level

    Arguments:
    p                : p-quantile e.g. F(m_p) = P(X < m_p) = p for 0 < p < 1
    n                : number of samples
    j_selection      :
    k_selection      :
    confidence_levels:

    Returns a tuple:
    (j_selection,k_selection,confidence_levels)

    """
    # Now, to keep only the one that is as symmetric as possible
    # around the percentile indices. Calculate the indices for
    # selecting the percentile value
    k_p = np.floor(p*n + (1-p))
    k_pp = np.ceil(p*n + (1-p))
    index_best_interval = np.argmin(np.abs((k_p-j_selection)-(k_selection-k_pp)))
    # print(k_p,k_pp)
    # print((k_p-j_selection),(k_selection-k_pp))
    
    j = j_selection[index_best_interval]
    k = k_selection[index_best_interval]
    level = confidence_levels[index_best_interval]
    return (j,k,level)

def ci(p,sigma,n):
    """Return index j and k that correspond to confidence interval of
    level sigma for percentile p*100 along with its respective
    confidence level

    Arguments:
    p    : p-quantile e.g. F(m_p) = P(X < m_p) = p for 0 < p < 1
    sigma: expected confidence level
    n    : number of samples

    Returns a tuple
    (j,k,level): index of lower CI value, index of upper CI value, confidence level
    
    """
    if n > 100:
        return _calculate_ci_approx(p,sigma,n)

    ci_pre_values = _calculate_ci(p,sigma,n)
    if ci_pre_values is not None:
        return _filter_ci(args.percentile,args.number_of_samples,ci_pre_values[0],ci_pre_values[1],ci_pre_values[2])
    else:
        return None

def main(args):
    j_k_level = ci(p=args.percentile,sigma=args.sigma,n=args.number_of_samples)
    if j_k_level is not None:
        print(j_k_level[0],j_k_level[1],j_k_level[2])
    else:
        print('No CI of {}th percentile found for n = {} and sigma = {}'.format(args.percentile*100,args.number_of_samples,args.sigma))

if __name__ == "__main__":
    args = setup_args()
    main(args)

import numpy as np
from hmmlearn.hmm import MultinomialHMM
from scipy.optimize import fmin_powell as fmin

class PSMC(MultinomialHMM):
    def __init__(self,
                 t,
                 theta,
                 rho,
                 algorithm='viterbi',
                 random_state=None,
                 n_iter=20, tol=0,
                 verbose=False):
        MultinomialHMM.__init__(self, n_components=len(t)+1,
                                algorithm=algorithm,
                                random_state=random_state,
                                n_iter=n_iter, tol=tol,
                                verbose=verbose)
        self.t = np.append(np.append([0], t), [np.inf])
        self.tau = np.diff(self.t)
        self.theta = theta
        self.rho = rho

    def _set_lambda(self, lambd):
        self.lambd = lambd
        n = self.n_components
        tau = self.tau
        rho = self.rho
        alpha = np.append([1], np.exp(-np.cumsum(tau / lambd)))
        mdiffalpha = - np.diff(alpha)
        beta = np.append([0], np.cumsum(lambd[:-1] * np.diff(1 / alpha[:-1])))
        C_pi = np.sum(lambd * mdiffalpha)
        C_sigma = 1 / (C_pi * rho) + 0.5
        alphaxtau = alpha[1:] * tau
        alphaxtau[-1] = 0
        pi = (mdiffalpha * (np.append([0], np.cumsum(tau))[:-1] + lambd) - alphaxtau) / C_pi
        sigma = (mdiffalpha / (C_pi * rho) + pi / 2) / C_sigma

        w = beta - lambd / alpha[:-1]
        z = mdiffalpha * w + tau
        d = (mdiffalpha * mdiffalpha * w + 2 * lambd * mdiffalpha - 2 * alphaxtau) / (C_pi * pi)
        def f(k, l):
            if k == l:
                return d[k]
            elif l < k:
                return mdiffalpha[k] / (C_pi * pi[k]) * z[l]
            elif l > k:
                return mdiffalpha[l] / (C_pi * pi[k]) * z[k]
        Q = np.zeros((n, n))
        for k in range(n):
            for l in range(n):
                Q[k,l] = f(k, l)
        z = pi / (C_sigma * sigma)
        P = z[:,np.newaxis] * Q + np.diag(1 - z)
        e = np.zeros((n, 2))
        e[:,1] = (1 - z) ** (theta / rho)
        e[:,0] = 1 - e[:,1]

        self.startprob_ = sigma
        self.transmat_ = P
        self.emissionprob_ = e

    def _do_mstep(self, stats):
        def Q(lambd):
            self._set_lambda(np.abs(lambd))
            # TODO
        res = fmin(Q, self.lambd, ftol=self.tol, full_output=self.verbose, disp=self.verbpose)
        lambd = np.abs(res.xopt)
        self._set_lambda(lambd)

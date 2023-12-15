import numpy as np
#class varbvs:
###
# Input Arguments
###
# x: n x p input matrix, where n is the number of samples, and p is the number of variables. X cannot be sparse.
# y: Vector of length n containing phenotype
# Z: n x q covariate data matrix, default is None
###
# Arguments for method fit
###
# sigma: initial value for sigma, default is None, if set to None, sigma will be estimated from data
# sa: initial value for sa, default is None, if set to None, sa wiil be initialized to 1
# logodds: initial value for logodds, default is None, if set to None, logodds will be initialized to a set of default values 
# tol: tolerance for convergence, default is 1e-5
# maxiter: maximum number of iterations, default is 1e5
# seed: random seed for initialization, default is None
###
#class attributes for summary
###
# n: number of samples
# p: number of variables
# q: number of covariates
# maxiter: maximum number of iterations
# tol: tolerance for convergence
# update_sigma: whether update sigma
# update_sa: whether update sa
# logodds: logodds for each setting of hyperparameters
# ns: number of settings of hyperparameters
# sigma: sigma for each setting of hyperparameters
# sa: sa for each setting of hyperparameters
# alpha: alpha for each setting of hyperparameters
# mu: mu for each setting of hyperparameters
# w: normalized weights compute from logw.
# betahat: posterior mean estimates of the coefficients for the variables
# beta_cov: posterior mean estimates of the coefficients for the covariates
# pip: "Averaged" posterior inclusion probabilities.
# logw: log likelihood for each hyperparameter setting (logw)
###
# Output Arguments
###
# pip: "Averaged" posterior inclusion probabilities.

class varbvs:
    def __init__(self, x, y, Z=None, family='gaussian'):
        self.n, self.p = x.shape
        self.family = family
        # preprocess data
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        if Z is not None:
            Z = np.concatenate((np.ones((self.n,1)),Z.astype(np.float64)),axis=1)
        else:
            Z = np.ones((self.n,1))
        self.Z = Z 
        self.q = Z.shape[1]
        
        if self.family == 'gaussian':
            ZtZ_inv = np.linalg.inv(Z.T.dot(Z))
            self.sZx = ZtZ_inv.dot(Z.T).dot(x)
            self.sZy = ZtZ_inv.dot(Z.T).dot(y)
            if Z.shape[1] > 1:
                self.x = x - Z.dot(self.sZx)
                self.y = y - Z.dot(self.sZy)
            else:
                self.x = x - np.mean(x,axis=0)
                self.y = y - np.mean(y)
        else:
            self.sZx = None
            self.sZy = None
        self.d = np.linalg.norm(self.x,axis=0) **2
        self.xty = self.x.T.dot(self.y)

    def outerloop(self, alpha, mu, sigma, sa, logodd, logdet):
        logodds = np.ones(self.p) * logodd
        logw, sigma, sa, alpha, mu, s, n_iter= self.varbvsnorm(alpha, mu, sigma, sa, np.log(10) * logodds)
        logw = logw - logdet/2
        mu_cov = self.sZy - self.sZx.dot(alpha * mu)
        return logw, sigma, sa, alpha, mu, s, mu_cov, n_iter

    def varbvsnorm(self, alpha, mu, sigma, sa, logodds):
        xr = self.x.dot(alpha * mu)
        logw = [float('-inf')]
        s = sa * sigma / (sa * self.d + 1)
        err = 0
        #print("sigma: {}".format(str(sigma)))
        #print("sa: {}".format(str(sa)))
        #print('logodds: {}'.format(str(logodds)))
        for i in range(self.maxiter):
            # e-step
            alpha0 = np.array(alpha)
            alpha, mu, xr = self.varbvsnorm_update(sigma, sa, logodds, alpha, mu, xr)
            logw.append(self.compute_lower_bound(xr, sigma, alpha, mu, logodds, s, sa))
            beta_var = alpha * (s + (1 - alpha) * mu**2)
            if self.update_sigma:
                sigma = (np.linalg.norm(self.y - xr) **2 + np.sum(self.d * beta_var) + alpha.dot(s + mu **2) / sa) / (self.n + alpha.sum())
                #print(sigma)
                s = sa * sigma / (sa * self.d + 1)
            if self.update_sa:
                sa = (alpha.dot(s + mu **2)+10) / (sigma * alpha.sum()+10)
                s = sa * sigma / (sa * self.d + 1)
            err = np.max(np.abs(alpha - alpha0))
            
            # check convergence
            if i > 1 and err < self.tol:
                n_iter = i+1
                break
            #if logw[-1] - logw[-2] < self.tol:
                #if logw[-1] < logw[-2]:
                    #print("lower bound decreases")
                #n_iter = i+1
                ##print('logw converged in {} iterations'.format(str(i)))
                #break
        logw = logw[-1]
        #print(s)
        return logw, sigma, sa, alpha, mu, s, n_iter
            


    def compute_lower_bound(self, xr, sigma, alpha, mu, logodds, s, sa):
        beta_var = alpha * (s + mu **2) - (alpha * mu) **2
        part1 = -self.n / 2 *np.log(2 * np.pi * sigma) - np.linalg.norm(self.y - xr) **2 / (2 * sigma) \
            - np.sum(self.d * beta_var) / (2 * sigma)
        part2 = np.sum((alpha - 1) * logodds - np.log(1 + np.exp(-logodds)))
        part3 = (np.sum(alpha) + np.sum(alpha * np.log(s / (sigma * sa))) - np.sum(alpha * (s +mu**2) / (sigma * sa))) / 2\
            - np.sum(alpha * np.log(alpha + np.finfo(np.float64).eps)) - np.sum((1 - alpha) * np.log(1 - alpha + np.finfo(np.float64).eps))
        #print(part1, part2, part3)
        return part1 + part2 + part3
        
    def varbvsnorm_update(self, sigma, sa, logodds, alpha, mu, xr):
        s = np.zeros(self.p)
        for i in range(self.p):
            s[i] = sigma * sa / (sa * self.d[i] + 1)
            beta_old = alpha[i] * mu[i]
            mu[i] = (s[i] / sigma) * (self.xty[i] + self.d[i] * beta_old - self.x[:, i].T.dot(xr))
            temp = logodds[i] + (np.log(s[i] / (sigma * sa)) + mu[i] **2 / s[i]) / 2
            alpha[i] = 1 / (1 + np.exp(-temp))
            beta_new = alpha[i] * mu[i]
            xr = xr + self.x[:, i] * (beta_new - beta_old)
        return alpha, mu, xr

    # method for fitting the model
    def fit(self, sigma=None, sa=None, logodds=None, tol=1e-5, maxiter=1e5, seed=None, verbose=False):
        if seed != None:
            np.random.seed(seed)
        self.maxiter = int(maxiter)
        self.tol = tol
        
        # initialize sigma, sa, logodds
        logodds = np.array([logodds]).reshape(-1)
        if logodds.any() == None:
            self.logodds = np.linspace(-np.log10(self.p),-1,20)
            self.ns = 20
        else:
            self.logodds = logodds
            self.ns = self.logodds.shape[0]
        
        if sigma == None:    
            self.sigma = np.ones(self.ns) * self.y.var()
            self.update_sigma = True
        else:
            self.sigma = np.ones(self.ns) * sigma
            self.update_sigma = False
        
        if sa == None:
            self.sa = np.ones(self.ns)
            self.update_sa = True
        else:
            self.sa = np.ones(self.ns) * sa
            self.update_sa = False
        
        # initialize alpha, mu
        self.alpha = np.random.rand(self.p,self.ns)
        self.alpha = self.alpha/self.alpha.sum(axis=0)
        self.mu = np.random.randn(self.p,self.ns)

        # first stage optimization
        if verbose:
            print("first stage optimization")
        logw = np.zeros(self.ns) # log likelihood for each hyperparameter setting (logw)
        s = np.zeros((self.p,self.ns)) # variances of the regression coefficients (s)
        mu_cov = np.zeros((self.q,self.ns)) # posterior mean estimates of the coefficients for the covariates

        sign, logdet = np.linalg.slogdet(self.Z.T.dot(self.Z))
        for i in range(self.ns):
            #print("itreration for {}th log odds".format(str(i)))
            logw[i], self.sigma[i], self.sa[i], self.alpha[:, i], self.mu[:, i], s[:, i], mu_cov[:, i], n_iter= \
                self.outerloop(self.alpha[:, i], self.mu[:, i], self.sigma[i], self.sa[i], self.logodds[i], logdet)
            if verbose:
                print("{}th setting: logw: {}, sigma: {}, sa: {}, n_iter: {}".format(str(i+1), str(logw[i]), str(self.sigma[i]), str(self.sa[i]), str(n_iter)))
        i = np.argmax(logw)
        self.alpha = np.repeat(self.alpha[:, i], self.ns).reshape(self.p, self.ns)
        self.mu = np.repeat(self.mu[:, i], self.ns).reshape(self.p, self.ns)
        self.sigma = np.ones(self.ns) * self.sigma[i]
        self.sa = np.ones(self.ns) * self.sa[i]

        # second stage optimization
        if verbose:
            print("second stage optimization")
        for i in range(self.ns):
            #print("itreration for {}th log odds".format(str(i+1)))
            logw[i], self.sigma[i], self.sa[i], self.alpha[:, i], self.mu[:, i], s[:, i], mu_cov[:, i], n_iter = \
                self.outerloop(self.alpha[:, i], self.mu[:, i], self.sigma[i], self.sa[i], self.logodds[i], logdet)
            if verbose:
                print("{}th setting: logw: {}, sigma: {}, sa: {}, n_iter: {}".format(str(i+1), str(logw[i]), str(self.sigma[i]), str(self.sa[i]), str(n_iter)))

        # average over ns
        max_logw = np.max(logw)
        self.logw = logw
        self.w = np.exp(logw - max_logw)
        self.w = self.w / np.sum(self.w)
        
        self.betahat = (self.mu * self.alpha).dot(self.w)
        self.beta_cov = mu_cov.dot(self.w)
        self.pip = self.alpha.dot(self.w)

        return self.pip
    
    # method for predict with new data
    def predict(self, x, Z=None):
        if Z is not None:
            Z = np.concatenate((np.ones((x.shape[0],1)),Z),axis=1)
        else:
            Z = np.ones((x.shape[0],1))
        return Z.dot(self.beta_cov) + x.dot(self.betahat)
    
    

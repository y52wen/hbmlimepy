import aesara.tensor as at
import numpy as np


#new class simple flogLike
class flogLike(at.Op):
    itypes = [at.dvector]  
    otypes = [at.dscalar]  

    def __init__(self, loglike):
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        (theta,) = inputs 
        logl = self.likelihood(theta)
        outputs[0][0] = np.array(np.float64(logl))  


        
class flogLike_with_grad(at.Op):
    itypes = [at.dvector]  
    otypes = [at.dscalar] 

    def __init__(self, loglike, loglike_grad):
        self.likelihood = loglike
        self.like_grad = loglike_grad
        self.logpgrad = fLogLikeGrad(self.like_grad)

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        logl = self.likelihood(theta)

        outputs[0][0] = np.array(np.float64(logl)) 
    
    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]
    
    
class fLogLikeGrad(at.Op):
    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, loglike_grad):
        self.like_grad = loglike_grad

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        grads = self.like_grad(theta)

        outputs[0][0] = np.float64(grads) # output the log-likelihood gradient
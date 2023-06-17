# This package uses perfect data (Cartesian) for inference and we only infer four parameters here.
#
# This is the basic non-hierachical inference.

import pymc as pm
from limepy import limepy
import aesara.tensor as at
import aesara
import jax
import numpy as np

from external_likeilhood_aesara import flogLike,flogLike_with_grad
from limepydf_interpolate_jax import limepy_interpolate

#Only estimate structural parameters with perfect data
def log_inf(x):
    return np.log(x) if np.all(x>0) else -float('Inf')

class log_limepy_df_with_data:
    def __init__(self,r,v):
        self.r = r
        self.v = v
        lp_interp = limepy_interpolate()
        self.my_df = lp_interp.my_df
    #using limepy df
    def log_df_limepy(self,theta):
        W0, g, log10M, log10rh = theta
        lp_m = limepy(W0, g, M=10**log10M, rh=10**log10rh)
        return np.sum(log_inf(lp_m.df(self.r,self.v)/(10**log10M)))
    #using interpolation df
    def log_df_interpolate_limepy(self,ii):
        return self.my_df(self.r,self.v,ii[:4])


class Bayesian_limepy_sampling:
    def __init__(self,r,v,test_param=None,whether_interpolate=True):        
        f_like = log_limepy_df_with_data(r,v)
        self.whether_interpolate = whether_interpolate
        #we use HMC for interpolated data
        if whether_interpolate:
            like_jit = jax.jit(f_like.log_df_interpolate_limepy)
            likegrad_jit = jax.jit(jax.grad(f_like.log_df_interpolate_limepy))
            #This is a very necessary step for multiprocessing!
            #It creates a function shared in all four processes instead dealing with function evaluation in each memory!
            print(like_jit(test_param))
            print(likegrad_jit(test_param))
            self.logl = flogLike_with_grad(like_jit,likegrad_jit)
        else:
        #non HMC for non interpolated data    
            self.logl = flogLike(f_like.log_df_limepy)
    
        container = np.load('interp_basic.npz')
        data = [container[key] for key in container]
        g_boundary = data[3]
        Wmin, Wmax = 1.4, 16
        N = 200
        W_a_np = np.linspace(Wmin,Wmax,N)
        self.coefficient_boundary = np.polyfit(W_a_np, g_boundary,10)
    
    #This is the aesara function used to cutoff Phi0 results
    def boundary_func_aesara(self,x,offset=0.2):
        #if we make coefficients as input, this will be a generic np.poly1d function
        #Here the coefficients are from small to big (from zero degree to higher degree)
        coefficients = at.as_tensor_variable(np.flip(self.coefficient_boundary))
        max_coefficients_supported = 10000
        components, updates = aesara.scan(fn=lambda coefficient, power, free_variable: coefficient * (free_variable ** power),
                                    outputs_info=None,
                                    sequences=[coefficients, at.arange(max_coefficients_supported)],
                                    non_sequences=x)
        polynomial = components.sum(axis=0)-offset
        return polynomial
    
    def sampling(self,ndraws=2000,nburns=2000,chains=4,target_accept=0.8,init_vals=None):
        basic_model = pm.Model()
        with basic_model:
            W0 = pm.Uniform("Phi0",lower=1.5,upper=14)
            g = pm.Uniform("g",lower=0.001,upper=self.boundary_func_aesara(W0))

            log10M = pm.Normal("log10M",mu=5.85,sigma=0.6)
            log10rh = pm.TruncatedNormal("log10rh", mu=0.7,sigma=0.3,lower=0,upper=1.5)
            
            theta = at.stack([W0,g,log10M,log10rh])

            pm.Potential("like", self.logl(theta))

            if self.whether_interpolate:
                idata = pm.sample(ndraws, tune=nburns, chains=chains, target_accept=target_accept)
            else:
                if init_vals is None:
                    idata = pm.sample(ndraws, tune=nburns, chains=chains)
                else:
                    idata = pm.sample(ndraws, tune=nburns, chains=chains, initvals=init_vals)
        return idata

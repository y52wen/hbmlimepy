import numpy as np
from limepydf_interpolate_jax import limepy_interpolate
from coordinate_transform_jax import StC, CtS

from jax import numpy as jnp
import jax
from math import pi

#With full data only estimate centers and structural parameters
class log_df_full_data:
    def __init__(self,vecx,vecv,data_type='complete',error_obs=None):
        #Only Need to initialize once to create a method for limepy interpolated df
        a = limepy_interpolate()
        self.my_df = a.my_df
        self.coefficient_boundary = a.coefficient_boundary
        if data_type=='complete':
            self.vec_x = jnp.array(vecx)
            self.vec_v = jnp.array(vecv)
            self.like = jax.jit(self.log_df_cartesian_complete)
            self.likegrad = jax.jit(jax.grad(self.log_df_cartesian_complete))
        else:
            self.Nop = jnp.shape(vecx)[1] 
            if data_type=='Cartesian':
                self.like = jax.jit(self.log_df_cartesian)
                self.likegrad = jax.jit(jax.grad(self.log_df_cartesian))
            else:
                self.p_obs = vecx[2]
                self.sigma_p = error_obs[2]
                self.like = jax.jit(self.log_df_sky)
                self.likegrad = jax.jit(jax.grad(self.log_df_sky))

    def log_df_cartesian_complete(self,ii):   
        r = jnp.sqrt((self.vec_x[0]-ii[4])**2+(self.vec_x[1]-ii[5])**2+(self.vec_x[2]-ii[6])**2)
        v = jnp.sqrt((self.vec_v[0]-ii[7])**2+(self.vec_v[1]-ii[8])**2+(self.vec_v[2]-ii[9])**2) 
        fr = self.my_df(r,v,ii[:4])
        return fr
    
    #now we completely do not know about rz distance!
    #so rz center is not included
    def log_df_cartesian(self,ii):  
        Np = self.Nop
        r = jnp.sqrt((ii[9:9+Np]-ii[4])**2+(ii[9+Np:9+2*Np]-ii[5])**2+(ii[9+2*Np:9+3*Np]-ii[6])**2)
        v = jnp.sqrt((ii[9+3*Np:9+4*Np]-ii[7])**2+(ii[9+4*Np:9+5*Np]-ii[8])**2+(ii[9+5*Np:9+6*Np])**2) 
        fr = self.my_df(r,v,ii[:4])
        return fr

    def log_df_sky(self,ii):  
        Np = self.Nop
        
        p_f = 1000/(ii[10+2*Np:10+3*Np]+1000/ii[6])
        xc_f,yc_f,zc_f,vxc_f,vyc_f,vzc_f = StC(jnp.array([ii[4],ii[5],ii[6],ii[7],ii[8],ii[9]]))
        x_f,y_f,z_f,vx_f,vy_f,vz_f = StC(jnp.array([ii[10:10+Np],ii[10+Np:10+2*Np],p_f,\
                                                   ii[10+3*Np:10+4*Np],ii[10+4*Np:10+5*Np],(ii[10+5*Np:10+6*Np]+ii[9])]))

        r = jnp.sqrt((x_f-xc_f)**2+(y_f-yc_f)**2+(z_f-zc_f)**2)
        v = jnp.sqrt((vx_f-vxc_f)**2+(vy_f-vyc_f)**2+(vz_f-vzc_f)**2) 
        
        #we have to correct for the coordinate sampling error!
        #The Jacobian term is included to correct for the sampling process
        fr = self.my_df(r,v,ii[:4])+jnp.sum(jnp.log(jnp.abs((ii[10+2*Np:10+3*Np]+1000/ii[6])**4*jnp.cos(ii[10+Np:10+2*Np]))))
        
        #Gaussian likelihood on individual parallax! 
        p_l = -jnp.sum(jnp.log(self.sigma_p*(2*pi)**(1/2))+1/2*((self.p_obs-p_f)/self.sigma_p)**2)
        return fr+p_l


from external_likeilhood_aesara import flogLike_with_grad

import aesara.tensor as at
import aesara
import arviz as az
import pymc as pm
import numpy as np

class Bayesian_sampling:
    def __init__(self,vecx,vecv,test_param,data_type='complete',error_obs=None):

        f_like = log_df_full_data(vecx,vecv,data_type=data_type,error_obs=error_obs)
        self.coefficient_boundary = f_like.coefficient_boundary
        
        #We need to compile the two functions before sending to pymc
        print(f_like.like(test_param))
        print(f_like.likegrad(test_param))
        self.logl = flogLike_with_grad(f_like.like,f_like.likegrad)
        
        self.error = error_obs
        self.vecx = vecx
        self.vecv = vecv

        #position mean, position max with respect to their arrays
        #a,d,p,va,vd,vR
        self.xmin, self.xmax = np.min(vecx,axis=1), np.max(vecx,axis=1)
        self.vmin, self.vmax = np.min(vecv,axis=1), np.max(vecv,axis=1)
    
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

    #we have already adopted a new prior here
    #should have used version control here
    #My mistake for not using version control!
    def sampling(self,ndraws=2000,nburns=2000,chains=4,target_accept=0.8):
        basic_model = pm.Model()
        Np = np.shape(self.vecx[0])[0]
        with basic_model:
            W0 = pm.Uniform("Phi0",lower=1.5,upper=14)
            g = pm.Uniform("g",lower=0.001,upper=self.boundary_func_aesara(W0))
            log10M = pm.Normal("log10M",mu=5.85,sigma=0.6)
            log10rh = pm.TruncatedNormal("log10rh", mu=0.7,sigma=0.3,lower=0,upper=1.5)

            ac = pm.Uniform("ac",lower=self.xmin[0],upper=self.xmax[0])
            dc = pm.Uniform("dc",lower=self.xmin[1],upper=self.xmax[1])
            pc  = pm.Uniform("pc",lower=self.xmin[2],upper=self.xmax[2])

            vac = pm.Uniform("vac",lower=self.vmin[0],upper=self.vmax[0])
            vdc = pm.Uniform("vdc",lower=self.vmin[1],upper=self.vmax[1])
            vRc = pm.Flat("vRc")   

            a = pm.Normal("a",mu=self.vecx[0],sigma=self.error[0])
            d = pm.Normal("d",mu=self.vecx[1],sigma=self.error[1])
            
            #star radius-GC center radius
            Rtc = pm.Flat("Rtc",shape=Np)

            va = pm.Normal("va",mu=self.vecv[0],sigma=self.error[3])
            vd = pm.Normal("vd",mu=self.vecv[1],sigma=self.error[4])

            #star riadial velocity-GC center radial velocity
            vRtc = pm.Flat("vRtc",shape=Np) 
            
            theta = at.concatenate([at.stack([W0,g,log10M,log10rh,ac,dc,pc,vac,vdc,vRc]),a,d,Rtc,va,vd,vRtc])
            pm.Potential("like", self.logl(theta))

            idata = pm.sample(ndraws,tune=nburns,chains=chains,cores=chains,initvals={"pc":np.mean(self.vecx[2]),\
                                                                  "Rtc":np.zeros(Np),"vRtc":np.zeros(Np)},\
                            target_accept=target_accept,jitter_max_retries=100)
        return idata
         


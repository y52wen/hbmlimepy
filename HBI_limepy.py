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
        self.boundary_func = a.boundary_func
        
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
                self.sigma_vR = jnp.array(error_obs[5])
                if jnp.all(self.sigma_vR==0):
                    self.vR_obs = jnp.zeros(len(vecx[2]))
                    #This is just to make sure the gradient still works in jnp.where!
                    #It does not really impact anything...
                    self.sigma_vR = 0.1
                else:
                    #note that the ones without measurements need to be set to 0 here!
                    self.vR_obs = jnp.array(vecv[2])
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
        #This will then add the future likelihood on virial velocity!
        #we can now handle data with or without missing velocities!
        vR_l = -jnp.sum(jnp.where(self.vR_obs==0,0,jnp.log(self.sigma_vR*(2*pi)**(1/2))+1/2*((self.vR_obs-(ii[10+5*Np:10+6*Np]+ii[9]))/self.sigma_vR)**2))
        return fr+p_l+vR_l


#For the naive fitting functions
#We take the observed parallaxes (3D), then using the 2D results for fitting
from limepy import limepy
from scipy.optimize import fmin    

def W0_g_rh_2D_og_fit(vecx,b_f_n):
    def og_to_xy(Sin):
        rad_to_mas = 180/pi*60*60*10**3 
        alpha,delta,p= Sin
        alpha = alpha/rad_to_mas #rad
        delta = delta/rad_to_mas #rad
        R = 1000/p 
        alphac = np.mean(alpha)
        deltac = np.mean(delta)
        #x = cos δ sin(α − αC )
        nx = jnp.cos(delta)*jnp.sin(alpha-alphac)
        x = R*nx
        #y = sin δ cos δC − cos δ sin δC cos(α − αC )
        ny = (jnp.sin(delta)*jnp.cos(deltac)\
                -jnp.cos(delta)*jnp.sin(deltac)*jnp.cos(alpha-alphac))
        y = R*ny
        return jnp.array([x,y])
    
    #the likelihood for the 2D parameters
    def minloglike(par, Rdat):
        # par is an array with parameters: W, g, rh
        m = limepy(par[0], par[1], M=1, rh=par[2], project=True) 
        # Return the minus log likelihood, note that the model is normalised to M=1 above
        return -sum(np.log(np.interp(Rdat, m.R, m.Sigma, right=1e-9)))
    
    pn = og_to_xy(jnp.array([vecx[0],vecx[1],vecx[2]]))
    r = np.sqrt(pn[0]**2 + pn[1]**2)
    param_array_try = np.stack(np.meshgrid(np.linspace(2,11.5,6),np.linspace(0.1,2.5,5),\
                                               np.linspace(1,15,7)), -1).reshape(-1, 3)

    #we can determine W0, g, and rh naively first to get the starting points
    #here 9 to change how many valid results one want to search to find minimum fvalue
    res_array = np.zeros((9,3))
    result_array = np.zeros(9)
    i,j = 0,0
    while j<np.shape(res_array)[0] and i<len(param_array_try):
        try:
            index = np.random.randint(0,len(param_array_try))
            x0 = param_array_try[index] # Starting values
            r = np.sqrt(pn[0]**2 + pn[1]**2)  
            result = fmin(minloglike, x0, args=(r,),full_output=True,disp=False)
            res_hold, fval = result[0], result[1]
            #we also include the prior cut here!
            if fval>0 and b_f_n(res_hold[0])>res_hold[1] and res_hold[0]>1.5 and res_hold[0]<14:
                res_array[j,:] = res_hold
                result_array[j] = fval
                j += 1
        except (ValueError,IndexError):
            pass
        i += 1
    if i==len(param_array_try):
        print('simple 2D fit initialization fails')
        return np.array([0])
    else:
        res = res_array[np.argmin(result_array),:] 
        print("Fitted Initial Result: W0 = %5.3f"%(res[0]),"; g = %5.3f"%(res[1]),"; rh = %5.3f "%(res[2]))
        return res    

    


from external_likeilhood_aesara import flogLike_with_grad

import aesara.tensor as at
import aesara
import arviz as az
import pymc as pm
import numpy as np

#we will need a version to support missing data!
class Bayesian_sampling:
    def __init__(self,vecx,vecv,test_param,data_type='complete',error_obs=None):

        f_like = log_df_full_data(vecx,vecv,data_type=data_type,error_obs=error_obs)
        self.coefficient_boundary = f_like.coefficient_boundary
        self.boundary_func_np = f_like.boundary_func
        
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
    def sampling(self,ndraws=2000,nburns=2000,chains=4,target_accept=0.8,no_init=True,whether_rand_seed=False):
        basic_model = pm.Model()
        Np = np.shape(self.vecx[0])[0]
        with basic_model:
            W0 = pm.Uniform("Phi0",lower=1.5,upper=14)
            g = pm.Uniform("g",lower=0.001,upper=self.boundary_func_aesara(W0))
            log10M = pm.Normal("log10M",mu=5.85,sigma=0.6)
            log10rh = pm.TruncatedNormal("log10rh", mu=0.7,sigma=0.3,lower=0,upper=1.5)

            ac = pm.Uniform("ac",lower=self.xmin[0],upper=self.xmax[0])
            dc = pm.Uniform("dc",lower=self.xmin[1],upper=self.xmax[1])
            #Here we need to make sure this is positive!!
            #Here is the modification on HBI!
            pc  = pm.Uniform("pc",lower=max(0.001,self.xmin[2]),upper=self.xmax[2])

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

            #We need to add another safe
            #If the more complicated initilialization scheme fails, just use the naive ones!
            #We need to change this!
            init_vals_simple = {"pc":np.mean(self.vecx[2]),"Rtc":np.zeros(Np),"vRtc":np.zeros(Np)}
            
            if no_init:
                idata = pm.sample(ndraws,tune=nburns,chains=chains,cores=chains,initvals=init_vals_simple,\
                                  target_accept=target_accept,jitter_max_retries=1000)
            else:
                res = W0_g_rh_2D_og_fit(self.vecx,self.boundary_func_np)
                if np.any(res==0):
                    print('We default back to only setting pc, Rtc and vRtc')
                    idata = pm.sample(ndraws,tune=nburns,chains=chains,cores=chains,initvals=init_vals_simple,\
                                  target_accept=target_accept,jitter_max_retries=1000)
                else:          
                    init_vals_complex={"Phi0":res[0],"g":res[1],"log10rh":np.log10(res[2]),\
                               "ac":np.mean(self.vecx[0]),"dc":np.mean(self.vecx[1]),"pc":np.mean(self.vecx[2]),\
                               "vac":np.mean(self.vecv[0]),"vdc":np.mean(self.vecv[1]),\
                               "a":self.vecx[0],"d":self.vecx[1],"Rtc":np.zeros(Np),\
                               "va":self.vecv[0],"vd":self.vecv[1],"vRtc":np.zeros(Np)}
                    try:
                        if whether_rand_seed:
                            seed = np.random.randint(0, 1000)
                            rng = np.random.default_rng(seed)
                            idata = pm.sample(ndraws,tune=nburns,chains=chains,cores=chains,initvals=init_vals_complex,\
                                              target_accept=target_accept,jitter_max_retries=1000,random_seed=rng)
                        else:
                            idata = pm.sample(ndraws,tune=nburns,chains=chains,cores=chains,initvals=init_vals_complex,\
                                              target_accept=target_accept,jitter_max_retries=1000)
                    except (RuntimeError,ValueError):
                        print('Using Complex Initial Result Sampling Fails. Try a different random seed for the complex initial')
                        try:
                            seed = np.random.randint(0, 100)
                            rng = np.random.default_rng(seed)
                            idata = pm.sample(ndraws,tune=nburns,chains=chains,cores=chains,initvals=init_vals_complex,\
                                      target_accept=target_accept,jitter_max_retries=1000,random_seed=rng)
                        except (RuntimeError,ValueError):
                            print('Still falis. We go back to simpler initial sampling.')
                            try:
                                if whether_rand_seed:
                                    seed = np.random.randint(0, 10000)
                                    rng = np.random.default_rng(seed)
                                    idata = pm.sample(ndraws,tune=nburns,chains=chains,cores=chains,initvals=init_vals_simple,\
                                                target_accept=target_accept,jitter_max_retries=1000,random_seed=rng)
                                else:  
                                            idata = pm.sample(ndraws,tune=nburns,chains=chains,cores=chains,initvals=init_vals_simple,\
                                                target_accept=target_accept,jitter_max_retries=1000) 
                            except (RuntimeError,ValueError):
                                print('Fails again. Try another random seed.')
                                seed = np.random.randint(0, 10)
                                rng = np.random.default_rng(seed)
                                idata = pm.sample(ndraws,tune=nburns,chains=chains,cores=chains,initvals=init_vals_simple,\
                                      target_accept=target_accept,jitter_max_retries=1000,random_seed=rng)
        return idata



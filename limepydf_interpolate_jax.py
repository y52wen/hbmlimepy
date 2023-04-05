import numpy as np
from jax import numpy as jnp
import jax
from linear_interpolate_jax import RegularGridInterpolator

class limepy_interpolate:
    def __init__(self,**kwargs):
        self._set_constants()
        self._read_int_array()
        self._interpolate_coefficients()
        self._interpolate_phi()
        
    def _set_constants(self):
        self.G_s = 1
        
        #reference rh and M used for interpolations
        self.rhref, self.Mref = 1.0, 1e5
        self.Wmin, self.Wmax = 1.4, 16
        self.gmin, self.gmax = 0.001, 3.49
        #Number of points for W and g interpolation arrays
        self.N = 200
        #np stands for numpy array (in contrast to JAX array)
        self.g_a_np = np.linspace(self.gmin,self.gmax,self.N)
        self.W_a_np = np.linspace(self.Wmin,self.Wmax,self.N)
        #JAX numpy array
        self.g_a = jnp.linspace(self.gmin,self.gmax,self.N)
        self.W_a = jnp.linspace(self.Wmin,self.Wmax,self.N)
        
        #Maximum cutoff radius that outputs zero
        self.rt_max = 69
        
        N1 = 250
        N2 = 1000
        self.Nr = N1+N2-1
        r_1 = np.linspace(0,self.rhref,N1)
        r_2 = np.linspace(self.rhref,self.rt_max,N2)
        self.r_a = np.r_[r_1[:-1],r_2]
    
        #Boundary functions for g_Phi0 space
    def _set_up_boundary_func(self,offset):
        z = np.polyfit(self.W_a_np, self.g_boundary,10)
        self.coefficient_boundary = z
        p_f = np.poly1d(z)
        self.boundary_func = p_f-offset
    
    
    #Cuts the 2D array with W0 as x and g as y
    def bound_cut(self,x,y,array,f_b):
        M,N = np.shape(x)[0],np.shape(y)[0]
        array_cut = np.zeros((M,N))
        for i in range(M):
            j = 0
            while y[j]<f_b(x[i]):
                array_cut[i,j] = array[i,j]
                j += 1
                if j>=N:
                    break
        return array_cut
    
    
    def _read_int_array(self):
        #read numerical values needed for interpolation!
        container = np.load('interp_basic.npz')
        data = [container[key] for key in container]
        rt_2d_np, A_2d_np, s2_2d_np =  data[0], data[1], data[2]
        self.g_boundary = data[3]
    
        self._set_up_boundary_func(offset=0.2)
        
        rt_2d_cut = self.bound_cut(self.W_a_np, self.g_a_np, np.transpose(rt_2d_np), self.boundary_func)
        s2_2d_cut = self.bound_cut(self.W_a_np, self.g_a_np, np.transpose(s2_2d_np), self.boundary_func)
        A_2d_cut = self.bound_cut(self.W_a_np,  self.g_a_np, np.transpose(A_2d_np),  self.boundary_func)

        self.rt_2d = jnp.asarray(rt_2d_cut)
        self.A_2d = jnp.asarray(A_2d_cut)
        self.s2_2d = jnp.asarray(s2_2d_cut)
    
    def _interpolate_coefficients(self):
        self.A_p = RegularGridInterpolator((self.W_a, self.g_a), self.A_2d)
        self.rt_p = RegularGridInterpolator((self.W_a, self.g_a), self.rt_2d)
        self.s2_p = RegularGridInterpolator((self.W_a, self.g_a), self.s2_2d)
    
    def _interpolate_phi(self):
        container = np.load('../phi_3d.npz')
        phi_3d = jnp.array([container[key] for key in container])
        self.phi_p = RegularGridInterpolator((self.W_a, self.g_a, self.r_a), phi_3d)
        
    #sum of all points in the distribution function, normalized by mass
    #this is in fact not the df, but the likelihood of df
    #This is the one used in MCMC sampling
    #Should change the name aftewards!
    def my_df(self,r,v,ii):
        rt_s = self.rt_p(jnp.transpose(ii[:2]))
        s2_s = self.s2_p(jnp.transpose(ii[:2]))
        A_s = self.A_p(jnp.transpose(ii[:2]))
        
        R_s = 10**ii[3]/self.rhref
        M_s = 10**ii[2]/self.Mref

        r_s = r/R_s
        v2_s = self.G_s*M_s/R_s
        s2 = s2_s*v2_s 
        A = A_s*M_s/(v2_s**1.5*R_s**3)
        
        def float_inf(ii,a,b):
            return -jnp.inf
        
        def sum_log_like(ii,E,A):
            return jnp.sum(jnp.log(jnp.exp(E)*jax.scipy.special.gammainc(ii[1],E)*A/(10**ii[2])))
        
        def df_all_r(ii,r_s,v):
            phi_s = self.phi_p(jnp.stack([jnp.ones_like(r_s)*ii[0],\
                                        jnp.ones_like(r_s)*ii[1],r_s],axis=-1))
            phi = phi_s*v2_s
            vesc2 = 2.0*phi
            v2 = v**2
            E = (phi-0.5*v2)/s2
            return jax.lax.cond(jnp.all(v2<vesc2),sum_log_like,float_inf,ii,E,A)

        return jax.lax.cond(jnp.all(r_s<rt_s),df_all_r,float_inf,ii,r_s,v) 
    
    #return just df, true df
    def my_df_func(self,r,v,ii):
        rt_s = self.rt_p(jnp.transpose(ii[:2]))
        s2_s = self.s2_p(jnp.transpose(ii[:2]))
        A_s = self.A_p(jnp.transpose(ii[:2]))
        
        R_s = 10**ii[3]/self.rhref
        M_s = 10**ii[2]/self.Mref

        r_s = r/R_s
        v2_s = self.G_s*M_s/R_s
        s2 = s2_s*v2_s 
        A = A_s*M_s/(v2_s**1.5*R_s**3)
        
        def float_inf(ii,a,b):
            return a
        
        def like_func(ii,E,A):
            return jnp.exp(E)*jax.scipy.special.gammainc(ii[1],E)*A
        
        def df_all_r(ii,r_s,v):
            phi_s = self.phi_p(jnp.stack([jnp.ones_like(r_s)*ii[0],\
                                        jnp.ones_like(r_s)*ii[1],r_s],axis=-1))
            phi = phi_s*v2_s
            vesc2 = 2.0*phi
            v2 = v**2
            E = (phi-0.5*v2)/s2
            return jax.lax.cond(jnp.all(v2<vesc2),like_func,float_inf,ii,E,A)

        return jax.lax.cond(jnp.all(r_s<rt_s),df_all_r,float_inf,ii,r_s,v)
from coordinate_transform_jax import StC, CtS

import numpy as np
from jax import numpy as jnp
from limepy import limepy, sample
from math import pi

#only contains numpy array here
#Here this is simply the Cartesian version
def simulate_limepy(struct_param,center_param,Np=1000,data_type='ski',\
                    include_error=False,error=None,seed=None):
    
    W0true, gtrue,log10Mtrue,rhtrue = struct_param
    k = limepy(W0true, gtrue, M=10**log10Mtrue, rh=rhtrue)
    ic = sample(k,N=Np)
    
    if data_type=='Cartesian':
        xtrue, ytrue,ztrue, vxtrue, vytrue, vztrue = center_param
        vecx_np = np.array([ic.x+xtrue,ic.y+ytrue,ic.z+ztrue])
        vecv_np = np.array([ic.vx+vxtrue,ic.vy+vytrue,ic.vz+vztrue])
        return vecx_np,vecv_np
    else:
        #center param with a,d (in radian),R(in pc),vx,vy,vz(km/s)
        #This input format probably needs some change
        rad_to_mas = 180/pi*60*60*10**3
        ac = center_param[0]*rad_to_mas
        dc = center_param[1]*rad_to_mas
        pc = 1000/center_param[2]
        
        po_c = StC(jnp.array([ac,dc,pc,1,1,1]))
        
        #cartesian center: in pc and km/s
        phase_car_c = np.array([po_c[0],po_c[1],po_c[2],center_param[3],center_param[4],center_param[5]])
        #sky center: in mas, mas/year and km/s
        phase_sky_c = np.array(CtS(jnp.array(phase_car_c)))
                               
        x_array,y_array,z_array = ic.x+phase_car_c[0],ic.y+phase_car_c[1],ic.z+phase_car_c[2]
        vx_array,vy_array,vz_array = ic.vx+phase_car_c[3],ic.vy+phase_car_c[4],ic.vz+phase_car_c[5]
        
        sky_array = np.array(CtS(jnp.array([x_array,y_array,z_array,vx_array,vy_array,vz_array])))                   
        if include_error:
            if not(seed==None):
                np.random.seed(seed)
            error_norm = np.random.normal(size=(6,Np))
            #a_obs, d_obs, p_obs, va_obs, vd_obs, vR_obs
            sky_obs = sky_array+error_norm*error
            #we return both the true positons and and the observed positions with error
            return phase_sky_c,sky_array[:3],sky_array[3:],sky_obs[:3],sky_obs[3:]                               
        else:
            #position and velocity
            return phase_sky_c,sky_array[:3],sky_array[3:]      
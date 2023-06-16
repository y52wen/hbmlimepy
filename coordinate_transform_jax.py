from jax import numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
from math import pi

rad_to_mas = 180/pi*60*60*10**3    
pc_to_km = 3.085677581*10**13
year_to_s = 365.25*24*60*60

def CtS(Cin):
    #pc, km/s
    x,y,z,vx,vy,vz = Cin

    R = jnp.sqrt(x**2+y**2+z**2) #pc
    delta_r = pi/2-jnp.arctan2(jnp.sqrt(x**2+y**2),z) #rad
    alpha_r = jnp.arctan2(y,x) #rad
    
    alpha = alpha_r*rad_to_mas #mas
    delta = delta_r*rad_to_mas #mas
    p = 1000/R #mas

    #vx,vy,vz km/s
    mud = 0-1/(1+(jnp.sqrt(x**2+y**2)/z)**2)*((x*vx+y*vy)/jnp.sqrt(x**2+y**2)*z-jnp.sqrt(x**2+y**2)*vz)/z**2
    mua = 1/(1+(y/x)**2)*(vy*x-vx*y)/x**2*jnp.cos(delta_r) #km/(pc*s)

    va = mua/pc_to_km*rad_to_mas*year_to_s #mas/year
    vd = mud/pc_to_km*rad_to_mas*year_to_s #mas/year
    vR = (x*vx+y*vy+z*vz)/R #km/s

    return jnp.array([alpha,delta,p,va,vd,vR])

def StC(Sin):
    #mas,mas/year,km/s
    alpha,delta,p,va,vd,vR = Sin
    alpha_r = alpha/rad_to_mas #rad
    delta_r = delta/rad_to_mas #rad
    R = 1000/p #pc
    x = R*jnp.cos(alpha_r)*jnp.cos(delta_r)
    y = R*jnp.sin(alpha_r)*jnp.cos(delta_r)
    z = R*jnp.sin(delta_r)

    mua = va/(rad_to_mas*year_to_s)*R*pc_to_km #km/s
    mud = vd/(rad_to_mas*year_to_s)*R*pc_to_km #km/s
    vx = vR*jnp.cos(alpha_r)*jnp.cos(delta_r)-mua*jnp.sin(alpha_r)-mud*jnp.cos(alpha_r)*jnp.sin(delta_r)
    vy = vR*jnp.sin(alpha_r)*jnp.cos(delta_r)+mua*jnp.cos(alpha_r)-mud*jnp.sin(alpha_r)*jnp.sin(delta_r)
    vz = vR*jnp.sin(delta_r)+mud*jnp.cos(delta_r)
    return jnp.array([x,y,z,vx,vy,vz])




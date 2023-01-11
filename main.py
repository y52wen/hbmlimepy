import numpy as np
from math import pi


from simulate_limepy_sky import simulate_limepy
from HBI_limepy import Bayesian_sampling


#simulation input
seed = 17
#W，g, log10M, rh
ip = [5.0,2.0,np.log10(1e5),3.0]
#a,d,R,vx,vy,vz
ic = [pi/3,pi/4,1004,0.5,0.3,0.4]
#a,d,p (mas),va,vd (mas/year)
error_sd = np.transpose(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0]]))


#generated simulated data
sky_c,truex,truev,datax,datav= simulate_limepy(ip,ic,Np=1000,data_type='ski',\
                    include_error=True,error=error_sd,seed=seed)
#precompile data (does not matter)
xhhh = np.concatenate([np.array(ip),sky_c,\
                       truex[0],truex[1],(1000/truex[2]-1000/sky_c[2]),\
                       truev[0],truev[1],(truev[2]-sky_c[5])])

#create sampling class
B_S = Bayesian_sampling(datax,datav[:2,:],xhhh,data_type='ski',error_obs=error_sd)

#sampling:
ndraws=2000
chains=4
idata = B_S.sampling(ndraws=ndraws,chains=chains)


#record/save results
stacked = idata.posterior
data_t = np.stack([stacked.log10M.values,stacked.Phi0.values,stacked.g.values,stacked.rh.values,\
                stacked.ac.values,stacked.dc.values,stacked.vac.values,stacked.vdc.values,1000/stacked.pc.values])

np.savetxt('s%d%d%d%d-%d.txt' %(ip[0],ip[1],ip[3],ip[2],seed),data_t.reshape((9,ndraws*chains)))
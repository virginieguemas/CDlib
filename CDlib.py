# This module contains functions to estimate bulk exchange coefficients from 
# observations and to compute them from various parameterisations.
#
# List of functions: - CDN          -  Neutral bulk transfer coefficient
#                    - CSN          -  Neutral bulk exchange coefficient 
#                                      for heat or humidity
#                    - Z0           -  Aerodynamic roughness length
#                    - ZS           -  Scalar roughness length for 
#                                      heat or humidity
#                    - U            -  Wind speed 
#                    - S            -  Potential temperature or 
#                                      specific humidity
#                    - LMO          -  Monin-Obukhov length
#                    - Rb           -  Bulk Richardson number
#                    - Rstar        -  Roughness Reynolds number
#                    - Thetavstar   -  Scaling virtual temperature
#                    - PSI          -  Stability correction (additive)
#                    - F            -  Stability correction as a ratio CD/CDN
#
# Author : Virginie Guemas - 2020
################################################################################
import numpy as np
import meteolib
import sys
g = 9.81 # Gravity 
k = 0.4 # Von Karman constant
################################################################################
def CDN(u=None, ustar=None, f=None, z0=None, z=None) :
   """
   This function computes the neutral bulk momentum exchange coefficient CDN either as a function of :
   - the aerodynamic roughness length z0 (in m),
   - the height z (in m),
   or as a function of : 
   - the horizontal wind speed u (in m/s), 
   - the friction velocity ustar (in m/s), 
   - the multiplicative stability correction f.

   Author : Virginie Guemas - October 2020 
   """

   if z0 is not None and z is not None:
     CDn = (k/np.log(z/z0))**2
   elif u is not None and ustar is not None and f is not None:
     CD = (ustar**2)/(u**2)
     CDn = CD/f
   else:
     sys.exit('CDn can be computed either from (z0,z) or from (u,ustar,f)')

   return CDn
################################################################################
def CSN(deltas=None, u=None, sstar=None, ustar=None, f=None, zs=None, z0=None, z=None) :
   """
   This function computes the neutral bulk exchange coefficient for either heat or humidity as a function of :
   - the scalar roughness length zs for either temperature or humidity (in m),
   - the aerodyamic roughness length z0 (in m),
   - the height z (in m),
   or as function of :
   - the difference in scalar deltas between height z and the surface (either potential temperature in Kelvin or specific humidity in kg/kg),
   - the horizontal wind speed u (in m/s),
   - the scaling parameter sstar (thetastar for temperature in Kelvin and qstar for humidity in kg/kg),
   - the friction velocity ustar (in m/s),
   - the multiplicative stability correction psi.

   Author : Virginie Guemas - October 2020 
   """

   if zs is not None and z0 is not None and z is not None:
     CSn = k**2/(np.log(z/zs)*np.log(z/z0))
   elif ustar is not None and sstar is not None and u is not None and deltas is not None and f is not None: 
     CS = -(ustar*sstar)/(u*deltas)
     CSn = CS/f
   else:
     sys.exit('CSn can be computed either from (zs,z0,z) or from (u,ustar,deltas,sstar,f')

   return CSn
################################################################################
def Z0(u=None, ustar=None, psi=None, CDN=None, z=None, T=None, alpha=None) :
   """
   This function computes the aerodynamic roughness length (in m) either as a function of :
   - the neutral bulk momentum exchange coefficient CDN,
   - the height z (in m),
   or as a function of (the relation between the observed wind profile and flux):
   - the horizontal wind speed u (in m/s), 
   - the friction velocity ustar (in m/s), 
   - the additive stability correction psi.
   - the height z (in m) 
   or as a function of (Smith, 1988):
   - the alpha = 0.011 as in Charnock (1955) or alpha = 0.013 as in Zeng et al (1998)  
   - the friction velocity ustar (in m/s),
   - the temperature T (in Kelvin)  

   Author : Virginie Guemas - October 2020 
   Modified : Virginie Guemas - December 2020 - Option Smith (1988) formula used in COARE 2.5 (Fairall, 1996) 
   """

   if CDN is not None and z is not None:
     z0 = z/np.exp(np.sqrt(k**2/CDN))
   elif u is not None and ustar is not None and psi is not None and z is not None:
     z0 = z/np.exp(u/ustar*k + psi)
   elif alpha is not None and ustar is not None and T is not None:
     z0 = alpha*ustar**2/g + 0.11*meteolib.NU(T)/ustar
   else:
     sys.exit('z0 can be computed either from (CDN,z) or from (u,ustar,psi,z) or from (alpha,ustar,T)')

   return z0
################################################################################
def ZS(deltas=None, sstar=None, psi=None, CSN=None, z0=None, z=None, rstar=None, ustar=None, T=None, s=None) :
   """
   This function returns the scalar roughness length (in m) for either heat or humidity as a function of :
   - the neutral bulk scalar exchange coefficient CHN (for heat) or CQN (for momentum),
   - the aerodyamic roughness length z0 (in m),
   - the height z (in m),
   or as a function of (the relation between the observed scalar profile and flux):
   - the height z (in m),
   - the difference in scalar deltas between height z and the surface (either potential temperature in Kelvin or specific humidity in kg/kg)
   - the scaling parameter sstar (for either temperature thetastar or humidity qstar),
   - the additive stability correction psi,
   or as a function of (Liu et al 1979 used in Fairall, 1996):
   - the temperature T (in Kelvin),
   - the roughness Reynolds number rstar,
   - the friction velocity ustar (in m/s),
   - s = 'T'/'Q' for heat/humidity (parameters a and b differ),
   or as a function of (Andreas, 1987):
   - the roughness Reynolds number rstar,
   - the aerodynamic roughness length z0 (in m),
   - s = 'T'/'Q' for heat/humidity (parameters a and b differ),
   or a function of (Brutsaert,1982):

   Author : Virginie Guemas - October 2020 
   Modified : Virginie Guemas - December 2020 - Option LKB (Liu et al, 1979) used in COARE2.5 (Fairall et al 1996) 
                                                Option Andreas (1987)
   """

   if CSN is not None and z0 is not None and z is not None:
     zs = z/np.exp(k**2/(ln(z/z0)*CSN))
   elif z is not None and deltas is not None and sstar is not None and psi is not None: 
     zs = z/np.exp(deltas/sstar*k + psi)
   elif rstar is not None and ustar is not None and T is not None:
     if s == 'T':
       a = np.where(rstar<0, np.nan, np.where(rstar<0.11, 0.177, np.where(rstar<0.825, 1.376, np.where(rstar<3., 1.026, np.where(rstar<10., 1.625, np.where(rstar<30., 4.661, 34.904))))))
       b = np.where(rstar<0, np.nan, np.where(rstar<0.11, 0, np.where(rstar<0.825, 0.929, np.where(rstar<3., -0.599, np.where(rstar<10., -1.018, np.where(rstar<30., -1.475, -2.067))))))
       # There is a gap in Liu et al (1979) - no values for a/b are given for 0.825 < rstar < 0.925.
       # Surely a typo, but between 0.825 and 0.925, I arbitrarily chose 0.825 as boundary between two
       # consecutive segments.
     elif s == 'Q':
       a = np.where(rstar<0, np.nan, np.where(rstar<0.11, 0.292, np.where(rstar<0.825, 1.808, np.where(rstar<3., 1.393, np.where(rstar<10., 1.956, np.where(rstar<30., 4.994, 30.79))))))
       b = np.where(rstar<0, np.nan, np.where(rstar<0.11, 0, np.where(rstar<0.825, 0.826, np.where(rstar<3., -0.528, np.where(rstar<10., -0.870, np.where(rstar<30., -1.297, -1.845))))))
     else:
       sys.exit('s should be T for temperature or Q for humidity')

     zs = meteolib.NU(T)/ustar* a*rstar**b  
   elif rstar is not None and z0 is not None:
     if s == 'T':
       b0 = np.where(rstar<0, np.nan, np.where(rstar<=0.135, 1.25, np.where(rstar<2.5, 0.149, 0.317)))
       b1 = np.where(rstar<0, np.nan, np.where(rstar<=0.135, 0., np.where(rstar<2.5, -0.55, -0.565)))
       b2 = np.where(rstar<0, np.nan, np.where(rstar<=0.135, 0., np.where(rstar<2.5, 0., -0.183)))
     elif s == 'Q':
       b0 = np.where(rstar<0, np.nan, np.where(rstar<=0.135, 1.61, np.where(rstar<2.5, 0.351, 0.396)))
       b1 = np.where(rstar<0, np.nan, np.where(rstar<=0.135, 0., np.where(rstar<2.5, -0.628, -0.512)))
       b2 = np.where(rstar<0, np.nan, np.where(rstar<=0.135, 0., np.where(rstar<2.5, 0., -0.18)))
     else:
       sys.exit('s should be T for temperature or Q for humidity')
     
     zs = z0*np.exp(b0 + b1*np.log(rstar) + b2*np.log(rstar)**2)
   else:
     sys.exit('zs can be computed either from (CSN,z0,z) or from (deltas,sstar,psi,z) or from (rstar,ustar,T,s) or from (rstar,z0,s)')

   return zs
################################################################################
def U(z, ustar, z0, psi=0) :
   """
   This funcion computes the wind speed (in m/s) at height z (in m) as a fonction of the friction velocity in ustar in (m/s), the aerodynamic roughness length z0 (in m), and the additive stability correction psi.

   Author : Virginie Guemas - October 2020 
   """

   u = ustar/k * (np.log(z/z0) - psi)

   return u
################################################################################
def UG(Q0v,h,thetav):
   """
   This function computes the corrected wind speed to account for gustiness as in Fairall et al (1996, 2003).
   It depends on :
   - the surface virtual temperature flux Q0v (K.m.s-1)
   - the convective boundary layer height h (in m)
   - the virtual potential temperature (in Kelvin).

   Author : Virginie Guemas - December 2020
   """
  
   ug = 1.25 (g/thetav*h*Q0v)**(1/3)

   return ug
################################################################################
def S(z, s0, sstar, zs, psi=0) :
   """
   This function computes either the potential temperature theta (in Kelvin) or the specific humidity (in kg/kg) as a function of :
   - the scaling parameter sstar (thetastar for temperature in Kelvin and qstar for humidity in kg/kg), 
   - the surface parameter s0 (surface potential temperature in Kelvin or surface specific humidity in kg/kg), 
   - the scalar roughness length for heat or humidity (in m),
   - the additive stability correction psi,
   - the height z (in m).

   Author : Virginie Guemas - October 2020 
   """

   s = s0 + sstar/k * (np.log(z/zs) - psi)

   return s
################################################################################
def LMO(ustar, thetav, thetavstar=None, Q0v=None) :
   """
   This function computes the Monin-Obukhov length (in meters) either as a function of the virtual temperature scaling parameter thetavstar (in Kelvin) or as a function of the surface virtual temperature flux (K.m.s-1). It also requires the friction velocity (in m.s-1) and the layer-average virtual potential temperature (in Kelvin). 

   Author : Virginie Guemas - October 2020 
   """

   if Q0v is None and thetavstar is None:
     sys.exit('At least one of thetavstar or Q0v should be provided')

   beta = g/thetav 

   if Q0v is not None:
     Lmo = -(ustar**3)/(k*beta*Q0v)
     if thetavstar is not None:
       Qvbis = - ustar*thetavstar
       if np.max(np.abs((Qvbis-Q0v)/Q0v)) > 0.001 :
         sys.exit('Q0v and -ustar*thetavstar differ')
   else:
     Lmo = (ustar**2)/(k*beta*thetavstar)

   return Lmo
################################################################################
def RB(thetav, Dthetav, u, v, z) :
   """
   This function computes the Bulk Richardson number as a function of the vitual potential temperature thetav (in Kelvin) and wind speed horizontal components u and v (in m/s) at height z (in m) and the difference in virtual potential temperature between the height z and the surface Dthetav (in Kelvin).

   Author : Virginie Guemas - October 2020 
   """
   
   meteolib.check_T(thetav)
 
   Rb = g/thetav * Dthetav*z/(u**2+v**2)

   return Rb
################################################################################
def Rstar(ustar, z0, T) :
   """
   This function computes the Roughness Reynolds number as a function of the friction velocity ustar (in m/s), the aerodynamic roughness length (in m) and the temperature (in Kelvin).

   Author : Virginie Guemas - October 2020 
   """

   Rs = ustar*z0/meteolib.NU(T)

   return Rs
################################################################################
def Thetavstar(thetastar, qstar, theta, q) : 
   """
   This function estimates the scaling virtual temperature parameter (in Kelvin) from the scaling temperature parameter (in Kelvin), the scaling humidity parameter (in kg/kg), the layer-average temperature (in Kelvin) and the layer-average specific humidity (in kg/kg).

   Author : Virginie Guemas - October 2020 
   """

   meteolib.check_q(q)
   meteolib.check_q(qstar)
   meteolib.check_T(theta)

   thetavstar = thetastar*(1+0.61*q) + 0.61*theta*qstar

   return thetavstar
################################################################################
def PSI(z, Lmo, stab=None, unstab=None) :
   """
   This function computes a stability correction as a function of Monin-Obukhov length. It takes four arguments:
   - z = height
   - Lmo = Monin-Obukhov length
   - stab = formulation for stable regimes : 'dyer-hicks'  -- Dyer and Hicks (1970)
                                             'large-pond'  -- Large and Pond (1982)
                                             'lettau'      -- Lettau (1979) 
                                             'holtslag-bruin' -- Holtslag and de Bruin (1988)
                                             'beljaars-holtslag' -- Beljaars and Holtslag (1991)
                                             'grachev'     -- Grachev (2007)
   - unstab = formulation for unstable regimes : 'businger-dyer' -- Paulson (1970)
                                                 'kansas'        -- Businger (1971)
                                                 'fairall'       -- Fairall (1996)

   Author : Virginie Guemas - September 2020
   Modified : Sebastien Blein - December 2020 (correct Beljaars and Holtslag 1991)
              Virginie Guemas - December 2020 (add kansas, fairall and holtslag-bruin)
   """
   np.seterr(invalid='ignore')

   if stab is None or unstab is None:
       sys.exit('Stability correction type has to be specified (e.g.: stab=\'beljaars-holtslag\' and unstab=\'businger-dyer\')')

   zeta = z/Lmo
  
   ###############################
   if unstab == 'businger-dyer':
     phiM = (1 - 16*zeta)**(-0.25)
     phiH = (1 - 16*zeta)**(-0.5)

     psiM = np.where (zeta<0, 2*np.log((1+phiM**(-1))/2) + np.log((1+phiM**(-2))/2) - 2*np.arctan(phiM**(-1)) + np.pi/2, 0.)
     psiH = np.where (zeta<0, 2*np.log((1+phiH**(-1))/2), 0.)
   ################################
   elif unstab == 'kansas':
     phiM = (1 - 15*zeta)**(-0.25)
     phiH = 0.74*(1 - 9*zeta)**(-0.5)

     psiM = np.where (zeta<0, 2*np.log((1+phiM**(-1))/2) + np.log((1+phiM**(-2))/2) - 2*np.arctan(phiM**(-1)) + np.pi/2, 0.)
     psiH = np.where (zeta<0, 1.74*np.log((1+0.74*phiH**(-1))/1.74)+0.26*np.log((1-0.74*phiH**(-1))/0.26), 0.)
   ################################
   elif unstab == 'fairall':
     y = (1 - 12.87*zeta)**(1/3) 

     psi = np.where (zeta<0, 1.5*np.log((y**2+y+1)/3) - np.sqrt(3)*np.arctan((2*y+1)/np.sqrt(3)) + np.pi/np.sqrt(3), 0.)

     psiM = np.where (zeta<0, 1/(1+zeta**2)*PSI(z, Lmo, stab, unstab = 'kansas')[0] + zeta**2/(1+zeta**2)*psi, 0.) 
     psiH = np.where (zeta<0, 1/(1+zeta**2)*PSI(z, Lmo, stab, unstab = 'kansas')[1] + zeta**2/(1+zeta**2)*psi, 0.) 
   ################################  
   elif  unstab == 'beljaars-holtslag':
     a,b,c,d = 1.,0.667,5.,0.35

     psiM = np.where(zeta<0, -(a * zeta + b*(zeta - c/d) * np.exp(-d * zeta) + b*c/d), 0.)
     psiH = np.where(zeta<0, -(a * zeta + b*(zeta - c/d) * np.exp(-d * zeta) + b*c/d), 0.)
     # Beljaars and Holtslag do not propose an option for unstable cases but only for stable cases. This formulation is here
     # only to reproduce Elvidge et al (2016).
   else:
     sys.exit('This option for unstable cases is not coded yet.')
   ################################
   ################################
   if stab == 'dyer-hicks':
     psiM = np.where (zeta>0, -5*zeta, psiM)
     psiH = np.where (zeta>0, -5*zeta, psiM)
   ################################
   elif stab == 'large-pond':
     psiM = np.where (zeta>0, -7*zeta, psiM)
     psiH = np.where (zeta>0, -7*zeta, psiM)
   ################################
   elif stab == 'lettau':
     x = 1 + 4.5*zeta

     psiM = np.where (zeta>0, np.log((x**0.5+1)/2) + 2*np.log((x**0.25+1)/2) -2*np.arctan(x**0.25) + np.pi/2 + 4/3*(1-x**0.75), psiM)
     psiH = np.where (zeta>0, 2*np.log((x**0.25+1)/2) -2*(x**1.5/3 + x**0.5 - 4/3), psiH)
   ################################
   elif stab == 'holtslag-bruin':
     psi = -3.75/0.35 - 0.7*zeta + 3.75/0.35*np.exp(-0.35*zeta) -0.75*zeta*np.exp(-0.35*zeta)

     psiM = np.where (zeta>0, psi, psiM)
     psiH = np.where (zeta>0, psi, psiH)
   ################################
   elif stab == 'beljaars-holtslag':
     a,b,c,d = 1.,0.667,5.,0.35

     psiM = np.where (zeta>0, -(a * zeta + b*(zeta - c/d) * np.exp(-d * zeta) + b*c/d), psiM)
     psiH = np.where (zeta>0, -((1+2.*a*zeta/3)**(3./2) + b*(zeta-c/d)*np.exp(-d*zeta) + b*c/d - 1), psiH)
   ################################
   elif stab == 'grachev':
     x = (zeta+1)**(1/3)

     psiM = np.where (zeta>0, -19.5*(x-1) + 3.25*0.3**(1/3)*(2*np.log((x+0.3**(1/3))/(1+0.3**(1/3))) - np.log((x**2-0.3**(1/3)*x+0.3**(2/3))/(1-0.3**(1/3)+0.3**(2/3))) +2*np.sqrt(3)*(np.arctan((2*x-0.3**(1/3))/(np.sqrt(3)*0.3**(1/3))) -np.arctan((2-0.3**(1/3))/(np.sqrt(3)*0.3**(1/3))))) , psiM)
     psiH = np.where (zeta>0, -2.5*np.log(1+3*zeta+zeta**2) +5/(2*np.sqrt(5))*(np.log((2*zeta+3-np.sqrt(5))/(2*zeta+3+np.sqrt(5))) -np.log((3-np.sqrt(5))/(3+np.sqrt(5)))), psiH)
   ################################
   else:
     sys.exit('This option for stable cases is not coded yet.')
   ################################
   ################################
   
   return (psiM,psiH)
################################################################################
def F(Rb, CDN, z, var='momentum', author='Louis') :
   """
   This function computes the CD/CDN ratio following Louis (1979) with either Louis original functions or later adjustments by other authors. It takes five arguments:
   - Rb = Richardson bulk number
   - CDN = neutral drag coefficient
   - z = height
   - var = 'momentum' / 'heat'
   - author = 'Louis' / 'LupkesGryanik'

   Author : Virginie Guemas - September 2020 
   """
  
   z0=CDlib.z0(CDN)

   if author == 'Louis' :
     c1 = 9.4
     if var == 'momentum' :
       alpha = 7.4
     elif var == 'heat' or var == 'moisture' :
       alpha = 5.3
     else:
       sys.exit('var can be momentum, heat or moisture only')
     
     c2 = c1*alpha*CDN*(z/z0)**0.5

     fstab = (1+c1/2*Rb)**(-2)

   elif author == 'LupkesGryanik' :
     if var == 'momentum' :
       c1 = 10
       alpha = 7.5
     elif var == 'heat' or var == 'moisture' :
       c1 = 15
       alpha = 5
     else:
       sys.exit('var can be momentum, heat or moisture only')
     
     c2 = c1*alpha*CDN*(z/z0+1)**0.5

     fstab = (1+10*Rb/np.sqrt(Rb+1))**(-1)

   else:
     sys.exit('Only Louis and LupkesGryanik version are coded for now')

   f = np.where(Rb < 0, 1 - (c1*Rb)/(1+c2*np.abs(Rb)**0.5), fstab)

   return f
################################################################################

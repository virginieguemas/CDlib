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
#                    - UG           -  Wind speed accounting for gustiness
#                    - S            -  Potential temperature or 
#                                      specific humidity
#                    - LMO          -  Monin-Obukhov length
#                    - LMOapprox    -  Monin-Obukhov length approximation
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
   or as a function of (the relation between observed wind profile and fluxes): 
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
   or as function of (the relation between observed meteorological parameters and fluxes):
   - the difference in scalar deltas between height z and the surface (either potential temperature in Kelvin or specific humidity in kg/kg),
   - the horizontal wind speed u (in m/s),
   - the scaling parameter sstar (thetastar for temperature in Kelvin and qstar for humidity in kg/kg),
   - the friction velocity ustar (in m/s),
   - the multiplicative stability correction f.

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
def Z0(method=None, u=None, ustar=None, psi=None, CDN=None, z=None, T=None, alpha=None) :
   """
   This function returns the aerodynamic roughness length (in m).
   With option method = 'CN', (correspondance between neutral exchange coefficient and roughness length),
   the function needs :
   - the neutral bulk momentum exchange coefficient CDN,
   - the height z (in m).
   With option method = 'obs' (from the relation between the observed wind profile and flux),
   the function needs :
   - the horizontal wind speed u (in m/s), 
   - the friction velocity ustar (in m/s), 
   - the additive stability correction psi.
   - the height z (in m).
   With option method = 'coare2.5' (Smith, 1988 formula used in COARE2.5):
   the function needs :
   - the alpha = 0.011 as in Charnock (1955), Smith (1988) and COARE2.5 or alpha = 0.013 as in Zeng et al (1998) or alpha = 0.018 as in Beljaars (1995) 
   - the friction velocity ustar (in m/s),
   - the temperature T (in Kelvin).
   With option method = 'coare3.0' (Fairall et al 2003),
   the function needs :
   - the 10m horizontal wind speed u (in m/s),
   - the friction velocity ustar (in m/s),
   - the temperature T (in Kelvin).
   With option method = 'tayloryelland' (Taylor and Yelland, 2001) implemented as an option in COARE3.0,
   the function needs :
   - the 10m horizontal wind speed u (in m/s),
   - the friction velocity ustar (in m/s),
   - the temperature T (in Kelvin).
   With option method = 'oost' (Oost et al, 2002) implemented as an option in COARE3.0,
   the function needs :
   - the 10m horizontal wind speed u (in m/s),
   - the friction velocity ustar (in m/s),
   - the temperature T (in Kelvin).

   Author : Virginie Guemas - October 2020 
   Modified : Virginie Guemas - December 2020 - Option Smith (1988) formula used in COARE 2.5 (Fairall, 1996) 
                                                Option COARE 3.0 (Fairall et al 2003)
                                                Options Taylor and Yelland (2001) and Oost et al (2002)
   """

   if method == 'CN': 
     if CDN is not None and z is not None:
       z0 = z/np.exp(np.sqrt(k**2/CDN))
     else: 
       sys.exit('With option method = \'CN\', input CDN and z are required.')

   ##################################
   elif method == 'obs':
     if u is not None and ustar is not None and psi is not None and z is not None:
       z0 = z/np.exp(u/ustar*k + psi)
     else: 
       sys.exit('With option method = \'obs\', input z, u, ustar and psi are required.')

   ##################################
   elif method == 'coare2.5':
     if alpha is not None and ustar is not None and T is not None:
       z0 = alpha*ustar**2/g + 0.11*meteolib.NU(T)/ustar
     else:
       sys.exit('With option method = \'coare2.5\', input alpha, ustar and T as required') 

   ##################################
   elif method == 'coare3.0':
     if u is not None and ustar is not None and T is not None:
       u10n = u
       err = 10
       while err>0.01: 
         alpha = np.where(u10n>18,0.018,np.where(u10n<10,0.011,0.011+(0.018-0.011)/(18-10)*(u10n-10)))
         z0 = alpha*ustar**2/g + 0.11*meteolib.NU(T)/ustar
         tmp = U(z=10, ustar = ustar, z0 = z0) 
         err = np.nanmax(abs(u10n-tmp))
         u10n = tmp
     else:    
       sys.exit('With option method = \'coare3.0\', input u, ustar and T are required') 

   ##################################
   elif method == 'tayloryelland':
     # What is coded here corresponds to what is reported in Fairall et al (2003) implemented as an option in COARE3.0
     # What appears in SURFEX documentation is not exactly the same: u is used instead of u10n and a different formula
     # is used for hs.
     if u is not None and ustar is not None and T is not None:
       u10n = u
       err = 10
       while err>0.01:
         Tp = 0.729*u10n
         hs = 0.0248*u10n**2
         Lp = (g*Tp**2)/(2*np.pi)
         z0 = 1200*hs*(hs/Lp)**4.5 + 0.11*meteolib.NU(T)/ustar
         tmp = U(z=10, ustar = ustar, z0 = z0) 
         err = np.nanmax(abs(u10n-tmp))
         u10n = tmp
     else:    
       sys.exit('With option method = \'tayloryelland\', input u, ustar and T are required') 

   ##################################
   elif method == 'oost':
     # What is coded here corresponds to what is reported in Fairall et al (2003) implemented as an option in COARE3.0.
     # What appears in SURFEX documentation is not exactly the same : u is used instead of U10n
     if u is not None and ustar is not None and T is not None:
       u10n = u
       err = 10
       while err>0.01:
         Tp = 0.729*u10n
         Lp = (g*Tp**2)/(2*np.pi)
         Cp = g*Tp/(2*np.pi)
         z0 = 50/(2*np.pi)*Lp*(ustar/Cp)**4.5 + 0.11*meteolib.NU(T)/ustar
         tmp = U(z=10, ustar = ustar, z0 = z0) 
         err = np.nanmax(abs(u10n-tmp))
         u10n = tmp
     else:    
       sys.exit('With option method = \'oost\', input u, ustar and T are required') 

   else:
     sys.exit('Valid methods are \'CN\', \'obs\', \'coare2.5\', \'coare3.0\',\'tayloryelland\',\'oost\'.')

   return z0
################################################################################
def ZS(method=None, deltas=None, sstar=None, psi=None, CSN=None, z0=None, z=None, rstar=None, ustar=None, T=None, s=None) :
   """
   This function returns the scalar roughness length (in m) for either heat or humidity.
   With option method = 'CN' (correspondance between neutral exchange coefficients and roughness length), 
   the function needs :
   - the neutral bulk scalar exchange coefficient CHN (for heat) or CQN (for momentum),
   - the aerodyamic roughness length z0 (in m),
   - the height z (in m).
   With option method = 'obs' (from the relation between the observed scalar profiles and fluxes),
   the function needs :
   - the height z (in m),
   - the difference in scalar deltas between height z and the surface (either potential temperature in Kelvin or specific humidity in kg/kg)
   - the scaling parameter sstar (for either temperature thetastar or humidity qstar),
   - the additive stability correction psi.
   With option method = 'LKB' (Liu et al 1979 used in COARE2.5),
   the function needs:
   - the temperature T (in Kelvin),
   - the roughness Reynolds number rstar,
   - the friction velocity ustar (in m/s).
   - s = 'T'/'Q' for heat/humidity (parameters a and b differ).
   With option method = 'andreas' (model from Andreas, 1987),
   the function needs:
   - the roughness Reynolds number rstar,
   - the aerodynamic roughness length z0 (in m),
   - s = 'T'/'Q' for heat/humidity (parameters b0, b1 and b2 differ).
   With option method = 'brutsaertgarratt' (Garratt, 1992 inspired from Brutsaert 1975 model),
   the function needs: 
   - the temperature T (in Kelvin),
   - the friction velocity ustar (in m/s).   
   - the aerodynamic roughness length z0 (in m),
   With option method = 'revisedbrutsaertgarratt' (updated coefficients as proposed in Fairall, 2003),
   the function needs: 
   - the temperature T (in Kelvin),
   - the friction velocity ustar (in m/s).   
   - the aerodynamic roughness length z0 (in m),
   With option method = 'simplebrutsaert' (simplified Brutsaert 1982 model used in several bulk algorithms presented in Fairall 2003),
   the function needs:
   - the temperature T (in Kelvin),
   - the friction velocity ustar (in m/s).   
   - s = 'T'/'Q' for heat/humidity (parameter r differ).
   With option method = 'clayson' (Clayson et al 1996),
   the function needs: Not coded yet

   With option method = 'mondonredelsperger' (Mondon and Redelsperger, 1998) proposed in SURFEX,
   the function needs:
   - the temperature T (in Kelvin),
   - the friction velocity ustar (in m/s).   
   - s = 'T'/'Q' for heat/humidity (parameter r differ).
   With option method = 'coare3.0' (Fairall et al, 2003),
   the function needs:
   - the temperature T (in Kelvin),
   - the friction velocity ustar (in m/s).   
   - the aerodynamic roughness length z0 (in m),

   Author : Virginie Guemas - October 2020 
   Modified : Virginie Guemas - December 2020 - Option LKB (Liu et al, 1979) used in COARE2.5 (Fairall et al 1996) 
                                                Option Andreas (1987)
                                                Option Brutsaert (1982) simplified model
                                                Option COARE 3.0 (Fairall et al, 2003) and (Garratt, 1992) and revised (Garratt, 1992) as suggested in Fairall et al, 2003
   """

   if method == 'CN': 
     if CSN is not None and z0 is not None and z is not None:
       zs = z/np.exp(k**2/(np.log(z/z0)*CSN))
     else: 
       sys.exit('With option method = \'CN\', input CSN, z0 and z are required.')

   ###################################
   elif method == 'obs':
     if z is not None and deltas is not None and sstar is not None and psi is not None: 
       zs = z/np.exp(deltas/sstar*k + psi)
     else: 
       sys.exit('With option method = \'obs\', input z, deltas, sstar and psi are required.')

   ###################################
   elif method == 'LKB': 
     if rstar is not None and ustar is not None and T is not None:
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
     else:
       sys.exit('With option method = \'LKB\', input rstar, ustar, T and s are required.')
  
   ###################################
   elif method == 'andreas': 
     if rstar is not None and z0 is not None:
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
       sys.exit('With option method = \'andreas\', input rstar, z0 and s are required.')

   ##################################
   elif method == 'brutsaertgarratt':
     if ustar is not None and z0 is not None and T is not None:
       Rr = ustar*z0/meteolib.NU(T)
       zs = z0*np.exp(2-2.28*Rr**0.25)
     else:
       sys.exit('With option method = \'brutsaertgarratt\', input ustar, z0 and T are required.')

   ##################################
   elif method == 'revisedbrutsaertgarratt':
     if ustar is not None and z0 is not None and T is not None:
       Rr = ustar*z0/meteolib.NU(T)
       zs = z0*np.exp(3.4-3.5*Rr**0.25)
     else:
       sys.exit('With option method = \'revisedbrutsaertgarratt\', input ustar, z0 and T are required.')

   ##################################
   elif method == 'simplebrutsaert':
     if ustar is not None and T is not None:
       if s == 'T':
         r = 0.4
       elif s == 'Q':
         r = 0.62
       else:
         sys.exit('s should be T for temperature or Q for humidity')
       zs = r*meteolib.NU(T)/ustar
     else:
       sys.exit('With option method = \'simplebrutsaert\', input ustar, T and s are required.')

   ##################################
   elif method == 'clayson':
     sys.exit('Sorry. Option method = \'clayson\' from Clayson et al (1996) is not coded yet. Coming soon')

   ##################################
   elif method == 'mondonredelsperger':
     if ustar is not None and T is not None:
       if s == 'T':
         zs = np.where(ustar>0.23, 0.14*meteolib.NU(T)/(ustar-0.2) + 7*10**(-6), 0.015*ustar**2/g+0.18*meteolib.NU(T)/ustar)
       elif s == 'Q':
         zs = np.where(ustar>0.23, 0.2*meteolib.NU(T)/(ustar-0.2) + 9*10**(-6), 0.0205*ustar**2/g+0.294*meteolib.NU(T)/ustar)
       else:
         sys.exit('s should be T for temperature or Q for humidity')
     else:
       sys.exit('With option method = \'mondonredelsperger\', input ustar, T and s are required.')

   ##################################
   elif method == 'coare3.0':
     if ustar is not None and z0 is not None and T is not None:
       Rr = ustar*z0/meteolib.NU(T)
       zs = 0.000055*Rr**(-0.6)
       zs = np.where(zs<0.0001,0.0001,zs)
     else:
       sys.exit('With option method = \'coare3.0\', input ustar, z0 and T are required.')

   else:
     sys.exit('Valid methods are \'CN\', \'obs\', \'LKB\', \'andreas\',\'brutsaertgarratt\',\'simplebrutsaert\',\'mondonredelsperger\',\'coare3.0\'.')

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
def UG(method=None, u=None, Q0v=None, h=None, thetav=None, beta=1.25):
   """
   This function computes the corrected wind speed to account for gustiness.
   With option method = 'godfreybeljaars' as in Fairall et al (1996, 2003),
   the function needs :
   - the horizontal wind speed u (in m/s),
   - the correction factor beta = 1.25 as in COARE2.5 or beta = 1 as in Zeng et al (1998) or beta = 1.2 as in Beljaars (1995)
   - the surface virtual temperature flux Q0v (K.m.s-1)
   - the convective boundary layer height h (in m), typically h=600m,
   - the virtual potential temperature thetav (in Kelvin).

   Author : Virginie Guemas - December 2020
   """
   
   if method == 'godfreybeljaars':
     if u is not None and thetav is not None and h is not None and Q0v is not None:
       ug = beta * (g/thetav*h*Q0v)**(1/3)
       ucor = np.sqrt(u**2+ug**2)
     else:
       sys.exit('With option method = \'godfreybeljaars\', input u, thetav, h and Q0v are required.')

   elif method == 'jordan':
     if u is not None:
       ucor = u + 0.5/np.cosh(u)
     else:
       sys.exit('With option method = \'jordan\', input u is required.')

   else:
     sys.exit('Valid methods are \'godfreybeljaars\' and \'jordan\'.')

   return ucor
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
   meteolib.check_T(thetav)

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
def LMOapprox(ustar, T, thetastar=None, qstar=None, Q0=None, E0=None) :
   """
   This function approximates the Monin-Obukhov length (in meters) as in Fairall et al (1996, 2003) either as of :
   - the friction velocity ustar (in m.s-1), 
   - the temperature scaling parameter thetastar (in Kelvin),
   - the humidity scaling parameter qstar (in kg.kg-1),
   - the temperature (in Kelvin).
   or as a function of :
   - the friction velocity ustar (in m.s-1),
   - the turbulent heat flux Q0 (in K.m.s-1),
   - the turbulent humidity flux E0 (in m.s-1),
   - the temperature (in Kelvin).

   Author : Virginie Guemas - January 2021
   """
   meteolib.check_T(T)
   
   if thetastar is None and Q0 is None:
     sys.exit('At least one of thetastar or Q0 should be provided')
   
   if qstar is None and E0 is None:
     sys.exit('At least one of qstar or E0 should be provided')

   beta = g/T

   if Q0 is not None and E0 is not None:
     Lmo = -(ustar**3)/(k*beta*(Q0+0.61*T*E0))
   elif thetastar is not None and qstar is not None:
     Lmo = (ustar**2)/(k*beta*(thetastar+0.61*T*qstar))
   elif Q0 is not None and qstar is not None:
     Lmo = -(ustar**3)/(k*beta*(Q0+0.61*T*(-ustar*qstar)))
   else:
     Lmo = -(ustar**3)/(k*beta*((-ustar*thetastar)+0.61*T*E0))

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
                                                 'kansas'        -- Businger et al (1971)
                                                 'fairall'       -- Fairall et al (1996)

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
   elif unstab == 'holtslag1990':
     # I need to integrate the phi
     psiM = None
     psiH = None
   ################################
   elif unstab == 'kaderyaglom1990':
     # I need to integrate the phi
     psiM = None
     psiH = None
   ################################
   elif unstab == 'fairall1996':
     y = (1 - 12.87*zeta)**(1/3) 

     psi = np.where (zeta<0, 1.5*np.log((y**2+y+1)/3) - np.sqrt(3)*np.arctan((2*y+1)/np.sqrt(3)) + np.pi/np.sqrt(3), 0.)

     psiM = np.where (zeta<0, 1/(1+zeta**2)*PSI(z, Lmo, stab, unstab = 'businger-dyer')[0] + zeta**2/(1+zeta**2)*psi, 0.) 
     psiH = np.where (zeta<0, 1/(1+zeta**2)*PSI(z, Lmo, stab, unstab = 'businger-dyer')[1] + zeta**2/(1+zeta**2)*psi, 0.) 
   ################################
   elif unstab == 'fairall2003':
     a={'m':10.15,'h':34.15}
     psi={}
     for s in ('m','h'):
       y = (1 - a[s]*zeta)**(1/3) 

       psi[s] = np.where (zeta<0, 1.5*np.log((y**2+y+1)/3) - np.sqrt(3)*np.arctan((2*y+1)/np.sqrt(3)) + np.pi/np.sqrt(3), 0.)

     psiM = np.where (zeta<0, 1/(1+zeta**2)*PSI(z, Lmo, stab, unstab = 'businger-dyer')[0] + zeta**2/(1+zeta**2)*psi['m'], 0.) 
     psiH = np.where (zeta<0, 1/(1+zeta**2)*PSI(z, Lmo, stab, unstab = 'businger-dyer')[1] + zeta**2/(1+zeta**2)*psi['h'], 0.) 
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
   elif stab == 'holtslag1990':
       # I need to integrate the phi
     psiM = None
     psiH = None
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
def BULK(u, deltaq, deltatheta, z) :








   return {'ustar':ustar, 'qstar':qstar, 'thetastar':thetastar, 'CDN':CDN, 'CHN':CHN}
################################################################################

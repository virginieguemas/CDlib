# This module contains functions to estimate bulk exchange coefficients from 
# observations and to compute them from various parameterisations.
#
# List of functions: - CDN          -  Neutral momentum bulk transfer coefficient 
#                    - CD           -  Momentum bulk transfer coefficient
#                    - CSN          -  Neutral scalar bulk transfer coefficient 
#                    - CS           -  Scalar bulk transfer coefficient
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
#                    - RB           -  Bulk Richardson number
#                    - Rstar        -  Roughness Reynolds number
#                    - Thetavstar   -  Scaling virtual temperature
#                    - WEBB         -  Webb correction to latent heat flux
#                    - HSR          -  Sensible heat flux due to rain
#                    - TAUR         -  Wind stress due to rain
#                    - PSI          -  Stability correction (additive)
#                    - F            -  Stability correction as a ratio CD/CDN
#                    - BULK         -  Bulk algorithms for turbulent fluxes
#
# Author : Virginie Guemas - 2020
# Modified : Sebastien Blein - January 2021 - Version that accepts unumpy arrays
#                              storing uncertainties as input and propagates those
#                              uncertainties until the output, still compatible
#                              with an usage without uncertainties
################################################################################
import numpy as np
import meteolib
import sys
from uncertainties import unumpy as unp

g = 9.81     # Gravity - Fairall et al 1996 use 9.72 instead
k = 0.4      # Von Karman constant
cpw = 4210   # J.K-1.kg-1 # specfic heat of liquid water
cp  = 1004   # J.K-1.kg-1 # specific heat of dry air
Rv  = 461.4  # J.K-1.kg-1
Ra  = 287    # J.K-1.kg-1 
dv  = 0.219  # cm2.s-1    # Water vapour diffusivity
dh  = 0.1906 # cm2.s-1    # Heat diffusivtity in air
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
   Modified : January 2021 - Sebastien Blein - Uncertainty propagation
   """

   if z0 is not None and z is not None:
     z0 = np.where(unp.nominal_values(z0)<=0,np.nan,z0)
     z0 = np.where(unp.nominal_values(z0)==np.inf,np.nan,z0)
     z = np.where(unp.nominal_values(z)<=unp.nominal_values(z0),np.nan,z)
     #
     CDn = (k/unp.log(z/z0))**2
   elif u is not None and ustar is not None and f is not None:
     CD = (ustar**2)/(u**2)
     CDn = CD/f
   else:
     sys.exit('CDn can be computed either from (z0,z) or from (u,ustar,f)')

   return CDn
################################################################################
def CD (CDN=None, psi=None) :
   """
   This function computes the bulk momentum transfer coefficient as a function of the
   neutral bulk momentum transfer coefficient CDN and the additive stability correction
   psi.

   Author : Virginie Guemas - January 2021 
   Modified : January 2021 - Sebastien Blein - Uncertainty propagation
   """
  
   if CDN is not None and psi is not None : 
     Cd = CDN / (1-unp.sqrt(CDN)/k*psi)**2
   else:
     sys.exit('CDN and psi are required to compute CD')

   return Cd
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
   Modified : January 2021 - Sebastien Blein - Uncertainty propagation
   """

   if zs is not None and z0 is not None and z is not None:
     zs = np.where(unp.nominal_values(zs)<=0,np.nan,zs)
     zs = np.where(unp.nominal_values(zs)==np.inf,np.nan,zs)
     z0 = np.where(unp.nominal_values(z0)<=0,np.nan,z0)
     z0 = np.where(unp.nominal_values(z0)==np.inf,np.nan,z0)
     z = np.where(unp.nominal_values(z)<=unp.nominal_values(z0),np.nan,z)
     z = np.where(unp.nominal_values(z)<=unp.nominal_values(zs),np.nan,z)
     CSn = k**2/(unp.log(z/zs)*unp.log(z/z0))
   elif ustar is not None and sstar is not None and u is not None and deltas is not None and f is not None: 
     CS = -(ustar*sstar)/(u*deltas)
     CSn = CS/f
   else:
     sys.exit('CSn can be computed either from (zs,z0,z) or from (u,ustar,deltas,sstar,f')

   return CSn
################################################################################
def CS (CDN, CSN, psiM, psiH) :
   """
   This function computes the bulk heat or humidity transfer coefficient as a function of the
   neutral bulk heat or humidity transfer coefficient CSN, the neutral bulk momentum transfer coefficient CDN
   and the additive stability correction psiM for momentum and psiH for heat or humidity.

   Author : Virginie Guemas - January 2021 
   Modified : January 2021 - Sebastien Blein - Uncertainty propagation
   """
  
   if CDN is not None and CSN is not None and psiM is not None and psiH is not None : 
     Cs = CSN / ((1-CSN/(k*unp.sqrt(CDN))*psiH)*(1-unp.sqrt(CDN)/k*psiM))
   else:
     sys.exit('CDN, CSN, psiM and psiH are required to compute CH or CE')

   return Cs
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
   Modified : December 2020 - Virginie Guemas - Option Smith (1988) formula used in COARE 2.5 (Fairall, 1996)
                                                Option COARE 3.0 (Fairall et al 2003)
                                                Options Taylor and Yelland (2001) and Oost et al (2002)
              January 2021  - Sebastien Blein - Uncertainty propagation
   """

   if method == 'CN': 
     if CDN is not None and z is not None:
       CDN = np.where(unp.nominal_values(CDN)==0,np.nan,CDN)
       CDN = np.where(unp.nominal_values(CDN)==np.inf,np.nan,CDN)
       z0 = z/unp.exp(unp.sqrt(k**2/CDN))
     else: 
       sys.exit('With option method = \'CN\', input CDN and z are required.')

   ##################################
   elif method == 'obs':
     if u is not None and ustar is not None and psi is not None and z is not None:
       ustar = np.where(unp.nominal_values(ustar)==0,np.nan,ustar)
       u = np.where(unp.nominal_values(u/ustar*k + psi)<=-1E3,np.nan,u)
       u = np.where(unp.nominal_values(u/ustar*k + psi)>=1E3,np.nan,u)
       #
       z0 = z/unp.exp(u/ustar*k + psi)
     else: 
       sys.exit('With option method = \'obs\', input z, u, ustar and psi are required.')

   ##################################
   elif method == 'coare2.5':
     if alpha is not None and ustar is not None and T is not None:
       ustar = np.where(unp.nominal_values(ustar)==0,np.nan,ustar)
       #
       z0 = alpha*ustar**2/g + 0.11*meteolib.NU(T)/ustar
     else:
       sys.exit('With option method = \'coare2.5\', input alpha, ustar and T as required') 

   ##################################
   elif method == 'coare3.0':
     if u is not None and ustar is not None and T is not None:
       ustar = np.where(unp.nominal_values(ustar)==0,np.nan,ustar)
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
       ustar = np.where(unp.nominal_values(ustar)==0,np.nan,ustar)
       u = np.where(unp.nominal_values(u)==0,np.nan,u)
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
       ustar = np.where(unp.nominal_values(ustar)==0,np.nan,ustar)
       u = np.where(unp.nominal_values(u)==0,np.nan,u)
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
   - the roughness Reynolds number rstar,
   - the aerodynamic roughness length z0 (in m),
   With option method = 'revisedbrutsaertgarratt' (updated coefficients as proposed in Fairall, 2003),
   the function needs: 
   - the roughness Reynolds number rstar,
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
   - the roughness Reynolds number rstar.

   Author : Virginie Guemas - October 2020 
   Modified : December 2020 - Virginie Guemas - Option LKB (Liu et al, 1979) used in COARE2.5 (Fairall et al 1996)
                                                Option Andreas (1987)
                                                Option Brutsaert (1982) simplified model
                                                Option COARE 3.0 (Fairall et al, 2003) and (Garratt, 1992) and revised (Garratt, 1992) as suggested in Fairall et al, 2003
              January 2021  - Sebastien Blein - Uncertainty propagation
   """

   if method == 'CN': 
     if CSN is not None and z0 is not None and z is not None:
       z0 = np.where(unp.nominal_values(z0)<=0,np.nan,z0)
       z0 = np.where(unp.nominal_values(z0)==np.inf,np.nan,z0)
       z = np.where(unp.nominal_values(z)<=unp.nominal_values(z0),np.nan,z)
       CSN = np.where(unp.nominal_values(CSN)<=0,np.nan,CSN)
       CSN = np.where(unp.nominal_values(CSN)==np.inf,np.nan,CSN)
       zs = z/unp.exp(k**2/(unp.log(z/z0)*CSN))
     else: 
       sys.exit('With option method = \'CN\', input CSN, z0 and z are required.')

   ###################################
   elif method == 'obs':
     if z is not None and deltas is not None and sstar is not None and psi is not None: 
       sstar = np.where(unp.nominal_values(sstar)==0,np.nan,sstar)
       tmp = unp.exp(deltas/sstar*k + psi)
       tmp = np.where(unp.nominal_values(tmp)==0,np.nan,tmp)
       zs = z/tmp
     else: 
       sys.exit('With option method = \'obs\', input z, deltas, sstar and psi are required.')

   ###################################
   elif method == 'LKB': 
     if rstar is not None and ustar is not None and T is not None:
       ustar = np.where(unp.nominal_values(ustar)==0,np.nan,ustar)
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
       rstar = np.where(unp.nominal_values(rstar)<=0,np.nan,rstar)
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
       zs = z0*unp.exp(b0 + b1*unp.log(rstar) + b2*unp.log(rstar)**2)
     else:
       sys.exit('With option method = \'andreas\', input rstar, z0 and s are required.')

   ##################################
   elif method == 'brutsaertgarratt':
     if rstar is not None and z0 is not None:
       zs = z0*unp.exp(2-2.28*rstar**0.25)
     else:
       sys.exit('With option method = \'brutsaertgarratt\', input rstar and z0 are required.')

   ##################################
   elif method == 'revisedbrutsaertgarratt':
     if rstar is not None and z0 is not None:
       zs = z0*unp.exp(3.4-3.5*rstar**0.25)
     else:
       sys.exit('With option method = \'revisedbrutsaertgarratt\', input rstar and z0 are required.')

   ##################################
   elif method == 'simplebrutsaert':
     if ustar is not None and T is not None:
       ustar = np.where(unp.nominal_values(ustar)==0,np.nan,ustar)
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
       ustar = np.where(unp.nominal_values(ustar)==0,np.nan,ustar)
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
     if rstar is not None:
       zs = 0.000055*rstar**(-0.6)
       zs = np.where(zs<0.0001,0.0001,zs)
     else:
       sys.exit('With option method = \'coare3.0\', input rstar is required.')

   else:
     sys.exit('Valid methods are \'CN\', \'obs\', \'LKB\', \'andreas\',\'brutsaertgarratt\',\'revisedbrutsaertgarratt\',\'simplebrutsaert\',\'mondonredelsperger\',\'coare3.0\'.')

   return zs
################################################################################
def U(z, ustar, z0, psi=0) :
   """
   This funcion computes the wind speed (in m/s) at height z (in m) as a fonction of the friction velocity in ustar in (m/s), the aerodynamic roughness length z0 (in m), and the additive stability correction psi.

   Author : Virginie Guemas - October 2020 
   Modified : January 2021  - Sebastien Blein - Uncertainty propagation
   """

   z0 = np.where(unp.nominal_values(z0)<=0,np.nan,z0)
   z0 = np.where(unp.nominal_values(z0)==np.inf,np.nan,z0)
   z = np.where(unp.nominal_values(z)<unp.nominal_values(z0),np.nan,z)
   u = ustar/k * (unp.log(z/z0) - psi)

   return u
################################################################################
def UG(method=None, u=None, h=None, Q0v=None, thetav=None, Q0=None, E0=None, T=None, beta=1.25, zeta=None):
   """
   This function computes the corrected wind speed to account for gustiness. It needs :
   - the stability parameter zeta to verify the stability conditions to apply a particular gustiness correction
   are verified.
   With option method = 'godfreybeljaars', (Godfrey and Beljaars, 1991) 
   the function needs : 
   - the horizontal wind speed u (in m/s),
   - the correction factor beta = 1.25 as in COARE2.5 or beta = 1 as in Zeng et al (1998) or beta = 1.2 as in Beljaars (1995)
   - the surface virtual temperature flux Q0v (K.m.s-1)
   - the convective boundary layer height h (in m), typically h=600m,
   - the virtual potential temperature thetav (in Kelvin).
   With option method = 'fairall' (approximation of godfreyBeljaars used in Fairall et al (1996, 2003),
   the function needs :
   - the horizontal wind speed u (in m/s),
   - the correction factor beta = 1.25 as in COARE2.5 or beta = 1 as in Zeng et al (1998) or beta = 1.2 as in Beljaars (1995)
   - the temperature T (in Kelvin),
   - the potential temperature flux Q0 (K.m.s-1),
   - the humidity flux E0 (m.s-1),
   - the convective boundary layer height h (in m), typically h=600m.
   With option method = 'jordan' (Jordan et al, 1999) used in Andreas et al (2010),
   the function needs:
   - the horizontal wind speed u (in m/s).

   Author : Virginie Guemas   - December 2020
   Modified : January 2021 - Virginie Guemas - Option fairall which approximates godfreybeljaars
              January 2021 - Sebastien Blein - Uncertainty propagation
                                               Verify whether stability allows gustiness correction
   """

   if zeta is None:
     sys.exit('Please provide zeta so that the function can verify whether the method selected is suitable for the stability')
   
   if method == 'godfreybeljaars':
     if u is not None and thetav is not None and h is not None and Q0v is not None:
       # Both 'godfreybeljaars' and 'fairall' methods to estimate the wind gustiness
       # correction are valid only in unstable cases.
       # Both conditions on zeta and Q0v are applied in case zeta is computed using
       # some other variables than Q0v (thetastar for example).
       # A Q0v_pos array has to be created as the np.where function calculates values everywhere. Otherwise the uncertainties module would rise math errors for the 1/3 power with negative Q0v values.
       Q0v_pos = np.where(Q0v>0, Q0v, np.nan)
       ug = np.where((zeta<0)&(Q0v>0), beta * (g/thetav*h*Q0v_pos)**(1/3), 0)
       ucor = unp.sqrt(u**2+ug**2)
     else:
       sys.exit('With option method = \'godfreybeljaars\', input u, thetav, h and Q0v are required.')

   elif method == 'fairall':
     if u is not None and T is not None and h is not None and Q0 is not None  and E0 is not None:
       Q0v = Q0+0.61*T*E0
       Q0v_pos = np.where(Q0v>0, Q0v, np.nan)
       ug = np.where((zeta<0)&(Q0v>0), beta * (g/T*h*Q0v_pos)**(1/3), 0.)
       ucor = unp.sqrt(u**2+ug**2)
     else:
       sys.exit('With option method = \'fairall\', input u, T, h, Q0 and E0 are required.')

   elif method == 'jordan':
     if u is not None:
       ucor = np.where(zeta>0, u + 0.5/unp.cosh(u), u)
     else:
       sys.exit('With option method = \'jordan\', input u is required.')

   else:
     sys.exit('Valid methods are \'godfreybeljaars\', \'fairall\' and \'jordan\'.')

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
   Modified : January 2021  - Sebastien Blein - Uncertainty propagation
   """

   zs = np.where(unp.nominal_values(zs)<=0,np.nan,zs)
   zs = np.where(unp.nominal_values(zs)==np.inf,np.nan,zs)
   z = np.where(unp.nominal_values(z)<unp.nominal_values(zs),np.nan,z)
   s = s0 + sstar/k * (unp.log(z/zs) - psi)

   return s
################################################################################
def LMO(ustar, thetav, thetavstar=None, Q0v=None) :
   """
   This function computes the Monin-Obukhov length (in meters) either as a function of the virtual temperature scaling parameter thetavstar (in Kelvin) or as a function of the surface virtual temperature flux (K.m.s-1). It also requires the friction velocity (in m.s-1) and the layer-average virtual potential temperature (in Kelvin). 

   Author : Virginie Guemas - October 2020 
   Modified : January 2021  - Sebastien Blein - Uncertainty propagation
   """
   meteolib.check_T(thetav)

   if Q0v is None and thetavstar is None:
     sys.exit('At least one of thetavstar or Q0v should be provided')

   beta = g/thetav 

   if Q0v is not None:
     Q0v = np.where(unp.nominal_values(Q0v)==0,np.nan,Q0v)
     Lmo = -(ustar**3)/(k*beta*Q0v)
     if thetavstar is not None:
       Qvbis = - ustar*thetavstar
       if np.nanmax(np.abs((Qvbis-Q0v)/Q0v)) > 0.001 :
         sys.exit('Q0v and -ustar*thetavstar differ')
   else:
     thetavstar = np.where(unp.nominal_values(thetavstar)==0,np.nan,thetavstar)
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
   Modified : January 2021  - Sebastien Blein - Uncertainty propagation
   """
   meteolib.check_T(T)
   
   if thetastar is None and Q0 is None:
     sys.exit('At least one of thetastar or Q0 should be provided')
   
   if qstar is None and E0 is None:
     sys.exit('At least one of qstar or E0 should be provided')

   beta = g/T

   if Q0 is not None and E0 is not None:
     Q0v = np.where(unp.nominal_values(Q0+0.61*T*E0)==0,np.nan,Q0+0.61*T*E0)
     Lmo = -(ustar**3)/(k*beta*Q0v)
   elif thetastar is not None and qstar is not None:
     thetavstar = np.where(unp.nominal_values(thetastar+0.61*T*qstar)==0,np.nan,thetastar+0.61*T*qstar)
     Lmo = (ustar**2)/(k*beta*thetavstar)
   elif Q0 is not None and qstar is not None:
     Q0v = np.where(unp.nominal_values(Q0+0.61*T*(-ustar*qstar))==0,np.nan,Q0+0.61*T*(-ustar*qstar))
     Lmo = -(ustar**3)/(k*beta*Q0v)
   else:
     Q0v = np.where(unp.nominal_values((-ustar*thetastar)+0.61*T*E0)==0,np.nan,(-ustar*thetastar)+0.61*T*E0)
     Lmo = -(ustar**3)/(k*beta*Q0v)

   return Lmo
################################################################################
def RB(thetav, Dthetav, u, v, z) :
   """
   This function computes the Bulk Richardson number as a function of the vitual potential temperature thetav (in Kelvin) and wind speed horizontal components u and v (in m/s) at height z (in m) and the difference in virtual potential temperature between the height z and the surface Dthetav (in Kelvin).

   Author : Virginie Guemas - October 2020 
   """
   
   meteolib.check_T(thetav)
 
   wspd2 = np.where(unp.nominal_values(u**2+v**2)==0,np.nan,u**2+v**2)
   Rb = g/thetav * Dthetav*z/wspd2

   return Rb
################################################################################
def Rstar(ustar, z0, T) :
   """
   This function computes the Roughness Reynolds number as a function of the friction velocity ustar (in m/s), the aerodynamic roughness length (in m) and the temperature (in Kelvin).

   Author : Virginie Guemas - October 2020 
   """

   z0 = np.where(unp.nominal_values(z0)<=0,np.nan,z0)
   z0 = np.where(unp.nominal_values(z0)==np.inf,np.nan,z0)
   z0 = np.where(unp.nominal_values(z0)>10,np.nan,z0)
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
def WEBB(E0,Q0,q,T) :
   """
   The total surface humidity flux contains a turbulent component and a component from the mean vertical wind speed. This mean vertical wind speed can be approximated as in Webb et al (1980). This function estimates the total surface humidity flux including the Webb effect as a function of :
   - the turbulent humidity flux E0 (in m.s-1),
   - the turbulent temperature flux Q0 (in K.m.s-1),
   - the specific humidity q (in kg.kg-1),
   - the temperature T (in Kelvin).

   Author : Virginie Guemas - January 2021
   """
   meteolib.check_q(q)
   meteolib.check_T(T)

   w = 1.61*E0 + (1.61*q)*Q0/T
   E0cor = EO + w*q

   return E0cor
################################################################################
def HSR(R, T, Ts, deltaq, P, lda=1, mu=1) :
   """
   Precipitation transfer heat to the ocean which should be accounted for in the sensible
   heat flux term. This function estimates the correction which should be added to 
   sensible heat flux according to Gosnell et al (1995) as used in Fairall et al (1996).
   This function needs :
   - the rain rate R (kg.s-1.m-2),
   - the rain (supposed equal to atmospheric) temperature T (in Kelvin),
   - the surface temperature Ts (in Kelvin),
   - the difference in specific humidity between atmosphere and surface deltaq (in kg.kg-1),
   - the pressure (in hPa),
   - the lambda parameter lda set to 1 for COARE, Rv/Ra for ECUME and 1 for ECUME6,
   - the mu parameter set to 1 for COARE, dh/dv for ECUME and ECUME6.

   Author : Virginie Guemas - January 2021
   """
   
   deltaT = T-Ts  # Difference between atmosphere/rain temperature and surface one

   Lv = meteolib.LV(T) # Latent heat of vaporization of rain

   qsat = meteolib.ES(T)*Ra/(P*Rv) # Saturation specific humidity

   dq = lda * Lv*qsat/(Rv*T**2) # Clausius-Clapeyron relation

   alpha = (1 + Lv/cp * dv/dh* dq) # wet-bulb factor

   deltaq = np.where(unp.nominal_values(deltaq)==0,np.nan,deltaq)
   B = mu*(cp/Lv)*deltaT/deltaq # Bowen ratio
           
   B = np.where(unp.nominal_values(B)==0,np.nan,B)
   Hsrain = - R*cpw*alpha*(1+1/B)*deltaT
  
   return Hsrain
################################################################################
def TAUR(u, R, gamma=0.85) :
   """
   Rainfall tends to increase surface drag. This functions computes the correction
   to be added to surface wind stress according to Fairall et al (1996) as a 
   function of :
   - the wind speed u (in m.s-1)
   - the precipitation rate (in kg.m-2.s-1),
   - the gamma parameter set to 0.85 in COARE and ECUME6, 1 in ECUME.

   Author : Virginie Guemas - January 2021
   """

   Taurain = gamma*R*u 

   return Taurain
################################################################################
def ZETA(z,Lmo) :
   """
   This function computes the Monin-Obukhov stability parameter zeta as a
   function of the height z and the Monin-Obukhov length Lmo.

   Author : Sebastien Blein - January 2021
   """

   Lmo = np.where(unp.nominal_values(Lmo)==0,np.nan,Lmo)
   zeta = z/Lmo

   return zeta
################################################################################
def PSI(zeta, gamma=5, stab=None, unstab=None) :
   """
   This function computes a stability correction as a function of Monin-Obukhov length. It takes four arguments:
   - zeta = z (height) / Lmo (Monin-Obukhov length)
   - stab = formulation for stable regimes : 'dyer-hicks'  -- Dyer and Hicks (1970)
                                             'lettau'      -- Lettau (1979) 
                                             'holtslag-bruin' -- Holtslag and de Bruin (1988)
                                             'beljaars-holtslag' -- Beljaars and Holtslag (1991)
                                             'grachev'     -- Grachev (2007)
   - unstab = formulation for unstable regimes : 'businger-dyer' -- Paulson (1970)
                                                 'fairall1996'   -- Fairall et al (1996)
                                                 'grachev2000'   -- Grachev et al (2000)
   - the gamma factor for the 'dyer-hicks' option which can range between about 5 
   (Webb, 1970; Businger et al., 1971; Dyer, 1974; Large and Pond,1981) and about 7 (Wieringa, 1980;
   Large and Pond, 1982; Högström, 1988) - 4.7 seems to be used in COARE2.5

   Author : Virginie Guemas - September 2020
   Modified : December 2020 - Sebastien Blein - correct Beljaars and Holtslag 1991
              December 2020 - Virginie Guemas - add kansas, fairall and holtslag-bruin
              January 2021  - Virginie Guemas - include factor gamma to tune the dyer-hicks option. Ex: 4.7 for COARE2.5
              January 2021  - Sebastien Blein - Uncertainty propagation
   """
   np.seterr(invalid='ignore')

   if stab is None or unstab is None:
       sys.exit('Stability correction type has to be specified (e.g.: stab=\'beljaars-holtslag\' and unstab=\'businger-dyer\')')

   ###############################
   # zeta discrimination between stab and unstab in order to avoid error with unp function when math domain is not Real.
   zeta_stab = np.where(zeta>0,zeta,np.nan)
   zeta_unstab = np.where(zeta<0,zeta,np.nan)

   ###############################
   if unstab == 'businger-dyer':
     phiM = (1 - 16*zeta_unstab)**(-0.25)
     phiH = (1 - 16*zeta_unstab)**(-0.5)

     psiM = np.where(zeta<0, 2*unp.log((1+phiM**(-1))/2) + unp.log((1+phiM**(-2))/2) - 2*unp.arctan(phiM**(-1)) + np.pi/2, 0.)
     psiH = np.where(zeta<0, 2*unp.log((1+phiH**(-1))/2), 0.)
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
     y = (1 - 12.87*zeta_unstab)**(1/3)

     psi = np.where(zeta<0, 1.5*unp.log((y**2+y+1)/3) - unp.sqrt(3)*unp.arctan((2*y+1)/unp.sqrt(3)) + np.pi/unp.sqrt(3), 0.)

     psiM = np.where(zeta<0, 1/(1+zeta_unstab**2)*PSI(zeta_unstab, stab = stab, unstab = 'businger-dyer')[0] + zeta_unstab**2/(1+zeta_unstab**2)*psi, 0.)
     psiH = np.where(zeta<0, 1/(1+zeta_unstab**2)*PSI(zeta_unstab, stab = stab, unstab = 'businger-dyer')[1] + zeta_unstab**2/(1+zeta_unstab**2)*psi, 0.)
   ################################
   elif unstab == 'grachev2000':
     a={'m':10.15,'h':34.15}
     psi={}
     for s in ('m','h'):
       y = (1 - a[s]*zeta_unstab)**(1/3)

       psi[s] = np.where(zeta<0, 1.5*unp.log((y**2+y+1)/3) - unp.sqrt(3)*unp.arctan((2*y+1)/unp.sqrt(3)) + np.pi/unp.sqrt(3), 0.)

     psiM = np.where(zeta<0, 1/(1+zeta_unstab**2)*PSI(zeta_unstab, stab = stab, unstab = 'businger-dyer')[0] + zeta_unstab**2/(1+zeta_unstab**2)*psi['m'], 0.)
     psiH = np.where(zeta<0, 1/(1+zeta_unstab**2)*PSI(zeta_unstab, stab = stab, unstab = 'businger-dyer')[1] + zeta_unstab**2/(1+zeta_unstab**2)*psi['h'], 0.)
   ################################  
   elif  unstab == 'beljaars-holtslag':
     a,b,c,d = 1.,0.667,5.,0.35

     psiM = np.where(zeta<0, -(a * zeta_unstab + b*(zeta_unstab - c/d) * unp.exp(-d * zeta_unstab) + b*c/d), 0.)
     psiH = np.where(zeta<0, -(a * zeta_unstab + b*(zeta_unstab - c/d) * unp.exp(-d * zeta_unstab) + b*c/d), 0.)
     # Beljaars and Holtslag do not propose an option for unstable cases but only for stable cases. This formulation is here
     # only to reproduce Elvidge et al (2016).
   else:
     sys.exit('This option for unstable cases is not coded yet.')
   ################################
   ################################
   if stab == 'dyer-hicks':
     psiM = np.where (zeta>0, -gamma*zeta_stab, psiM)
     psiH = np.where (zeta>0, -gamma*zeta_stab, psiM)
   ################################
   elif stab == 'lettau':
     x = 1 + 4.5*zeta_stab

     psiM = np.where (zeta>0, unp.log((x**0.5+1)/2) + 2*unp.log((x**0.25+1)/2) -2*unp.arctan(x**0.25) + np.pi/2 + 4/3*(1-x**0.75), psiM)
     psiH = np.where (zeta>0, 2*unp.log((x**0.25+1)/2) -2*(x**1.5/3 + x**0.5 - 4/3), psiH)
   ################################
   elif stab == 'holtslag-bruin':
     psi = -3.75/0.35 - 0.7*zeta_stab + 3.75/0.35*unp.exp(-0.35*zeta_stab) -0.75*zeta_stab*unp.exp(-0.35*zeta_stab)

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

     psiM = np.where (zeta>0, -(a * zeta_stab + b*(zeta_stab - c/d) * unp.exp(-d * zeta_stab) + b*c/d), psiM)
     psiH = np.where (zeta>0, -((1+2.*a*zeta_stab/3)**(3./2) + b*(zeta_stab-c/d)*unp.exp(-d*zeta_stab) + b*c/d - 1), psiH)
   ################################
   elif stab == 'grachev':
     x = (zeta_stab+1)**(1/3)

     psiM = np.where (zeta>0, -19.5*(x-1) + 3.25*0.3**(1/3)*(2*unp.log((x+0.3**(1/3))/(1+0.3**(1/3))) - unp.log((x**2-0.3**(1/3)*x+0.3**(2/3))/(1-0.3**(1/3)+0.3**(2/3))) +2*unp.sqrt(3)*(unp.arctan((2*x-0.3**(1/3))/(unp.sqrt(3)*0.3**(1/3))) -unp.arctan((2-0.3**(1/3))/(unp.sqrt(3)*0.3**(1/3))))) , psiM)
     psiH = np.where (zeta>0, -2.5*unp.log(1+3*zeta_stab+zeta_stab**2) +5/(2*unp.sqrt(5))*(unp.log((2*zeta_stab+3-unp.sqrt(5))/(2*zeta_stab+3+unp.sqrt(5))) -unp.log((3-unp.sqrt(5))/(3+unp.sqrt(5)))), psiH)
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
   Modified : January 2021  - Sebastien Blein - Uncertainty propagation
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
     
     z0 = np.where(unp.nominal_values(z0)==0,np.nan,z0)
     c2 = c1*alpha*CDN*(z/z0)**0.5

     tmp = np.where(unp.nominal_values(1+c1/2*Rb)==0,np.nan,1+c1/2*Rb)
     fstab = 1/tmp**2

   elif author == 'LupkesGryanik' :
     if var == 'momentum' :
       c1 = 10
       alpha = 7.5
     elif var == 'heat' or var == 'moisture' :
       c1 = 15
       alpha = 5
     else:
       sys.exit('var can be momentum, heat or moisture only')
     
     z0 = np.where(unp.nominal_values(z0)==0,np.nan,z0)
     c2 = c1*alpha*CDN*(z/z0+1)**0.5

     Rb = np.where(unp.nominal_values(Rb)==-1,np.nan,Rb)
     tmp = np.where(unp.nominal_values(1+10*Rb/unp.sqrt(Rb+1))==0,np.nan,1+10*Rb/unp.sqrt(Rb+1))
     fstab = 1/tmp

   else:
     sys.exit('Only Louis and LupkesGryanik version are coded for now')

   dunstab = np.where(unp.nominal_values(1+c2*np.abs(Rb)**0.5)==0,np.nan,1+c2*np.abs(Rb)**0.5)
   f = np.where(Rb < 0, 1 - (c1*Rb)/dunstab, fstab)

   return f
################################################################################
def BULK(z, u, theta, thetas, q, qs, T, method='coare2.5') :
   """
   This function applies one of the COARE or ECUME algorithms to estimate bulk
   turbulent fluxes of velocity, heat and humidity as well as the associated transfer 
   coefficients. It needs :
   - the atmospheric measurement height z (in m),
   - the wind speed at height z (in m.s-1),
   - the potential temperature at height z (in Kelvin),
   - the potential temperature at the surface (in Kelvin),
   - the specific humidity at height z (in kg.kg-1),
   - the specific humidity at the surface (in kg.kg-1)
   - the layer-averaged temperature T (in Kelvin).

   Warning: The Webb correction for latent heat flux, the precipitation correction for
   sensible heat flux, the warm-layer and cool-skin corrections for surface temperature
   are not included.

   Author : Virginie Guemas - January 2021  
   Modified : January 2021  - Sebastien Blein - Uncertainty propagation
   """
   deltatheta = theta - thetas
   deltaq = q - qs

   # First guess of wind gustiness correction and neutral bulk transfer coefficients
   Ucor = unp.sqrt(u**2+0.5**2)
   cdn = 0.0015
   chn = 0.001
   cen = 0.001
   # I could not find the first guess of neutral bulk transfer coefficients in Fairall et al (1996, 2003).
   # Those choices are arbitrary.

   if method == 'coare2.5':
     # First guess of bulk transfer coefficients is neutral bulk transfer coefficients
     cd = cdn
     ch = chn
     ce = cen

   elif method == 'coare3.0':
     # First guess based on Grachev and Fairall (1997) estimate of stability
     Rb = RB(thetav = T, Dthetav = meteolib.Thetav(theta,q) - meteolib.Thetav(thetas,qs), u = u, v = 0, z = z) 
     # Fairall et al 2003 use T instead of thetav in the estimate of beta = g/thetav
     Rb = np.where(unp.nominal_values(Rb)==4.5,np.nan,Rb)
     zeta = 10*Rb/(1+Rb/(-4.5))
     (psiM, psiH) = PSI(zeta, gamma = 4.7, stab='beljaars-holtslag', unstab='grachev2000')
     cd = CD (CDN = cdn, psi = psiM)
     ch = CS (CDN = cdn, CSN = chn, psiM = psiM, psiH = psiH)
     ce = CS (CDN = cdn, CSN = cen, psiM = psiM, psiH = psiH)

   else:
      sys.exit('Only coare2.5 and coare3.0 are coded for now')

   # First guess of bulk turbulent fluxes
   ustar = unp.sqrt(cd * Ucor**2)
   thetastar = ch/unp.sqrt(cd) * deltatheta
   qstar = ce/unp.sqrt(cd) * deltaq

   # A few choices of methods used in the various COARE algorithms
   zsmod = {'coare2.5':'LKB','coare3.0':'coare3.0'}
   psiunstab = {'coare2.5':'fairall1996','coare3.0':'grachev2000'}
   psistab = {'coare2.5':'dyer-hicks','coare3.0':'beljaars-holtslag'}
   ncount = {'coare2.5':20,'coare3.0':10}

   # Iterations
   count = 0 
   while count < ncount[method]:
     # Monin-Obukhov length depends on turbulent fluxes
     lmo = LMOapprox (ustar = ustar, T = T, thetastar = thetastar, qstar = qstar)
     zeta = ZETA(z, lmo)
     # Aerodynamic roughness depends on friction velocity
     z0 = Z0(method = method, alpha=0.011, u = u, ustar = ustar, T = T)
     # Roughness Reynolds number depends on friction velocity and aerodynamic roughness
     rstar = Rstar(ustar = ustar, z0 = z0, T = T)
     # Scalar roughness depend on Roughness Reynolds number and friction velocity
     z0T = ZS(method = zsmod[method], T = T, rstar = rstar, ustar = ustar, s='T')
     z0q = ZS(method = zsmod[method], T = T, rstar = rstar, ustar = ustar, s='Q')
     # Neutral transfer coefficients depend on roughness lengths
     Cdn = CDN (z0 = z0, z = z)
     Chn = CSN (zs = z0T, z0 = z0, z = z) 
     Cen = CSN (zs = z0q, z0 = z0, z = z) 
     # Stability correction depends on Monin-Obukov length
     (psiM, psiH) = PSI(zeta, gamma = 4.7, stab = psistab[method], unstab = psiunstab[method])
                  # I am unsure about the 4.7 factor which is not stated clearly in Fairall et al 1996
     # Transfer coefficients depend on neutral transfer coefficients and stability corrections
     Cd = CD (CDN = Cdn, psi = psiM)
     Ch = CS (CDN = Cdn, CSN = Chn, psiM = psiM, psiH = psiH)
     Ce = CS (CDN = Cdn, CSN = Cen, psiM = psiM, psiH = psiH)
     # Updated estimates of turbulent fluxes
     ustar = unp.sqrt(Cd * Ucor**2)
     thetastar = Ch/unp.sqrt(Cd) * deltatheta
     qstar = Ce/unp.sqrt(Cd) * deltaq
     # Update corrected wind speed for gustiness
     Ucor = np.where(zeta<0, UG(method='fairall', u = u, h=600, T = T, E0 = -ustar*qstar, Q0 = -ustar*thetastar, beta = 1.25, zeta = zeta), u)
     # Cool-skin is not implemented
     count = count + 1
   # Webb correction, precipitation correction and warm-layer corrections should be included when getting out of the loop 
   # I prefer leaving them out of the function for now.


   return {'ustar':ustar, 'qstar':qstar, 'thetastar':thetastar, 'CDN':Cdn, 'CHN':Chn, 'CEN':Cen}
################################################################################

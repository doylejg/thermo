#! /usr/bin/env python
"""thermo.py

   Copyright (C) 2009 Jonathan Doyle
   All rights reserved.
   
   Jonathan G. Doyle <doylejg@dal.ca>
   Department of Physics and Atmospheric Science,
   Dalhousie University, Halifax, Nova Scotia, Canada, B3H 3J5

NOTICE

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

REFERENCE

   Saturation vapour pressure forumations are from:
    1) http://cires.colorado.edu/~voemel/vp.html
    2) Bohren & Albrecht, Atmospheric Thermodynamics, 1998

   Much of this code is taken from thermo code written by Thomas J. Duck
   <tom.duck@dal.ca>
   
DESCRIPTION

   
"""
import numpy

# Physical constants
Rd = 287.04  # Dry gas constant (J/kg/K)
Rv = 461.50  # Gas constant for water vapour (J/kg/K)
cpd = 1005.  # Specific heat capacity for dry gas at 0C (J/kg/K)
cw = 4218.   # Specific heat capacity for water at 0C (J/kg/K)
cpv = 1850.  # Specific heat capacity for water vapour at 0C (J/kg/K)
es0 = 6.11   # Saturation vapour pressure (hPa) at 0 C
lv0 = 2.501e6  # Enthalpy of vaporization (J/kg) at 0 C
eps = Rd/Rv
Na = 6.02214199e23 # Avogadro's number (/mol)
Mair = 0.02894     # Molecular weight of air (kg/mol)
kb = 1.3806504e-23 # Boltzmans constant (J/K)

def get_number_density(mass_density):
    """Converts mass density (kg/m3) to number density (/m3)."""
    return mass_density * Na/Mair

def get_mass_density(number_density):
    """Converts number density (/m3) to mass density (kg/m3)."""
    return number_density * Mair/Na
    
def _es_simple(T):
    """Saturation vapour pressure (hPa) for temperature T (K)"""
    return es0*numpy.exp(19.83-5417./T)

def _es_murphy_and_koop(T):
    """Murpy and Koop 2005 Vapour pressure (hPa) for temperautre T (K)"""
    e = numpy.exp(54.842763 - 6763.22 / T - 4.21 * numpy.log(T) + 0.000367 * T \
               + numpy.tanh(0.0415 * (T - 218.8)) \
               * (53.878 - 1331.22 / T - 9.44523 * numpy.log(T) + 0.014025 * T))
    return e/100.

def _es_bohren_and_albrecht(T):
    """Returns the saturation vapour pressure (hPa) for temperature T (K)."""
    # Eq. 5.67 in Bohren & Albrecht.
    return es0 * numpy.exp( 6808.*(1./273.15-1./T) - 5.09*numpy.log(T/273.15) )

def _es_WMO_2000(T):
    """WMO 2000 vapour pressure over liquid water from -100 -> 0 C
    WMO based its recommendation on a paper by Goff (1957), which is shown
    here. The recommendation published by WMO (1988) has several
    typographical errors and cannot be used. A corrigendum (WMO, 2000) shows
    the term +0.42873 10-3 (10(-4.76955*(1-273.16/T)) - 1) in the fourth line
    compared to the original publication by Goff (1957). Note the different
    sign of the exponent. The earlier 1984 edition shows the correct formula.
    """
    return 10.**( 10.79574 * (1-273.16/T) \
           - 5.02800 * numpy.log10(T/273.16) \
           + 1.50475e-4 * (1 - 10**(-8.2969*(T/273.16-1))) \
           + 0.42873e-3 * (10**(-4.76955*(1-273.16/T)) - 1) \
           + 0.78614 )

def _es_hyland_and_wexler(T):
    """Hyland and Wexler, 1983 saturation vapour pressure (hPa) for temperautre
    T (K)"""
    return numpy.exp(-0.58002206e4 / T
              + 0.13914993e1 \
              - 0.48640239e-1 * T\
              + 0.41764768e-4 * T**2\
              - 0.14452093e-7 * T**3\
              + 0.65459673e1 *numpy.log(T))/100.

def _es_goff_and_gratch(T):
    """Smithsonian Tables, 1984, after Goff and Gratch, 1946"""
    return 10**(-7.90298 * (373.16/T-1) \
                + 5.02808 * numpy.log10(373.16/T) \
                - 1.3816e-7 * (10**(11.344 * (1-T/373.16))-1)\
                + 8.1328e-3 * (10**(-3.49149 * (373.16/T-1))-1)\
                + numpy.log10(1013.246) )

def _es_goff(T):
    """Goff, 1957"""
    return 10.**( 10.79574 * (1-273.16/T) \
                  - 5.02800 * numpy.log10(T/273.16) \
                  + 1.50475e-4 * (1 - 10**(-8.2969*(T/273.16-1))) \
                  + 0.42873e-3 * (10**(+4.76955*(1-273.16/T)) - 1) \
                  + 0.78614 )

def vapour_pressure(T,form='murphykoop'):
    """
    Return the (saturation) vapour pressure (hPa) for temperature T (K)

    Different forms of the equation:
    WMO: WMO 2000
    bohrenalbrecht: Eq. 5.67 in Bohren & Albrecht
    murphykoop: Murpy and Koop 2005
    hylandwexler: Hyland and Wexler 1983
    simple: Eq. 5.68 from Bohren and Albrecht
    goff: Goff, 1957
    goffgratch: Smithsonian Tables, 1984, after Goff and Gratch, 1946
    """
    if form=='WMO':
        return _es_WMO_2000(T)
    elif form=='bohrenalbrecht':
        return _es_bohren_and_albrecht(T)
    elif form=='murphykoop':
        return _es_murphy_and_koop(T)
    elif form=='hylandwexler':
        return _es_hyland_and_wexler(T)
    elif form=='goff':
        return _es_goff(T)
    elif form=='goffgratch':
        return _es_goff_and_gratch(T)
    elif form=='simple':
        return _es_simple(T)

# Wrap vapour_pressure(...)
es = vapour_pressure

def mixing_ratio(p,T):
    """
    Watervapour mixing ratio (actual or saturated)
    p:  Pressure (hPa)
    T: Tempearture (K) (Dewpoint temperature for real mixing ratio)
    
    Returns: Saturated or actual water vapour mixing ratio (g/kg)
    """
    e = vapour_pressure(T)
    return eps * e / (p-e)

# Wrap mixing_ratio(...)
w = property(mixing_ratio,None,None,"Watervapour mixing ratio (g/kg)")
get_w = property(mixing_ratio,None,None,"Watervapour mixing ratio (g/kg)")

def actual_w(p,Td):
    """
    Actual watervapour mixing ratio
    p: Pressure (hPa)
    Td: Dewpoint Temperature (K)
    
    Returns: Actual watervapour mixing ratio (g/kg)
    """
    return mixing_ratio(p,Td)

def relative_humidity(p,T,Td):
    """
    Get RH
    p: Pressure (hPa)
    T: Temperature (K)
    Td: Dewpoint Temperature (K)    
    
    Returns: Relative humidity (%)
    """
    e = vapour_pressure(p,Td)
    es = vapour_pressure(p,T)
    return e / es * (p-es) / (p-e) * 100.
    
def convert_to_mixing_ratio(RH,p,T):
    """
    Get RH from w
    RH: Relative humidity (%)
    p: Pressure (hPa)
    T: Tempearture
    
    Returns: Actual mixing ratio (g/kg)
    """
    ws = get_w(p,T)
    return RH * ws / 100.

def convert_to_relative_humidity(w,T,p):
    """
    Get RH from w
    w: Actual mixing (g/kg)
    p: Pressure (hPa)
    T: Tempearture (K)
    
    Returns: Relative humidity (%)
    """
    ws = get_w(p,T)
    return w / ws * 100.

def _nv_T_Td(T,Td):
    """Water vapour number density as a function of T and Td (K)"""
    return es(Td)*100/kb/T

def _nv_w_p_T(w,p,T):
    """Water vapour number density as a function of mixing ratio w (g/kg),
    pressure p (hPa) and temperature T (K)"""
    _w = w/1000. # g/kg -> g/g
    _p = p*100. # hPa -> Pa
    return _w*_p/(_w+eps)/kb/T

def _nv_w_dens(w,dens):
    """Water vapour number density as a function of mixing ratio w (g/kg),
    density dens (/m3)"""
    _w = w/1000. # g/kg -> g/g
    return _w*dens/(_w+eps)*Rv/kb

def _nv_r_p_T_Td(r,p,T,Td):
    """Water vapour number density as a function of relative humidity r
    (fractional percent), pressure p (hPa) tempearture T (K) and dewpoint
    temperature T (K)"""
    if r>1:
        print("Warning relative humidity is greater than 100% (%.3f)"%r)
        exit()
    _p = p*100. # hPa -> Pa
    return r*_p*es(Td)/(es(Td)*(r-1)+_p)/kb/T

def watervapour_number_density(T=None,Td=None,dens=None,p=None,w=None,r=None,
                               RH=None):
    """Calculate the number density of water given different input
    thermodynamic fields."""
    if T!=None and Td!=None:
        return _nv_T_Td(T,Td)
    elif w!=None and p!=None and T!=None:
        return _nv_w_p_T(w,p,T)
    elif w!=None and dens!=None:
        return _nv_w_dens(w,dens)
    elif (r!=None or RH!=None) and p!=None and Td!=None and T!=None:
        if RH!=None:
            r = RH/100.
        return _nv_r_p_Td_T(r,p,Td,T)
    else:
        msg = "Not enough parameters given, you provided:\n"\
              +"T: %s, Td: %s, dens: %s, p: %s, \nw: %s, r: %s, RH: %s" % \
              (T,Td,dens,p,w,r,RH)
        raise ValueError(msg)
        
# Wrap watervapour number density
nv = watervapour_number_density

def plot_Fig5_4():
    """Plots Figure 5.4 from Bohren and Albrecht"""
    Tc = numpy.arange(-30,31,1)
    T = Tc + 273.15

    es_67 = _es_bohren_and_albrecht(T)
    es_68 = _es_simple(T)

    fig = pylab.figure()
    ax = fig.add_subplot(111)

    ax.plot(Tc,es_67,'k-',label=r'Eq. 5.67')
    ax.plot(Tc,es_68,'k--',label=r'Eq. 5.68')

    ax.set_ylabel(r'Saturation Vpour Pressure (mb)')
    ax.set_xlabel(r'Temperature ($^o$C)')
    
    int_fmt = matplotlib.ticker.FormatStrFormatter('% d')
    ax.xaxis.set_major_formatter(int_fmt)
    ax.yaxis.set_major_formatter(int_fmt)

    prop = matplotlib.font_manager.FontProperties(size='large')
    legend = ax.legend(loc='upper left',prop=prop)
    legend.draw_frame(False)
    

def plot_comparisons():
    """Plots the comparison from the different vapour-pressures"""
    Tc = numpy.arange(-100,41,1)
    T = Tc + 273.15

    es = {}
    es['Bohren and Albercht'] = _es_bohren_and_albrecht(T)
    es['Simple'] =              _es_simple(T)
    es['Murphy and Koop'] =     _es_murphy_and_koop(T)
    es['WMO 2000'] =            _es_WMO_2000(T)
    es['Hyland and Wexler'] =   _es_hyland_and_wexler(T)
    es['Goff'] =                _es_goff(T)
    es['Goff and Gratch'] =     _es_goff_and_gratch(T)
    
    fig = pylab.figure()
    ax = fig.add_subplot(111)

    ref_key = 'Murphy and Koop'
    es_ref = es[ref_key]

    for key in es.keys():
        diff = (es[key]-es_ref)/es_ref * 100
        ax.plot(Tc,diff,lw=2,label=r'%s'%key)

    ax.set_ylim(-25,25)
        
    ax.set_ylabel(r'Percent devaiation from %s'%ref_key)
    ax.set_xlabel(r'Temperature ($^o$C)')
    
    int_fmt = matplotlib.ticker.FormatStrFormatter('% d')
    ax.xaxis.set_major_formatter(int_fmt)
    ax.yaxis.set_major_formatter(int_fmt)

    prop = matplotlib.font_manager.FontProperties(size='large')
    legend = ax.legend(loc='upper right',prop=prop)
    legend.draw_frame(False)
    
        

#----------------------------------------#
# Below are taken from Tom's thermo code #
#----------------------------------------#

def dry_adiabat(T0,p):
    """Returns dry adiabat (K) at given pressures p for temperature T0 (K).

    Pressures should be ordered from high to low.
    """
    return T0*(p/p[0])**(Rd/cpd)


def moist_adiabat(T0,p,Tf):
    """Returns dry adiabat (K) at given pressures p for temperature T0 (K).
    
    Tf (K) is an estimate of the form for the moist adiabat at pressures p.  
    For low values of T0 a dry adiabat can be used.  For higher values of 
    T0, use the moist adiabat for a nearby (previously calculated) adiabat.

    Pressures p (hPa) should be ordered from high to low.
    """

    # All equations named below are from in Bohren & Albrecht.

    # Array for the moist adiabat (K)
    T = numpy.zeros(len(p),dtype=numpy.float)
    T[0] = T0

    # Determine the moise adiabat from the bottom up
    for i in range(1,len(p)):

        T[i] = T[i-1] + Tf[i]-Tf[i-1]  # Initial guess

        Tav = (T[i]+T[i-1])/2. # Layer average temperature
        pav = (p[i]+p[i-1])/2. # Layer average pressure (hPa)

        # Saturation vapour pressure (hPa) at the upper [i] and lower [i-1]
        # pressure levels (eq. 5.67)
        es_upper = vapour_pressure(T[i])
        es_lower = vapour_pressure(T[i-1])

        # Saturation mass mixing ratio  at the upper [i] and lower [i-1]
        # pressure levels, and the layer average (eq. 5.14)
        ws_upper = 0.622*es_upper/p[i] # (p[i]-es_upper)
        ws_lower = 0.622*es_lower/p[i-1] # (p[i-1]-es_lower)
        ws = (ws_upper+ws_lower)/2.

        # Layer average pseudoadiabatic specific heat capacity (J/kg/K)
        cp = cpd + ws * cw

        # Enthalpy of vaporization (J/kg) at the upper [i] and lower [i-1]
        # pressure levels, and the layer average (eq. 5.64)
        lv_upper = lv0 + (cpv-cw)*(T[i]-273.15)
        lv_lower = lv0 + (cpv-cw)*(T[i-1]-273.15)
        lv = (lv_upper+lv_lower)/2.

        # Integrate eq. 6.130.  To do this, d(lv*ws/T) needs to be expanded,
        # and substitutions must be made for dlv and dws in terms of dT and
        # dp.  The expansion of dws requires use of equation 5.14 and the
        # Clausius-Clapeyron equation 5.48.  The expansion of dlv only 
        # requires equation 5.64. In the end there should only be 
        # differentials dT and dp; temperature and pressure may then be 
        # integrated upward from the bottom boundary.
        dp = (p[i]-p[i-1])  # (hPa)
        T[i] = T[i-1] + (Rd/pav + lv*ws/(Tav*pav))*dp / \
            (cp/Tav + ws*(cpv-cw)/Tav + ws*lv**2/(Rv*Tav**3) - ws*lv/Tav**2)

    return T


def const_ws_isolpleth(ws,p):
    """Returns tems (K) at pres p (hPa) for saturation mixing ratio ws (g/kg).

    The pressures should be ordered from high to low
    """

    # We need to solve equation 6.147 in Bohren & Albrecht for the 
    # temperatures given the pressure values.  Find the zeros of
    # RHS - LHS = 0 using scipy.optimize.brentq.

    ws /= 1000.  # Change to kg/kg

    # The range of temperatures to consider
    T1 = -150. + 273.15
    T2 = 50. + 273.15

    def f(T,p):
        # The function to invert
        es = vapour_pressure(T)
        return  es + 0.622*es/ws - p
    
    T = numpy.zeros(len(p),dtype=numpy.float)
    for i,p_ in enumerate(p):
        T[i] = scipy.optimize.brentq(f,T1,T2,args=(p_))

    return T


def get_plcl(p,T,dewpoint):
    """Returns the origin and lifting condensation level pressures."""

    p0 = plcl = 0.
    Txmax = 0.

    n = numpy.searchsorted(-p,-500.)

    for i in range(n):

        # Get the dry adiabat and constant mixing ratio isopleth and
        # see where they cross
        x1 = dry_adiabat(T[i],p[i:])
        ws = mixing_ratio(p[i],dewpoint[i])
        x2 = const_ws_isolpleth(ws,p[i:])
        d = x2-x1 # Difference
        assert( ((d[1:]-d[:-1])>0.).all() )        
        # Crossing pressure of x1 and x2
        px = 10**numlib.lininterp2(d,numpy.log10(p[i:]),0.) 
        Tx = numlib.lininterp2(d,x1,0.) # Crossing temperature

        # Get the moist adiabat extending from crossing pressure 
        # and temperature
        j = numpy.searchsorted(-p,-px)  # Index for the crossing pressure
        pupper = numpy.hstack(([px],p[j:]))  # Pressures at and above lcl
        assert( ((pupper[:-1]-pupper[1:])>0.).all() )
        Tlast = dry_adiabat(Tx,pupper)

        for k in range(3):
            Tmoist_x = moist_adiabat(Tx,pupper,Tlast)
            Tlast = Tmoist_x


        # Choose the right-most moist adiabat
        if Txmax < Tmoist_x[-1]:
            Txmax = Tmoist_x[-1]
            p0 = p[i]
            plcl = px
                   
    return p0, plcl


#---------------------------------------------------------------------------


def plot_skewTlogp(title,p,z,T,dewpoint,pmin=None,pmax=None,
                   T0min=-50,T0max=50.):
    """Plots skew-T log-p diagram using matplotlib.

    p: pressures (hPa) array
    z: heights (m) array
    T: temperatures (C) array
    dewpoint: dew points (C) array
    pmin: The minimum pressure to plot (hPa)
    pmax: The maximum pressure to plot (hPa)
    T0min: The minimum surface temperature to plot (C)
    T0max: The maximum surface temperature to plot (C)

    Returns a handle to the matplotlib figure.
    """

    # Get the minimum and maximum pressures
    if pmin == None:
        pmin = p.min()
    if pmax == None:
        pmax = p.max()


    # Condition the data
    indices = numpy.where(numpy.logical_and(numpy.logical_and(
            numpy.logical_and(T<1000.,dewpoint<1000.),p>pmin),p<pmax))[0]
    p = p.take(indices)
    z = z.take(indices)
    T = T.take(indices)
    dewpoint = dewpoint.take(indices)
    for i in range(1,len(p)):
        if p[i]==p[i-1]:
            p[i] *= 0.999999

    fig = pylab.figure()

    # Get the lifting condensation level parameters
    p0,plcl = get_plcl(p,T+273.15,dewpoint+273.15)
    i0,ilcl = arrayutils.get_indices(-p,-p0,-plcl)
    zlcl = numlib.lininterp2(-numpy.log10(p),z,-numpy.log10(plcl))

    T0range = T0max-T0min # Surface temperature range (C)
    Tmax = 50.   # Maximum temperature (C)
    Tmin = -150. # Minimum temperature (C)


    # Skew function to skew temperatures
    def skew(z,xmin=T0min,xmax=T0max):
        return (z-z[0])/float(z[-1]-z[0])*(xmax-xmin)


    # Set up the plot
    xticks = [tem for tem in range(-40,50,10)]
    yticks = [10**logp for logp in numpy.arange(math.log10(p[0]),0.,-0.15)]

    # Pressure grid lines
    for y in yticks:
        pylab.semilogy(numpy.array([T0min,T0max]),numpy.array([y,y]),
                       color=rgbcolors.get_color('grey'))

    # Skew-T grid lines
    for T0 in numpy.arange(Tmin,Tmax,10.):
        Tc = T0 * numpy.ones(len(p))
        pylab.semilogy(Tc+skew(z),p,color=rgbcolors.get_color('grey'))

    def lininterp(x,y,xnew):
        """Linear interpolation of y = f(x) at xnew."""
        tck = scipy.interpolate.splrep(x,y,s=0,k=1)
        return scipy.interpolate.splev(xnew,tck)

        
    # Dry adiabats
    for T0 in numpy.arange(-40.,700.,10.):

        # Thin out the contours at high altitudes
        if T0 > 100 and (T0 % 15) != 0:
            continue

        Tdry = dry_adiabat(T0+273.15,p)-273.15
        pylab.semilogy(Tdry+skew(z), p, color=rgbcolors.get_color('green'))

        # Label the adiabats
        Tlabel = T0max-0.05*T0range
        T_ = (Tdry+skew(z))[::-1]
        T_.sort()
        if Tdry[0]>Tmax and T0%20==0:
            plabel = lininterp(T_, p[::-1], Tlabel)
            if plabel>p[-1]:
                pylab.text(Tlabel,plabel,'%.0f'%T0,
                           color=rgbcolors.get_color('green'),
                           backgroundcolor=rgbcolors.get_color('white'),
                           horizontalalignment='center',
                           verticalalignment='center',
                           size='small'
                           )

    # Dry adiabat to lifting condensation level
    plower = numpy.hstack((p[i0:ilcl],[plcl]))  # Pressures at and below lcl
    zlower = numpy.hstack((z[:ilcl],[zlcl,z[-1]]))  # Heights including zlcl
    Tdry_lcl = dry_adiabat(T[i0]+273.15,plower)-273.15
    pylab.semilogy(Tdry_lcl+skew(zlower)[i0:ilcl+1], plower,
                   color=rgbcolors.get_color('green'),linewidth=2)


    # Moist adiabats
    Tlast = dry_adiabat(-40.+273.15,p)
    for T0 in numpy.arange(-40.,350.,5.):
        Tmoist = moist_adiabat(T0+273.15,p,Tlast)-273.15
        pylab.semilogy(Tmoist+skew(z), p, linestyle='--',
                       color=rgbcolors.get_color('blue'))
        Tlast = Tmoist

    # Moist adiabat from lifting condensation level
    pupper = numpy.hstack(([plcl],p[ilcl:]))  # Pressures at and above lcl
    zupper = numpy.hstack(([z[0],zlcl],z[ilcl:]))  # Heights including zlcl
    Tlast = dry_adiabat(Tdry_lcl[-1]+273.15,pupper)
    for k in range(3):
        Tmoist_lcl = moist_adiabat(Tdry_lcl[-1]+273.15,pupper,Tlast)
        Tlast = Tmoist_lcl
    Tmoist_lcl -= 273.15
    pylab.semilogy(Tmoist_lcl+skew(zupper)[1:], pupper, linewidth=2,
                   color=rgbcolors.get_color('blue'))


    # Constant ws isopleths
    for ws in [3.e-7,1.e-6,3.e-6,1.e-5,3.e-5,0.0001,0.0003,0.001,0.003,0.01,
               0.03,0.1,0.3,1.,2.5,5.,10.,15.,25.]:
             
        Tws = const_ws_isolpleth(ws,p)-273.15
        pylab.semilogy(Tws+skew(z), p, linestyle='--',
                       color=rgbcolors.get_color('red'))

        # Label the isopleths
        if ws < 0.0001:
            continue
        for Tlabel in [T0min+0.05*T0range,T0max-0.1*T0range]:
            T_ = Tws+skew(z)
            T_.sort() # Ignore minor inversion problems in calculation of Tws
            plabel = lininterp(T_, p, Tlabel)
            if p[0]>plabel>p[-1]:
                pylab.text(Tlabel,plabel,'%2g'%ws,
                           color=rgbcolors.get_color('red'),
                           backgroundcolor=rgbcolors.get_color('white'),
                           horizontalalignment='center',
                           verticalalignment='center',
                           size='small'
                           )


    # Constant ws isopleth to lifting condensation level
    ws = mixing_ratio(p[i0],dewpoint[i0]+273.15)
    Tws_lcl = const_ws_isolpleth(ws,plower)-273.15
    pylab.semilogy(Tws_lcl+skew(zlower)[i0:ilcl+1], plower, linewidth=2,
                   color=rgbcolors.get_color('red'))


    # Temperature profile
    pylab.semilogy(T+skew(z),p,color=rgbcolors.get_color('black'),
                   marker='o',linewidth=2) 

    # Dew point profile
    pylab.semilogy(dewpoint+skew(z),p,color=rgbcolors.get_color('black'),
                   marker='o',linewidth=2)

    pylab.xlabel('Temperature (C)')
    pylab.ylabel('Pressure (hPa)')
    pylab.title(title)

    pylab.xticks(xticks,['%.0f'%tem for tem in xticks])
    pylab.yticks(yticks,['%.0f'%pres for pres in yticks])

    pylab.xlim(T0min,T0max)
    pylab.ylim(pmax+1.,pmin)

    # Plot heights on the right axis at the same levels as the pressure ticks
    ax2 = pylab.twinx()
    pylab.ylabel('Height (km)')
    pylab.semilogy()
    logp = numpy.log(p)
    for i in range(1,len(logp)): # Deal with identical pressure values
        if logp[i]>=logp[i-1]:
            logp[i] = logp[i-1]*0.999999
    ylabels = ['%.1f'%y for y in \
                   lininterp(-logp,z/1000.,-numpy.log(yticks))]
    pylab.yticks(yticks,ylabels)
    pylab.xlim(T0min,T0max)
    pylab.ylim(pmax+1.,pmin)

    return fig

if __name__ == "__main__":

    import matplotlib
    matplotlib.rc('text', usetex=True)
    import pylab

    plot_Fig5_4()
    plot_comparisons()
    pylab.show()

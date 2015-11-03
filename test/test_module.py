#! /usr/bin/python

"""thermo.py: Unit tests for the thermo module.

  Copyright (C) 2007 Jonathan G. Doyle
  All rights reserved.
  
  Jonathan G. Doyle <doylejg@dal.ca>
  Department of Physics and Atmospheric Science,
  Dalhousie University, Halifax, Nova Scotia, Canada, B3H 3J5


NOTICE

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License version 2 as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
  MA  02111-1307  USA
"""

import unittest

import sys; sys.path.append('..')

import os.path
import stat
import grp
import getpass
import shutil

import thermo
import numpy

# Tom's packages
import arrayutils
import datafile
from modelatm import ussa

class Measurement(object):
    def __init__(self,data,err=None,z=None,t=None,date=None):
        self.data = data
        self.err = err
        self.z = z
        self.t = t
        self.date = date

def get_radiosonde(radiosonde_path,site_elevation=.01):
    """
    Modified version from Tom's quicklook
    VERSION = '2008-11-04'
    """
    df = datafile.NumpyDataFile(radiosonde_path,6)
    p = df.float[0]*100.                  # Pressures (Pa)
    z = df.float[1]/1000.-site_elevation  # Heights (km) relative to ground
    T = df.float[2] + 273.15              # Temperatures (K)
    Td = df.float[3]                      # Dewpoint Temperatures (K)
    RH = df.float[4]                      # Relative Humidity (%)
    w = df.float[5]                       # Mixing ratio (g/kg)
    den = p/(287.*T)                      # Densities (kg/m3)

    
    # Fix up any problems
    # Data in wrong order
    z,p,T,Td,RH,w,den = arrayutils.multisort(z,p,T,Td,RH,w,den) 
    indices = numpy.unique1d(z,return_index=True)[1] # Unique z values only
    z,p,T,Td,RH,w,den = arrayutils.multitake(indices,z,p,T,Td,RH,w,den)

    # Fill in the higher altitudes using model data
    model = ussa.SubarcticWinter()
    zbreak = z[-1]
    n = model.z.searchsorted(zbreak)
    if model.z[n] == zbreak:
        n+=1
    numpy.append(z,model.z[n:])
    numpy.append(p,model.p[n:]*100.)
    numpy.append(T,model.T[n:])
    numpy.append(Td,model.TD[n:])
    numpy.append(den,model.dens[n:])
    numpy.append(RH,model.RH[n:])
    
    # Create the Measurement object
    rsdata = Measurement({'p':p,'T':T,'Td':Td,'RH':RH,'den':den},z=z)

    # Save the break point between measurement and model
    rsdata.zbreak = zbreak

    return rsdata
    
# Get Radiosonde data
radiosonde_path = '2009-03-06-12z.txt'
rsdata = get_radiosonde(radiosonde_path)

den = rsdata.data['den']
p = rsdata.data['p']
T = rsdata.data['T']
Td = rsdata.data['Td']
RH = rsdata.data['RH']        

forms = ['simple','WMO','bohrenalbrecht','murphykoop']

thres = 1E-15

class TestThermoFunctions(unittest.TestCase):

    def test_number_density_mass_density(self):
        """Test number_density and mass_density"""

        # Excecute the function 
        num_den = thermo.get_number_density(den)
        mass_den = thermo.get_mass_density(num_den)
        
        # Boolean checks to verify the fuciton behaved correctly
        self.assert_( numpy.alltrue(numpy.abs(den-mass_den) < thres) )

    def test_relative_humidity(self):
        """Test relative_humidity"""
        print p,T,Td
        calc_RH = thermo.relative_humidity(p,T,Td)

        self.assert_( numpy.alltrue(numpy.abs(RH-calc_RH)<thres) )
        

def run_tests():


    # Do the tests
    print '\nthermo module tests:'
    suite = unittest.makeSuite(TestThermoFunctions)
    ret = unittest.TextTestRunner(verbosity=2).run(suite)


    return ret

if __name__ == '__main__':
    sys.exit(len(run_tests().failures))

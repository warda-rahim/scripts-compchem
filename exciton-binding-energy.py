#!/usr/bin/env python3                                                                                                                  

"""Calculate exciton binding energy from effective mass theory"""

################################################################################
#                                             #
################################################################################

import math as m
import numpy as np
import scipy.constants as sc
import statistics as stats
#import matplotlib.pyplot as plt
import argparse


######################## Set up optional arguments #############################


parser = argparse.ArgumentParser(description='Calculates the exciton binding energy')

parser.add_argument('--electronmasses', metavar='mass', nargs="+",
                    default=[1.029, 0.497], type=float,
                    help='Electron effectives masses along different directions')
parser.add_argument('--holemasses', metavar='mass', nargs="+", 
                    default=[1.216, 24.781], type=float, 
		    help='Hole effective masses along different directions')
parser.add_argument('--m_e',  metavar='me*', 
                    default=0.335, type=float,
		    help='Harmonic mean of electron effective mass')
parser.add_argument('--m_h', metavar='mh*', 
                    default=1.159, type=float,
		    help='Harmonic mean of hole effective mass')
parser.add_argument('--s', metavar='dielectric', nargs="+",
                    default=4.28683, type=float,
                    help='Static (low frequency) dielectric constants along x,y,z')
parser.add_argument('--o', metavar='dielectric', nargs="+",
                    default=3.680235, type=float,
		    help='Optical (high frequency) dielectric constants along x,y,z')
parser.add_argument('--staticdiel', metavar='avg_dielectric',
                    default=4.28683, type=float,
		    help='Average static (low frequency) dielectric constant')
parser.add_argument('--opticaldiel', metavar='avg_dielectric',
                    default=3.680235, type=float,
		    help='Optical (high frequency) dielectric constant')
                      
args = parser.parse_args()




#calculate the harmonic mean of electron effective mass
if args.electronmasses is None:
    m_e = args.m_e
else:
    m_e = stats.harmonic_mean(args.electronmasses)
    print("m_e:", m_e)

#calculate the harmonic mean of hole effective mass
if args.holemasses is None:
    m_h = arg.m_h
else:
    m_h = stats.harmonic_mean(args.holemasses)
    print("m_h:",m_h)

#calculate the reduced effective mass in SI units (times by free mass of an electron)
m = ((m_e * m_h) / (m_e + m_h)) * (sc.electron_mass)
print("m:",m)

#calculate average dielectric constants
if args.s is None:
    staticdiel = args.staticdiel
else: 
    staticdiel = np.mean(args.s)

if args.o is None:
    opticaldiel = args.opticaldiel
else: 
    opticaldiel = np.mean(args.o)

#calculate effective Bohr diameter of thermal exciton in metres
#a_0 = (4 * pi * (opticaldiel * sc.epsilon_0) * (sc.hbar)**2 ) / (reduced effective mass * (sc.e)**2)
a_0 = (4 * (sc.pi) * (args.staticdiel * sc.epsilon_0) * (sc.hbar)**2 ) / (m * (sc.e)**2)
print("Thermal a_0:", a_0)

#calculate thermal exciton binding energy in Joules
E = 1/a_0 * ((sc.e)**2 / (8 * args.staticdiel * sc.epsilon_0 * sc.pi))
print("Thermal E in Joules:", E)

#exciton binding energy in eV
E = E * (1 / sc.eV)
print("Thermal E in eV:", E)

#calculate effective Bohr radius of optical exciton in metres
#a_0 = (4 * pi * (opticaldiel * sc.epsilon_0) * (sc.hbar)**2 ) / (m * (sc.e)**2)
a_0 = (4 * (sc.pi) * (args.opticaldiel * sc.epsilon_0) * (sc.hbar)**2 ) / (m * (sc.e)**2)
print("Optical a_0:", a_0)

#calculate optical exciton binding energy in Joules
E = 1/a_0 * ((sc.e)**2 / (8 * args.opticaldiel * sc.epsilon_0 * sc.pi))
print("Optical E in Joules:", E)

#optical exciton binding energy in eV
E = E * (1 / sc.eV)
print("Optical E in eV:", E)

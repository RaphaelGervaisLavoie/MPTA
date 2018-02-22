''' 
MPTA_Functions_Pure_Gas.py


This file provide the functions used in the implementation of the MPTA for pure gas.

 Version: 1.0
 Last Update: 2018-feb-19
 License: MIT
 Author: Raphaël Gervais Lavoie (raphael.gervaislavoie@uqtr.ca)
 
 
 If you find this code usefull, please cite the folowing reference:

"Numerical implementation of the multicomponent potential theory of adsorption in Python using the NIST Refprop", 
 Raphaël Gervais Lavoie, Mathieu Ouellet, Jean Hamelin and Pierre Bénard,  
 Commun. Comput. Phys., Vol. 23, No. 5, 2018, pp. 1602--1625,  
 DOI: 10.4208/cicp.OA-2017-0012
'''

import refprop
import numpy
import scipy

def pressure(d, T, x, P):
    '''
        Return the function P(d,t,x) - P = 0 to be solve for unknown density d.
        
        Inputs:
            d	Fluid molar density [mol/L]
            T	Fluid temperature [K]
            x	Composition array [array of molar fraction]
            P	Fluid pressure [kPa]
        Output:
            Return the function P(d,t,x) - P = 0 to be solve for unknown density d.
    '''
    return refprop.press(T, float(d), x)['p'] - P


def d_max(T, x):
    '''
        Maximal fluid density allowed by the Refprop.
        
        Inputs:
            T	Fluid temperature [K]
            x	Composition array [array of molar fraction]
        Output
            d	Fluid molar density [mol/L]
    '''
    return refprop.limitx(x, htype='EOS', t=T, D=0, p=0)['Dmax']


def chem(T, d, x):
    '''
        Fluid chemical potential for each component.
        
        Inputs:
            T	Fluid temperature [K]
            d	Fluid molar density [mol/L]
            x	Composition array [array of molar fraction]
        Output:
            u	Array containing the chemical potential of each fluid component [J/mol]
    '''
    return refprop.chempot(T, float(d), x)['u']


def DRA(z0, eps0, z, beta):
    '''
        Dubinin–Radushkevich–Astakhov potential.
        
        This is the fluid--surface interaction potential.
        
        Inputs:
            z0		Limiting microporous volume [cm³/g]
            eps0	Characteristic energy of adsorption [J/mol]
            z		Microporous volume [cm²/g]
            beta	Heterogeneity parameter (usually set to 2 for activated carbon)
        Output:
            epsilon	Adsorbent surface potential [J/mol]
    '''
    return eps0*numpy.log(z0/z)**(1.0/beta)


def d_pure(z, T, z0, eps0, beta, dB, xB, d0, d_liq):
    '''
        Compute the pure gas density in the adsorbed phase.
        
        Inputs:
            z		Microporous volume (0 <= z <= z0) [cm²/g]
            T		Fluid temperature [K]
            z0		Limiting microporous volume [cm³/g]
            eps0	Characteristic energy of adsorption [J/mol]
            beta	Heterogeneity parameter (usually set to 2 for activated carbon)
            dB		Fluid density in the bulk phase [mol/L]
            xB		Bulk phase composition array [array of molar fraction]
            d0		Initial guess for the density [mol/L]
            d_liq	Liquid density at dew point [mol/L]
        Output:
            d_z		Fluid density in the adsorbed phase at specific value z [mol/L]
    '''
    def f(d_z):
        y = chem(T, dB, xB)[0] + DRA(z0, eps0, z, beta) - chem(T, d_z, xB)[0]
        return y
    if chem(T, dB, xB)[0] + DRA(z0, eps0, z, beta) >= chem(T, d_max(T, xB), xB)[0]:
        return d_max(T, xB)
    if chem(T, dB, xB)[0] + DRA(z0, eps0, z, beta) > chem(T, d_liq, xB)[0] and d0 < d_liq:
        d0 = 1.1*d_liq
    return scipy.optimize.fsolve(f, d0)[0]
    
    
def N_ex_pure(T, z0, eps0, beta, dB, xB, d_liq, N=300):
    '''
        Compute the excess (Gibbs) adsorption.
        
        Inputs:
            T		Fluid temperature [K]
            z0		Limiting microporous volume [cm³/g]
            eps0	Characteristic energy of adsorption [J/mol]
            beta	Heterogeneity parameter (usually set to 2 for activated carbon)
            dB		Fluid density in the bulk phase [mol/L]
            xB		Bulk phase composition array [array of molar fraction]
            d_liq	Liquid density at dew point [mol/L]
        Optional:
            N		The number of subintervals in 0<=z<=z0
        Output:
            N_ex	Excess adsorption [mol/Kg]
    '''
    delta = z0 / N
    d0 = dB
    integral = 0
    for i in range(0, N):
        d_z = d_pure(z0 - i*delta - delta/2, T, z0, eps0, beta, dB, xB, d0, d_liq)
        integral += d_z*delta
        d0 = d_z
    return integral - dB*z0


def pure_fit(params, T, dataD, dataAd, xB, d_liq):
    '''
        By minimizing this function, the model is fited on experiental data.
        
        Inputs:
            params		Dictionary data structure containing the parameters z0, eps0, and beta.
            T			Fluid temperature [K]
            dataD		Array containing experimental data for bulk phase density [array of mol/L]
            dataAd		Array containing experiment data for excess adsorption [array of mol/Kg]
            xB			Bulk phase composition array [array of molar fraction]
            d_liq		Liquid density at dew point [mol/L]
        Output:
            diffrerence	Array containing the difference between the model and the experimental values.
    '''
    value = params.valuesdict()
    z0 = value['z0']
    eps0 = value['eps0']
    beta = value['beta']
    difference = []
    for i in range(0, len(dataD)):
        difference.append(N_ex_pure(T, z0, eps0, beta, dataD[i], xB, d_liq) - dataAd[i])
    return difference
    

def mean_error(dataExp, dataModel):
    '''
    	Return the mean error between the two arrays of data.
        
        Inputs:
        	dataExp		Array containing experiment data for excess adsorption.
            dataModel	Array containing data for excess adsorption computed by the MPTA model.
        Output:
        	Mean error in %.
    '''
    d = 0
    if len(dataExp) != len(dataModel):
        return print('*****The two dataset dont have the same lenght...*****')
    for i in range(0, len(dataExp)):
        d += abs((dataExp[i] - dataModel[i])/dataExp[i])
    return 100*d/len(dataExp)
    
    
    
    
    
    

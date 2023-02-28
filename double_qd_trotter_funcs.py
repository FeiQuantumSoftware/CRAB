import numpy as np
from numba import jit, njit
from numpy.linalg import det

#%%

@njit

def utrotter(pulse, h0, hd, t):
    # Trotterized unitary
    N = len(pulse)
    if N == 0:
        return np.diag(np.ones(4,dtype='complex128'))
    deltat = t/N
    h0diag = np.diag(h0)
    ed,ev = np.linalg.eig(hd)
    u0 = ev.T.conj() * np.exp(-1j * deltat * h0diag) @ ev
    u0half = ev.T.conj() * np.exp(-1j * deltat/2 * h0diag)
    u = u0half

    for phi in pulse:
        u = u0 * np.exp(-1j * deltat * phi * ed) @ u

    u = u0half.T.conj() @ u
    return u

@njit
def uGRAPE(pulse, h0, hd, t):
    # trotterization of u for Grape
    u = np.diag(np.ones(4,dtype='complex128'))
    N = len(pulse)
    if N == 0:
        return u
    deltat = t/N
    
    for phi in pulse:
        ed, ev = np.linalg.eig(h0 + phi*hd)
        u = (ev * np.exp(-1j * deltat * ed)) @ ev.T.conj() @ u
        
    return u

@njit
def crab_pulse(coeff, pulse, freq, h0, hd, t):
    # updated crab pulses
    N = len(pulse)
    c_pulse = np.zeros(N)
    deltat = t/N
    for i in range(len(c_pulse)):
        c_cor = 1
        for j in range(int(len(coeff) / 2)):
            c_cor += (coeff[int(2 * j)] * np.sin(freq[j] * (i)*deltat) + 
                        coeff[int(2 * j + 1)] * np.cos(freq[j] * (i)*deltat))
        c_pulse[i] = pulse[i] * c_cor
    return c_pulse

@njit
def uCRAB(coeff, pulse, freq, h0, hd, t):
    # trotterized u for crab
    N = len(pulse)
    if N == 0:
        return np.diag(np.ones(4,dtype='complex128'))
    deltat = t/N
    h0diag = np.diag(h0)
    ed,ev = np.linalg.eig(hd)
    u0 = ev.T.conj() * np.exp(-1j * deltat * h0diag) @ ev
    u0half = ev.T.conj() * np.exp(-1j * deltat/2 * h0diag)
    u = u0half

    for i in range(len(pulse)):
        c_cor = 1
        for j in range(int(len(coeff) / 2)):
            c_cor += (coeff[int(2 * j)] * np.sin(freq[j] * (i)*deltat) + 
                        coeff[int(2 * j + 1)] * np.cos(freq[j] * (i)*deltat))
        u = u0 * np.exp(-1j * deltat * pulse[i] * c_cor * ed) @ u

    u = u0half.T.conj() @ u
    return u

@njit
def du(i, pulse, h0, hd, t):
    # returns dU/d\Omega_i, i.e. the i-th component of the gradient of U w.r.t. \Omega
    N = len(pulse)
    deltat = t/N
    h0diag = np.diag(h0)
    sandwich = -1j*deltat*np.diag(np.exp(1j * deltat/2 * h0diag)) @ hd * np.exp(-1j * deltat/2 * h0diag)
    dui = utrotter(pulse[i+1:], h0,hd,t-(i)*deltat) @ sandwich @ utrotter(pulse[:i+1],h0,hd,(i)*deltat)
    return dui

@njit
def duG(i, pulse, h0, hd, t):
    # du for gradient for Grape
    N = len(pulse)
    deltat = t/N
    dui = -1j * deltat * uGRAPE(pulse[i+1:], h0,hd,t-(i)*deltat) @ hd @ uGRAPE(pulse[:i+1],h0,hd,(i)*deltat)
    return dui

# Magic matrix - for basis change into "magic" Bell basis
q = np.array([
[1,0,0,1j],
[0,1j,1,0],
[0,1j,-1,0],
[1,0,0,-1j]
], dtype='complex128')/np.sqrt(2)

@njit
def m(pulse, h0, hd, t):
    # Makhlin matrix - M - U_B^T U_B
    ub = q.T.conj() @ utrotter(pulse, h0, hd, t) @ q
    return ub.T @ ub

@njit
def mG(pulse, h0, hd, t):
    # Makhlin matrix - M - U_B^T U_B
    ub = q.T.conj() @ uGRAPE(pulse, h0, hd, t) @ q
    return ub.T @ ub

@njit
def mC(coeff, pulse, freq, h0, hd, t):
    # Makhlin matrix - M - U_B^T U_B
    ub = q.T.conj() @ uCRAB(coeff, pulse, freq, h0, hd, t) @ q
    return ub.T @ ub

@njit
def dm(i, pulse, h0, hd, t):
    # returns dM/d\Omega_i, i.e. the i-th component of the gradient of M w.r.t. \Omega
    ub = q.T.conj() @ utrotter(pulse, h0, hd, t) @ q
    dmPart = ub.T @ q.T.conj() @ du(i, pulse, h0, hd, t) @ q
    return dmPart + dmPart.T

@njit
def dmG(i, pulse, h0, hd, t):
    # returns dM/d\Omega_i, i.e. the i-th component of the gradient of M w.r.t. \Omega
    ub = q.T.conj() @ uGRAPE(pulse, h0, hd, t) @ q
    dmPart = ub.T @ q.T.conj() @ duG(i, pulse, h0, hd, t) @ q
    return dmPart + dmPart.T

@njit
def g1(pulse, h0, hd, t):
    # first Makhlin invariant
    detu = det(utrotter(pulse, h0, hd, t))
    return np.trace(m(pulse, h0, hd, t))**2 / (16*detu)

@njit
def g2(pulse, h0, hd, t):
    # second Makhlin invariant
    detu = det(utrotter(pulse, h0, hd, t))
    return ( np.trace(m(pulse, h0, hd, t))**2 -
    np.trace(m(pulse, h0, hd, t)@m(pulse, h0, hd, t)) ) / (4*detu)

@njit
def g1G(pulse, h0, hd, t):
    # first Makhlin invariant
    detu = det(uGRAPE(pulse, h0, hd, t))
    return np.trace(mG(pulse, h0, hd, t))**2 / (16*detu)

@njit
def g2G(pulse, h0, hd, t):
    # second Makhlin invariant
    detu = det(uGRAPE(pulse, h0, hd, t))
    return ( np.trace(mG(pulse, h0, hd, t))**2 -
    np.trace(mG(pulse, h0, hd, t)@mG(pulse, h0, hd, t)) ) / (4*detu)

@njit
def g1C(coeff, pulse, freq, h0, hd, t):
    # first Makhlin invariant
    detu = det(uCRAB(coeff, pulse, freq, h0, hd, t))
    return np.trace(mC(coeff, pulse, freq, h0, hd, t))**2 / (16*detu)

@njit
def g2C(coeff, pulse, freq, h0, hd, t):
    # second Makhlin invariant
    detu = det(uCRAB(coeff, pulse, freq, h0, hd, t))
    return ( np.trace(mC(coeff, pulse, freq, h0, hd, t))**2 -
    np.trace(mC(coeff, pulse, freq, h0, hd, t)@mC(coeff, pulse, freq, h0, hd, t)) ) / (4*detu)

@njit
def cost(pulse, h0, hd, t):
    # cost function for finding a gate locally equivalent to CNOT - C = |G1| + |1-G2|
    return np.abs(g1(pulse, h0, hd, t)) + np.abs(1-g2(pulse, h0, hd, t))

@njit
def costG(pulse, h0, hd, t):
    # cost function for finding a gate locally equivalent to CNOT - C = |G1| + |1-G2|
    return np.abs(g1G(pulse, h0, hd, t)) + np.abs(1-g2G(pulse, h0, hd, t))

@njit
def costC(coeff, pulse, freq, h0, hd, t):
    # cost function for finding a gate locally equivalent to CNOT - C = |G1| + |1-G2|
    return np.abs(g1C(coeff, pulse, freq, h0, hd, t)) + np.abs(1-g2C(coeff, pulse, freq, h0, hd, t))

@njit
# This assumes det(U) constant
def gradCost(pulse, h0, hd, t):
    # gradient of cost function w.r.t. \Omega - gives full array dC/d\Omega_i
    grad = np.zeros(len(pulse), dtype='float64')
    phi1 = np.angle(g1(pulse, h0, hd, t))
    phi2 = np.angle(1-g2(pulse, h0, hd, t))
    mg = m(pulse, h0, hd, t)
    for i in range(len(grad)):
        dmi = dm(i, pulse, h0, hd, t)
        grad[i] = np.real(
        np.trace(mg)*np.trace(dmi)*( np.exp(-1j*phi1) - 4*np.exp(-1j*phi2) ) +
        np.trace(mg @ dmi)*4*np.exp(-1j*phi2)
        )/8
    return grad

@njit
# This assumes det(U) constant
def gradCostG(pulse, h0, hd, t):
    # gradient of cost function w.r.t. \Omega - gives full array dC/d\Omega_i
    grad = np.zeros(len(pulse), dtype='float64')
    phi1 = np.angle(g1G(pulse, h0, hd, t))
    phi2 = np.angle(1-g2G(pulse, h0, hd, t))
    mg = mG(pulse, h0, hd, t)
    for i in range(len(grad)):
        dmi = dmG(i, pulse, h0, hd, t)
        grad[i] = np.real(
        np.trace(mg)*np.trace(dmi)*( np.exp(-1j*phi1) - 4*np.exp(-1j*phi2) ) +
        np.trace(mg @ dmi)*4*np.exp(-1j*phi2)
        )/8
    return grad
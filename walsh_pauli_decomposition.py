import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def decompose_harmonic_polynomial(N, interpolate=False):
    ### Constructs the Walsh-Pauli decomposition of $(x-1/2)^2$
    decomp = {}
    
    if interpolate:
        decomp['I' * N] = (1 + (2 / 4**N))/12
    else:
        decomp['I' * N] = 1/12
                
    for l1 in range(1, N+1):
        if interpolate:
            pauli_str_1q = ['I'] * N
            pauli_str_1q[l1-1] = 'Z'
            pauli_str_1q = ''.join(pauli_str_1q)
            pauli_str_1q = pauli_str_1q[::-1] ## qiskit in most-significant bit first convention
            decomp[pauli_str_1q] =  2**(-(N+l1+1))
            
        for l2 in range(l1+1, N+1):
            pauli_str = ['I'] * N
            pauli_str[l1-1] = 'Z'
            pauli_str[l2-1] = 'Z'
            pauli_str = ''.join(pauli_str)
            pauli_str = pauli_str[::-1] ## qiskit in most-significant bit first convention
            coeff =  (2**(- (l1 + l2 +1)))
            decomp[pauli_str] = coeff
    return decomp

def decompose_potential_term(N, V0=1.0, interpolate=False):
    factor = V0 
    decomp = decompose_harmonic_polynomial(N, interpolate)
    decomp = {pauli_str: factor * coeff for pauli_str, coeff in decomp.items()}
    return decomp

def decompose_laplacian_term(N, interpolate=False):
    factor = 4**N *2*np.pi**2 
    decomp = decompose_harmonic_polynomial(N, interpolate)
    decomp = {pauli_str: factor * coeff for pauli_str, coeff in decomp.items()}
    return decomp


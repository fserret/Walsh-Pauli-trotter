import numpy as np 

def remove_global_phase(x):
    return np.exp(-1j*np.angle(x[0])) * x

def construct_qft_matrix(n_qubits):
    N = 2 ** n_qubits
    k_indices = np.arange(N)
    j_indices = np.arange(N).reshape(-1, 1)
    exponent_matrix = j_indices * k_indices
    omega = np.exp(2 * np.pi * 1j / N)
    qft_matrix = omega ** exponent_matrix
    return qft_matrix / np.sqrt(N)

def reverse_bits(x, N):
    return int(format(x, f'0{int(N)}b')[::-1], 2)

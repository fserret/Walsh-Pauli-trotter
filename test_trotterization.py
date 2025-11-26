
import unittest
import numpy as np
from walsh_potential_pauli import build_walsh_pauli_potential, build_classical_diagonal_potential
from walsh_laplacian_pauli import build_walsh_pauli_matrix as build_walsh_pauli_kinetic
from walsh_laplacian_pauli import build_classical_diagonal_matrix as build_classical_diagonal_kinetic
from walsh_trotterization import create_trotter_step_circuit
from circuit_constructor import create_pauli_rotation_circuit, create_fourier_basis_rotation_circuit
from walsh_pauli_decomposition import decompose_potential_term, decompose_laplacian_term
from qiskit.quantum_info import Statevector, Operator
from utils import *

class TestTrotterization(unittest.TestCase):
    def setUp(self):
        self.N = 5
        self.V0 = 0
        self.interpolate = False
        self.dt = 0.0002
        
        self.P = build_walsh_pauli_potential(self.N, V0=self.V0)
        self.K = build_walsh_pauli_kinetic(self.N)
        self.P_interp = build_classical_diagonal_potential(self.N, V0=self.V0)
        self.K_interp = build_classical_diagonal_kinetic(self.N)

        # DFT is the conjugate of QFT --> We go from x to exp basis hence we need the rows to be the conjugate of the functions
        self.DFT = construct_qft_matrix(self.N).conj().transpose()
        X_0 = np.kron(np.array([[0, 1], [1, 0]]), np.eye(2**(self.N - 1)))
        # negative frequencies on the left
        self.DFT = np.dot(X_0, self.DFT)
        
        self.K_x = np.dot(self.DFT.T.conj(), self.K).dot(self.DFT)
        self.K_x_interp = np.dot(np.dot(self.DFT.T.conj(), self.K_interp), self.DFT)
        
        X = np.linspace(0, 1, 2**self.N, endpoint=False)
        self.psi = np.exp(-((X - 0.5)**2) / (2 * 0.1**2))
        self.psi = self.psi / np.linalg.norm(self.psi)
        self.state = Statevector(self.psi)

        self.potential_decomp = decompose_potential_term(self.N, self.V0, interpolate=self.interpolate)
        self.kinetic_decomp = decompose_laplacian_term(self.N, interpolate=self.interpolate)


    def test_potential_decomposition(self):
        """Test that the potential is correctly represented by Walsh-Pauli in the qiskit msb basis."""
        Hadamard = np.asarray([[1, 1], [1, -1]]) / 2**0.5
        HN = 1
        for i in range(self.N):
            HN = np.kron(HN, Hadamard)
        
        indices = {int(''.join(['0' if i == 'I' else '1' for i in string[::-1]]), base=2): coeff for string, coeff in self.potential_decomp.items()}
        
        reconstructed_potential = np.sum([HN[:, index] * coeff * 2**(self.N / 2) for index, coeff in indices.items()], axis=0)
        
        self.assertTrue(np.allclose(np.diag(self.P), reconstructed_potential.real))


    def test_potential_rotation_angles(self):
        """Test that the angles for the rotation for the potential term are correct."""
        potential_circuit_half = create_pauli_rotation_circuit(self.N, self.dt / 2, self.potential_decomp)
        state_T = self.state.evolve(potential_circuit_half)
        
        exp_pot = np.diag(np.exp(-1j * np.diag(self.P) * self.dt / 2))
        
        angles_circuit = np.angle((state_T.data) / self.state.data * np.exp(-1j * np.angle((state_T.data) / self.state.data)[0]))
        angles_expected = np.angle(np.diag(exp_pot) * np.exp(-1j * np.angle(np.diag(exp_pot))[0]))

        self.assertTrue(np.allclose(angles_circuit, angles_expected))


    def test_kinetic_rotation_angles(self):
        """Test that the angles in the QFT basis are correct."""
        kinetic_circuit = create_fourier_basis_rotation_circuit(self.N, self.dt, self.kinetic_decomp)
        state_K = self.state.evolve(kinetic_circuit)
        exp_kin = np.dot(self.DFT.T.conj(), np.diag(np.exp(-1j * np.diag(self.K) * self.dt))).dot(self.DFT)
        
        # Calculate angles, handling potential division by zero
        with np.errstate(divide='ignore', invalid='ignore'):

            dft_psi = self.DFT @ self.psi
            dft_state_K = self.DFT @ state_K.data
            
            angles_circuit = np.angle(dft_state_K / dft_psi * np.exp(-1j * np.angle(dft_state_K / dft_psi)[0]))

            dft_exp_kin_psi = self.DFT @ exp_kin @ self.psi
            angles_expected = np.angle(dft_exp_kin_psi / dft_psi * np.exp(-1j * np.angle(dft_exp_kin_psi / dft_psi)[0]))

        # Replace NaNs with 0, as they occur where the state is ~0 and the angle is thus undefined.
        angles_circuit = np.nan_to_num(angles_circuit)
        angles_expected = np.nan_to_num(angles_expected)

        self.assertTrue(np.allclose(angles_circuit, angles_expected, atol=1e-7))

    def test_potential_unitarity(self):
        """Verify that the potential evolution operator is unitary."""
        potential_circuit_half = create_pauli_rotation_circuit(self.N, self.dt / 2, self.potential_decomp)
        op = Operator(potential_circuit_half)
        self.assertTrue(op.is_unitary(atol=1e-9))

    def test_kinetic_unitarity(self):
        """Verify that the kinetic evolution operator is unitary."""
        kinetic_circuit = create_fourier_basis_rotation_circuit(self.N, self.dt, self.kinetic_decomp)
        op = Operator(kinetic_circuit)
        self.assertTrue(op.is_unitary(atol=1e-9))

    def test_trotter_step_evolution(self):
        """Verify that a single Trotter step circuit evolves the state correctly compared to numpy."""
        psi = self.state.data.copy()
        trotter_step_circuit = create_trotter_step_circuit(self.N, self.dt, self.V0, interpolate=self.interpolate)
        state_end_circuit = self.state.evolve(trotter_step_circuit)

        exp_kin_diag_op = np.diag(np.exp(-1j * np.diag(self.K) * self.dt))
        exp_kin = np.dot(np.dot(self.DFT.T.conj(), exp_kin_diag_op), self.DFT)
        
        exp_pot_op = np.diag(np.exp(-1j * np.diag(self.P) * self.dt / 2))
        
        trotter_step_matrix = np.dot(np.dot(exp_pot_op, exp_kin), exp_pot_op)
        psi_end_numpy = np.dot(trotter_step_matrix, psi)
        remove_global_phase = lambda x: np.exp(-1j*np.angle(x[0])) * x

        self.assertTrue(np.allclose(remove_global_phase(state_end_circuit.data), remove_global_phase(psi_end_numpy), atol=1e-7))

if __name__ == '__main__':
    unittest.main()


from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate


def create_pauli_rotation_circuit(N, dt, decomp):
    qc = QuantumCircuit(N)
    
    for pauli_str, coeff in decomp.items():
        
        phase = coeff * dt  
        z_qubits = [i for i, p in enumerate(pauli_str) if p == 'Z'] 
        if len(z_qubits) == 0:
            continue
        
        q1 = z_qubits[0]
        if len(z_qubits) == 1:
            qc.rz(2 * phase, q1)
            
        elif len(z_qubits) >= 2:
            for q2 in z_qubits[1:]: 
                qc.cx(q2, q1)
                
            qc.rz(2 * phase, q1)
            
            for q2 in z_qubits[1:]: 
                qc.cx(q2, q1)
    
    return qc

def create_fourier_basis_rotation_circuit(N, dt, decomp):
    qft_gate = QFTGate(N).inverse() # QFT^{-1} = DFT
    QFT_basis_decomp = {pauli_str: coeff for pauli_str, coeff in decomp.items()} # QFT inverses the bitstring ## NOT TRUE IN QISKIT!! This is correct because qiskit is in MSB to LSB indexing
    qc = QuantumCircuit(N)
    qc.append(qft_gate, range(N)) # 
    qc.x(N-1)  # Negative frequencies first (0,.. ,2**N/2-1, -2**N/2,-(2**N/2-1), ..., -1) --> (-2**N/2, ..., 2**N/2-1), equivalent to X_0 in lsb 
    qc.compose(create_pauli_rotation_circuit(N, dt, QFT_basis_decomp), inplace=True)
    qc.x(N-1)  
    qc.append(qft_gate.inverse(), range(N))    
       
    return qc
"""
Hardware-Aligned Graph-Constrained Warm-Started QAOA (HG-WS-QAOA).
Implements the warm-started QAOA circuit with:
1. Safe-exploration Bloch sphere state preparation: U_prep = prod_l Ry(theta_l)
2. Ising Problem Unitary: U(H_C, gamma) = exp(-i * gamma * H_C)
3. Custom Warm-Start Mixer: U_M(beta) = prod_l Ry(theta_l) Rz(-2*beta) Ry(-theta_l)
Preserves the continuous DC-OPF prior without ejection into invalid disconnected subspaces.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from ..hamiltonian.ising_formulation import IsingHamiltonian


class WarmStartQaoaCircuit:
    """
    Constructs parameterized Warm-Started QAOA circuits with custom Bloch mixers.
    """

    def __init__(
        self,
        hamiltonian: IsingHamiltonian,
        bloch_angles: Dict[int, float],
        depth_p: int = 1
    ):
        self.hamiltonian = hamiltonian
        self.num_qubits = hamiltonian.num_qubits
        self.depth_p = depth_p

        # Extract bloch angles in order of qubits
        self.thetas = []
        for q_idx, lid in enumerate(hamiltonian.candidate_lines):
            theta = bloch_angles.get(lid, np.pi / 2.0)
            self.thetas.append(theta)

    def build_circuit(
        self,
        gamma_values: Optional[List[float]] = None,
        beta_values: Optional[List[float]] = None,
        include_measurements: bool = True
    ) -> Tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
        """
        Builds the parameterized or bound WS-QAOA quantum circuit.
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits if include_measurements else 0)

        # 1. Warm-Start State Preparation: Ry(theta_i) on each qubit
        for i in range(self.num_qubits):
            qc.ry(self.thetas[i], i)
        qc.barrier(label="WarmStart_Init")

        gamma_params = []
        beta_params = []

        # 2. Alternating layers
        for layer in range(self.depth_p):
            # Parameters for layer
            if gamma_values is not None:
                gamma = gamma_values[layer]
            else:
                gamma = Parameter(f"gamma_{layer}")
                gamma_params.append(gamma)

            if beta_values is not None:
                beta = beta_values[layer]
            else:
                beta = Parameter(f"beta_{layer}")
                beta_params.append(beta)

            # Problem Unitary: exp(-i * gamma * H_C)
            # Linear terms: Rz(2 * gamma * h_i)
            for i, h in self.hamiltonian.linear_terms.items():
                if abs(h) > 1e-7:
                    qc.rz(2.0 * gamma * h, i)

            # Quadratic terms: Rzz(2 * gamma * J_ij)
            # Rzz implemented via cx -> rz -> cx
            for (i, j), J in self.hamiltonian.quadratic_terms.items():
                if abs(J) > 1e-7:
                    qc.cx(i, j)
                    qc.rz(2.0 * gamma * J, j)
                    qc.cx(i, j)

            qc.barrier(label=f"Cost_p{layer}")

            # Custom Warm-Start Mixer Unitary:
            # U_M(beta) = prod_l Ry(theta_l) Rz(-2 * beta) Ry(-theta_l)
            for i in range(self.num_qubits):
                th = self.thetas[i]
                qc.ry(-th, i)
                qc.rz(-2.0 * beta, i)
                qc.ry(th, i)

            qc.barrier(label=f"WS_Mixer_p{layer}")

        # 3. Measurement
        if include_measurements:
            qc.measure(range(self.num_qubits), range(self.num_qubits))

        return qc, gamma_params, beta_params

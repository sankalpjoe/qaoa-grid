"""
Hardware-Native Heavy-Hex Mapping for IBM Eagle (127q) and Heron (133q).
Manages transpilation onto heavy-hex coupling topologies using Sabre layout and routing.
Analyzes 2-qubit gate depth, SWAP gate insertion, and native gate decomposition (ECR / CZ).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap


@dataclass
class TranspilationResult:
    target_architecture: str
    original_depth: int
    transpiled_depth: int
    two_qubit_gate_count: int
    single_qubit_gate_count: int
    swap_count_estimate: int
    operation_counts: Dict[str, int]
    circuit: QuantumCircuit


class HeavyHexMapper:
    """
    Handles hardware-native compilation and topological routing on IBM Heavy-Hex devices.
    """

    def __init__(
        self,
        architecture: str = "ibm_heron",  # "ibm_eagle" or "ibm_heron"
        lattice_distance: int = 3         # 3 -> 19q, 5 -> 57q, 7 -> 115q
    ):
        self.architecture = architecture.lower()
        self.lattice_distance = lattice_distance

        # Build Heavy-Hex coupling map
        self.coupling_map = CouplingMap.from_heavy_hex(distance=lattice_distance)
        self.num_hardware_qubits = self.coupling_map.size()

        # Target native basis gates:
        # Eagle uses ECR as native 2Q gate; Heron uses CZ as native 2Q gate
        if "heron" in self.architecture:
            self.basis_gates = ["rz", "sx", "x", "cz"]
            self.two_qubit_gate_name = "cz"
        else:
            self.basis_gates = ["rz", "sx", "x", "ecr"]
            self.two_qubit_gate_name = "ecr"

    def transpile_circuit(
        self,
        qc: QuantumCircuit,
        optimization_level: int = 3,
        seed_transpiler: int = 42
    ) -> TranspilationResult:
        """
        Transpiles a logical QAOA circuit onto the heavy-hex coupling map.
        Uses Sabre layout and routing with high optimization level.
        """
        orig_depth = qc.depth()

        # Transpile onto heavy-hex coupling map
        transpiled_qc = transpile(
            qc,
            coupling_map=self.coupling_map,
            basis_gates=self.basis_gates,
            optimization_level=optimization_level,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=seed_transpiler
        )

        ops = transpiled_qc.count_ops()
        two_q_count = ops.get(self.two_qubit_gate_name, 0)
        single_q_count = sum(cnt for gate, cnt in ops.items() if gate != self.two_qubit_gate_name and gate != "barrier" and gate != "measure")

        # Estimate SWAP insertions: in heavy-hex, each logical non-adjacent 2Q interaction
        # expands into SWAPs. Each SWAP decomposes into 3 native 2Q gates (ECR or CZ).
        # We estimate excess 2Q gates above base interaction count.
        logical_two_q_count = qc.count_ops().get("cx", 0) + qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0)
        excess_2q = max(0, two_q_count - logical_two_q_count)
        swap_estimate = excess_2q // 3

        return TranspilationResult(
            target_architecture=f"{self.architecture} (Heavy-Hex {self.num_hardware_qubits}Q)",
            original_depth=orig_depth,
            transpiled_depth=transpiled_qc.depth(),
            two_qubit_gate_count=two_q_count,
            single_qubit_gate_count=single_q_count,
            swap_count_estimate=swap_estimate,
            operation_counts=dict(ops),
            circuit=transpiled_qc
        )

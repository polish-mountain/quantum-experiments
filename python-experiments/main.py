from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library.standard_gates import RC3XGate
from qiskit_aer import AerSimulator
import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
from collections import Counter

# Use Aer's AerSimulator
simulator = AerSimulator()

qreg_q = QuantumRegister(12, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.h(qreg_q[2])
circuit.h(qreg_q[3])
circuit.h(qreg_q[5])
circuit.h(qreg_q[6])
circuit.u(pi / 2, pi / 2, pi / 2, qreg_q[11])
circuit.cx(qreg_q[8], qreg_q[9])
circuit.h(qreg_q[10])
circuit.h(qreg_q[8])
circuit.u(pi / 2, pi / 2, pi / 2, qreg_q[9])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.x(qreg_q[3])
circuit.x(qreg_q[6])
circuit.u(pi / 2, pi / 2, pi / 2, qreg_q[11])
circuit.u(pi / 2, pi / 2, pi / 2, qreg_q[8])
circuit.h(qreg_q[3])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.x(qreg_q[11])
circuit.x(qreg_q[6])
circuit.cx(qreg_q[9], qreg_q[10])
circuit.cx(qreg_q[8], qreg_q[9])
circuit.cx(qreg_q[3], qreg_q[4])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.h(qreg_q[11])
circuit.cx(qreg_q[6], qreg_q[7])
circuit.h(qreg_q[3])
circuit.u(pi / 2, pi / 2, pi / 2, qreg_q[6])
circuit.cx(qreg_q[10], qreg_q[11])
circuit.cx(qreg_q[7], qreg_q[8])
circuit.cx(qreg_q[4], qreg_q[5])
circuit.h(qreg_q[3])
circuit.u(pi / 2, pi / 2, pi / 2, qreg_q[7])
circuit.h(qreg_q[3])
circuit.cx(qreg_q[3], qreg_q[4])
circuit.cx(qreg_q[4], qreg_q[5])

circuit.measure_all(add_bits=True)


# Compile the circuit for the support instruction set (basis_gates)
# and topology (coupling_map) of the backend
compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the aer simulator
job = simulator.run(compiled_circuit, shots=10000)

# Grab results from the job
result = job.result()

print(result.get_counts())

points = {}
for value, count in result.get_counts().items():
    truncated_value = value[:circuit.num_qubits]
    split = len(truncated_value) // 2
    x = int(truncated_value[:split], base=2)
    y = int(truncated_value[split:], base=2)
    points[(x, y)] = count

x_max = 2**split - 1
y_max = 2**(len(truncated_value) - split) - 1

xs = [point[0]/x_max for point in points]
ys = [point[1]/y_max for point in points]
sizes = [size for size in points.values()]
    
plt.scatter(xs, ys, sizes)
plt.xlim((-0.05, 1.05))
plt.ylim((-0.05, 1.05))
plt.show()

# # Importing standard python libraries
# import numpy as np
# from math import pi,sqrt
# import matplotlib.pyplot as plt

# # Importing standard Qiskit libraries
# from qiskit import QuantumCircuit, transpile, Aer, IBMQ, execute, QuantumRegister
# from qiskit.tools.jupyter import *
# from qiskit.visualization import * # plot_bloch_multivector
# from qiskit.providers.aer import QasmSimulator
# from qiskit.quantum_info import Statevector

# # Start with an one qubit quantum circuit yielding a nice fractal. Change the circuit as you like.
# circuit = QuantumCircuit(1,1)
# circuit.h(0)
# circuit.u(pi/4, -pi/3, pi/8, 0)

# qc1 = circuit

# # Run the circuit with the state vector simulator to obtain a noise-free fractal.
# backend = Aer.get_backend('statevector_simulator')
# out = execute(qc1,backend).result().get_statevector()
# print(out)

# # Extract the first element of the state vector as z0 and the second element as z1.
# z0 = out.data[0]
# z1 = out.data[1]

# # Goal: One complex number for the Julia set fractal. 
# if z1.real != 0 or z1.imag != 0:
#     z = z0/z1
#     z = round(z.real, 2) + round(z.imag, 2) * 1j
# else:
#      z = 0 

# print("z= ",z)

# # Define the size
# size = 1000
# heightsize = size
# widthsize = size


# def julia_set(c=z, height=heightsize, width=widthsize, x=0, y=0, zoom=1, max_iterations=100):

#     # To make navigation easier we calculate these values
#     x_width = 1.5
#     y_height = 1.5*height/width
#     x_from = x - x_width/zoom
#     x_to = x + x_width/zoom
#     y_from = y - y_height/zoom
#     y_to = y + y_height/zoom
    
#     # Here the actual algorithm starts and the z paramter is defined for the Julia set function
#     x = np.linspace(x_from, x_to, width).reshape((1, width))
#     y = np.linspace(y_from, y_to, height).reshape((height, 1))
#     z = x + 1j * y
    
#     # Initialize c to the complex number obtained from the quantum circuit
#     c = np.full(z.shape, c)
    
#     # To keep track in which iteration the point diverged
#     div_time = np.zeros(z.shape, dtype=int)
    
#     # To keep track on which points did not converge so far
#     m = np.full(c.shape, True, dtype=bool)
    
#     for i in range(max_iterations):
#         z[m] = z[m]**2 + c[m] 
#         m[np.abs(z) > 2] = False
#         div_time[m] = i
#     return div_time


# # plot the Julia set fractal
# plt.imshow(julia_set(), cmap='magma') # viridis', 'plasma', 'inferno', 'magma', 'cividis'
# plt.show()

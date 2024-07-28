import numpy as np

from sympy import Matrix
from sympy.physics.quantum import TensorProduct

from qiskit import QuantumCircuit, transpile

from qiskit.circuit import parameter
from qiskit.circuit import Parameter, ParameterVector

from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate, RZZGate

from qiskit_algorithms.gradients import SPSAEstimatorGradient

from qiskit_aer import AerSimulator, StatevectorSimulator

import matplotlib.pyplot as plt

import data_script as ds
from optimizer import adam


class TSPSolver:
    def __init__(
        self, flow, distance, beta, theta1, theta2, alpha, expected_result, loss
    ):
        self.__affinity = TensorProduct(flow, distance)
        self.__flow = flow
        self.__beta = beta
        self.__theta1 = theta1
        self.__theta2 = theta2
        self.__alpha = alpha
        self.__expected_result = expected_result
        self.__loss = loss
        self.__num_qubits = int(np.power(flow.shape[0], 2))
        self.__referential_circuit, self.__variational_circuit = self.__create_circuit()

    def __create_circuit(self):
        # Initialise circuit of qubits equal to n1 dimension of the affinity matrix
        ref_circuit = QuantumCircuit(self.__num_qubits)
        var_circuit = QuantumCircuit(self.__num_qubits)

        # Create encoding layer
        self.__encoding_layer(ref_circuit)

        # Create constrain layer
        self.__constraint_layer(var_circuit)

        # Create perceptron layer
        self.__perceptron_layer(var_circuit)

        # Create pooling layer
        self.__pooling_layer(var_circuit)

        return ref_circuit, var_circuit

    def update_referential_layer(self, new_dist, new_result):
        self.__expected_result = new_result
        self.__affinity = TensorProduct(self.__flow, new_dist)
        self.__ref_circuit = QuantumCircuit(self.__num_qubits)
        self.__encoding_layer(self.__ref_circuit)

    def update_variational_layers(self, params):
        self.__beta = params[0]
        self.__theta1 = params[1]
        self.__theta2 = params[2]
        self.__alpha = params[3]
        self.__variational_circuit = QuantumCircuit(self.__num_qubits)
        self.__constraint_layer(self.__variational_circuit)
        self.__perceptron_layer(self.__variational_circuit)
        self.__pooling_layer(self.__variational_circuit)

    def variational_layers_loss(self):
        self.__statevect()
        self.__y_matrix()
        self.cost_function()

        return self.__loss

    # Combine the two parts of the circuit and add a measurement
    def ansatz(self):
        ansat = self.__referential_circuit.compose(self.__variational_circuit)
        ansat.measure_all()
        return ansat

    def __encoding_layer(self, ref_circuit):
        num_qubits = len(ref_circuit.qubits)
        ref_circuit.h([qubit_no for qubit_no in range(num_qubits)])

        # Apply R_Y gate to diagonal qubits and R_ZZ to off-diagonal qubits
        for i in range(num_qubits):
            for j in range(i, num_qubits):
                if j == i:
                    ref_circuit.ry(self.__affinity[i][i], i)
                    continue

                ref_circuit.append(RZZGate(self.__affinity[i][j]), [i, j])

    def __constraint_layer(self, var_circuit):
        # Define multi-control R_X gate for 2 * (Num qubits per column/row - 1) controls
        MCRXGate = RXGate(self.__beta).control(
            2 * (int(np.sqrt(self.__num_qubits)) - 1), ctrl_state=0
        )

        x_matrix = np.array([i for i in range(self.__num_qubits)]).reshape(
            int(np.sqrt(self.__num_qubits)), int(np.sqrt(self.__num_qubits))
        )

        # Loop over all the qubits and create MCRX Gates with neighbours
        for i in range(int(np.sqrt(self.__num_qubits))):
            for j in range(int(np.sqrt(self.__num_qubits))):
                controls = self.__get_neighbours(x_matrix, i, j) + [x_matrix[i][j]]
                var_circuit.append(MCRXGate, controls)

        var_circuit.barrier()

    def __perceptron_layer(self, var_circuit):
        for qubit in range(self.__num_qubits):
            var_circuit.append(RZGate(self.__theta1), [qubit])
            var_circuit.append(RYGate(self.__theta2), [qubit])

        var_circuit.barrier()

    def __pooling_layer(self, var_circuit):
        for qubit in range(self.__num_qubits):
            var_circuit.append(RYGate(self.__alpha), [qubit])

    # Calculates the binary cross-entropy loss
    def cost_function(self):
        X = list(self.__expected_result.flatten())
        Y = list(self.__y_matrix().flatten())
        for i in range(len(X)):
            self.__loss += -X[i] * np.log(Y[i]) + (1 - X[i]) * np.log(1 - Y[i])

        return self.__loss

    def __sinkhorn_normalization(
        self, matrix, epsilon=1e-3, max_iters=100, constraint_epsilon=1e-9
    ):
        assert matrix.shape[0] == matrix.shape[1]

        u = np.ones(matrix.shape[0])
        v = np.ones(matrix.shape[0])

        # Perform Sinkhorn iterations
        for _ in range(max_iters):
            u_new = 1 / (np.dot(matrix, v) + epsilon)
            v_new = 1 / (np.dot(matrix.T, u_new) + epsilon)

            # Check convergence
            if np.allclose(u_new, u) and np.allclose(v_new, v):
                break

            u = u_new
            v = v_new

        # Compute the Sinkhorn normalised matrix
        normalized_matrix = np.diag(u_new) @ matrix @ np.diag(v_new)

        # Apply constraint on zero terms
        normalized_matrix[normalized_matrix < constraint_epsilon] = constraint_epsilon

        # Renormalise the matrix to ensure it remains doubly stochastic
        row_sums = np.sum(normalized_matrix, axis=1)
        col_sums = np.sum(normalized_matrix, axis=0)
        normalized_matrix /= np.sqrt(np.outer(row_sums, col_sums))

        return normalized_matrix

    # Reshape the result matrix in the
    def __y_matrix(self):
        n = int(np.sqrt(self.__num_qubits))

        latex_string = (
            self.__statevect().data
        )  # Stores the statevector as a latex string5
        latex_list = list(latex_string)  # This object stores the statevector as a list

        "Extracting only the states of the qubits"
        a = latex_list.index("|")
        b = latex_list.index("\\")
        y = latex_list[a + 1 : b]
        y1 = np.array([int(i) for i in y])
        y1 = y1[::-1]
        Y1 = y1.reshape(n, n)  # Reshaping the measured states of qubits to a matrix

        # Apply Sinkhorn algorithm to converge to a doubly stochastic matrix with the matching constraint
        X_ds = self.__sinkhorn_normalization(Y1)
        return X_ds

    def __statevect(self):
        simulator = StatevectorSimulator()
        ansatz_trans = transpile(self.ansatz(), simulator)
        result = simulator.run(ansatz_trans).result()
        statevector = result.get_statevector()
        return statevector.draw("latex")

    # TODO: Rewrite this function
    def __get_neighbours(self, x_matrix, row_idx, col_idx):
        neighbors = []
        num_rows = len(x_matrix)
        num_cols = len(x_matrix[0])

        # Add neighbors from the same row
        for j in range(num_cols):
            if j != col_idx:  # Exclude the element itself
                neighbors.append(x_matrix[row_idx][j])

        # Add neighbors from the same column
        for i in range(num_rows):
            if i != row_idx:  # Exclude the element itself
                neighbors.append(x_matrix[i][col_idx])

        return neighbors


def generate_flow_matrix(num_nodes, edges):
    fmatrix = np.zeros((4, 4))

    for edge in edges:
        u, v = edge
        fmatrix[u][v] = 1
        fmatrix[v][u] = 1

    return fmatrix


def train_model(data_file):
    # Number of nodes
    nodes = 4

    # Load data from excel file
    training_data = ds.read_data(data_file, "Training data")

    # Edge connections
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    # Generate flow matrix
    flow = generate_flow_matrix(nodes, edges)

    beta = 1.39957953
    theta1 = 1.9026521
    theta2 = 2.22944427
    alpha = 5.56530919
    loss = 0

    solver = TSPSolver(
        flow,
        training_data["Distance Matrix"][0],
        beta,
        theta1,
        theta2,
        alpha,
        training_data["Expected Result"][0],
        loss,
    )

    def objective_function(params):
        print(params)

        # Set theta1 and theta2 values
        # Update the beta_para for MCRX gate
        # Run the quantum neural network and compute loss
        solver.update_variational_layers(params)
        solver.ansatz()
        # plt.show()
        obj_loss = solver.variational_layers_loss()
        return obj_loss

    def gradient_function(params):
        epsilon = 1e-6  # pert value

        initial_loss = objective_function(params)

        params_plus = params.copy() + epsilon

        loss_plus = objective_function(params_plus)

        gradient = (loss_plus - initial_loss) / epsilon

        return gradient

    solver.ansatz()

    params = np.array([beta, theta1, theta2, alpha])

    # Instantiate Adam optimizer
    adam_optimizer = adam(
        solver,
        maxiter=5,
        tol=1e-06,
        lr=0.1,
        beta_1=0.9,
        beta_2=0.99,
        noise_factor=1e-08,
        eps=1e-10,
        amsgrad=False,
        snapshot_dir=None,
    )

    # Run optimization on all the data
    adam_optimizer.minimize(
        objective_function, params, gradient_function, training_data
    )
    return params


if __name__ == "__main__":
    print(train_model("TSP-4.xlsx"))

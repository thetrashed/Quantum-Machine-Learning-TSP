import numpy as np
from qiskit_algorithms.optimizers import ADAM


class adam(ADAM):
    def __init__(
        self,
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
    ):
        super().__init__(1, tol, lr, beta_1, beta_2, noise_factor, eps, amsgrad, snapshot_dir)
        self.__maxiter = maxiter
        self.__solver = solver
        

    def minimize(self, fun, params, jac, dataset):
        i = 0
        while i < self.__maxiter:
            for data in dataset.itertuples():
                self.__solver.update_referential_layer(data[1], data[2])
                params = super().minimize(fun, params, jac).x
                print(params)

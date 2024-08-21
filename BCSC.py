import numpy as np
import cvxpy as cp

class Sinkhorn():

    def __init__(self, A, B, T, N):
        
        self.nx = A.shape[1] # state dimension
        self.nu = B.shape[1] # input dimension
        self.x0 = np.ones((self.nx, 1)) # initial conditions
        self.T = T
        self.N = N

        # Definition of the stacked system dynamics over the control horizon
        self.A_bl = np.kron(np.eye(T), A)
        self.B_bl = np.kron(np.eye(T), B)

        # Identity matrix and block-downshift operator
        self.I = np.eye(self.nx*T)
        self.Z = np.block([[np.zeros((self.nx, self.nx*(T-1))), np.zeros((self.nx, self.nx))], [np.eye(self.nx*(T-1)),  np.zeros((self.nx*(T-1), self.nx))]])

    # Implement golden search method to find the best value for lambda over a given interval.
    # Adapted from https://drlvk.github.io/nm/section-golden-section.html
    # For a reference https://en.wikipedia.org/wiki/Golden-section_search#Iterative_algorithm
    def solve_golden_search(self, lam_min, lam_max, rho, eps, data, tol = 1e-1):

        gr = .5*(3-np.sqrt(5))
        a = lam_min
        b = lam_max
        interval = b-a
        c = lam_min + gr*interval
        d = lam_max - gr*interval
        fc = self.solve_SDP(c, rho, eps, data)
        fd = self.solve_SDP(d, rho, eps, data)

        while np.abs(b-a) >= tol:
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = a + gr*(b-a)
                fc = self.solve_SDP(c, rho, eps, data) 
            else:
                a = c
                c = d
                fc = fd
                d = b - gr*(b-a)
                fd = self.solve_SDP(d, rho, eps, data)
        return b
    
    # Solve the optimization problem from Sinkhorn control with fixed value of lambda (dual variable)
    def solve_SDP(self, lam, rho, eps, data):

        # Define the decision variables of the optimization problem
        Phi_x = cp.Variable((self.nx*self.T, self.nx*self.T))
        Phi_u = cp.Variable((self.nu*self.T, self.nx*self.T))  

        self.Phi = cp.vstack([Phi_x, Phi_u])
        s = cp.Variable(self.N)
        z = cp.Variable(self.N)
        Q = cp.Variable((self.nx*self.T,self.nx*self.T), PSD = True)
        P = cp.Variable((self.nx*self.T,self.nx*self.T), PSD = True)

        # define the objective function
        obj = lam*rho + cp.sum(s)/s.size

        constraints = [P == lam*np.eye(self.nx*self.T)-Q]

        # Impose the achievability constraints
        constraints += [(self.I - self.Z @ self.A_bl) @ Phi_x - self.Z @ self.B_bl @ Phi_u == self.I]

        # Impose the causal sparsities on the closed loop responses
        for i in range(self.T-1):
            for j in range(i+1, self.T): # Set j from i+2 for non-strictly causal controller (first element in w is x0)
                constraints += [Phi_x[i*self.nx:(i+1)*self.nx, j*self.nx:(j+1)*self.nx] == np.zeros((self.nx, self.nx))]
                constraints += [Phi_u[i*self.nu:(i+1)*self.nu, j*self.nx:(j+1)*self.nx] == np.zeros((self.nu, self.nx))]

        for i in range(len(data)):
            # Get the i-th datapoint
            xi_hat = data[i]  
            # First inequality constraint
            inequality = eps*lam*np.log(lam)*.5*self.nx + z[i] - lam*eps*.5*cp.log_det(P)
            constraints += [s[i] >= inequality]

            # First LMI constraint
            tmp1 = cp.hstack([P, lam*xi_hat])
            tmp2 = cp.hstack([lam*xi_hat.T, z[i] + lam*xi_hat.T@xi_hat])
            LMI = cp.vstack([tmp1, tmp2])
            constraints += [LMI >> 0]

            # Second LMI constraint
            tmp3 = cp.hstack([Q, self.Phi.T])
            tmp4 = cp.hstack([self.Phi, np.eye((self.nu+self.nx)*self.T)])
            LMI2 = cp.vstack([tmp3, tmp4])
            constraints += [LMI2 >> 0]

        myprob = cp.Problem(cp.Minimize(obj), constraints)

        if not myprob.is_dcp():
             print("Error: the problem is not DCP.")

        # Solve the problem using mosek
        print('=====================================')
        print("Solving the optimization problem...")
        print('=====================================')
        res = myprob.solve(solver='MOSEK', verbose=True)
        print('Value of lambda: ', lam, 'Result: ', res)
        return res
    
    def LQG(self):

        # Define the decision variables of the optimization problem
        Phi_x = cp.Variable((self.nx*self.T, self.nx*self.T))
        Phi_u = cp.Variable((self.nu*self.T, self.nx*self.T)) 
    
        # Define the objective function
        obj = cp.norm(0.3*cp.vstack([Phi_x, Phi_u]), 'fro')

        # Impose the achievability constraints
        constraints = [(self.I - self.Z @ self.A_bl) @ Phi_x - self.Z @ self.B_bl @ Phi_u == self.I]

        # Impose the causal sparsities on the closed loop responses
        for i in range(self.T-1):
            for j in range(i+1, self.T): # Set j from i+2 for non-strictly causal controller (first element in w is x0)
                constraints += [Phi_x[i*self.nx:(i+1)*self.nx, j*self.nx:(j+1)*self.nx] == np.zeros((self.nx, self.nx))]
                constraints += [Phi_u[i*self.nu:(i+1)*self.nu, j*self.nx:(j+1)*self.nx] == np.zeros((self.nu, self.nx))]

        myprob = cp.Problem(cp.Minimize(obj), constraints)

        if not myprob.is_dcp():
            print("Error: the problem is not DCP.")
        # Solve the problem using mosek and bcd for multiconvex programming
        print('Solving the problem...')
        res = myprob.solve(solver='MOSEK')
        print('problem solved')
        return res**2
    
def main():

    np.random.seed(0)

    A = np.array([[0.7, 0.2, 0], [0.3, 0.7, -0.1], [0, -0.2, 0.8]])
    B = np.array([[1, 0.2], [2, 0.3], [1.5, 0.5]])
    T = 15 # control horizon

    d = A.shape[1]
    N = 10 # number of datapoints
    data_points = []
    for i in range(N):
        # Gaussian samples
        random_vec = np.random.default_rng().normal(0, .3, size=(d*T, 1))
        # Uniform (-1, 1) samples
        # random_vec = .15*np.random.default_rng().random(size=(d*T, 1))
        data_points.append(random_vec)

    eps = 1e-1  # regularization parameter
    rho = 1e-1
    rho_bar = rho + eps*d*.5*(np.log(eps*np.pi)) # Sinkhorn radius
    lam_min = 0
    lam_max = 50

    controller = Sinkhorn(A, B, T, N)
    controller.solve_golden_search(lam_min, lam_max, rho_bar**2, eps, data_points)  
    res_LQG = controller.LQG()
    print(np.linalg.norm(controller.Phi.value, 'fro'), res_LQG)
    return

if __name__ == "__main__":
    main()

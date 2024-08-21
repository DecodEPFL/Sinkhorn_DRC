import numpy as np
import cvxpy as cp
from collections import deque 

class Sinkhorn_controller():

    def __init__(self, A, B, T, N, len_buffer):
        
        # self.sys = ct.StateSpace(A, B, C, 0)
        self.nx = A.shape[1] # state dimension
        self.nu = B.shape[1] # input dimension
        self.x0 = np.ones((self.nx, 1)) # initial conditions
        self.T = T
        # Definition of the stacked system dynamics over the control horizon

        self.A_bl = np.kron(np.eye(T), A)
        self.B_bl = np.kron(np.eye(T), B)

        # Identity matrix and block-downshift operator
        self.I = np.eye(self.nx*T)
        self.Z = np.block([[np.zeros((self.nx, self.nx*(T-1))), np.zeros((self.nx, self.nx))], [np.eye(self.nx*(T-1)),  np.zeros((self.nx*(T-1), self.nx))]])

        # Define the decision variables of the optimization problem
        Phi_x = cp.Variable((self.nx*self.T, self.nx*self.T))
        Phi_u = cp.Variable((self.nu*self.T, self.nx*self.T))  

        self.Phi = cp.vstack([Phi_x, Phi_u])
        lam = cp.Variable((), nonneg=True)
        s = cp.Variable(N)
        z = cp.Variable(N)
        Q = cp.Variable((self.nx*self.T,self.nx*self.T), PSD = True)
        P = cp.Variable((self.nx*self.T,self.nx*self.T), PSD = True)
        self.var = {'Phi_x': Phi_x, 
                    'Phi_u': Phi_u,
                    'lam': lam, 
                    's': s, 
                    'z': z, 
                    'Q': Q, 
                    'P': P}
        
        # Initiliaze all variables 
        for _, var in self.var.items():
            self.init(var)

        # create buffers to store past values of optimization variables
        self.buf_lambda = deque(len_buffer*[self.var['lam'].value])
        self.buf_s = deque(len_buffer*[self.var['s'].value])
        self.buf_phix = deque(len_buffer*[self.var['Phi_x'].value])
        self.buf_phiu = deque(len_buffer*[self.var['Phi_u'].value])
        self.buf_P = deque(len_buffer*[self.var['P'].value])
        self.buf_Q = deque(len_buffer*[self.var['Q'].value])
        self.buf_z = deque(len_buffer*[self.var['z'].value])

    # solve the problem with block coordinate descent method
    def optimization_solve(self, max_it, precision, eps, rad, datas):
            
        # iterate the bidirectional method
        it = 0
        while it < max_it:

            # Find the optimal variables fixed the value of lambda
            Phi_x_s, Phi_u_s, s_s, z_s, P_s, Q_s = self.solve_SDP(self.buf_lambda[-1], rad, eps, datas)
            print(s_s)
            self.update_buffers(Phi_x_s, Phi_u_s, s_s, z_s, P_s, Q_s)

            # Find the best lambda fixed all the other variables to the previous iteration value
            lambda_s = self.solve_lambda(self.buf_s[-1], self.buf_z[-1], self.buf_P[-1], self.buf_Q[-1], rad, eps, datas)

            self.update_lam_buffer(lambda_s)

            it += 1

    # Initialization of the variables with standard normal entries
    def init(self, var):
        if var.sign == "NONNEGATIVE":
            var.value = np.abs(np.random.standard_normal(var.shape))
        elif var.attributes['PSD'] == True:
            dummy = np.asmatrix(np.random.standard_normal(var.shape))
            matrix = dummy.T*dummy
            var.value = np.asarray(matrix)
        elif var.sign == "NONPOSITIVE":
            var.value = -np.abs(np.random.standard_normal(var.shape))
        else:
            var.value = np.random.standard_normal(var.shape)

    # Update the value of current optimal lambda
    def update_lam_buffer(self, val):
            self.buf_lambda.append(val)
            self.buf_lambda.popleft()    

    # Update the value of current optimal variables 
    def update_buffers(self, phi_x, phi_u, s, z, P, Q):
        self.buf_phix.append(phi_x)
        self.buf_phix.popleft()
        self.buf_phiu.append(phi_u)
        self.buf_phiu.popleft()
        self.buf_s.append(s)
        self.buf_s.popleft()
        self.buf_z.append(z)
        self.buf_z.popleft()
        self.buf_P.append(P)
        self.buf_P.popleft()
        self.buf_Q.append(Q)
        self.buf_Q.popleft()

    # Fixed lambda the optimization program is an SDP
    def solve_SDP(self, lam, rho, eps, datas):

        # define the objective function
        obj = lam*rho + cp.norm(self.var['s'], 1)/self.var['s'].size

        constraints = [self.var['P'] == lam*np.eye(self.nx*self.T)-self.var['Q']]

        # Impose the achievability constraints
        constraints += [(self.I - self.Z @ self.A_bl) @ self.var['Phi_x'] - self.Z @ self.B_bl @ self.var['Phi_u'] == self.I]

        # Impose the causal sparsities on the closed loop responses
        for i in range(self.T-1):
            for j in range(i+1, self.T): # Set j from i+2 for non-strictly causal controller (first element in w is x0)
                constraints += [self.var['Phi_x'][i*self.nx:(i+1)*self.nx, j*self.nx:(j+1)*self.nx] == np.zeros((self.nx, self.nx))]
                constraints += [self.var['Phi_u'][i*self.nu:(i+1)*self.nu, j*self.nx:(j+1)*self.nx] == np.zeros((self.nu, self.nx))]

        for i in range(len(datas)):
            # Get the i-th datapoint
            xi_hat = datas[i]  
            # First inequality constraint
            inequality = eps*lam*np.log(lam)*.5*self.nx + self.var['z'][i] - lam*eps*.5*cp.log_det(self.var['P'])
            constraints += [self.var['s'][i] >= inequality]

            # First LMI constraint
            tmp1 = cp.hstack([self.var['P'], lam*xi_hat])
            tmp2 = cp.hstack([lam*xi_hat.T, self.var['z'][i] + lam*xi_hat.T@xi_hat])
            LMI = cp.vstack([tmp1, tmp2])
            constraints += [LMI >> 0]

            # Second LMI constraint
            tmp3 = cp.hstack([self.var['Q'], self.Phi.T])
            tmp4 = cp.hstack([self.Phi, np.eye((self.nu+self.nx)*self.T)])
            LMI2 = cp.vstack([tmp3, tmp4])
            constraints += [LMI2 >> 0]

        myprob = cp.Problem(cp.Minimize(obj), constraints)

        if not myprob.is_dcp():
             print("Error: the problem is not DCP.")

        # Solve the problem using mosek
        res = myprob.solve(solver='MOSEK')

        return self.var['Phi_x'].value, self.var['Phi_u'].value, self.var['s'].value, self.var['z'].value, self.var['P'].value, self.var['Q'].value
    
    # Solve in the direction of lambda
    def solve_lambda(self, s, z, P, Q, rho, eps, datas):

        # define the objective function
        obj = self.var['lam']*rho + np.linalg.norm(s, 1)/s.size

        constraints = [P == self.var['lam']*np.eye(self.nx*self.T)-Q]

        for i in range(len(datas)):
            # Get the i-th datapoint
            xi_hat = datas[i]  
            # First inequality constraint
            inequality = eps*.5*self.nx*(cp.kl_div(self.var['lam'], 1) - 1 + self.var['lam']) + z[i] - self.var['lam']*eps*.5*np.linalg.slogdet(P)[1]
            constraints += [s[i] >= inequality]

            # LMI constraint
            tmp1 = cp.hstack([P, self.var['lam']*xi_hat])
            tmp2 = cp.hstack([self.var['lam']*xi_hat.T, z[i] + self.var['lam']*xi_hat.T@xi_hat])
            LMI = cp.vstack([tmp1, tmp2])
            constraints += [LMI >> 0]

        myprob = cp.Problem(cp.Minimize(obj), constraints)

        if not myprob.is_dcp():
             print("Error: the problem is not DCP.")

        # Solve the problem using mosek
        res = myprob.solve(solver='MOSEK')

        return self.var['lam'].value
    
def main():

    np.random.seed(0)

    A = np.array([[0.7, 0.2, 0], [0.3, 0.7, -0.1], [0, -0.2, 0.8]])
    B = np.array([[1, 0.2], [2, 0.3], [1.5, 0.5]])
    T = 10 # control horizon

    d = A.shape[1]
    N = 10
    len_buffer = 10 # memory of past optimization variables' values
    data_points = []
    for i in range(N):
        # Gaussian samples
        random_vec = np.random.default_rng().standard_normal(size=(d*T, 1))

        # Uniform (-1, 1) samples
        # random_vec = 2*np.random.default_rng().random(size=(d*T, 1))-1
        data_points.append(random_vec)

    eps = 1e-1  # regularization parameter
    rho = 0.1
    rho_bar = rho + eps*d*.5*(np.log(eps*np.pi)) # Sinkhorn radius
    max_it = 100 # max number of iterations

    controller = Sinkhorn_controller(A, B, T, N, len_buffer)
    controller.optimization_solve(max_it, 0., eps, rho_bar**2, data_points)
    return

if __name__ == "__main__":
    main()
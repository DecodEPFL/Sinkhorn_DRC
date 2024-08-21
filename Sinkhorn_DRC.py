import numpy as np
import cvxpy as cp
import dmcp

class Sinkhorn_controller():

    def __init__(self, A, B, T):
        
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

    def causal_unconstrained_Sink(self, datas, radius, eps, max_it):
        # Define the decision variables of the optimization problem
        Phi_x = cp.Variable((self.nx*self.T, self.nx*self.T))
        Phi_u = cp.Variable((self.nu*self.T, self.nx*self.T))  

        Phi = cp.vstack([Phi_x, Phi_u])
        N = len(datas)
        lam = cp.Variable((), nonneg=True)
        s = cp.Variable(N)
        z = cp.Variable(N)
        Q = cp.Variable((self.nx*self.T,self.nx*self.T), PSD = True)
        P = cp.Variable((self.nx*self.T,self.nx*self.T), PSD = True)

        obj = lam*radius + cp.norm(s, 1)

        constraints = [lam >= cp.lambda_sum_largest(Q,1)]
        constraints += [P == np.eye(self.nx*self.T)-Q]

        # Impose the achievability constraints
        constraints += [(self.I - self.Z @ self.A_bl) @ Phi_x - self.Z @ self.B_bl @ Phi_u == self.I]

        # Impose the causal sparsities on the closed loop responses
        for i in range(self.T-1):
            for j in range(i+1, self.T): # Set j from i+2 for non-strictly causal controller (first element in w is x0)
                constraints += [Phi_x[i*self.nx:(i+1)*self.nx, j*self.nx:(j+1)*self.nx] == np.zeros((self.nx, self.nx))]
                constraints += [Phi_u[i*self.nu:(i+1)*self.nu, j*self.nx:(j+1)*self.nx] == np.zeros((self.nu, self.nx))]

        for i in range(len(datas)):
            xi_hat = datas[i]
            inequality = eps*(cp.kl_div(lam, 1) - 1 + lam)*.5*self.nx + z[i] - lam*eps*.5*cp.log_det(P)
            constraints += [s[i] >= inequality]
            tmp1 = cp.hstack([P, lam*xi_hat])
            tmp2 = cp.hstack([lam*cp.transpose(xi_hat), z[i] + lam*xi_hat.T@xi_hat])
            LMI = cp.vstack([tmp1, tmp2])
            constraints += [LMI >> 0]
            tmp3 = cp.hstack([Q, Phi.T])
            tmp4 = cp.hstack([Phi, np.eye((self.nu+self.nx)*self.T)])
            LMI2 = cp.vstack([tmp3, tmp4])
            constraints += [LMI2 >> 0]

        myprob = cp.Problem(cp.Minimize(obj), constraints)
        if not dmcp.is_dmcp(myprob):
            print("Error: the problem is not DMCP.")
        # Solve the problem using mosek and bcd for multiconvex programming
        print('Solving the problem...')
        res = myprob.solve(method = 'bcd', solver = 'MOSEK', max_iter = max_it)
        # Extract the closed-loop responses corresponding to a unconstrained causal 
        # linear controller that is optimal
        Phi_x = Phi_x.value 
        Phi_u = Phi_u.value
        print('problem solved')
        return [Phi_x, Phi_u, res]
    
    def causal_unconstrained_Wass(self, datas, radius):
        # Define the decision variables of the optimization problem
        Phi_x = cp.Variable((self.nx*self.T, self.nx*self.T))
        Phi_u = cp.Variable((self.nu*self.T, self.nx*self.T))  

        Phi = cp.vstack([Phi_x, Phi_u])
        N = len(datas)
        lam = cp.Variable((), nonneg=True)
        s = cp.Variable(N)
        Q = cp.Variable((self.nx*self.T,self.nx*self.T), PSD = True)

        obj = lam*radius + cp.norm(s, 1)

        constraints = [lam >= 0]

        # Impose the achievability constraints
        constraints += [(self.I - self.Z @ self.A_bl) @ Phi_x - self.Z @ self.B_bl @ Phi_u == self.I]

        # Impose the causal sparsities on the closed loop responses
        for i in range(self.T-1):
            for j in range(i+1, self.T): # Set j from i+2 for non-strictly causal controller (first element in w is x0)
                constraints += [Phi_x[i*self.nx:(i+1)*self.nx, j*self.nx:(j+1)*self.nx] == np.zeros((self.nx, self.nx))]
                constraints += [Phi_u[i*self.nu:(i+1)*self.nu, j*self.nx:(j+1)*self.nx] == np.zeros((self.nu, self.nx))]

        for i in range(len(datas)):
            xi_hat = datas[i]
            constraints += [s[i] >= 0]
            tmp1 = cp.hstack([lam*np.eye(self.nx*self.T) - Q, lam*xi_hat])
            tmp2 = cp.hstack([lam*cp.transpose(xi_hat), s[i] + lam*xi_hat.T@xi_hat])
            LMI = cp.vstack([tmp1, tmp2])
            constraints += [LMI >> 0]
            tmp3 = cp.hstack([Q, Phi.T])
            tmp4 = cp.hstack([Phi, np.eye((self.nu+self.nx)*self.T)])
            LMI2 = cp.vstack([tmp3, tmp4])
            constraints += [LMI2 >> 0]

        myprob = cp.Problem(cp.Minimize(obj), constraints)
        if not myprob.is_dcp():
            print("Error: the problem is not DCP.")
        # Solve the problem using mosek and bcd for multiconvex programming
        print('Solving the problem...')
        res = myprob.solve(solver='MOSEK')
        # Extract the closed-loop responses corresponding to a unconstrained causal 
        # linear controller that is optimal
        Phi_x = Phi_x.value 
        Phi_u = Phi_u.value
        print('problem solved')
        return [Phi_x, Phi_u, res]
    
    def LQG(self):

        # Define the decision variables of the optimization problem
        Phi_x = cp.Variable((self.nx*self.T, self.nx*self.T))
        Phi_u = cp.Variable((self.nu*self.T, self.nx*self.T)) 
    
        # Define the objective function
        obj = cp.norm(cp.vstack([Phi_x, Phi_u]), 'fro')

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
    T = 10 # control horizon

    d = A.shape[1]
    N = 10
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
    max_it = 300

    controller = Sinkhorn_controller(A, B, T)
    [Phi_x, Phi_u, obj] = controller.causal_unconstrained_Sink(data_points, rho_bar**2, eps, max_it)
    [Phi_x_W, Phi_u_W, obj_W] = controller.causal_unconstrained_Wass(data_points, rho**2)
    lqg_cost = controller.LQG()
    Sink_dro_cost = np.linalg.norm(np.vstack([Phi_x, Phi_u]), 'fro')
    Wass_dro_cost = np.linalg.norm(np.vstack([Phi_x_W, Phi_u_W]), 'fro')
    print(lqg_cost)
    print(Sink_dro_cost**2)
    print(Wass_dro_cost**2)
    print(obj)
    print(obj_W)
    return

if __name__ == "__main__":
    main()

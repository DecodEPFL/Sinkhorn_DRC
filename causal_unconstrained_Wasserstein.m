function [Phi_x, Phi_u, objective] = causal_unconstrained_Wasserstein(sys, sls, opt, rho)
%CAUSAL_UNCONSTRAINED computes an unconstrained causal linear control
    
    % Define the decision variables of the optimization problem
    Phi_x = sdpvar(sys.n*opt.T, sys.n*opt.T, 'full');
    Phi_u = sdpvar(sys.m*opt.T, sys.n*opt.T, 'full');
    Phi = [Phi_x; Phi_u];
    
    % Define the objective function
    s = sdpvar(opt.N, 1);
    gamma = sdpvar();
    Q = sdpvar(sys.n*opt.T, sys.n*opt.T, 'symmetric');

    constraints = [];

    % define the objective function
    objective = gamma*rho^2 + sum(s)/opt.N;

    % Impose the achievability constraints
    constraints = [constraints, (sls.I - sls.Z*sls.A)*Phi_x - sls.Z*sls.B*Phi_u == sls.I];
    % Impose the causal sparsities on the closed loop responses
    for i = 0:opt.T-2
        for j = i+1:opt.T-1 % Set j from i+2 for non-strictly causal controller (first element in w is x0)
            constraints = [constraints, Phi_x((1+i*sys.n):((i+1)*sys.n), (1+j*sys.n):((j+1)*sys.n)) == zeros(sys.n, sys.n)];
            constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.n):((j+1)*sys.n)) == zeros(sys.m, sys.n)];
        end
    end

    for i=1:opt.N
        % Get the i-th datapoint
        xi_hat = opt.data{i};
        constraints = [constraints, s(i) >=0];

        % First LMI constraint

        tmp1 = [gamma*eye(sys.n*opt.T)-Q, gamma*xi_hat]; 
        tmp2 = [gamma*xi_hat', s(i) + gamma*(xi_hat'*xi_hat)];
        LMI = [tmp1; tmp2];
        constraints = [constraints, LMI >= 0];

        % Second LMI constraint
        tmp3 = [Q, Phi'];
        tmp4 = [Phi, eye((sys.m+sys.n)*opt.T)];
        LMI2 = [tmp3; tmp4];
        constraints = [constraints, LMI2 >= 0];
    end
    
    % Solve the optimization problem
    fprintf('=====================================')
    fprintf("Solving the optimization problem...")
    fprintf('=====================================')
    options = sdpsettings('verbose', 2, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    % Extract the closed-loop responses corresponding to a unconstrained causal 
    % linear controller that is optimal
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    % Extract the cost incurred by an unconstrained causal linear controller
    objective = value(objective);

end
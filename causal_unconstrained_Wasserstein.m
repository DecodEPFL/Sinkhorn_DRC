function [Phi_x, Phi_u, objective] = causal_unconstrained_Wasserstein(sys, sls, opt, rho)
%CAUSAL_UNCONSTRAINED computes an unconstrained causal linear control
    
    % Define the decision variables of the optimization problem
    Phi_x = sdpvar(sys.d*opt.N, sys.d*opt.N, 'full');
    Phi_u = sdpvar(sys.m*opt.N, sys.d*opt.N, 'full');
    Phi = [Phi_x; Phi_u];
    
    s = sdpvar(opt.n, 1);
    gamma = sdpvar();
    Q = sdpvar(sys.d*opt.N, sys.d*opt.N, 'symmetric');

    constraints = [];

    % define the objective function
    objective = gamma*rho + sum(s)/opt.n;

    % Second LMI constraint
    D_half = sqrtm(opt.C);
    LMI2 = [Q, Phi'*D_half'; D_half*Phi, eye((sys.m+sys.d)*opt.N)];
    constraints = [constraints, LMI2 >= 0];

    % Impose the achievability constraints
    constraints = [constraints, (sls.I - sls.Z*sls.A)*Phi_x - sls.Z*sls.B*Phi_u == sls.E];
    % Impose the causal sparsities on the closed loop responses
    for i = 0:opt.N-2
        for j = i+1:opt.N-1 % Set j from i+2 for non-strictly causal controller (first element in w is x0)
            constraints = [constraints, Phi_x((1+i*sys.d):((i+1)*sys.d), (1+j*sys.d):((j+1)*sys.d)) == zeros(sys.d, sys.d)];
            constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.d):((j+1)*sys.d)) == zeros(sys.m, sys.d)];
        end
    end

    for i=1:opt.n
        % Get the i-th datapoint
        xi_hat = opt.data{i};

        % First LMI constraint
        LMI = [gamma*eye(sys.d*opt.N)-Q, gamma*xi_hat; gamma*xi_hat', s(i) + gamma*(xi_hat'*xi_hat)];
        constraints = [constraints, LMI >= 0];
    end
    
    % Solve the optimization problem
    fprintf('=====================================')
    fprintf("Solving the optimization problem...")
    fprintf('=====================================')
    options = sdpsettings('verbose', 2, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        disp(sol.problem)
        error('Something went wrong...');
    end
    
    % Extract the closed-loop responses corresponding to a unconstrained causal 
    % linear controller that is optimal
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    % Extract the cost incurred by an unconstrained causal linear controller
    objective = value(objective);

end
function [Phi_x, Phi_u, objective] = causal_unconstrained_Wasserstein_v2(sys, sls, opt, rho)
    %CAUSAL_UNCONSTRAINED computes the unconstrained Wasserstein linear
    %controller
    
    % Define the decision variables of the optimization problem
    Phi_u = sdpvar(sys.m*opt.N, sys.d + sys.p*(opt.N-1), 'full');
    Phi_x = (sls.I - sls.Z*sls.A) \ (sls.Z*sls.B*Phi_u + sls.E); % Phi_x is given as function of Phi_u
    Phi = [Phi_x; Phi_u];

    s = sdpvar(opt.n, 1);
    gamma = sdpvar();
    Q = sdpvar(sys.d + sys.p*(opt.N-1), sys.d + sys.p*(opt.N-1), 'symmetric');

    % define the objective function
    objective = gamma*rho + sum(s)/opt.n;

    % define the constraints
    constraints = [];
    
    % second LMI constraint
    D_half = sqrtm(opt.C);
    LMI2 = [Q, Phi'*D_half'; D_half*Phi, eye((sys.m+sys.d)*opt.N)];
    constraints = [constraints, LMI2 >= 0];

    % Impose the causal sparsities on the closed loop responses
    for i = 0:opt.N-2
        for j = i+1:opt.N-1 % Set j from i+2 for non-strictly causal controller (first element in w is x0)
            constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.p):((j+1)*sys.p)) == zeros(sys.m, sys.p)];
        end
    end
    
    % Loop for each datapoint 
    for i=1:opt.n
        % Get the i-th datapoint
        xi_hat = opt.data{i};

        % First LMI constraint
        LMI = [gamma*eye(sys.d + sys.p*(opt.N-1))-Q, gamma*xi_hat; gamma*xi_hat', s(i) + gamma*(xi_hat'*xi_hat)];
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
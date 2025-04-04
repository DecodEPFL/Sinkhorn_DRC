function [Phi_x, Phi_u, ret] = nominal_unconstrained(sys, sls, opt)
    % Compute the unconstrained linear controller using the empirical
    % center distribution

    % Define the decision variables of the optimization problem
    Phi_u = sdpvar(sys.m*opt.N, sys.d + sys.p*(opt.N-1), 'full');
    Phi_x = (sls.I - sls.Z*sls.A) \ (sls.Z*sls.B*Phi_u + sls.E); % Phi_x is given as function of Phi_u
    Phi = [Phi_x; Phi_u];

    % define the objective function
    objective = 0;
    for i=1:opt.n
        % Get the i-th datapoint
        xi_hat = opt.data{i};
        tmp = Phi*xi_hat;
        objective = objective + tmp'*opt.C*tmp;
    end

    constraints = [];

    % Impose the causal sparsities on the closed loop responses
    for i = 0:opt.N-2
        for j = i+1:opt.N-1 % Set j from i+2 for non-strictly causal controller (first element in w is x0)
            constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.p):((j+1)*sys.p)) == zeros(sys.m, sys.p)];
        end
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
    ret = value(objective)/opt.n;
end
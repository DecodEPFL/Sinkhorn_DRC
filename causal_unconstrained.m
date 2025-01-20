function [Phi_x, Phi_u, Q, objective] = causal_unconstrained(sys, sls, opt, lam, rho, eps)
%CAUSAL_UNCONSTRAINED computes an unconstrained causal linear control
    
    % Define the decision variables of the optimization problem
    m = sys.d*opt.N; % state dimension times length of horizon
    Phi_x = sdpvar(m, m, 'full');
    Phi_u = sdpvar(sys.m*opt.N, m, 'full');
    Phi = [Phi_x; Phi_u];

    % Define the objective function
    s = sdpvar(opt.n, 1);
    z = sdpvar(opt.n, 1);
    Q = sdpvar(m, m, 'symmetric');
    
    constraints = [];
    % define the objective function
    objective = lam*rho + sum(s)/opt.n;

    % Second LMI constraint
    D_half = sqrtm(opt.C);
    LMI2 = [Q, Phi'*D_half'; D_half*Phi, eye((sys.m+sys.d)*opt.N)];
    constraints = [constraints, LMI2 >= 0];

    % Impose the achievability constraints
    constraints = [constraints, (sls.I - sls.Z*sls.A)*Phi_x - sls.Z*sls.B*Phi_u == sls.I];
    % Impose the causal sparsities on the closed loop responses
    for i = 0:opt.N-2
        for j = i+1:opt.N-1 % Set j from i+2 for non-strictly causal controller (first element in w is x0)
            constraints = [constraints, Phi_x((1+i*sys.d):((i+1)*sys.d), (1+j*sys.d):((j+1)*sys.d)) == zeros(sys.d, sys.d)];
            constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.d):((j+1)*sys.d)) == zeros(sys.m, sys.d)];
        end
    end

    nonlin = eps * lam * .5 * m * (log(lam * eps * pi) - log(geomean(lam * eye(m) - Q)));

    for i=1:opt.n
        % Get the i-th datapoint
        wi_hat = opt.data{i};
        % First inequality constraint
        constraints = [constraints, s(i) >= z(i) + nonlin];

        % First LMI constraint
        LMI = [lam*eye(m)-Q, lam*wi_hat; lam*wi_hat', z(i) + lam*(wi_hat'*wi_hat)];
        constraints = [constraints, LMI >= 0];
    end
    
    % Solve the optimization problem
    fprintf('=====================================')
    fprintf("Solving the optimization problem...")
    fprintf('=====================================')
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    % options.MSK_DPAR_INTPNT_CO_TOL_NEAR_REL = 1e15;

    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        if sol.problem == 1
            objective = inf;
            return
        elseif sol.problem == 4
            objective = value(objective);
        else
            string = yalmiperror(sol.problem);
            error(string);
        end
    end

    Q = value(Q);
    if lam <= max(eigs(Q))
        objective = inf;
    end

    fprintf('Value of lambda: %s Result: %s\n', num2str(lam), num2str(value(objective)));
    
    % Extract the closed-loop responses corresponding to a unconstrained causal 
    % linear controller that is optimal
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);

    % s = value(s);
    % z = value(z);
    Q = value(Q);

    % Extract the cost incurred by an unconstrained causal linear controller
    objective = value(objective);

end
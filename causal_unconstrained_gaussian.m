function [Phi_x, Phi_u, Q, objective] = causal_unconstrained_gaussian(sys, sls, opt, lam, rho, eps)
%CAUSAL_UNCONSTRAINED computes an unconstrained causal linear control
    
    % Define the decision variables of the optimization problem
    m = sys.d + sys.p*(opt.N-1);
    Phi_u = sdpvar(sys.m*opt.N, m, 'full');
    Phi_x = (sls.I - sls.Z*sls.A) \ (sls.Z*sls.B*Phi_u + sls.E); % Phi_x is given as function of Phi_u
    Phi = [Phi_x; Phi_u];

    s = sdpvar(opt.n, 1);
    z = sdpvar(opt.n, 1);
    Q = sdpvar(m, m, 'symmetric');
    
    % define the objective function
    objective = lam*rho + sum(s)/opt.n;

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

    matr = lam .* (eye(m) + .5 * eps .* inv(opt.Sigma)) - Q;

    nonlin = eps * lam * .5 * m * (log(lam * eps * .5) - log(geomean(matr))) - eps * lam * .5 * logdet(opt.Sigma);

    for i=1:opt.n
        % Get the i-th datapoint
        wi_hat = opt.data{i};
        % First inequality constraint
        constraints = [constraints, s(i) >= z(i) + nonlin];

        % First LMI constraint
        tmp = lam * (wi_hat + eps * .5 .* opt.Sigma \ opt.m);
        LMI = [matr, tmp; tmp', z(i) + lam * norm(wi_hat, 2)^2 + eps * lam * .5 .* opt.m' * (opt.Sigma \ opt.m)];
        constraints = [constraints, LMI >= 0];
    end
    
    % Solve the optimization problem
    fprintf('=====================================')
    fprintf("Solving the optimization problem...")
    fprintf('=====================================')
    options = sdpsettings('verbose', 2, 'solver', 'mosek');
    % options.MSK_DPAR_INTPNT_CO_TOL_NEAR_REL = 1e15;

    sol = optimize(constraints, objective, options);
    
    % Q = value(Q);
    % if lam * (eye(m) + .5*eps./opt.Sigma) <= Q
    %     objective = inf;
    % end

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

    fprintf('Value of lambda: %s Result: %s\n', num2str(lam), num2str(value(objective)));
    
    % Extract the closed-loop responses corresponding to a unconstrained causal 
    % linear controller that is optimal
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    Q = value(Q);

    % Extract the cost incurred by an unconstrained causal linear controller
    objective = value(objective);

end
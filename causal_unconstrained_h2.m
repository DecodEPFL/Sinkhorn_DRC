function [Phi_x, Phi_u, objective] = causal_unconstrained_h2(sys, sls, opt, flag)
%CAUSAL_UNCONSTRAINED computes an unconstrained causal linear control
%policy that is optimal either in the H2 or in the Hinf sense
    
    % Define the decision variables of the optimization problem
    Phi_x = sdpvar(sys.d*opt.N, sys.d*opt.N, 'full');
    Phi_u = sdpvar(sys.m*opt.N, sys.d*opt.N, 'full');
    
    % Define the objective function
    if strcmp(flag, 'H2')
        objective = norm(sqrtm(opt.C)*[Phi_x; Phi_u]*sqrtm(opt.Sigma), 'fro');
    elseif strcmp(flag, 'Hinf')
        objective = norm(sqrtm(opt.C)*[Phi_x; Phi_u], 2);
    else
        error('Something went wrong...');
    end
   
    constraints = [];
    % Impose the achievability constraints
    constraints = [constraints, (sls.I - sls.Z*sls.A)*Phi_x - sls.Z*sls.B*Phi_u == sls.I];
    % Impose the causal sparsities on the closed loop responses
    for i = 0:opt.N-2
        for j = i+1:opt.N-1 % Set j from i+2 for non-strictly causal controller (first element in w is x0)
            constraints = [constraints, Phi_x((1+i*sys.d):((i+1)*sys.d), (1+j*sys.d):((j+1)*sys.d)) == zeros(sys.d, sys.d)];
            constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.d):((j+1)*sys.d)) == zeros(sys.m, sys.d)];
        end
    end
    
    % Solve the optimization problem
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    % Extract the closed-loop responses corresponding to a unconstrained causal 
    % linear controller that is optimal either in the H2 or in the Hinf sense
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    objective = value(objective)^2; % Extract the H2- or Hinf-optimal cost incurred by an unconstrained causal linear controller

end
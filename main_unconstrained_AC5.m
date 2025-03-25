close all; clearvars; clc;
rng(1234);             % Set random seed for reproducibility
%% Numerical validation with Aircraft model 4

%            A -- real (nx times nx) matrix
%            B -- real (nx times nu) matrix
%            C -- real (ny times nx) matrix
%           B1 -- real (nx times nw) matrix
%           C1 -- real (nz times nx) matrix
%          D11 -- real (nz times nw) matrix
%          D12 -- real (nz times nu) matrix
%          D21 -- real (ny times nw) matrix
%           nx -- dimension of the 'state'
%           nu -- dimension of the 'control'
%           ny -- dimension of the 'measurement'
%           nw -- dimension of the 'noise'
%           nz -- dimension of the 'regulated output'
%
%  Note: From the output data of COMPleib, one can define a control
%        system of the form
%
%        (d/dt)x(t) =  A*x(t)+ B1*w(t)+ B*u(t);  (x - state, u - control, w- noise)
%              z(t) = C1*x(t)+D11*w(t)+D12*u(t); (z - reg. output)
%              y(t) =  C*x(t)+D21*w(t).          (y - measurement)

rho = {1e-10, 1e-2, 1e-1}; % radius

sys.eps = {1e-15}; % Regularization parameter
epsilonbis = linspace(1e-8, 1, 200);

[A,B1,B,~,C,~,~,D21,nx,nw,nu,~,~] = COMPleib('AC4');

sysd = c2d(ss(A, [B B1], C, [[0;0] D21]), 0.01);

sys.d = nx;
sys.m = nu;
sys.p = nw;

% Definition of the parameters of the optimization problem
opt.Qt = eye(sys.d); % Stage cost: state weight matrix
opt.Rt = eye(sys.m); % Stage cost: input weight matrix

opt.N = 15; % Control horizon

mean_vector = zeros(nw, 1);
opt.m = 0.*ones((opt.N-1) * sys.p + sys.d, 1); % mean of the reference probability
opt.Sigma_t = [1e-1 0; 0 0.00001]; % standard deviation of the reference probability
opt.Sigma = blkdiag(0.5*eye(sys.d) ,kron(eye(opt.N-1), opt.Sigma_t)); 

opt.Q = kron(eye(opt.N), opt.Qt); % State cost matrix
opt.R = kron(eye(opt.N), opt.Rt); % Input cost matrix
opt.C = blkdiag(opt.Q, opt.R); % Cost matrix

%% Generation of noise samples

opt.n = 20; % Number of noise datapoints

% Preallocate the cell array to store trajectories
noise_trajectories = cell(opt.n, 1);

for i = 1:opt.n
    % Generate the initial condition (d-dimensional vector)
    initial_condition = 0.5*randn(sys.d, 1);
    
    % Generate noise samples ((N-1) x p matrix)
    noise_samples = mvnrnd(mean_vector, opt.Sigma_t, opt.N-1);
    
    % Concatenate initial condition and noise samples
    trajectory = [initial_condition; noise_samples(:)];
    
    % Store the trajectory
    noise_trajectories{i} = trajectory;
end

opt.data = noise_trajectories;

%% Definition of the stacked system dynamics over the control horizon
sls.A = kron(eye(opt.N), sysd.A);
sls.B = kron(eye(opt.N), sysd.B(:, 1:nu));
sls.E = blkdiag(eye(sys.d), kron(eye(opt.N-1), sysd.B(:, nu+1:nw+nu)));

% Identity matrix and block-downshift operator
sls.I = eye(sys.d*opt.N);
sls.Z = [zeros(sys.d, sys.d*(opt.N-1)) zeros(sys.d, sys.d); eye(sys.d*(opt.N-1)) zeros(sys.d*(opt.N-1), sys.d)];

%% Radius feasibility check
norm_samples = 0;

% compute the norm of the samples
for i = 1:length(opt.data)
    norm_samples = norm_samples + norm(opt.data{i}, 2)^2;
end

% Preallocate result for rhobis
rhobis = zeros(size(epsilonbis));  % Same size as epsilonbis
d = (opt.N-1) * sys.p + sys.d;

% Loop over each epsilon value
for j = 1:length(epsilonbis)
    epsilon = epsilonbis(j);  % Select one epsilon value at a time

    % Precompute inverse term (I + (epsilon / 2) * Sigma^-1)
    inv_term = eye((opt.N-1) * sys.p + sys.d) + (epsilon / 2) .* inv(opt.Sigma);

    % Initialize part4 sum for this epsilon
    part4 = 0;

    % Sum over k
    for k = 1:opt.n
        xi_term = opt.data{k} + (epsilon / 2) .* (opt.Sigma \ opt.m);
        
        % Compute sum_term for this epsilon
        sum_term = xi_term' * (inv_term \ xi_term);
        part4 = part4 + sum_term;
    end

    % Compute part1, part2, part3 for this epsilon
    part1 = -(epsilon * d / 2) * log(epsilon / 2);
    part2 = (epsilon / 2) * logdet(opt.Sigma + (epsilon / 2) * eye((opt.N-1) * sys.p + sys.d));
    part3 = (epsilon / 2) * (opt.m' * (opt.Sigma \ opt.m));

    % Final part4 term (normalized)
    part4 = (-part4 + norm_samples)/opt.n;

    % Compute final result for this epsilon
    rhobis(j) = part1 + part2 + part3 + part4;
end

% Plot the results
figure()
semilogx(epsilonbis, rhobis)
grid on
xlabel('epsilon')
ylabel('rhobis')
title('rhobis vs epsilon')

for i=1:length(rho)
    for j=1:length(sys.eps)

        epsilon = sys.eps{j};
        Sigma = opt.Sigma;
        m = opt.m;

        % Initialize the fourth part (sum over k)
        part4 = 0;
        
        % Inverse term I + (ε / 2) * Sigma^-1
        inv_term = eye(d) + (epsilon / 2) .* inv(Sigma);

        for k=1:opt.n
            xi_term = opt.data{k} + (epsilon / 2) * (Sigma \ m);
            
            % Compute the expression inside the sum
            sum_term = opt.data{k}' * (inv_term \ opt.data{k});

            % Accumulate the sum
            part4 = part4 + sum_term;
        end

        part1 = -(epsilon * d / 2) * log(epsilon / 2);

        % Second part: - (ε / 2) * log(det(Sigma + (ε / 2) * I))
        part2 = (epsilon / 2) * logdet(Sigma + (epsilon / 2) * eye(d));
    
        % Third part: - (ε / 2) * norm(m)^2 with Sigma^-1
        part3 = (epsilon / 2) * (m' * (Sigma \ m));
    
        % Final sum term
        part4 = (-part4 + norm_samples) / opt.n;
    
        % Final result
        result = part1 + part2 + part3 + part4;

        if result > rho{i}
            error('The radius value does not guarantee feasibility ...');
        end
    end
end

%% Wasserstein DRControl unconstrained
cost_W = {};
Phi_W = {};
for i=1:length(rho)
    [Phi_x, Phi_u, ret] = causal_unconstrained_Wasserstein_v2(sys, sls, opt, rho{i});
    cost_W{i} = ret;
    Phi_W{i} = [Phi_x; Phi_u];
end

%% Nominal controller
[Phi_x_nominal, Phi_u_nominal, cost_nominal] = nominal_unconstrained(sys, sls, opt);

%% H2 controller
[Phi_x_h2, Phi_u_h2, cost_h2] = causal_unconstrained_h2(sys, sls, opt, 'H2');

%% Sinkhorn DR Finite Horizon Control
Phi = {};
cost_S = {};
lambdas = {};
Qs = {};
tol = 1e-1; % Golden search tolerance 
lam_min = 0; % Starting value for lambda in the golden search
lam_max = 10000; % Final value for lambda

for i=1:length(sys.eps)
    eps = sys.eps{i};
    for j=1:length(rho)
        tic;
        [ret, lambda_opt, Q_opt, Phi_x, Phi_u] = goldenSearch(rho{j}, eps, tol, lam_min, lam_max, sys, sls, opt, true);
        elapsed = toc;
        disp(['Elapsed time: ', num2str(elapsed), ' seconds']);
        Phi{i,j} = [Phi_x; Phi_u];
        lambdas{i, j} = lambda_opt;
        Qs{i, j} = Q_opt;
        cost_S{i, j} = ret;
    end
end
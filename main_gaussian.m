close all; clearvars; clc;
% addpath('./functions') % Add path to the folder with auxiliary functions
rng(1234);             % Set random seed for reproducibility

%% Definition of the underlying discrete-time LTI system
rho = {.1}; % Sinkhorn radius

% sys.eps = num2cell(logspace(-5, 0, 5)); % Regularization parameter
sys.eps = {0.001, 0.01, 0.1};
epsilonbis = linspace(1e-5, 100, 200);
% System dynamics
sys.A = [1 1; -1 0];
sys.B = [0;1];

sys.d = size(sys.A, 1);   % Order of the system: state dimension
sys.m = size(sys.B, 2);   % Number of input channels
sys.p = sys.d;
sys.E = eye(sys.d);

% Definition of the parameters of the optimization problem
opt.Qt = eye(sys.d); % Stage cost: state weight matrix
opt.Rt = eye(sys.m); % Stage cost: input weight matrix

opt.N = 2; % Control horizon

opt.Q = kron(eye(opt.N), opt.Qt); % State cost matrix
opt.R = kron(eye(opt.N), opt.Rt); % Input cost matrix
opt.C = blkdiag(opt.Q, opt.R); % Cost matrix

opt.m = zeros(sys.d*opt.N, 1); 
opt.Sigma_t = 0.1*eye(sys.d);
opt.Sigma = kron(eye(opt.N), opt.Sigma_t);

%% Generation of noise samples

opt.n = 4; % Number of noise datapoints
mean_vector = zeros(sys.d, 1); % Zero mean

% Generate random noise datapoints
data_points = cell(opt.n, 1); % Initialize an empty cell array
for i = 1:opt.n
    % Gaussian samples
    trajectory = mvnrnd(mean_vector, opt.Sigma_t, opt.N); % Size: N x d
    data_points{i} = reshape(trajectory', [], 1);
end
opt.data = data_points;


%% Radius feasibility check
norm_samples = 0;

% compute the norm of the samples
for i = 1:length(opt.data)
    norm_samples = norm_samples + norm(opt.data{i}, 2)^2;
end

% Preallocate result for rhobis
rhobis = zeros(size(epsilonbis));  % Same size as epsilonbis
d = opt.N * sys.d;

% Loop over each epsilon value
for j = 1:length(epsilonbis)
    epsilon = epsilonbis(j);  % Select one epsilon value at a time

    % Precompute inverse term (I + (epsilon / 2) * Sigma^-1)
    inv_term = eye(opt.N * sys.d) + (epsilon / 2) .* inv(opt.Sigma);

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
    part2 = (epsilon / 2) * logdet(opt.Sigma + (epsilon / 2) * eye(opt.N * sys.d));
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


%% Definition of the stacked system dynamics over the control horizon
sls.A = kron(eye(opt.N), sys.A);
sls.B = kron(eye(opt.N), sys.B);
sls.E = blkdiag(eye(sys.d), kron(eye(opt.N-1), sys.E));

% Identity matrix and block-downshift operator
sls.I = eye(sys.d*opt.N);
sls.Z = [zeros(sys.d, sys.d*(opt.N-1)) zeros(sys.d, sys.d); eye(sys.d*(opt.N-1)) zeros(sys.d*(opt.N-1), sys.d)];

%% Sinkhorn DR Finite Horizon Control
Phi = {};
cost = {};
lambdas = {};
Qs = {};
tol = 1e-3; % Golden search tolerance 
lam_min = 0; % Starting value for lambda in the golden search
lam_max = 100; % Final value for lambda

for i=1:length(sys.eps)
    eps = sys.eps{i};
    for j=1:length(rho)
        [ret, lambda_opt, Q_opt, Phi_x, Phi_u] = goldenSearch(rho{j}, eps, tol, lam_min, lam_max, sys, sls, opt, true);
        Phi{i,j}.x = Phi_x;
        Phi{i, j}.u = Phi_u;
        lambdas{i, j} = lambda_opt;
        Qs{i, j} = Q_opt;
        cost{i, j} = ret;
    end
end

%% Nominal controller
[Phi_x_nominal, Phi_u_nominal, cost_nominal] = nominal_unconstrained(sys, sls, opt);
Phi_nominal.x = Phi_x_nominal;
Phi_nominal.u = Phi_u_nominal;

%% H2 controller
[Phi_x_h2, Phi_u_h2, cost_h2] = causal_unconstrained_h2(sys, sls, opt, 'H2');

%% Wasserstein DRControl unconstrained
cost_W = {}; cost2 = {};
Phi_W = {}; Phi_W2 = {};
for i=1:length(rho)
    tic;
    [Phi_x, Phi_u, ret] = causal_unconstrained_Wasserstein(sys, sls, opt, rho{i});
    elapsed = toc;
    disp(['Elapsed time: ', num2str(elapsed), ' seconds']);
    tic;
    [Phi_x2, Phi_u2, ret2] = causal_unconstrained_Wasserstein_v2(sys, sls, opt, rho{i});
    elapsed = toc;
    disp(['Elapsed time: ', num2str(elapsed), ' seconds']);
    cost_W{i} = ret;
    cost2{i} = ret2;
    Phi_W{i}.x = Phi_x;
    Phi_W{i}.u = Phi_u;
    Phi_W2{i}.x = Phi_x2;
    Phi_W2{i}.u = Phi_u2;
end

%% Compute the performances given the true distribution
D_half = sqrtm(opt.C);
% Compute the Sinkhorn cost
for i=1:length(sys.eps)
    for j=1:length(rho)
        tmp = Phi{i, j};
        X = D_half * [tmp.x;tmp.u];
        true_cost{i, j} = norm(X * sqrtm(opt.Sigma), 'fro')^2;
    end
end

% Compute the Wasserstein cost
for k = 1:length(rho)
    tmp = Phi_W{k};
    X = D_half * [tmp.x; tmp.u];
    true_cost_W{k} = norm(X * sqrtm(opt.Sigma), 'fro')^2;
end

X = D_half * [Phi_x_h2; Phi_u_h2];
true_cost_H2 = norm(X * sqrtm(opt.Sigma), 'fro')^2;

X = D_half * [Phi_nominal.x; Phi_nominal.u];
true_cost_nominal = norm(X * sqrtm(opt.Sigma), 'fro')^2;

% %% Plot cost comparison
% 
% % Define the figure dimensions in inches
% width = 3.5; % One column width
% height = 3.5 * 0.75; % Aspect ratio of 4:3, adjust as needed
% 
% % Create the figure
% figure('Units', 'inches', 'Position', [1, 1, width, height]);
% 
% hold on;
% 
% % Define colors for each radius to keep them consistent across plots
% colors = lines(3);  % Generates 3 distinguishable colors (one for each radius)
% 
% % Plot lines for each radius with corresponding epsilon values
% for radiusIdx = 1:length(rho)
%     % semilogx(epsilons_fine, cost_fine(:, radiusIdx), 'LineWidth', 1.5, 'Color', colors(radiusIdx, :));
%     plot(cell2mat(sys.eps), cell2mat(cost(:, radiusIdx)), 'LineWidth', 1, 'Color', colors(radiusIdx, :));
% end
% 
% set(gca, 'XScale', 'log');
% 
% % Now plot horizontal dashed lines for cost_W (same color as the lines for each radius)
% for radiusIdx = 1:length(rho)
%     yline(cell2mat(cost_W(radiusIdx)), '--', 'LineWidth', 1, 'Color', colors(radiusIdx, :));
% end
% 
% % H2 line and letter on top
% yline(cost_h2, '--', 'LineWidth', 1, 'Color', 'k');
% text(1e-3, cost_h2*1.3, '$\mathcal{H}_2$', 'HorizontalAlignment', 'center', 'FontSize', 8, 'Interpreter', 'latex');
% 
% % Add labels, title, and legend
% xlabel('\epsilon', 'Interpreter','tex', 'FontSize', 8);
% ylabel('Cost', 'FontSize', 8);
% % title('Worst case cost vs \epsilon for different radii', 'FontSize', 16, 'Interpreter','tex');
% for i = 1:length(rho)
%     legend_labels{i} = ['\rho = ', num2str(rho{i})];  % Convert each rho value to a string and format it
% end
% 
% % Customize the x-axis ticks
% xticks = 10.^(-5:4);
% set(gca, 'XTick', xticks); % Set x-axis ticks
% set(gca, 'XTickLabel', num2cell(-5:4), 'FontSize', 8); % Set only exponents as labels
% 
% % Add the legend to the plot
% legend(legend_labels, 'Location', 'Best', 'FontSize', 8);
% 
% % yLimits = ylim;  % Get current limits
% % % xLimits = xlim;
% % padding = 0.1 * (yLimits(2) - yLimits(1));  % 10% padding
% % ylim([yLimits(1), yLimits(2)]);  % Apply padding
% xlim([1e-5, sys.eps{end}]);
% 
% % Display grid
% grid on;
% 
% hold off;
% set(gcf, 'PaperPositionMode', 'auto');
% exportgraphics(gcf, 'cost_epsilon_h2.pdf', 'ContentType', 'vector');

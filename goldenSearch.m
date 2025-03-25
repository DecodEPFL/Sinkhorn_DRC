function [cost, lambda_opt, Q_opt, Phi_x, Phi_u] = goldenSearch(rho, eps, tol, lam_min, lam_max, sys, sls, opt)
    % Taken from https://drlvk.github.io/nm/section-golden-section.html
    % For a reference https://en.wikipedia.org/wiki/Golden-section_search#Iterative_algorithm

    gr = .5*(3-sqrt(5));
    a = lam_min; b = lam_max;
    interval = b - a;
    c = a + gr*interval;
    d = b - gr*interval;
    [~,~,~,fc] = causal_unconstrained_gaussian(sys, sls, opt, c, rho, eps);
    [~,~,~,fd] = causal_unconstrained_gaussian(sys, sls, opt, d, rho, eps);
    
    while abs(b-a) >= tol
        if fc < fd
            b = d;
            d = c;
            fd = fc;
            c = a + gr*(b-a);
            [Phi_x, Phi_u, Q, fc] = causal_unconstrained_gaussian(sys, sls, opt, c, rho, eps);
        else
            a = c;
            c = d;
            fc = fd;
            d = b - gr*(b-a);
            [Phi_x, Phi_u, Q, fd] = causal_unconstrained_gaussian(sys, sls, opt, d, rho, eps);
        end
    end
    cost = fd;
    lambda_opt = d;
    Q_opt = Q;
end
function [cost, lambda_opt, Phi_x, Phi_u] = goldenSearch(rho, tol, lam_min, lam_max, sys, sls, opt)
    % Taken from https://drlvk.github.io/nm/section-golden-section.html
    % For a reference https://en.wikipedia.org/wiki/Golden-section_search#Iterative_algorithm

    gr = .5*(3-sqrt(5));
    a = lam_min; b = lam_max;
    interval = b - a;
    c = a + gr*interval;
    d = b - gr*interval;
    [~,~,fc] = causal_unconstrained(sys, sls, opt, c, rho);
    [~,~,fd] = causal_unconstrained(sys, sls, opt, d, rho);
    
    while abs(b-a) >= tol
        if fc < fd
            b = d;
            d = c;
            fd = fc;
            c = a + gr*(b-a);
            [Phi_x, Phi_u, fc] = causal_unconstrained(sys, sls, opt, c, rho);
        else
            a = c;
            c = d;
            fc = fd;
            d = b - gr*(b-a);
            [Phi_x, Phi_u, fd] = causal_unconstrained(sys, sls, opt, d, rho);
        end
    end
    cost = fd;
    lambda_opt = d;
end
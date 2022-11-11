function [Phi_x, Phi_u, objective] = ftc_unconstrained(sys, sls, opt, Phi_benchmark)

    % Define the decision variables of the optimization problem
    Phi_x = sdpvar(sys.n*opt.T, sys.n*opt.T, 'full');
    Phi_u = sdpvar(sys.m*opt.T, sys.n*opt.T, 'full');
    
    lambda = sdpvar(1, 1, 'full'); % Maximum eigenvalue to be minimized
    
    % Define the objective function
    objective = lambda;
   
    constraints = [];
    % Impose the achievability constraints
    constraints = [constraints, (sls.I - sls.Z*sls.A)*Phi_x - sls.Z*sls.B*Phi_u == sls.I];
    % Impose the causal sparsities on the closed loop responses
    for i = 0:opt.T-2
        for j = i+1:opt.T-1 % Set j from i+2 for non-strictly causal controller (first element in w is x0)
            constraints = [constraints, Phi_x((1+i*sys.n):((i+1)*sys.n), (1+j*sys.n):((j+1)*sys.n)) == zeros(sys.n, sys.n)];
            constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.n):((j+1)*sys.n)) == zeros(sys.m, sys.n)];
        end
    end
    % Impose the constraints deriving from the Schur complement
    P = [eye((sys.n+sys.m)*opt.T) sqrtm(opt.C)*([Phi_x; Phi_u] - [Phi_benchmark.x; Phi_benchmark.u]); ([Phi_x; Phi_u] - [Phi_benchmark.x; Phi_benchmark.u])'*sqrtm(opt.C) lambda*eye(sys.n*opt.T)];
    constraints = [constraints, P >= 0];
    constraints = [constraints, lambda >= 0];
    
    % Solve the optimization problem
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    objective = value(objective);
    
end
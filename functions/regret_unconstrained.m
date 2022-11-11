function [Phi_x, Phi_u, objective] = regret_unconstrained(sys, sls, opt, Phi_benchmark)
%REGRET_UNCONSTRAINED computes a regret-optimal unconstrained causal linear
%control policy with respect to the given benchmark

    % Define the decision variables of the optimization problem
    Phi_x = sdpvar(sys.n*opt.T, sys.n*opt.T, 'full');
    Phi_u = sdpvar(sys.m*opt.T, sys.n*opt.T, 'full');
    
    lambda = sdpvar(1, 1, 'full'); % Maximum eigenvalue to be minimized
    
    % Compute the matrix that defines the quadratic form measuring the cost incurred by the benchmark controller
    J_benchmark = [Phi_benchmark.x; Phi_benchmark.u]'*opt.C*[Phi_benchmark.x; Phi_benchmark.u];
    
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
    P = [eye((sys.n+sys.m)*opt.T) sqrtm(opt.C)*[Phi_x; Phi_u]; [Phi_x; Phi_u]'*sqrtm(opt.C) lambda*eye(sys.n*opt.T) + J_benchmark];
    constraints = [constraints, P >= 0];
    constraints = [constraints, lambda >= 0];
    
    % Solve the optimization problem
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    % Extract the closed-loop responses corresponding to a regret-optimal
    % unconstrained causal linear control policy with respect to the given benchmark
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    objective = value(objective);
    
end
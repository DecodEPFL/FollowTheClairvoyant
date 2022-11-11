function [Phi_x, Phi_u, objective] = noncausal_constrained_preview(sys, sls, opt, p, flag)
    
    % Define the decision variables of the optimization problem
    Phi_x = sdpvar(sys.n*opt.T, sys.n*opt.T, 'full');
    Phi_u = sdpvar(sys.m*opt.T, sys.n*opt.T, 'full');
    
    % Define the objective function
    if strcmp(flag, 'H2')
        objective = norm(sqrtm(opt.C)*[Phi_x; Phi_u], 'fro');
    elseif strcmp(flag, 'Hinf')
        objective = norm(sqrtm(opt.C)*[Phi_x; Phi_u], 2);
    else
        error('Something went wrong...');
    end
    
    constraints = [];
    % Impose the achievability constraints
    constraints = [constraints, (sls.I - sls.Z*sls.A)*Phi_x - sls.Z*sls.B*Phi_u == sls.I];
    % Impose the causal sparsities on the closed loop responses according
    % to the preview parameter
    for i = 0:opt.T-2-p
        for j = i+1+p:opt.T-1 % Set j from i+2 for non-strictly causal controller (first element in w is x0)
            constraints = [constraints, Phi_x((1+i*sys.n):((i+1)*sys.n), (1+j*sys.n):((j+1)*sys.n)) == zeros(sys.n, sys.n)];
            constraints = [constraints, Phi_u((1+i*sys.m):((i+1)*sys.m), (1+j*sys.n):((j+1)*sys.n)) == zeros(sys.m, sys.n)];
        end
    end
    % Impose the polytopic safety constraints
    z = sdpvar(size(sls.Hw, 1), size(sls.H, 1), 'full'); % Define the dual variables
    for i = 1:size(sls.H, 1)
        constraints = [constraints, z(:, i)'*sls.hw <= sls.h(i)];
        constraints = [constraints, z(:, i) >= 0];
    end
    constraints = [constraints, sls.H*[Phi_u; Phi_x] == z'*sls.Hw];
    
    % Solve the optimization problem
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    objective = value(objective)^2; 

end
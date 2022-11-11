function [Phi_x, Phi_u, objective] = noncausal_constrained(sys, sls, opt, flag)
%NONCAUSAL_CONSTRAINED computes a constrained clairvoyant linear control 
%policy that is optimal either in the H2 or in the Hinf sense
    
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
    
    % Extract the closed-loop responses corresponding to a constrained
    % clairvoyant linear controller that is optimal either in the H2 or in the Hinf sense
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    objective = value(objective)^2; % Extract the H2- or Hinf-optimal cost incurred by a constrained clairvoyant linear controller

end
function [Phi_x, Phi_u, obj_h2, obj_hinf] = noncausal_unconstrained(sys, sls, opt)
%NONCAUSAL_UNCONSTRAINED computes the unconstrained clairvoyant controller 
%that selects the globally optimal dynamic sequence of control actions in 
%hindsight
    
    % Define the decision variables of the optimization problem
    Phi_x = sdpvar(sys.n*opt.T, sys.n*opt.T, 'full');
    Phi_u = sdpvar(sys.m*opt.T, sys.n*opt.T, 'full');
    
    % Define the objective function
    objective = norm(sqrtm(opt.C)*[Phi_x; Phi_u], 'fro');
    
    constraints = [];
    % Impose the achievability constraints
    constraints = [constraints, (sls.I - sls.Z*sls.A)*Phi_x - sls.Z*sls.B*Phi_u == sls.I];
    
    % Solve the optimization problem
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    % Extract the closed-loop responses corresponding to the unconstrained clairvoyant controller
    Phi_x = value(Phi_x); 
    Phi_u = value(Phi_u);
    
    obj_h2 = value(objective)^2;                       % Extract the H2-optimal   cost incurred by the unconstrained clairvoyant controller
    obj_hinf = norm(sqrtm(opt.C)*[Phi_x; Phi_u], 2)^2; % Compute the Hinf-optimal cost incurred by the unconstrained clairvoyant controller  

end
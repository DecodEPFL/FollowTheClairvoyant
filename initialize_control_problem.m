function [sys, opt, sls] = initialize_control_problem(experiment)
    
    if ~(experiment == 1 || experiment == 2)
        error('Unsupported argument value, exiting...')
    end
    sys.rho = 1.05; % Spectral radius
    sys.A = sys.rho*[0.7 0.2 0; 0.3 0.7 -0.1; 0 -0.2 0.8];
    sys.B = [1 0.2; 2 0.3; 1.5 0.5];

    sys.n = size(sys.A, 1);   % Order of the system: state dimension
    sys.m = size(sys.B, 2);   % Number of input channels
    sys.x0 = zeros(sys.n, 1); % Initial condition

    sys.Hu = [eye(sys.m); -eye(sys.m)]; % Polytopic constraints: Hu * u <= hu
    sys.hu = (10*(experiment == 1) + 5*(experiment == 2))*ones(size(sys.Hu, 1), 1);

    sys.Hx = [eye(sys.n); -eye(sys.n)]; % Polytopic constraints: Hx * x <= hx
    sys.hx = 10*ones(size(sys.Hx, 1), 1);

    sys.Hw = [eye(sys.n); -eye(sys.n)]; % Polytopic disturbance set: Hw * w <= hw 
    sys.hw = 1*ones(size(sys.Hw, 1), 1);
    % Definition of the parameters of the optimization problem
    opt.Qt = eye(sys.n); % Stage cost: state weight matrix
    opt.Rt = eye(sys.m); % Stage cost: input weight matrix

    opt.T = 30; % Control horizon

    opt.Q = kron(eye(opt.T), opt.Qt); % State cost matrix
    opt.R = kron(eye(opt.T), opt.Rt); % Input cost matrix
    opt.C = blkdiag(opt.Q, opt.R); % Cost matrix
    % Definition of the stacked system dynamics over the control horizon
    sls.A = kron(eye(opt.T), sys.A);
    sls.B = kron(eye(opt.T), sys.B);

    sls.I = eye(sys.n*opt.T); % Identity matrix and block-downshift operator
    sls.Z = [zeros(sys.n, sys.n*(opt.T-1)) zeros(sys.n, sys.n); eye(sys.n*(opt.T-1)) zeros(sys.n*(opt.T-1), sys.n)];

    % Polytopic disturbance description and safety constraints
    sls.Hu = kron(eye(opt.T), sys.Hu);
    sls.hu = kron(ones(opt.T, 1), sys.hu);

    sls.Hx = kron(eye(opt.T), sys.Hx);
    sls.hx = kron(ones(opt.T, 1), sys.hx);

    sls.H = blkdiag(sls.Hu, sls.Hx);
    sls.h = [sls.hu; sls.hx];

    sls.Hw = kron(eye(opt.T), sys.Hw);
    sls.hw = kron(ones(opt.T, 1), sys.hw);
end


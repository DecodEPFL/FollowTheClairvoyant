%% Martin et al. - "Follow the Clairvoyant: an Imitation Learning Approach to Optimal Control"
clc; close all; clear;
addpath('./functions') % Add path to the folder with auxiliary functions
rng(1234);             % Set random seed for reproducibility
%% Experiment 1: performance comparison against H2 and Hinfinity for deterministic and stochastic disturbance profiles
% If available, load the data and the controllers
source_file = 'data_T30_rho1p05_hu10_hx10_hw1.mat';
if isfile(source_file)
    load(source_file);
else
    [sys, opt, sls] = initialize_control_problem(1);
    % Computation of the optimal clairvoyant unconstrained policy
    [Phi_nc_unc.x, Phi_nc_unc.u, obj_nc.unc_h2, obj_nc.unc_hinf] = noncausal_unconstrained(sys, sls, opt);
    % Computation of the optimal clairvoyant constrained H2 policy
    [Phi_nc_con_h2.x, Phi_nc_con_h2.u, obj_nc.con_h2] = noncausal_constrained(sys, sls, opt, 'H2');
    % Computation of H2 and Hinf constrained controllers
    [Phi_c_con_h2.x,   Phi_c_con_h2.u,   obj_c.con_h2]   = causal_constrained(sys, sls, opt, 'H2');
    [Phi_c_con_hinf.x, Phi_c_con_hinf.u, obj_c.con_hinf] = causal_constrained(sys, sls, opt, 'Hinf');
    % Computation of regret-optimal and follow-the-clairvoyant constrained controllers
    [Phi_reg_con_nc_con_h2.x, Phi_reg_con_nc_con_h2.u, obj_reg.con_nc_con_h2] = regret_constrained(sys, sls, opt, Phi_nc_con_h2);
    [Phi_ftc_con_nc_con_h2.x, Phi_ftc_con_nc_con_h2.u, obj_ftc.con_nc_con_h2] =    ftc_constrained(sys, sls, opt, Phi_nc_con_h2);
    save data_T30_rho1p05_hu10_hx10_hw1
end
clear source_file
% Verify that FTC attains minimal regret as predicted by Theorem 3
if max(max(Phi_nc_con_h2.u - Phi_nc_unc.u)) < 1e-06
    fprintf('Worst-case regret of regret-optimal policy: %5.2f\n',         max(eig([Phi_reg_con_nc_con_h2.x; Phi_reg_con_nc_con_h2.u]'*opt.C*[Phi_reg_con_nc_con_h2.x; Phi_reg_con_nc_con_h2.u] - [Phi_nc_con_h2.x; Phi_nc_con_h2.u]'*opt.C*[Phi_nc_con_h2.x; Phi_nc_con_h2.u])))
    fprintf('Worst-case regret of follow-the-clairvoyant policy: %5.2f\n', max(eig([Phi_ftc_con_nc_con_h2.x; Phi_ftc_con_nc_con_h2.u]'*opt.C*[Phi_ftc_con_nc_con_h2.x; Phi_ftc_con_nc_con_h2.u] - [Phi_nc_con_h2.x; Phi_nc_con_h2.u]'*opt.C*[Phi_nc_con_h2.x; Phi_nc_con_h2.u])))
    fprintf('------------------------------------------------------\n\n')
end

disturbance.profiles = ["Gaussian: N(0,1)" "Uniform: U(0.5, 1)" "Uniform: U(0, 1)" "Constant" "Sinusoidal wave" "Sawtooth wave" "Step function" "Stairs function" "Worst-case"];
disturbance.stochastic = [1000*ones(3, 1); ones(6, 1)]; % Maximum number of iterations per profile

for i = 1:size(disturbance.profiles, 2) % Iterate over all different disturbance profiles
    for j = 1:disturbance.stochastic(i) 
        % Sample a disturbance realization
        if i == 1     % Gaussian: N(0, 1)
            w = [sys.x0; randn(sys.n*(opt.T - 1), 1)];
        elseif i == 2 % Uniform: U(0.5, 1)
            w = [sys.x0; 0.5 + rand(sys.n*(opt.T - 1), 1)*0.5];
        elseif i == 3 % Uniform: U(0, 1)
            w = [sys.x0; rand(sys.n*(opt.T - 1), 1)];
        elseif i == 4 % Constant at 1
            w = [sys.x0; ones(sys.n*(opt.T - 1), 1)];
        elseif i == 5 % Sinusoidal wave
            w = [sys.x0; kron(ones(sys.n, 1), sin(0.1*(1:opt.T-1)'))];
        elseif i == 6 % Sawtooth wave
            w = [sys.x0; kron(ones(sys.n, 1), sawtooth(0.1*(1:opt.T-1), 1)')];
        elseif i == 7 % Step function
            w = [sys.x0; zeros(sys.n*(opt.T - 1 - floor(opt.T/2)), 1); ones(sys.n*floor(opt.T/2), 1)];
        elseif i == 8 % Stairs function taking values in the set {-1, 0, 1}
            w = [sys.x0; -ones(sys.n*(opt.T - 1 - 2*floor(opt.T/3)), 1); zeros(sys.n*floor(opt.T/3), 1); ones(sys.n*floor(opt.T/3), 1)]; 
        else          % Worst-case disturbance: adversarial selection for all the three safe control laws  
            c_con_hinf.w = getWorstCaseRealization(sys, opt, Phi_c_con_hinf);
            c_con_h2.w   = getWorstCaseRealization(sys, opt, Phi_c_con_h2);
            ftc_con_nc_con_h2.w = getWorstCaseRealization(sys, opt, Phi_ftc_con_nc_con_h2);
        end
        if i ~= 9 % Always simulate the three considered safe control policies with the same disturbance sequence, except when dealing with the worst-case of each of them
            c_con_hinf.w = w/norm(w); c_con_h2.w = w/norm(w); ftc_con_nc_con_h2.w = w/norm(w);
        end
        % Vectorize the sampled disturbance sequence for convenience
        c_con_hinf.w = c_con_hinf.w(:); c_con_h2.w = c_con_h2.w(:); ftc_con_nc_con_h2.w   = ftc_con_nc_con_h2.w(:); 
        % Simulate the closed-loop system with the H2 and Hinf constrained controllers
        c_con_h2.cum_costs(j)   = evaluate_policy(opt, Phi_c_con_h2, c_con_h2.w); 
        c_con_hinf.cum_costs(j) = evaluate_policy(opt, Phi_c_con_hinf, c_con_hinf.w);
        % Simulate the closed-loop system with the follow-the-clairvoyant policy
        ftc_con_nc_con_h2.cum_costs(j)   = evaluate_policy(opt, Phi_ftc_con_nc_con_h2,   ftc_con_nc_con_h2.w);
    end
    % Compute the mean cumulative cost incurred by the H2 and Hinf constrained controllers
    c_con_h2.avg_cost   = mean(c_con_h2.cum_costs);
    c_con_hinf.avg_cost = mean(c_con_hinf.cum_costs);
    % Compute the mean cumulative cost incurred by the follow-the-clairvoyant policy
    ftc_con_nc_con_h2.avg_cost = mean(ftc_con_nc_con_h2.cum_costs);
    % Display the average incurred control costs 
    fprintf('%s\n\n', disturbance.profiles(i))
    fprintf('Constrained H2: %f\n', c_con_h2.avg_cost)
    fprintf('Constrained Hinf: %f\n', c_con_hinf.avg_cost)
    fprintf('Constrained FTC: %f\n\n', ftc_con_nc_con_h2.avg_cost)

    if c_con_h2.avg_cost < c_con_hinf.avg_cost && c_con_h2.avg_cost < ftc_con_nc_con_h2.avg_cost
        fprintf('Percentage increase: Hinf/H2: %5.2f   ', 100 * (c_con_hinf.avg_cost        - c_con_h2.avg_cost) / c_con_h2.avg_cost)
        fprintf('FTC/H2: %5.2f',                          100 * (ftc_con_nc_con_h2.avg_cost - c_con_h2.avg_cost) / c_con_h2.avg_cost)
    elseif ftc_con_nc_con_h2.avg_cost < c_con_hinf.avg_cost && ftc_con_nc_con_h2.avg_cost <  c_con_h2.avg_cost 
        fprintf('Percentage increase: H2/FTC: %5.2f   ', 100 * (c_con_h2.avg_cost   - ftc_con_nc_con_h2.avg_cost) / ftc_con_nc_con_h2.avg_cost)
        fprintf('Hinf/FTC: %5.2f   ',                    100 * (c_con_hinf.avg_cost - ftc_con_nc_con_h2.avg_cost) / ftc_con_nc_con_h2.avg_cost)
    else
        fprintf('Percentage increase: H2/Hinf: %5.2f   ', 100 * (c_con_h2.avg_cost          - c_con_hinf.avg_cost) / c_con_hinf.avg_cost)
        fprintf('FTC/Hinf: %5.2f',                        100 * (ftc_con_nc_con_h2.avg_cost - c_con_hinf.avg_cost) / c_con_hinf.avg_cost)
    end
    fprintf('\n------------------------------------------------------\n\n')
    clear c_con_h2 c_con_hinf ftc_con_nc_con_h2; % Clear variables corresponding to past disturbances profiles
end
clear disturbance i j w;
%% Experiment 2: performance comparison against regret-optimal policies in constrained scenarios
clear; rng(1234); % Clear and set random seed for reproducibility
% If available, load the data and the controllers
source_file = 'data_T30_rho1p05_hu5_hx10_hw1.mat';
if isfile(source_file)
    load(source_file);
else
    [sys, opt, sls] = initialize_control_problem(2);
    % Computation of the optimal clairvoyant constrained H2 policy with limited preview
    [Phi_nc_con_h2_p.x, Phi_nc_con_h2_p.u, obj_nc_p.con_h2] = noncausal_constrained_preview(sys, sls, opt, 5, 'H2');
    % Computation of regret-optimal and follow-the-clairvoyant constrained controllers
    [Phi_reg_con_nc_con_h2_p.x, Phi_reg_con_nc_con_h2_p.u, obj_reg.con_nc_con_h2_p] = regret_constrained(sys, sls, opt, Phi_nc_con_h2_p);
    [Phi_ftc_con_nc_con_h2_p.x, Phi_ftc_con_nc_con_h2_p.u, obj_ftc.con_nc_con_h2_p] =    ftc_constrained(sys, sls, opt, Phi_nc_con_h2_p);
    % Generate challenging disturbance realizations to compare the performance of regret-optimal and follow-the-clairvoyant control policies
    counter = 0; % Counter for the number of disturbance realizations that nearly activate the constraints
    N = 1000;    % Number of disturbance realizations to collect before exiting
    cost_reg_con_nc_con_h2_p = zeros(N, opt.T);  % Cost incurred by the regret-optimal policy
    cost_ftc_con_nc_con_h2_p = zeros(N, opt.T);  % Cost incurred by the follow-the-clairvoyant policy
    track_reg_con_nc_con_h2_p = zeros(N, opt.T); % Tracking error incurred by the regret-optimal policy 
    track_ftc_con_nc_con_h2_p = zeros(N, opt.T); % Tracking error incurred by the follow-the-clairvoyant policy
    ws = []; % List of disturbance realizations that nearly activate the constraints
    while counter < N
        % Randomly sample a disturbance sequence from one of the vertices of the disturbance set
        w = sign(rand(sys.n*opt.T, 1) - 0.5); % Make sure that each entry is either 1 or -1
        % Compute the input and state trajectories of the clairvoyant benchmark policy
        x_nc_con_h2_p = Phi_nc_con_h2_p.x * w; u_nc_con_h2_p = Phi_nc_con_h2_p.u * w;
        % Verify whether the clairvoyant trajectories nearly activate the constraints (note that the state and input components are uniform across the different components)
        if max(x_nc_con_h2_p) >= 0.9*sys.hx(1) || min(x_nc_con_h2_p) <= -0.9*sys.hx(1) || max(u_nc_con_h2_p) >= 0.9*sys.hu(1) || min(u_nc_con_h2_p) <= -0.9*sys.hu(1)
            flag = 0; % Make sure not to count the same disturbance realization twice...
            for i = 1:size(ws, 2)
                if w == ws(:, i)
                    flag = 1;
                end
            end
            if ~flag % If the current disturbance realization was not already counted...
                ws = [ws w]; counter = counter + 1; % Add the current disturbance realization to the list and increase the counter
                % Compute the input and state trajectories of the regret-optimal policy
                x_reg_con_nc_con_h2_p = Phi_reg_con_nc_con_h2_p.x * w; u_reg_con_nc_con_h2_p = Phi_reg_con_nc_con_h2_p.u * w;
                % Compute the input and state trajectories of the follow-the-clairvoyant policy
                x_ftc_con_nc_con_h2_p = Phi_ftc_con_nc_con_h2_p.x * w; u_ftc_con_nc_con_h2_p = Phi_ftc_con_nc_con_h2_p.u * w;
                % Compute the control cost and the tracking error incurred  along the horizon
                for j = 1:opt.T
                    cost_reg_con_nc_con_h2_p(counter, j) = x_reg_con_nc_con_h2_p(1:sys.n*j)'*x_reg_con_nc_con_h2_p(1:sys.n*j) + u_reg_con_nc_con_h2_p(1:sys.m*j)'*u_reg_con_nc_con_h2_p(1:sys.m*j);
                    cost_ftc_con_nc_con_h2_p(counter, j) = x_ftc_con_nc_con_h2_p(1:sys.n*j)'*x_ftc_con_nc_con_h2_p(1:sys.n*j) + u_ftc_con_nc_con_h2_p(1:sys.m*j)'*u_ftc_con_nc_con_h2_p(1:sys.m*j);
                    track_reg_con_nc_con_h2_p(counter, j) = (x_reg_con_nc_con_h2_p(1:sys.n*j) - x_nc_con_h2_p(1:sys.n*j))'*(x_reg_con_nc_con_h2_p(1:sys.n*j) - x_nc_con_h2_p(1:sys.n*j)) + (u_reg_con_nc_con_h2_p(1:sys.m*j) - u_nc_con_h2_p(1:sys.m*j))'*(u_reg_con_nc_con_h2_p(1:sys.m*j) - u_nc_con_h2_p(1:sys.m*j));
                    track_ftc_con_nc_con_h2_p(counter, j) = (x_ftc_con_nc_con_h2_p(1:sys.n*j) - x_nc_con_h2_p(1:sys.n*j))'*(x_ftc_con_nc_con_h2_p(1:sys.n*j) - x_nc_con_h2_p(1:sys.n*j)) + (u_ftc_con_nc_con_h2_p(1:sys.m*j) - u_nc_con_h2_p(1:sys.m*j))'*(u_ftc_con_nc_con_h2_p(1:sys.m*j) - u_nc_con_h2_p(1:sys.m*j));
                end    
            end
        end
    end
    clear i j counter w ws flag x_nc_con_h2_p u_nc_con_h2_p x_reg_con_nc_con_h2_p u_reg_con_nc_con_h2_p x_ftc_con_nc_con_h2_p u_ftc_con_nc_con_h2_p
    save data_T30_rho1p05_hu5_hx10_hw1
end
clear source_file
% Plot the simulation results: pad with an initial zero to obtain nicer graphics
t = 0:opt.T; % Time horizon with one extra entry for visualization
cost_reg_con_nc_con_h2_p = [zeros(size(cost_reg_con_nc_con_h2_p, 1), 1) cost_reg_con_nc_con_h2_p];
cost_ftc_con_nc_con_h2_p = [zeros(size(cost_ftc_con_nc_con_h2_p, 1), 1) cost_ftc_con_nc_con_h2_p];
track_reg_con_nc_con_h2_p = [zeros(size(track_reg_con_nc_con_h2_p, 1), 1) track_reg_con_nc_con_h2_p];
track_ftc_con_nc_con_h2_p = [zeros(size(track_ftc_con_nc_con_h2_p, 1), 1) track_ftc_con_nc_con_h2_p];

figure % Performance comparison: incurred control cost
dcost_mean = mean(cost_reg_con_nc_con_h2_p - cost_ftc_con_nc_con_h2_p); 
dcost_std  =  std(cost_reg_con_nc_con_h2_p - cost_ftc_con_nc_con_h2_p); 
dcost_mean_std = [dcost_mean + dcost_std, fliplr(dcost_mean - dcost_std)];
dcost_pos = [max(max(cost_reg_con_nc_con_h2_p - cost_ftc_con_nc_con_h2_p), 0), max(fliplr(min(cost_reg_con_nc_con_h2_p - cost_ftc_con_nc_con_h2_p)), 0)];
dcost_neg = [min(max(cost_reg_con_nc_con_h2_p - cost_ftc_con_nc_con_h2_p), 0), min(fliplr(min(cost_reg_con_nc_con_h2_p - cost_ftc_con_nc_con_h2_p)), 0)];
tick_5p = 0.05*mean(cost_ftc_con_nc_con_h2_p(:, end));
tick_10p = 0.1*mean(cost_ftc_con_nc_con_h2_p(:, end));

plot(t, dcost_mean, 'k', 'LineWidth', 1)
hold on
fill([t, fliplr(t)], dcost_mean_std, 'k', 'FaceAlpha', 0.1, 'LineStyle', 'none')
grid on; grid minor;
fill([t, fliplr(t)], dcost_pos, 'g', 'FaceAlpha', 0.15, 'LineStyle', 'none')
fill([t, fliplr(t)], dcost_neg, 'r', 'FaceAlpha', 0.15, 'LineStyle', 'none')
ylim([-tick_5p 2*tick_10p])
yticks([-tick_5p 0 tick_5p tick_10p tick_5p+tick_10p 2*tick_10p])
yticklabels({'-5\%', '0', '5\%', '10\%', '15\%', '20\%'})
set(gca,'TickLabelInterpreter','latex')
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 12)
ylabel('$\Delta \bar{J}_t$', 'Interpreter', 'latex', 'FontSize', 12)

figure % Performance comparison: incurred tracking error
dtrack_mean = mean(track_reg_con_nc_con_h2_p - track_ftc_con_nc_con_h2_p);
dtrack_std  =  std(track_reg_con_nc_con_h2_p - track_ftc_con_nc_con_h2_p);
dtrack_mean_std = [dtrack_mean + dtrack_std, fliplr(dtrack_mean - dtrack_std)];
dtrack_pos = [max(max(track_reg_con_nc_con_h2_p - track_ftc_con_nc_con_h2_p), 0), max(fliplr(min(track_reg_con_nc_con_h2_p - track_ftc_con_nc_con_h2_p)), 0)];
dtrack_neg = [min(max(track_reg_con_nc_con_h2_p - track_ftc_con_nc_con_h2_p), 0), min(fliplr(min(track_reg_con_nc_con_h2_p - track_ftc_con_nc_con_h2_p)), 0)];
tick_5p = 0.05*mean(track_ftc_con_nc_con_h2_p(:, end));
tick_10p = 0.1*mean(track_ftc_con_nc_con_h2_p(:, end));
plot(t, dtrack_mean, 'k', 'LineWidth', 1)
hold on
fill([t, fliplr(t)], dtrack_mean_std, 'k', 'FaceAlpha', 0.1, 'LineStyle', 'none')
grid on; grid minor;
fill([t, fliplr(t)], dtrack_pos, 'g', 'FaceAlpha', 0.15, 'LineStyle', 'none')
fill([t, fliplr(t)], dtrack_neg, 'r', 'FaceAlpha', 0.15, 'LineStyle', 'none')
ylim([-tick_10p 4*tick_10p])
yticks([-tick_10p 0 tick_10p 2*tick_10p 3*tick_10p 4*tick_10p])
yticklabels({'-10\%', '0', '10\%', '20\%', '30\%', '40\%'})
set(gca,'TickLabelInterpreter','latex')
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 12)
ylabel('$\Delta \bar{E}_t$', 'Interpreter', 'latex', 'FontSize', 12)
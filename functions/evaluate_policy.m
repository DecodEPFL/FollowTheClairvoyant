function [cum_cost] = evaluate_policy(opt, Phi, w)
%EVALUATE_POLICY computes the cumulative cost incurred applying the policy 
%corresponding to the closed-loop responses in Phi in response to 
%the disturbance realization w
    
    % Compute the input-state trajectory associated with the disturbance w 
    x = Phi.x * w; 
    u = Phi.u * w; 
    
    % Compute the incurred cumulative cost
    cum_cost = [x; u]'*opt.C*[x; u];
    
end
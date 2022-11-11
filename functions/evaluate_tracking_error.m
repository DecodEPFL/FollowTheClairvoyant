function [tracking_error] = evaluate_tracking_error(opt, Phi, Psi, w)
  
    % Compute the incurred cumulative cost
    tracking_error = w'*[Phi.x - Psi.x; Phi.u - Psi.u]'*opt.C*[Phi.x - Psi.x; Phi.u - Psi.u]*w;
    
end
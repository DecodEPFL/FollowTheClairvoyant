function w = getWorstCaseRealization(sys, opt, Phi)
    cost_qf = [Phi.x; Phi.u]'*opt.C*[Phi.x; Phi.u];
    [evectors, evalues] = eig(cost_qf(sys.n+1:end, sys.n+1:end), 'vector');
    [~, index] = max(evalues);
    w = [sys.x0; evectors(:, index)]; 
end


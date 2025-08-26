function [rob, grad] = computeSTLRobustnessGradient(phi, t, x)
    % phi : STL_Formula object
    % t   : time vector
    % x   : signal (1 x N)
    
    phi_type = get_type(phi);
    
    switch lower(phi_type)
        case 'predicate'
            expr = get_expr(phi);  % get the raw expression like "x > 3"
            [rob, grad] = eval_predicate(expr, x);

        case 'and'
            children = get_children(phi);
            [r1, g1] = computeSTLRobustnessGradient(children{1}, t, x);
            [r2, g2] = computeSTLRobustnessGradient(children{2}, t, x);
            [rob, w] = softmin([r1; r2]);
            grad = w(1)*g1 + w(2)*g2;

        case 'or'
            children = get_children(phi);
            [r1, g1] = computeSTLRobustnessGradient(children{1}, t, x);
            [r2, g2] = computeSTLRobustnessGradient(children{2}, t, x);
            [rob, w] = softmax([r1; r2]);
            grad = w(1)*g1 + w(2)*g2;

        case 'not'
            child = get_children(phi);
            [r, g] = computeSTLRobustnessGradient(child{1}, t, x);
            rob = -r;
            grad = -g;

        case '=>'
            children = get_children(phi);
            lhs = children{1};
            rhs = children{2};
            % Convert A => B to (!A or B)
            not_lhs = STL_Formula('tmp_not', ['!(', char(lhs), ')']);
            or_formula = STL_Formula('tmp_or', [char(not_lhs), ' or (', char(rhs), ')']);
            [rob, grad] = computeSTLRobustnessGradient(or_formula, t, x);

        case 'always'
            child = get_children(phi);
            interval = get_interval(phi);
            idx = find(t >= interval(1) & t <= interval(2));
            [r_full, g_full] = computeSTLRobustnessGradient(child{1}, t, x);
            [rob, w] = softmin(r_full(idx));
            grad = zeros(size(x));
            for i = 1:length(idx)
                grad(idx(i)) = w(i) * g_full(idx(i));
            end

        case 'eventually'
            child = get_children(phi);
            interval = get_interval(phi);
            idx = find(t >= interval(1) & t <= interval(2));
            [r_full, g_full] = computeSTLRobustnessGradient(child{1}, t, x);
            [rob, w] = softmax(r_full(idx));
            grad = zeros(size(x));
            for i = 1:length(idx)
                grad(idx(i)) = w(i) * g_full(idx(i));
            end

        otherwise
            error(['Unsupported STL operator: ', phi_type]);
    end
end

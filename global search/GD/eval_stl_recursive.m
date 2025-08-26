function [rob, grad] = eval_stl_recursive(phi, x_struct)
    phi = STL_Formula('phi', phi);

    op_type = get_type(phi);

    switch op_type
        case 'predicate'
            expr = disp(phi);
            expr = strip_outer_parentheses(expr);
            % disp('in');
            % disp(expr);
            [rob, grad] = eval_predicate(expr, x_struct);
            % disp(rob);
            % disp(grad);
        case 'not'
            children = get_children(phi);
            [rob1, grad1] = eval_stl_recursive(children{1}, x_struct);
            rob = -rob1;
            grad = structfun(@(g) -g, grad1, 'UniformOutput', false);

        case 'and'
            children = get_children(phi);
            [rob1, grad1] = eval_stl_recursive(children{1}, x_struct);
            [rob2, grad2] = eval_stl_recursive(children{2}, x_struct);
            [rob, grad] = smooth_min(rob1, grad1, rob2, grad2);

        case 'or'
            children = get_children(phi);
            [rob1, grad1] = eval_stl_recursive(children{1}, x_struct);
            [rob2, grad2] = eval_stl_recursive(children{2}, x_struct);
            [rob, grad] = smooth_max(rob1, grad1, rob2, grad2);

        case '=>'
            % A => B ≡ not(A) or B
            children = get_children(phi);
            a = children{1};
            b = children{2};
            not_a = STL_Formula('tmp', 'not (' + disp(a) + ')');
            new_phi = STL_Formula('tmp2', '(' + disp(not_a) + ') or (' + disp(b) + ')');
            [rob, grad] = eval_stl_recursive(new_phi, x_struct);

        case 'always'
            children = get_children(phi);
            [rob_, grad_] = eval_stl_recursive(children{1}, x_struct);
            [rob, grad] = smooth_min_all(rob_, grad_);

        case 'eventually'
            children = get_children(phi);
            % disp(children{1});
            % disp('adsfadsf');
            [rob_, grad_] = eval_stl_recursive(children{1}, x_struct);
            [rob, grad] = smooth_max_all(rob_, grad_);

        otherwise
            error("Unsupported operator: " + op_type);
    end
end


function [rob, grad] = smooth_min(r1, g1, r2, g2)
    alpha = 10;
    w1 = exp(-alpha*r1);
    w2 = exp(-alpha*r2);
    Z = w1 + w2;
    rob = (w1.*r1 + w2.*r2) ./ Z;

    grad = struct();
    fields = fieldnames(g1);
    for i = 1:length(fields)
        grad.(fields{i}) = (w1 .* g1.(fields{i}) + w2 .* g2.(fields{i})) ./ Z ...
                         + alpha * (rob - r1) .* g1.(fields{i}) ...
                         + alpha * (rob - r2) .* g2.(fields{i});
    end
end

function [rob, grad] = smooth_max(r1, g1, r2, g2)
    [rob, grad] = smooth_min(-r1, structfun(@(g) -g, g1, 'UniformOutput', false), ...
                             -r2, structfun(@(g) -g, g2, 'UniformOutput', false));
    rob = -rob;
    grad = structfun(@(g) -g, grad, 'UniformOutput', false);
end

function [rob, grad] = smooth_min_all(r_arr, g_arr)
    rob = r_arr(1);
    grad = g_arr;
    for i = 2:length(r_arr)
        [rob, grad] = smooth_min(rob, grad, r_arr(i), g_arr);
    end
end

function [rob, grad] = smooth_max_all(r_arr, g_arr)
    [rob, grad] = smooth_max(r_arr(1), g_arr);
    for i = 2:length(r_arr)
        [rob, grad] = smooth_max(rob, grad, r_arr(i), g_arr);
    end
end


function out = strip_outer_parentheses(s)
    % Remove outermost parentheses from a string if they exist
    s = strtrim(s); % remove whitespace

    if startsWith(s, '(') && endsWith(s, ')')
        % Check if parentheses are balanced
        depth = 0;
        for i = 1:length(s)
            if s(i) == '('
                depth = depth + 1;
            elseif s(i) == ')'
                depth = depth - 1;
            end

            % If we close the outermost ( before the end, it's not safe to strip
            if depth == 0 && i < length(s)
                out = s;
                return;
            end
        end

        % If we get here, the outermost parentheses enclose the full string
        out = s(2:end-1);
    else
        out = s;
    end
end

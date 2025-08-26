function [rob, grad] = eval_predicate(expr, x_struct)
    % Evaluate predicate robustness and gradient
    % expr: e.g. 'x1[t] + x2[t] <= 1'
    % x_struct: struct of signal traces, each 1xN
    % disp(expr)
    expr = strrep(expr, ' ', '');  % remove whitespace
    % disp(expr);
    pattern = '(.*?)(<=|>=|==|<|>|=)(.*)';
    tokens = regexp(expr, pattern, 'tokens');
    
    if isempty(tokens)
        error(['Cannot parse predicate: ', expr]);
    end

    lhs_str = tokens{1}{1};
    % disp(lhs_str);
    op = tokens{1}{2};
    % disp(op);
    rhs_str = tokens{1}{3};
    % disp(rhs_str);
    fn = fieldnames(x_struct);
    N = length(x_struct.(fn{1})); % number of time steps
    rob = zeros(1, N);
    grad = struct();  % each field is signal name → gradient (1xN)

    sig_names = fieldnames(x_struct);
    for s = 1:length(sig_names)
        grad.(sig_names{s}) = zeros(1, N);
    end

    % finite diff epsilon
    eps = 1e-6;

    for i = 1:N
        vars = get_timestep(x_struct, i);
        % disp(vars);
        
        lhs_val = eval_expr(lhs_str, vars);
        rhs_val = eval_expr(rhs_str, vars);

        switch op
            case {'>', '>='}
                rob(i) = lhs_val - rhs_val;
                rob_expr = ['(', lhs_str, ') - (', rhs_str, ')'];
            case {'<', '<='}
                rob(i) = rhs_val - lhs_val;
                rob_expr = ['(', rhs_str, ') - (', lhs_str, ')'];
            case {'==', '='}
                rob(i) = -abs(lhs_val - rhs_val);
                rob_expr = ['-abs((', lhs_str, ') - (', rhs_str, '))'];
            otherwise
                error(['Unsupported operator: ', op]);
        end

        % Gradient via finite differences
        for s = 1:length(sig_names)
            var_name = sig_names{s};
            vars_eps = vars;
            vars_eps.(var_name) = vars_eps.(var_name) + eps;
            rob1 = eval_expr(rob_expr, vars_eps);
            rob0 = eval_expr(rob_expr, vars);
            grad.(var_name)(i) = (rob1 - rob0) / eps;
        end
    end
end


function val = eval_expr(expr_str, vars)
    % disp(expr_str);
    % Dynamically evaluate an expression string using signal values in vars
    expr_str = replace(expr_str, '[t]', '');  % remove [t] for eval
    for name = fieldnames(vars)'
        var = name{1};
        expr_str = strrep(expr_str, var, sprintf('vars.%s', var));
    end
    % disp(expr_str);
    % disp(vars.x1);
    % disp(vars.x2);
    val = eval(expr_str);
    % disp(val);
end


function step_vars = get_timestep(x_struct, t_idx)
    sig_names = fieldnames(x_struct);
    for i = 1:length(sig_names)
        step_vars.(sig_names{i}) = x_struct.(sig_names{i})(t_idx);
    end
end



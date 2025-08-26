function P = build_koopman_P(A, B, c, N)
% build_koopman_P
%   Construct Koopman affine maps P for N steps
%
% Inputs:
%   A - Koopman matrix for g(x0)
%   B - Input matrix
%   c - Constant offset (vector)
%   N - Horizon length
%
% Output:
%   P - cell array of length N, each with fields .A, .B, .c
    load('testkoopModel.mat', 'koopModel');
    N = koopModel.N;
    A = koopModel.A;
    A_ = A;
    B = koopModel.B;
    % c = koopModel.c;
    P = cell(1, N);
    for i = 1:N
        P{i}.A = A_ * A;
        P{i}.B = B;
        P{i}.c = c;
    end
end

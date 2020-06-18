% IDS/ACM/CS 158: Fundamentals of Statistical Learning
% PS5, Problem 1: Maximal Margin Hyperplane
% Author: Michael Li, mlli@caltech.edu
%-----------------------------------------------------------------------
clear;

D = readmatrix('dataset7.csv');
X = D(:, 1:end-1);
ys = D(:,end);
g_plus = D(ys == 1, 1:end-1);
g_minus = D(ys == -1, 1:end-1);

N = size(D,1);
p = size(D(1,1:end-1), 2);
X_with_bias = [ones(N,1),X];

% Primal values for beta
primal_margin = quadprog(eye(p+1), zeros(p+1, 1), ys.*X_with_bias*-1, -1+zeros(N, 1));
fprintf("\nPrimal Maximal Margin Hyperplane Beta: \n")
disp(primal_margin)

x = linspace(-3, 8, 10000);
f=@(x) (-primal_margin(2) / primal_margin(3))*x - (primal_margin(1) / primal_margin(3));
Y=f(x);

% Dual approach for beta
H = (ys*transpose(ys)) .* (X*transpose(X));
dual_margin = quadprog(H, -1*ones(1,N), zeros(1,N), 0, transpose(ys), 0, zeros(N,1), 10^10*ones(N,1));

% Find beta from lambdas
support_vecs = X(abs(dual_margin) > 10^-5, :);
beta = sum(dual_margin .* ys .* X, 1);
beta0 = -1/2 * (min(beta*transpose(g_plus)) + max(beta*transpose(g_minus)));
dual_beta = [beta0 beta];
fprintf("\nDual Maximal Margin Hyperplane Beta: \n")
disp(dual_beta)

% plot
figure
hold on
plot(x, Y, 'k')
plot(g_plus(:,1), g_plus(:, 2), 'or')
plot(g_minus(:,1), g_minus(:, 2), 'ob')
plot(support_vecs(:,1), support_vecs(:,2), 'og')
title('Dataset 7 with Maximal Margin Hyperplane and Support Vectors')
xlabel('X1')
ylabel('X2')

% primal margin hyperplane beta = [-13.6254, 2.7269, 3.2707]
% dual margin hyperplane beta = [-13.6254, 2.7269, 3.2707]

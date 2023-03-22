%Bayesian Dictionary Learning on Tubal-Transform Tensor Factorization 
% with shared lambda
% [X, S, Out] = VBDL(Y, opts)
%----------------------------------------------------------------
% Input:
%   Y:      input corrupted tensor
%   opts:   optional parameters
%       
% Output:
%   X:      low-rank component 
%   S:      sparse component
%   Out:    the output information
%----------------------------------------------------------------
% Copyright(c) 2022 Qilun Luo 
% All Rights Reserved.

function [X, S, Out] = VBDL(Y, opts)

%% Parameters
max_iter = 100;
tol = 1e-4;
init = 2; % initial: 1-max likelihood 2-random
r = min(size(Y, [1,2]));
debug = 1;
Prune = 0;
it_step = 10;
LMAX_ = 1e4;

% Hyper-priors
a0_lambda = 1e-1;
b0_lambda = 1e-3;
a0_gamma = 1e-1;
b0_gamma = 1e-3;
a0_beta = 1e-1;
b0_beta = 1e-3;
a0_tau = 1e-1;
b0_tau = 1e-3;

[n1, n2, n3] = size(Y);
YNorm = norm(Y(:));
YNorm2 = YNorm^2;
Yscale2 = (YNorm2/numel(Y));
Yscale = sqrt(Yscale2);

if ~exist('opts', 'var')
    opts = [];
end 
if  isfield(opts, 'max_iter');      max_iter = opts.max_iter;           end
if  isfield(opts, 'tol');           tol = opts.tol;                     end
if  isfield(opts, 'init');          init = opts.init;                   end
if  isfield(opts, 'r');             r = opts.r;                         end
if  isfield(opts, 'Prune');         Prune = opts.Prune;                 end
if  isfield(opts, 'debug');         debug = opts.debug;                 end
if  isfield(opts, 'it_step');       it_step = opts.it_step;             end
if  isfield(opts, 'LMAX_');         LMAX_ = opts.LMAX_;                 end
if  isfield(opts, 'a0_lambda');     a0_lambda = opts.a0_lambda;         end
if  isfield(opts, 'b0_lambda');     b0_lambda = opts.b0_lambda;         end
if  isfield(opts, 'a0_gamma');      a0_gamma = opts.a0_gamma;           end
if  isfield(opts, 'b0_gamma');      b0_gamma = opts.b0_gamma;           end
if  isfield(opts, 'a0_beta');       a0_beta = opts.a0_beta;             end
if  isfield(opts, 'b0_beta');       b0_beta = opts.b0_beta;             end
if  isfield(opts, 'a0_tau');        a0_tau = opts.a0_tau;               end
if  isfield(opts, 'b0_tau');        b0_tau = opts.b0_tau;               end
if  isfield(opts, 'lambda')          
    lambda = opts.lambda;                     
else
    lambda = rand(r, 1).*(a0_lambda./b0_lambda);
end
if  isfield(opts, 'gamma')          
    gamma = opts.gamma;                     
else
    gamma = rand(n1, n2, n3).*(a0_gamma./b0_gamma);
end
if  isfield(opts, 'beta')          
    beta = opts.beta;                     
else
    beta = a0_beta/b0_beta;
end
if  isfield(opts, 'tau')          
    tau = opts.tau;                     
else
    tau = a0_tau/b0_tau;
end


%% Initialization
% Factor tensors
switch init
    case 1
        % Dictionary matrix
        D = (rand(n3));
        YD = reshape((D\reshape(Y, [n1*n2, n3])')', [n1, n2, n3]);
        U = zeros(n1, r, n3);
        V = zeros(n2, r, n3);
        for k=1:n3
            [Uk, Sk, Vk] = svds(YD(:,:,k), r);
            U(:,:, k) = Uk*(Sk.^(0.5));
            V(:,:, k) = Vk*(Sk.^(0.5));
            
        end
        X = reshape(reshape(pagemtimes(U, 'none', V, 'ctranspose'), [n1*n2, n3])*D', [n1, n2, n3]);
    case 2
        D = rand(n3);
        U = rand(n1, r, n3);
        V = rand(n2, r, n3);
        X = reshape(reshape(pagemtimes(U, 'none', V, 'ctranspose'), [n1*n2, n3])*D', [n1, n2, n3]);  
end

% Sparse component
S = 1e-3*rand(n1, n2, n3);
% S = rand(n1, n2, n3)./(a0_gamma./b0_gamma);

Qv_inv = eye(r*n3);
Qd_inv = eye(n3);
Vm = reshape(V, [n2, r*n3])';
RD = n3*Qd_inv + D'*D;
RV = n2*Qv_inv + Vm*Vm';

%% Iteration
Out.rrse = [];
Out.rse = [];
Out.psnr = [];

for it = 1:max_iter
    S0 = S;
    X0 = X;
    XX = Y-S;
    
    %% Update U
    Qu = tau*RV.*(kron(RD, ones(r))) + diag(kron(ones(n3, 1), lambda));
    Qu_inv = eye(r*n3)/Qu;
    Tv = reshape(pagemtimes(reshape(kron(D, ones(1,r)), [n3, 1, r*n3]), ...
        reshape(V, [1, n2, r*n3])), [n3*n2, r*n3]);
    X1 = reshape(permute(XX, [3,2,1]),[n3*n2, n1]);
    Um = tau*Qu_inv*Tv'*X1;
    U = reshape(Um', [n1, r, n3]);
    RU = n1*Qu_inv + Um*Um';
    
    %% Update V
    Qv = tau*RU.*(kron(RD, ones(r))) + diag(kron(ones(n3, 1), lambda));
    Qv_inv = eye(r*n3)/Qv;
    Tu = reshape(pagemtimes(reshape(kron(D, ones(1,r)), [n3, 1, r*n3]), ...
        reshape(U, [1, n1, r*n3])), [n3*n1, r*n3]);
    X2 = reshape(permute(XX, [3,1,2]), [n3*n1, n2]);
    Vm = tau*Qv_inv*Tu'*X2;
    V = reshape(Vm', [n2, r, n3]);
    RV = n2*Qv_inv + Vm*Vm';
    
    %% Update D
    RUV0 = RU.*RV;
    RUV = squeeze(sum(reshape(RUV0, [r, n3, r, n3]), [1,3]));
    Qd = tau*RUV + beta*eye(n3);
    Qd_inv = eye(n3)/Qd;
    Z = pagemtimes(U, 'none', V, 'ctranspose');
    Z3 = reshape(Z, [n1*n2, n3]);
    X3 = reshape(XX, [n1*n2, n3]);
    D = tau*(X3'*Z3)*Qd_inv;
    RD = n3*Qd_inv + D'*D;

    %% Compute X
    X = reshape(Z3*D', [n1, n2, n3]);
    
    %% Update S
    Qs_inv = 1./(tau+gamma);
    S = tau*(Y-X).*Qs_inv;
    
    %% Update tau
    tau_chg = norm(Y(:)-S(:)-X(:))^2 + sum(Qs_inv, 'all')...
        - norm(X(:))^2 + trace(RD*RUV);
    a_tau = a0_tau + (n1*n2*n3)/2;
    b_tau = b0_tau + tau_chg/2;
    tau = a_tau/b_tau;
    
    %% Update lambda
    a_lambda = a0_lambda + (n1+n2)*n3/2;
    LU = reshape(permute(reshape(RU, [r, n3, r, n3]), [1,3,2,4]), [r^2, n3^2]);
    LV = reshape(permute(reshape(RV, [r, n3, r, n3]), [1,3,2,4]), [r^2, n3^2]);
    b_lambda = b0_lambda + (sum(LU(1:r+1:r^2, 1:n3+1:n3^2) ...
        + LV(1:r+1:r^2, 1:n3+1:n3^2), 2))/2;
    lambda = a_lambda./b_lambda(:);
    
    %% Update gamma
    a_gamma = a0_gamma + 1/2;
    b_gamma = b0_gamma + (S.*S+Qs_inv)/2;
    gamma = a_gamma./b_gamma;
    
    %% Update beta
    a_beta = a0_beta + n3^2/2;
    b_beta = b0_beta + trace(RD)/2;
    beta = a_beta./b_beta;
    
    %% Prune
    if Prune
        LMAX = min(lambda) * LMAX_;
        ind = find(lambda<LMAX);
        if length(ind)<r
            U = U(:,ind,:);
            V = V(:,ind,:);
            Z = pagemtimes(U, 'none', V, 'ctranspose');
            Z3 = reshape(Z, [n1*n2, n3]);
            X = reshape(Z3*D', [n1, n2, n3]);
            TRV = reshape(RV, [r, n3, r, n3]);
            r = length(ind);
            RV = reshape(TRV(ind, :, ind, :), r*n3, r*n3);
            lambda = lambda(ind);
        end
    end
    
    %% Check Convergence
    X_chg = norm(X(:)-X0(:))/norm(X0(:));
    S_chg = norm(S(:)-S0(:))/norm(S0(:));
    rrse = norm(X(:)-X0(:), 'fro')/norm(X0(:), 'fro');

    Out.rrse = [Out.rrse; rrse];
    rse = nan;
    mypsnr = nan;
    if isfield(opts, 'Xtrue')
        Xtrue = opts.Xtrue;
        rse = norm(X(:)-Xtrue(:), 'fro')/norm(Xtrue(:), 'fro');
        mypsnr = psnr(abs(X), abs(Xtrue), max(Xtrue(:)));
        Out.rse = [Out.rse; rse];
        Out.psnr = [Out.psnr; mypsnr];
    end

    if (debug && mod(it, it_step)==0)
        fprintf(['Iter %d: rrse=%.4f, rse = %.4f, psnr=%.4f, r = %d, ' ...
            'X_chg = %.5f, S_chg = %.5f, tau=%.2f.\n'], ...
            it, rrse, rse, mypsnr, r, X_chg, S_chg, tau);
    end

    if max([X_chg, S_chg, rrse])<tol
        break
    end         
end

%% Record the model
model.X = X;
model.S = S;
model.U = U;
model.V = V;
model.D = D;
model.r = r;
Out.model = model;








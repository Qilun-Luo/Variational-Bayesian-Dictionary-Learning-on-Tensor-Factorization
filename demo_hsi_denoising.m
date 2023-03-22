%% Test algs on hyperspectral images denoising

% clc
clear
close all

rng('default') % For reproducibility

%% Path Settings
addpath(genpath('utils/'))
addpath(genpath('algs/'));

% root_path = 'E:/data';
root_path = 'data';
data_path = fullfile(root_path, 'HSI');


data_name = {
    'Urban.mat',
    'Cuprite.mat',
    'Samson.mat',
    'Salinas.mat',
    'PaviaU.mat',
    'Indian_pines.mat',
};

% test_list = 1:6;
test_list = 3:3;

%% Algs settings
flag_VBDL = 1; % Proposed

%% Iteration
for i = test_list
    %% Loading data
    load(fullfile(data_path, data_name{i}));
    normalize = max(T(:));
    X = T/normalize;

    [n1,n2,n3] = size(X);
    Xn = X;
    
    rhos = 0.3;
    ind = find(rand(n1*n2*n3,1)<rhos);
    Xn(ind) = X(ind) + randn(length(ind),1);

    %% record 
    alg_name = {};
    alg_result = {};
    alg_out = {};
    alg_rse = {};
    alg_cpu = {};
    alg_ssim = {};
    alg_psnr = {};
    alg_cnt = 1;

    %% -- Sample
    X_dif_sample = Xn - X;
    X_rse_sample = norm(X_dif_sample(:))/norm(X(:));
    X_psnr_sample = psnr(Xn, X, max(X(:)));
    X_ssim_sample = ssim(Xn, X);
    % record
    alg_name{alg_cnt} = 'Sample';
    alg_result{alg_cnt} = Xn;
    alg_out{alg_cnt} = Xn;
    alg_cpu{alg_cnt} = 0;
    alg_rse{alg_cnt} = X_rse_sample;
    alg_ssim{alg_cnt} = X_ssim_sample;
    alg_psnr{alg_cnt} = X_psnr_sample;
    alg_cnt = alg_cnt + 1;

    %% VBDL
    if flag_VBDL
        opts = [];
        opts.tol = 1e-4;
        opts.max_iter = 100;
        opts.init = 2;
        opts.a0_lambda = 1e0;
        opts.b0_lambda = 1e0;
        opts.a0_gamma = 1e-1;
        opts.b0_gamma = 1e-6;
        opts.a0_beta = 1e0;
        opts.b0_beta = 1e0;
        opts.a0_tau = 1e0;
        opts.b0_tau = 1e0;

        opts.debug = 0;
        opts.Prune = 1;
        opts.it_step = 10;
        opts.LMAX_ = 1e4;
        opts.r = 10;
%         opts.r = min(size(X, [1,2]));
        opts.Xtrue = X;

        opts.gamma = rand(n1, n2, n3)*1e3;
        opts.beta = 1e0;
        opts.tau = 1e2;
        opts.lambda = rand(opts.r, 1);

        alg_name{alg_cnt} = 'VBDL';
        fprintf('Processing method: %12s\n', alg_name{alg_cnt});
        t_VBDL = tic;
        [X_VBDL, S_VBDL, Out_VBDL] = VBDL(Xn, opts);
        X_dif_VBDL = X_VBDL - X;
        X_rse_VBDL = norm(X_dif_VBDL(:))/norm(X(:));
        X_psnr_VBDL = psnr(abs(X_VBDL), X);
        X_ssim_VBDL = ssim(abs(X_VBDL), X);
        % record
        alg_result{alg_cnt} = X_VBDL;
        alg_out{alg_cnt} = Out_VBDL;
        alg_cpu{alg_cnt} = toc(t_VBDL);
        alg_rse{alg_cnt} = X_rse_VBDL;
        alg_ssim{alg_cnt} = X_ssim_VBDL;
        alg_psnr{alg_cnt} = X_psnr_VBDL;
        alg_cnt = alg_cnt + 1;
    end

    %% result table
    flag_report = 1;
    fprintf('Test on data: %s\n', data_name{i});

    if flag_report
        fprintf('%12s\t%4s\t%4s\t%4s\t%4s\n',...\
            'Algs', 'CPU', 'RSE', 'PSNR', 'SSIM');
        for j = 1:alg_cnt-1
            fprintf('%12s\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
                alg_name{j}, alg_cpu{j}, alg_rse{j}, alg_psnr{j}, alg_ssim{j});
        end
    end

    
end



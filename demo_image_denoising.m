%% Test algs on color image denoising

% clc
clear
close all

rng('default') % For reproducibility

%% Path Settings
addpath(genpath('utils/'))
addpath(genpath('algs/'));

% root_path = 'E:/data';
root_path = 'data';
data_path = fullfile(root_path, 'denoise_all');

ext_name = 'jpg';

imageNames = dir(fullfile(data_path, strcat('*.',ext_name)));
imageNames = {imageNames.name}';

%% Algs settings
flag_VBDL = 1;

result_all = [];

%% Iteration

% --- Run all the 200 images
% for i = 1:length(imageNames)

% --- Run the first 6 images
for i = 1:6

    im_name = fullfile(data_path,imageNames{i});
    fprintf('Runing the image %d: %s\n', i, im_name);
    X = double(imread(im_name))/255;
    maxP = max(abs(X(:)));
    [n1,n2,n3] = size(X);

    rhos = 0.3;


    Xn = X;    
    ind = find(rand(n1*n2,1)<rhos);
    for j = 1:n3
        tmp = X(:,:,j);
        tmp(ind) = tmp(ind) + randn(length(ind),1);
        Xn(:,:,j) = tmp;
    end    

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

    %% VBDL (Proposed)
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
        opts.r = 100;
%         opts.r = min(size(X, [1,2]));
        opts.Xtrue = X;

        opts.gamma = rand(n1, n2, n3)*1e3;
        opts.beta = 1e0;
        opts.tau = 1e3;
        opts.lambda = rand(opts.r, 1)*1e-2;


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
    if flag_report
        fprintf('%12s\t%4s\t%4s\t%4s\t%4s\n',...\
            'Algs', 'CPU', 'RSE', 'PSNR', 'SSIM');
        for j = 1:alg_cnt-1
            fprintf('%12s\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
                alg_name{j}, alg_cpu{j}, alg_rse{j}, alg_psnr{j}, alg_ssim{j});
        end
    end
    
    flag_draw = 0;
    if flag_draw
        alg_name_new = cat(2, 'Original', alg_name);
        alg_result_new = cat(2, X, alg_result);
        nrow = floor(sqrt(alg_cnt));
        ncol = ceil(alg_cnt/nrow);
        figure
        hold on
        for j = 1:alg_cnt
            subplot(nrow, ncol, j);
            imshow((alg_result_new{j}));
            title(alg_name_new{j});
        end
        hold off
    end
    
    save_flag = 0;
    
    if save_flag
        for j = 1:alg_cnt-1
            if isfield(result_all, alg_name{j})
                result_all.(alg_name{j}).image_name{end+1} = imageNames{i};
                result_all.(alg_name{j}).cpu(end+1) = alg_cpu{j};
                result_all.(alg_name{j}).rse(end+1) = alg_rse{j};
                result_all.(alg_name{j}).psnr(end+1) = alg_psnr{j};
                result_all.(alg_name{j}).ssim(end+1) = alg_ssim{j};
            else
                result_all.(alg_name{j}).image_name = {imageNames{i}};
                result_all.(alg_name{j}).cpu = alg_cpu{j};
                result_all.(alg_name{j}).rse= alg_rse{j};
                result_all.(alg_name{j}).psnr= alg_psnr{j};
                result_all.(alg_name{j}).ssim= alg_ssim{j};
            end
        end
    end
    
end
if save_flag
    save_path = 'output/results_image_denoising';
    if ~exist(save_path, 'dir')
	    mkdir(save_path)
    end
    save(fullfile(save_path, 'demo_image_denoising_result_all.mat'), 'result_all');
end

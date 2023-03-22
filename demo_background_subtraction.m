%% Test algs on background subtraction
clear
close all

addpath('utils')
addpath(genpath('algs/'));

%% load data
cate_list = {
    'shadow',
    'thermal',
    'shadow',
    'turbulence',
    'shadow',
};
file_list = {
    'busStation',
    'park',
    'peopleInShade',
    'turbulence3',
    'cubicle',
};
range_list = {
    [900, 1000],
    [300, 400],
    [900, 1000],
    [900, 1000],
    [1100, 1200],
};

res_draw = 0; % result plot: 0 - No; 1- Yes

%% Algs settings
flag_VBDL = 1; % Proposed

% root_path = 'E:/data';
root_path = 'data';

% t_list = 1:5;
t_list = 2:2;

for t = t_list

        %% Path setting
        category = cate_list{t};
        idxFrom = range_list{t}(1);
        idxTo = range_list{t}(2);
        file_name = file_list{t};

        data_path = fullfile(root_path, 'dataset2014', 'dataset');
        dataset_name = fullfile(category, file_name, 'input');
        output_path = fullfile(root_path, 'dataset2014', 'results');
        output_folder = fullfile(category, file_name);
        tmp_save_path = fullfile(root_path, 'tmp' ,'quantitative');

        if ~exist(tmp_save_path, 'dir')
            mkdir(tmp_save_path);
        end
        mat_file = fullfile(tmp_save_path, strcat(file_name, '_', num2str(idxFrom), '_', num2str(idxTo)));
        ext_name = 'jpg';
        show_flag = 0;
        if ~exist(strcat(mat_file, '_rgb.mat'), 'file')
            [X, height, width, imageNames] = load_video_for_quantitative(data_path, dataset_name, ext_name, show_flag, idxFrom, idxTo);
            save(strcat(mat_file, '_rgb.mat'), 'X', 'height', 'width', 'imageNames');
        else
            load(strcat(mat_file, '_rgb.mat'));
        end
        [height_width, dims, nframes] = size(X);

        %% Recorder
        alg_name = {};
        alg_result = {};
        alg_out = {};
        alg_cpu = {};
        alg_cnt = 1;

        %% VBDL
        if flag_VBDL
            Y = permute(X, [1,3,2]);
            % Y = reshape(X, [height, width, dims*nframes]);
            [n1,n2,n3] = size(Y);

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
            opts.r = 1;
    %         opts.r = min(size(X, [1,2]));

            opts.gamma = ones(n1, n2, n3)*1e4;
            opts.beta = 1e0;
            opts.tau = 1e3;
            opts.lambda = 0.5*rand(opts.r, 1);
 
            alg_name{alg_cnt} = 'VBDL';
            fprintf('Processing method: %12s\n', alg_name{alg_cnt});
            t_VBDL = tic;
            
            
            [X_VBDL, S_VBDL, Out_VBDL] = VBDL(Y, opts);

            X_VBDL = permute(X_VBDL, [1,3,2]);
            S_VBDL = permute(S_VBDL, [1,3,2]);

            X_VBDL = reshape(X_VBDL, [height_width, dims, nframes]);
            S_VBDL = reshape(S_VBDL, [height_width, dims, nframes]);

            if res_draw
                figure
            end
            for i = 1:nframes
                S_VBDL_frame = reshape(S_VBDL(:, :, i), [height, width, dims]);
                Tmask_VBDL = medfilt2(double(hard_threshold(mean(S_VBDL_frame,3))),[5 5]);

                if res_draw
                    subplot(1,3,1)
                    imshow(uint8(reshape(X_VBDL(:, :, i), [height, width, dims])))
                    subplot(1,3,2)  
                    imshow(uint8(S_VBDL_frame));
                    subplot(1,3,3)
                    imshow(Tmask_VBDL)
                end

                save_path = fullfile(output_path, output_folder, alg_name{alg_cnt});
                if ~exist(save_path, 'dir')
                    mkdir(save_path);
                end
                imwrite(Tmask_VBDL, fullfile(save_path, strcat('b', imageNames{i})));
                pause(0.01)
            end

            % record
            alg_result{alg_cnt} = S_VBDL;
            alg_out{alg_cnt} = Out_VBDL;
            alg_cpu{alg_cnt} = toc(t_VBDL);
            alg_cnt = alg_cnt + 1;
        end

        %% Compute quantitative measures
        extension = '.jpg';
        range = [idxFrom, idxTo];

        videoPath = fullfile(data_path, category, file_name);
        binaryFolder = fullfile(output_path, category, file_name);

        fprintf('===================================================\n')
        fprintf('Category: %s\tDateset: %s\n', category, file_name)
        fprintf('Alg_name\tCPU\tRecall\tPrecision\tFMeasure\n')
        for i = 1:alg_cnt-1
            [confusionMatrix, stats] = compute_measures(videoPath, fullfile(binaryFolder, alg_name{i}), range, extension);
            fprintf('%s\t%.2f\t%.4f\t%.4f\t%.4f\t\n', alg_name{i},alg_cpu{i}, stats(1), stats(6), stats(7))
        end

        
end


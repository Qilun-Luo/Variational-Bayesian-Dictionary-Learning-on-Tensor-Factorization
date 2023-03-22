function [X, height, width, imageNames] = load_video_for_quantitative(data_path, file_name, ext_name, show_flag, idxFrom, idxTo)
    imageNames = dir(fullfile(data_path, file_name, strcat('*.',ext_name)));
    imageNames = {imageNames.name}';
    imageNames = imageNames(idxFrom:idxTo);
    [height, width, dims] = size(imread(fullfile(data_path, file_name, imageNames{1})));
    nframes = length(imageNames);
    X = zeros(height*width, dims, nframes);
    if(show_flag)
        figure
    end
    for i = 1:length(imageNames)
       img = imread(fullfile(data_path,file_name,imageNames{i}));
       if(show_flag)
           imshow(img)
       end
       X(:, :, i) = reshape(img, [height*width, dims]);
    end
    
end
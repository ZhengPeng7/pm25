clear, clc;
%% Save TBV and entropy values.
root_path = '../../datasets/PM2.5data/fog_1508_data/';
scenes =  dir(fullfile(root_path, 'fog_*'));
scenes = {scenes.name};
TBVs = [];
entros = [];
for idx_scene = 1 : 1 : length(scenes)
    display(strcat(num2str(idx_scene), "-th scene."));
    scene = scenes{idx_scene};
    scene_dir = fullfile(root_path, scene);
    image_names = dir(fullfile(scene_dir, '*.jpg'));
    image_names = {image_names.name};
    for idx_image = 1 : 1 : length(image_names)
        image_name = image_names{idx_image};
        image_path = fullfile(scene_dir, image_name);
        image = imread(image_path);
        image = image(125:857-1, :, :);
        image_gray = rgb2gray(image);
        TBV = image_TBV_computing(image_gray);
        TBVs = [TBVs; TBV];
        entro = entropy(image);
        entros = [entros; entro];
    end
end

dlmwrite("TBVs.txt", TBVs);
dlmwrite('entropies.txt', entros);

%% Plot Ground Truth Bounding Boxes on PETS-S2L1 Sequence
% Reads gt.txt and draws bounding boxes on each frame.

clear; close all; clc;

%% Paths
gtFile   = fullfile('PETS-S2L1', 'gt', 'gt.txt');
imgDir   = fullfile('Crowd_PETS', 'S2', 'L1', 'Time_12-34', 'View_001');

%% Read ground truth
% Columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
gt = readmatrix(gtFile);

% Only keep entries with confidence == 1 (valid GT)
gt = gt(gt(:,7) == 1, :);

%% Get unique frame numbers
frames = unique(gt(:,1));

%% Display each frame with bounding boxes
figure('Name', 'GT Bounding Boxes', 'NumberTitle', 'off');

for i = 1:length(frames)
    frameNum = frames(i);

    % Read the image
    imgFile = fullfile(imgDir, sprintf('frame_%04d.jpg', frameNum - 1));
    if ~isfile(imgFile)
        continue;
    end
    img = imread(imgFile);

    % Get GT entries for this frame
    idx = gt(:,1) == frameNum;
    bboxes = gt(idx, 3:6);  % [left, top, width, height]
    ids    = gt(idx, 2);

    % Display image
    imshow(img); hold on;
    title(sprintf('Frame %d', frameNum), 'FontSize', 14);

    % Draw each bounding box
    for j = 1:size(bboxes, 1)
        rectangle('Position', bboxes(j,:), ...
                  'EdgeColor', 'g', 'LineWidth', 2);
        text(bboxes(j,1), bboxes(j,2) - 5, sprintf('ID:%d', ids(j)), ...
             'Color', 'yellow', 'FontSize', 8, 'FontWeight', 'bold');
    end
    hold off;

    drawnow;
    pause(0.05);  % ~20 fps playback
end

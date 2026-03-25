%% Task 2: Pedestrian Detection and Tracking using Background Subtraction
% Detects pedestrians via background subtraction + morphological filtering,
% extracts bounding boxes, and assigns labels to each detection.

clear; close all; clc;

%% Parameters (tune these as needed)
params.bgFrames      = 50;      % number of frames to build background model
params.fgThreshold   = 50;      % threshold on absolute difference (grayscale)
params.minBlobArea   = 800;     % minimum blob area to count as pedestrian
params.maxBlobArea   = 30000;   % maximum blob area (reject very large blobs)
params.seClose       = strel('disk', 7);   % structuring element for closing
params.seOpen        = strel('disk', 3);   % structuring element for opening

%% Paths
imgDir  = fullfile('Crowd_PETS', 'S2', 'L1', 'Time_12-34', 'View_001');
nFrames = 795;
%hey

%% Step 1: Build background model (median of first N frames)
fprintf('Building background model from %d frames...\n', params.bgFrames);
sample = imread(fullfile(imgDir, 'frame_0000.jpg'));
[H, W, ~] = size(sample);

bgStack = zeros(H, W, params.bgFrames, 'uint8');
for i = 1:params.bgFrames
    img = imread(fullfile(imgDir, sprintf('frame_%04d.jpg', i - 1)));
    bgStack(:,:,i) = rgb2gray(img);
end
bgModel = median(bgStack, 3);  % median background (robust to moving objects)
fprintf('Background model built.\n');

%% Step 2: Process each frame - detect and display bounding boxes
figure('Name', 'Pedestrian Detection & Tracking', 'NumberTitle', 'off', ...
       'Position', [100 100 1100 500]);

detectionLabel = 0;  % global label counter

for f = 1:nFrames
    % Read frame
    imgFile = fullfile(imgDir, sprintf('frame_%04d.jpg', f - 1));
    if ~isfile(imgFile)
        continue;
    end
    img = imread(imgFile);
    gray = rgb2gray(img);

    % --- Foreground segmentation ---
    % Absolute difference from background model
    fgMask = abs(double(gray) - double(bgModel)) > params.fgThreshold;

    % Morphological cleanup: close gaps, then remove small noise
    fgMask = imclose(fgMask, params.seClose);
    fgMask = imopen(fgMask, params.seOpen);

    % Fill holes inside blobs
    fgMask = imfill(fgMask, 'holes');

    % --- Blob analysis: extract bounding boxes ---
    stats = regionprops(fgMask, 'BoundingBox', 'Area', 'Centroid');

    % Filter by area
    areas = [stats.Area];
    valid = (areas >= params.minBlobArea) & (areas <= params.maxBlobArea);
    stats = stats(valid);

    % --- Display ---
    subplot(1,2,1);
    imshow(fgMask);
    title(sprintf('Foreground Mask - Frame %d', f), 'FontSize', 12);

    subplot(1,2,2);
    imshow(img); hold on;
    title(sprintf('Detections - Frame %d  (%d found)', f, length(stats)), ...
          'FontSize', 12);

    for j = 1:length(stats)
        detectionLabel = detectionLabel + 1;
        bb = stats(j).BoundingBox;
        ct = stats(j).Centroid;

        % Draw bounding box
        rectangle('Position', bb, 'EdgeColor', 'r', 'LineWidth', 2);

        % Draw label
        text(bb(1), bb(2) - 8, sprintf('%d', detectionLabel), ...
             'Color', 'cyan', 'FontSize', 9, 'FontWeight', 'bold', ...
             'BackgroundColor', [0 0 0 0.4]);

        % Draw centroid
        plot(ct(1), ct(2), 'r+', 'MarkerSize', 8, 'LineWidth', 1.5);
    end
    hold off;

    drawnow;
    pause(0.03);
end

fprintf('Done. Total detections across all frames: %d\n', detectionLabel);

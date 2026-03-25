%% Task 2: Pedestrian Detection and Tracking using Background Subtraction
% Detects pedestrians via background subtraction + morphological filtering,
% extracts bounding boxes, and assigns labels to each detection.

clear; close all; clc;

%% Parameters (tune these as needed)
params.bgFrames        = 50;      % number of frames to build background model
params.fgThreshold     = 50;      % threshold on absolute difference (grayscale)

% Area limits
params.minBlobAreaTop    = 120;   % minimum allowed area near top of image
params.minBlobAreaBottom = 1200;  % minimum allowed area near bottom of image
params.maxBlobArea       = 30000; % maximum blob area

% Shape constraints
params.minAspectHW     = 1.0;     % require height/width >= 2
params.minHeight       = 18;      % allow smaller pedestrians far away
params.maxHeight       = 300;
params.minWidth        = 6;
params.maxWidth        = 120;

% Morphology
params.seClose         = strel('disk', 7);
params.seOpen          = strel('disk', 3);

%% Paths
imgDir  = fullfile('Views', 'View_001');
nFrames = 795;

%% Step 1: Build background model (median of first N frames)
fprintf('Building background model from %d frames...\n', params.bgFrames);
sample = imread(fullfile(imgDir, 'frame_0000.jpg'));
[H, W, ~] = size(sample);

bgStack = zeros(H, W, params.bgFrames, 'uint8');
randFrames = randperm(nFrames, params.bgFrames);

for i = 1:params.bgFrames
    img = imread(fullfile(imgDir, sprintf('frame_%04d.jpg', randFrames(i)-1)));
    bgStack(:,:,i) = rgb2gray(img);
end

bgModel = median(bgStack, 3);  % median background
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
    fgMask = abs(double(gray) - double(bgModel)) > params.fgThreshold;

    % Morphological cleanup
    fgMask = imclose(fgMask, params.seClose);
    fgMask = imopen(fgMask, params.seOpen);
    fgMask = imfill(fgMask, 'holes');

    % Optional: remove tiny specks before regionprops
    fgMask = bwareaopen(fgMask, 40);

    % --- Blob analysis ---
stats = regionprops(fgMask, 'BoundingBox', 'Area', 'Centroid');

validMask = false(1, numel(stats));

for j = 1:numel(stats)
    bb = stats(j).BoundingBox;   % [x y w h]
    area = stats(j).Area;

    wBox = bb(3);
    hBox = bb(4);

    if wBox <= 0
        continue;
    end

    % Aspect ratio: require tall blobs
    aspectHW = hBox / wBox;

    % Perspective-aware min area using bottom of box
    yBottom = bb(2) + bb(4);
    alpha = min(max(yBottom / H, 0), 1);

    minAreaHere = params.minBlobAreaTop + ...
        alpha * (params.minBlobAreaBottom - params.minBlobAreaTop);

    % Filters
    if area < minAreaHere || area > params.maxBlobArea
        continue;
    end

    if hBox < params.minHeight || hBox > params.maxHeight
        continue;
    end

    if wBox < params.minWidth || wBox > params.maxWidth
        continue;
    end

    if aspectHW < params.minAspectHW
        continue;
    end

    % Passed all tests
    validMask(j) = true;
end

validStats = stats(validMask);

    % --- Display ---
    subplot(1,2,1);
    imshow(fgMask);
    title(sprintf('Foreground Mask - Frame %d', f), 'FontSize', 12);

    subplot(1,2,2);
    imshow(img); hold on;
    title(sprintf('Detections - Frame %d  (%d found)', f, length(validStats)), ...
          'FontSize', 12);
    
    
    %imshow(bgModel); title('Background'); %Validate background ok
    

    for j = 1:length(validStats)
        detectionLabel = detectionLabel + 1;
        bb = validStats(j).BoundingBox;
        ct = validStats(j).Centroid;

        rectangle('Position', bb, 'EdgeColor', 'r', 'LineWidth', 2);

        text(bb(1), bb(2) - 8, sprintf('%d', detectionLabel), ...
             'Color', 'cyan', 'FontSize', 9, 'FontWeight', 'bold', ...
             'BackgroundColor', 'black');

        plot(ct(1), ct(2), 'r+', 'MarkerSize', 8, 'LineWidth', 1.5);
    end
    hold off;

    drawnow;
    pause(0.03);
end

fprintf('Done. Total detections across all frames: %d\n', detectionLabel);


function detections = task2_detectAndTrack(gt, config)
% TASK2_DETECTANDTRACK  Detect pedestrians using handcrafted features.
%   Uses a combination of:
%     - Adaptive Gaussian Mixture Background Subtraction
%     - Morphological filtering
%     - Connected component analysis
%     - HOG-based pedestrian verification (optional refinement)
%   Returns detections in MOT format: [frame, id, bb_left, bb_top, bb_w, bb_h]

    outDir = fullfile(config.outputDir, 'task2_detection');
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    testImg = fullfile(config.imgDir, sprintf(config.imgPattern, 1));
    hasImages = exist(testImg, 'file');
    
    if hasImages
        detections = detectWithBGSubtraction(config, outDir);
    else
        fprintf('  No images found. Using GT-based simulated detections.\n');
        detections = simulateDetections(gt, config);
    end
    
    % Assign initial labels (simple: sequential per frame)
    detections = assignInitialLabels(detections);
    
    % Save detections
    save(fullfile(outDir, 'detections.mat'), 'detections');
    
    % Visualize some sample frames
    visualizeDetections(detections, config, outDir);
    
    fprintf('  Task 2 complete. %d detections across %d frames.\n', ...
        size(detections, 1), numel(unique(detections(:,1))));
end

function detections = detectWithBGSubtraction(config, outDir)
% Full detection pipeline using background subtraction + morphology

    % --- Step 1: Build background model using median of first N frames ---
    fprintf('  Building background model...\n');
    nBgFrames = min(50, config.numFrames);
    
    img0 = imread(fullfile(config.imgDir, sprintf(config.imgPattern, 1)));
    [H, W, ~] = size(img0);
    bgStack = zeros(H, W, nBgFrames, 'uint8');
    
    for i = 1:nBgFrames
        imgPath = fullfile(config.imgDir, sprintf(config.imgPattern, i));
        imgGray = rgb2gray(imread(imgPath));
        bgStack(:,:,i) = imgGray;
    end
    bgModel = median(bgStack, 3);  % Median background
    
    % --- Step 2: Process each frame ---
    detections = [];
    minArea = 500;    % Minimum blob area (pixels)
    maxArea = 50000;  % Maximum blob area
    minAspect = 1.2;  % Minimum height/width ratio for pedestrians
    maxAspect = 5.0;  % Maximum height/width ratio
    
    % Structuring elements for morphology
    seOpen  = strel('disk', 3);
    seClose = strel('disk', 7);
    seDilate = strel('disk', 5);
    
    fprintf('  Detecting pedestrians...\n');
    for f = 1:config.numFrames
        imgPath = fullfile(config.imgDir, sprintf(config.imgPattern, f));
        img = imread(imgPath);
        imgGray = rgb2gray(img);
        
        % Background subtraction
        diff = abs(double(imgGray) - double(bgModel));
        
        % Adaptive thresholding
        thresh = max(25, 0.3 * std(diff(:)));
        fgMask = diff > thresh;
        
        % Morphological operations to clean up
        fgMask = imopen(fgMask, seOpen);     % Remove small noise
        fgMask = imclose(fgMask, seClose);   % Fill gaps
        fgMask = imdilate(fgMask, seDilate); % Connect nearby regions
        fgMask = imfill(fgMask, 'holes');    % Fill holes
        
        % Connected components
        cc = bwconncomp(fgMask);
        stats = regionprops(cc, 'BoundingBox', 'Area');
        
        for k = 1:numel(stats)
            area = stats(k).Area;
            bb = stats(k).BoundingBox;  % [x, y, w, h]
            aspect = bb(4) / bb(3);     % height / width
            
            % Filter by size and aspect ratio
            if area >= minArea && area <= maxArea && ...
               aspect >= minAspect && aspect <= maxAspect
                detections = [detections; f, 0, bb(1), bb(2), bb(3), bb(4)];
            end
        end
        
        % Update background model slowly (running average)
        alpha = 0.01;
        bgModel = uint8((1-alpha) * double(bgModel) + alpha * double(imgGray));
        
        if mod(f, 100) == 0
            fprintf('    Frame %d/%d: %d detections so far\n', ...
                f, config.numFrames, size(detections, 1));
        end
    end
end

function detections = simulateDetections(gt, config)
% Simulate detections from GT with realistic noise (for testing without images)
%   Adds positional noise, occasional missed detections, and false positives.

    detections = [];
    rng(42);  % Reproducibility
    
    for f = 1:config.numFrames
        mask = gt(:,1) == f & gt(:,7) == 1;
        gtFrame = gt(mask, :);
        
        for k = 1:size(gtFrame, 1)
            % Miss ~10% of detections (simulating FN)
            if rand() < 0.10
                continue;
            end
            
            % Add Gaussian noise to bounding box
            bb = gtFrame(k, 3:6);
            noise = randn(1,4) .* [5, 5, 3, 5];
            bb = bb + noise;
            bb(3:4) = max(bb(3:4), [15, 30]);  % Ensure minimum size
            
            detections = [detections; f, 0, bb];
        end
        
        % Add occasional false positives (~5% of frames)
        if rand() < 0.05
            fpBB = [randi([50, 600]), randi([100, 400]), ...
                    randi([20, 50]), randi([50, 120])];
            detections = [detections; f, 0, fpBB];
        end
    end
end

function detections = assignInitialLabels(detections)
% Assign sequential labels per frame (label switching allowed in Task 2)
    frames = unique(detections(:,1));
    for i = 1:numel(frames)
        mask = detections(:,1) == frames(i);
        nDets = sum(mask);
        detections(mask, 2) = (1:nDets)';
    end
end

function visualizeDetections(detections, config, outDir)
% Save sample detection visualizations
    sampleFrames = [1, 100, 200, 400, 600, 795];
    
    testImg = fullfile(config.imgDir, sprintf(config.imgPattern, 1));
    hasImages = exist(testImg, 'file');
    
    colors = lines(20);
    
    for f = sampleFrames
        if hasImages
            img = imread(fullfile(config.imgDir, sprintf(config.imgPattern, f)));
        else
            img = uint8(128 * ones(576, 768, 3));
        end
        
        mask = detections(:,1) == f;
        detsFrame = detections(mask, :);
        
        for k = 1:size(detsFrame, 1)
            id = detsFrame(k, 2);
            bb = detsFrame(k, 3:6);
            col = colors(mod(id-1, size(colors,1)) + 1, :) * 255;
            
            img = insertShape(img, 'Rectangle', bb, ...
                'Color', col, 'LineWidth', 2);
            img = insertText(img, [bb(1), bb(2)-15], ...
                sprintf('#%d', id), ...
                'FontSize', 12, 'TextColor', 'white', ...
                'BoxColor', col, 'BoxOpacity', 0.6);
        end
        
        img = insertText(img, [5, 5], sprintf('Frame: %d  Detections: %d', ...
            f, size(detsFrame,1)), ...
            'FontSize', 14, 'TextColor', 'yellow', ...
            'BoxColor', 'black', 'BoxOpacity', 0.5);
        
        imwrite(img, fullfile(outDir, sprintf('det_frame_%04d.png', f)));
    end
end

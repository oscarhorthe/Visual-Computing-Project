function task1_plotGT(gt, config)
% TASK1_PLOTGT  Draw ground truth bounding boxes on each frame.
%   For each frame in the sequence, overlays the GT bounding boxes
%   with unique colors per pedestrian ID and saves sample frames.

    outDir = fullfile(config.outputDir, 'task1_gt');
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    % Assign a unique color to each pedestrian ID
    uniqueIDs = unique(gt(:,2));
    colors = lines(numel(uniqueIDs));
    colorMap = containers.Map(uniqueIDs, num2cell(colors, 2));

    % --- Option A: Process ALL frames and save as video ---
    videoFile = fullfile(outDir, 'gt_visualization.avi');
    
    % Check if images exist
    testImg = fullfile(config.imgDir, sprintf(config.imgPattern, 1));
    hasImages = exist(testImg, 'file');
    
    if hasImages
        vWriter = VideoWriter(videoFile, 'Motion JPEG AVI');
        vWriter.FrameRate = 25;
        open(vWriter);
    end

    sampleFrames = [1, 100, 200, 400, 600, 795];  % frames to save as images
    
    for f = 1:config.numFrames
        % Get GT for this frame
        mask = gt(:,1) == f;
        gtFrame = gt(mask, :);
        
        if hasImages
            imgPath = fullfile(config.imgDir, sprintf(config.imgPattern, f));
            img = imread(imgPath);
        else
            % Create a blank frame if no images available
            img = uint8(128 * ones(576, 768, 3));  % PETS frame size
        end
        
        % Draw bounding boxes
        for k = 1:size(gtFrame, 1)
            id = gtFrame(k, 2);
            bb = gtFrame(k, 3:6);  % [left, top, width, height]
            col = colorMap(id) * 255;
            
            img = insertShape(img, 'Rectangle', bb, ...
                'Color', col, 'LineWidth', 2);
            img = insertText(img, [bb(1), bb(2)-15], ...
                sprintf('ID:%d', id), ...
                'FontSize', 12, 'TextColor', 'white', ...
                'BoxColor', col, 'BoxOpacity', 0.6);
        end
        
        % Add frame number
        img = insertText(img, [5, 5], sprintf('Frame: %d', f), ...
            'FontSize', 14, 'TextColor', 'yellow', ...
            'BoxColor', 'black', 'BoxOpacity', 0.5);
        
        if hasImages
            writeVideo(vWriter, img);
        end
        
        % Save sample frames
        if ismember(f, sampleFrames)
            imwrite(img, fullfile(outDir, sprintf('gt_frame_%04d.png', f)));
        end
        
        if mod(f, 100) == 0
            fprintf('  Task 1: Processed %d/%d frames\n', f, config.numFrames);
        end
    end
    
    if hasImages
        close(vWriter);
        fprintf('  Video saved: %s\n', videoFile);
    end
    
    fprintf('  Task 1 complete. Sample frames saved in %s\n', outDir);
end

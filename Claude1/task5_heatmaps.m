function task5_heatmaps(detections, config)
% TASK5_HEATMAPS  Generate occupancy heatmaps of pedestrian trajectories.
%   Creates:
%     (i)  Static heatmap: accumulated over all frames
%     (ii) Dynamic heatmap: evolving over time (video)

    outDir = fullfile(config.outputDir, 'task5_heatmaps');
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    % Frame dimensions (PETS S2.L1 View001 typical size)
    H = 576; W = 768;
    
    testImg = fullfile(config.imgDir, sprintf(config.imgPattern, 1));
    if exist(testImg, 'file')
        img0 = imread(testImg);
        [H, W, ~] = size(img0);
    end
    
    % Compute centroids (bottom-center of bounding box = foot position)
    cx = detections(:,3) + detections(:,5)/2;
    cy = detections(:,4) + detections(:,6);
    frames = detections(:,1);
    
    % Gaussian kernel parameters
    sigma = 20;  % Standard deviation of Gaussian
    kernelSize = 6 * sigma + 1;
    [Xk, Yk] = meshgrid(-3*sigma:3*sigma, -3*sigma:3*sigma);
    gaussKernel = exp(-(Xk.^2 + Yk.^2) / (2 * sigma^2));
    
    %% --- (i) Static Heatmap ---
    fprintf('  Generating static heatmap...\n');
    heatmap = zeros(H, W);
    
    for k = 1:numel(cx)
        x = round(cx(k));
        y = round(cy(k));
        
        % Bounds checking
        x1 = max(1, x - 3*sigma); x2 = min(W, x + 3*sigma);
        y1 = max(1, y - 3*sigma); y2 = min(H, y + 3*sigma);
        
        kx1 = x1 - (x - 3*sigma) + 1; kx2 = kernelSize - ((x + 3*sigma) - x2);
        ky1 = y1 - (y - 3*sigma) + 1; ky2 = kernelSize - ((y + 3*sigma) - y2);
        
        if x1 <= x2 && y1 <= y2 && kx1 >= 1 && ky1 >= 1
            heatmap(y1:y2, x1:x2) = heatmap(y1:y2, x1:x2) + ...
                gaussKernel(ky1:ky2, kx1:kx2);
        end
    end
    
    % Normalize
    heatmap = heatmap / max(heatmap(:));
    
    % Visualize
    fig1 = figure('Visible', 'off', 'Position', [100 100 900 650]);
    
    if exist(testImg, 'file')
        bgImg = imread(testImg);
    else
        bgImg = uint8(128 * ones(H, W, 3));
    end
    
    imshow(bgImg); hold on;
    hImg = imagesc(heatmap);
    colormap('jet');
    set(hImg, 'AlphaData', heatmap * 0.7);  % Transparency
    colorbar('FontSize', 10);
    title('Static Occupancy Heatmap (All Frames)', 'FontSize', 14);
    hold off;
    
    saveas(fig1, fullfile(outDir, 'heatmap_static.png'));
    close(fig1);
    
    %% --- (ii) Dynamic Heatmap (video) ---
    fprintf('  Generating dynamic heatmap video...\n');
    
    videoFile = fullfile(outDir, 'heatmap_dynamic.avi');
    vWriter = VideoWriter(videoFile, 'Motion JPEG AVI');
    vWriter.FrameRate = 25;
    open(vWriter);
    
    decayFactor = 0.95;  % How fast old contributions fade
    dynamicHeat = zeros(H, W);
    sigmaD = 15;  % Slightly smaller for dynamic
    kernelSizeD = 6 * sigmaD + 1;
    [Xd, Yd] = meshgrid(-3*sigmaD:3*sigmaD, -3*sigmaD:3*sigmaD);
    gaussKernelD = exp(-(Xd.^2 + Yd.^2) / (2 * sigmaD^2));
    
    for f = 1:config.numFrames
        % Decay previous heat
        dynamicHeat = dynamicHeat * decayFactor;
        
        % Add current frame detections
        mask = frames == f;
        for k = find(mask)'
            x = round(cx(k));
            y = round(cy(k));
            
            x1 = max(1, x - 3*sigmaD); x2 = min(W, x + 3*sigmaD);
            y1 = max(1, y - 3*sigmaD); y2 = min(H, y + 3*sigmaD);
            
            kx1 = x1 - (x - 3*sigmaD) + 1; kx2 = kernelSizeD - ((x + 3*sigmaD) - x2);
            ky1 = y1 - (y - 3*sigmaD) + 1; ky2 = kernelSizeD - ((y + 3*sigmaD) - y2);
            
            if x1 <= x2 && y1 <= y2 && kx1 >= 1 && ky1 >= 1
                dynamicHeat(y1:y2, x1:x2) = dynamicHeat(y1:y2, x1:x2) + ...
                    gaussKernelD(ky1:ky2, kx1:kx2);
            end
        end
        
        % Normalize for display
        if max(dynamicHeat(:)) > 0
            dispHeat = dynamicHeat / max(dynamicHeat(:));
        else
            dispHeat = dynamicHeat;
        end
        
        % Create RGB heatmap overlay
        heatRGB = ind2rgb(uint8(dispHeat * 255), jet(256));
        
        if exist(testImg, 'file')
            imgPath = fullfile(config.imgDir, sprintf(config.imgPattern, f));
            if exist(imgPath, 'file')
                bgImg = im2double(imread(imgPath));
            end
        else
            bgImg = 0.5 * ones(H, W, 3);
        end
        
        % Blend
        alpha = repmat(dispHeat * 0.7, [1, 1, 3]);
        blended = uint8(255 * ((1 - alpha) .* bgImg + alpha .* heatRGB));
        
        % Add text
        blended = insertText(blended, [5, 5], sprintf('Frame: %d', f), ...
            'FontSize', 14, 'TextColor', 'yellow', ...
            'BoxColor', 'black', 'BoxOpacity', 0.5);
        
        writeVideo(vWriter, blended);
        
        % Save snapshots
        if ismember(f, [1, 200, 400, 600, 795])
            imwrite(blended, fullfile(outDir, sprintf('heatmap_dynamic_%04d.png', f)));
        end
        
        if mod(f, 200) == 0
            fprintf('    Frame %d/%d\n', f, config.numFrames);
        end
    end
    
    close(vWriter);
    
    %% --- Heatmap per pedestrian ID ---
    fprintf('  Generating per-ID heatmaps...\n');
    uniqueIDs = unique(detections(:,2));
    
    if numel(uniqueIDs) <= 20  % Only if reasonable number
        fig2 = figure('Visible', 'off', 'Position', [100 100 1200 800]);
        nPlots = min(numel(uniqueIDs), 12);
        nCols = ceil(sqrt(nPlots));
        nRows = ceil(nPlots / nCols);
        
        for i = 1:nPlots
            id = uniqueIDs(i);
            idMask = detections(:,2) == id;
            
            idHeat = zeros(H, W);
            for k = find(idMask)'
                x = round(cx(k)); y = round(cy(k));
                x1 = max(1, x-3*sigma); x2 = min(W, x+3*sigma);
                y1 = max(1, y-3*sigma); y2 = min(H, y+3*sigma);
                kx1 = x1-(x-3*sigma)+1; kx2 = kernelSize-((x+3*sigma)-x2);
                ky1 = y1-(y-3*sigma)+1; ky2 = kernelSize-((y+3*sigma)-y2);
                if x1<=x2 && y1<=y2 && kx1>=1 && ky1>=1
                    idHeat(y1:y2,x1:x2) = idHeat(y1:y2,x1:x2) + ...
                        gaussKernel(ky1:ky2,kx1:kx2);
                end
            end
            
            subplot(nRows, nCols, i);
            imagesc(idHeat); colormap('jet');
            title(sprintf('ID %d', id), 'FontSize', 10);
            axis image; axis off;
        end
        sgtitle('Per-Pedestrian Heatmaps', 'FontSize', 14);
        saveas(fig2, fullfile(outDir, 'heatmaps_per_id.png'));
        close(fig2);
    end
    
    fprintf('  Task 5 complete. Results saved in %s\n', outDir);
end

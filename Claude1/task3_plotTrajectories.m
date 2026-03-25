function task3_plotTrajectories(detections, config)
% TASK3_PLOTTRAJECTORIES  Visualize pedestrian trajectories dynamically.
%   Plots trajectory paths with a fading tail effect to avoid clutter.
%   Creates both a static overview and a dynamic video.

    outDir = fullfile(config.outputDir, 'task3_trajectories');
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    % --- Get trajectory centroids ---
    % Centroid = (bb_left + bb_w/2, bb_top + bb_h)  (bottom-center of BB)
    centroids = [detections(:,1:2), ...
                 detections(:,3) + detections(:,5)/2, ...
                 detections(:,4) + detections(:,6)];
    % centroids: [frame, id, cx, cy]
    
    uniqueIDs = unique(detections(:,2));
    colors = hsv(numel(uniqueIDs));
    
    %% --- 1. Static trajectory plot (all trajectories on one image) ---
    fprintf('  Generating static trajectory plot...\n');
    
    testImg = fullfile(config.imgDir, sprintf(config.imgPattern, 1));
    if exist(testImg, 'file')
        bgImg = imread(testImg);
    else
        bgImg = uint8(128 * ones(576, 768, 3));
    end
    
    fig1 = figure('Visible', 'off', 'Position', [100 100 900 650]);
    imshow(bgImg); hold on;
    title('All Pedestrian Trajectories (Static)', 'FontSize', 14);
    
    for i = 1:numel(uniqueIDs)
        id = uniqueIDs(i);
        mask = centroids(:,2) == id;
        traj = centroids(mask, 3:4);
        
        if size(traj, 1) > 1
            plot(traj(:,1), traj(:,2), '-', 'Color', colors(i,:), ...
                'LineWidth', 2);
            % Mark start and end
            plot(traj(1,1), traj(1,2), 'o', 'Color', colors(i,:), ...
                'MarkerSize', 8, 'MarkerFaceColor', colors(i,:));
            plot(traj(end,1), traj(end,2), 's', 'Color', colors(i,:), ...
                'MarkerSize', 8, 'MarkerFaceColor', colors(i,:));
        end
    end
    
    legend_str = arrayfun(@(x) sprintf('ID %d', x), uniqueIDs, 'UniformOutput', false);
    % Only show legend for reasonable number of IDs
    if numel(uniqueIDs) <= 20
        legend(legend_str, 'Location', 'eastoutside', 'FontSize', 8);
    end
    hold off;
    
    saveas(fig1, fullfile(outDir, 'trajectories_static.png'));
    close(fig1);
    
    %% --- 2. Dynamic trajectory plot (fading tail, video) ---
    fprintf('  Generating dynamic trajectory video...\n');
    
    tailLength = 30;  % Number of frames to show in the trail
    
    videoFile = fullfile(outDir, 'trajectories_dynamic.avi');
    vWriter = VideoWriter(videoFile, 'Motion JPEG AVI');
    vWriter.FrameRate = 25;
    open(vWriter);
    
    for f = 1:config.numFrames
        fig2 = figure('Visible', 'off', 'Position', [100 100 768 576]);
        
        if exist(testImg, 'file')
            imgPath = fullfile(config.imgDir, sprintf(config.imgPattern, f));
            if exist(imgPath, 'file')
                bgImg = imread(imgPath);
            end
        end
        
        imshow(bgImg); hold on;
        
        for i = 1:numel(uniqueIDs)
            id = uniqueIDs(i);
            mask = centroids(:,2) == id & ...
                   centroids(:,1) >= max(1, f - tailLength) & ...
                   centroids(:,1) <= f;
            traj = centroids(mask, [1, 3, 4]);
            traj = sortrows(traj, 1);
            
            if size(traj, 1) > 1
                % Draw trail with fading alpha
                for j = 2:size(traj, 1)
                    alpha = j / size(traj, 1);  % Fade in
                    plot(traj(j-1:j, 2), traj(j-1:j, 3), '-', ...
                        'Color', [colors(i,:), alpha], ...
                        'LineWidth', max(1, 3 * alpha));
                end
            end
            
            % Draw current position marker
            currMask = centroids(:,2) == id & centroids(:,1) == f;
            if any(currMask)
                pos = centroids(currMask, 3:4);
                plot(pos(1), pos(2), 'o', 'Color', colors(i,:), ...
                    'MarkerSize', 8, 'MarkerFaceColor', colors(i,:));
            end
        end
        
        title(sprintf('Frame %d / %d', f, config.numFrames), 'FontSize', 12);
        hold off;
        
        frame = getframe(fig2);
        writeVideo(vWriter, frame.cdata);
        close(fig2);
        
        if mod(f, 100) == 0
            fprintf('    Frame %d/%d\n', f, config.numFrames);
        end
    end
    
    close(vWriter);
    
    %% --- 3. Trajectory direction plot ---
    fprintf('  Generating trajectory direction plot...\n');
    fig3 = figure('Visible', 'off', 'Position', [100 100 900 650]);
    
    if exist(testImg, 'file')
        imshow(imread(fullfile(config.imgDir, sprintf(config.imgPattern, 1)))); 
    else
        imshow(uint8(128 * ones(576, 768, 3)));
    end
    hold on;
    
    for i = 1:numel(uniqueIDs)
        id = uniqueIDs(i);
        mask = centroids(:,2) == id;
        traj = centroids(mask, 3:4);
        
        if size(traj, 1) > 5
            % Draw arrows showing direction
            step = max(1, floor(size(traj,1) / 10));
            for j = 1:step:size(traj,1)-step
                dx = traj(j+step, 1) - traj(j, 1);
                dy = traj(j+step, 2) - traj(j, 2);
                quiver(traj(j,1), traj(j,2), dx, dy, 0, ...
                    'Color', colors(i,:), 'LineWidth', 1.5, ...
                    'MaxHeadSize', 2);
            end
        end
    end
    title('Trajectory Directions', 'FontSize', 14);
    hold off;
    
    saveas(fig3, fullfile(outDir, 'trajectories_directions.png'));
    close(fig3);
    
    fprintf('  Task 3 complete. Results saved in %s\n', outDir);
end

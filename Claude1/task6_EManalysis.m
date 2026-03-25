function task6_EManalysis(detections, config)
% TASK6_EMANALYSIS  Statistical analysis of trajectories using EM algorithm.
%   Applies Gaussian Mixture Models (GMM) via EM to:
%     - Cluster trajectory endpoints (entry/exit points)
%     - Model spatial distribution of pedestrian positions
%     - Identify dominant movement corridors

    outDir = fullfile(config.outputDir, 'task6_EM');
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    
    H = 576; W = 768;
    
    testImg = fullfile(config.imgDir, sprintf(config.imgPattern, 1));
    if exist(testImg, 'file')
        img0 = imread(testImg);
        [H, W, ~] = size(img0);
    end

    %% --- 1. Cluster ALL pedestrian positions using GMM ---
    fprintf('  Fitting GMM to all pedestrian positions...\n');
    
    cx = detections(:,3) + detections(:,5)/2;
    cy = detections(:,4) + detections(:,6)/2;
    positions = [cx, cy];
    
    % Try different numbers of components
    maxK = 8;
    BIC = zeros(1, maxK);
    AIC = zeros(1, maxK);
    
    for k = 1:maxK
        try
            gmm = fitgmdist(positions, k, 'Replicates', 5, ...
                'RegularizationValue', 0.01, 'Options', ...
                statset('MaxIter', 500));
            BIC(k) = gmm.BIC;
            AIC(k) = gmm.AIC;
        catch
            BIC(k) = inf;
            AIC(k) = inf;
        end
    end
    
    [~, bestK_BIC] = min(BIC);
    fprintf('    Best K by BIC: %d\n', bestK_BIC);
    
    % Fit with best K
    gmmBest = fitgmdist(positions, bestK_BIC, 'Replicates', 10, ...
        'RegularizationValue', 0.01, 'Options', statset('MaxIter', 500));
    
    clusterIdx = cluster(gmmBest, positions);
    
    % Plot BIC/AIC
    fig1 = figure('Visible', 'off', 'Position', [100 100 800 400]);
    subplot(1,2,1);
    plot(1:maxK, BIC, 'b-o', 'LineWidth', 2); hold on;
    plot(bestK_BIC, BIC(bestK_BIC), 'r*', 'MarkerSize', 15);
    xlabel('Number of Components (K)'); ylabel('BIC');
    title('BIC vs Number of Gaussian Components');
    grid on; hold off;
    
    subplot(1,2,2);
    plot(1:maxK, AIC, 'r-o', 'LineWidth', 2); hold on;
    [~, bestK_AIC] = min(AIC);
    plot(bestK_AIC, AIC(bestK_AIC), 'b*', 'MarkerSize', 15);
    xlabel('Number of Components (K)'); ylabel('AIC');
    title('AIC vs Number of Gaussian Components');
    grid on; hold off;
    
    saveas(fig1, fullfile(outDir, 'model_selection_BIC_AIC.png'));
    close(fig1);
    
    % Plot clustered positions on frame
    fig2 = figure('Visible', 'off', 'Position', [100 100 900 650]);
    
    if exist(testImg, 'file')
        imshow(imread(testImg)); hold on;
    else
        imagesc(ones(H, W, 3) * 0.5); hold on;
        axis image;
    end
    
    colors = lines(bestK_BIC);
    for k = 1:bestK_BIC
        mask = clusterIdx == k;
        scatter(positions(mask,1), positions(mask,2), 10, colors(k,:), ...
            'filled', 'MarkerFaceAlpha', 0.3);
    end
    
    % Draw Gaussian ellipses
    for k = 1:bestK_BIC
        mu = gmmBest.mu(k, :);
        Sigma = gmmBest.Sigma(:,:,k);
        drawEllipse(mu, Sigma, colors(k,:));
    end
    
    title(sprintf('GMM Clustering (K=%d) of Pedestrian Positions', bestK_BIC), ...
        'FontSize', 14);
    legend(arrayfun(@(k) sprintf('Cluster %d (w=%.2f)', k, ...
        gmmBest.ComponentProportion(k)), 1:bestK_BIC, 'UniformOutput', false), ...
        'Location', 'eastoutside');
    hold off;
    
    saveas(fig2, fullfile(outDir, 'gmm_position_clusters.png'));
    close(fig2);
    
    %% --- 2. Cluster trajectory entry/exit points ---
    fprintf('  Clustering entry/exit points...\n');
    
    uniqueIDs = unique(detections(:,2));
    entryPoints = [];
    exitPoints = [];
    
    for i = 1:numel(uniqueIDs)
        id = uniqueIDs(i);
        mask = detections(:,2) == id;
        traj = detections(mask, :);
        traj = sortrows(traj, 1);
        
        if size(traj, 1) >= 2
            % Entry point: first detection centroid
            entryPoints = [entryPoints; ...
                traj(1,3)+traj(1,5)/2, traj(1,4)+traj(1,6)/2];
            % Exit point: last detection centroid
            exitPoints = [exitPoints; ...
                traj(end,3)+traj(end,5)/2, traj(end,4)+traj(end,6)/2];
        end
    end
    
    if size(entryPoints, 1) >= 3
        nClusters = min(4, size(entryPoints, 1));
        
        gmmEntry = fitgmdist(entryPoints, nClusters, 'Replicates', 5, ...
            'RegularizationValue', 1);
        gmmExit = fitgmdist(exitPoints, nClusters, 'Replicates', 5, ...
            'RegularizationValue', 1);
        
        fig3 = figure('Visible', 'off', 'Position', [100 100 1200 500]);
        
        subplot(1,2,1);
        if exist(testImg, 'file')
            imshow(imread(testImg)); hold on;
        else
            imagesc(ones(H, W, 3)*0.5); hold on; axis image;
        end
        entryCluster = cluster(gmmEntry, entryPoints);
        gscatter(entryPoints(:,1), entryPoints(:,2), entryCluster, ...
            lines(nClusters), 'o', 10);
        for k = 1:nClusters
            drawEllipse(gmmEntry.mu(k,:), gmmEntry.Sigma(:,:,k), lines(nClusters));
        end
        title('Entry Points (GMM Clustered)', 'FontSize', 12);
        hold off;
        
        subplot(1,2,2);
        if exist(testImg, 'file')
            imshow(imread(testImg)); hold on;
        else
            imagesc(ones(H, W, 3)*0.5); hold on; axis image;
        end
        exitCluster = cluster(gmmExit, exitPoints);
        gscatter(exitPoints(:,1), exitPoints(:,2), exitCluster, ...
            lines(nClusters), 's', 10);
        for k = 1:nClusters
            drawEllipse(gmmExit.mu(k,:), gmmExit.Sigma(:,:,k), lines(nClusters));
        end
        title('Exit Points (GMM Clustered)', 'FontSize', 12);
        hold off;
        
        saveas(fig3, fullfile(outDir, 'entry_exit_clusters.png'));
        close(fig3);
    end
    
    %% --- 3. Velocity/direction analysis ---
    fprintf('  Analyzing velocity distributions...\n');
    
    velocities = [];
    for i = 1:numel(uniqueIDs)
        id = uniqueIDs(i);
        mask = detections(:,2) == id;
        traj = detections(mask, :);
        traj = sortrows(traj, 1);
        
        if size(traj, 1) >= 2
            cx_traj = traj(:,3) + traj(:,5)/2;
            cy_traj = traj(:,4) + traj(:,6)/2;
            
            dx = diff(cx_traj);
            dy = diff(cy_traj);
            velocities = [velocities; dx, dy];
        end
    end
    
    if size(velocities, 1) >= 5
        nVelClusters = min(4, floor(size(velocities,1)/3));
        gmmVel = fitgmdist(velocities, nVelClusters, 'Replicates', 5, ...
            'RegularizationValue', 0.1);
        velCluster = cluster(gmmVel, velocities);
        
        fig4 = figure('Visible', 'off', 'Position', [100 100 800 600]);
        gscatter(velocities(:,1), velocities(:,2), velCluster, ...
            lines(nVelClusters), '.', 8);
        hold on;
        for k = 1:nVelClusters
            drawEllipse(gmmVel.mu(k,:), gmmVel.Sigma(:,:,k), ...
                lines(nVelClusters));
        end
        xlabel('Velocity X (pixels/frame)'); ylabel('Velocity Y (pixels/frame)');
        title('Velocity Distribution (GMM Clustered)', 'FontSize', 14);
        grid on; axis equal; hold off;
        
        saveas(fig4, fullfile(outDir, 'velocity_distribution.png'));
        close(fig4);
    end
    
    %% --- Save report ---
    fid = fopen(fullfile(outDir, 'em_analysis_report.txt'), 'w');
    fprintf(fid, 'EM Statistical Analysis Report\n');
    fprintf(fid, '==============================\n\n');
    fprintf(fid, 'Position GMM:\n');
    fprintf(fid, '  Best K (BIC): %d\n', bestK_BIC);
    for k = 1:bestK_BIC
        fprintf(fid, '  Component %d: weight=%.3f, mu=[%.1f, %.1f]\n', ...
            k, gmmBest.ComponentProportion(k), gmmBest.mu(k,1), gmmBest.mu(k,2));
    end
    fprintf(fid, '\nTotal trajectories analyzed: %d\n', numel(uniqueIDs));
    fprintf(fid, 'Total position samples: %d\n', size(positions, 1));
    fclose(fid);
    
    fprintf('  Task 6 complete. Results saved in %s\n', outDir);
end

function drawEllipse(mu, Sigma, color)
% Draw 2-sigma confidence ellipse for a 2D Gaussian
    theta = linspace(0, 2*pi, 100);
    circle = [cos(theta); sin(theta)];
    
    [V, D] = eig(Sigma);
    ellipse = mu' + 2 * V * sqrt(D) * circle;
    
    plot(ellipse(1,:), ellipse(2,:), '-', 'Color', color(1,:), 'LineWidth', 2);
end

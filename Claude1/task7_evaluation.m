function task7_evaluation(gt, detections, config)
% TASK7_EVALUATION  Evaluate detection performance against ground truth.
%   Computes:
%     (i)  Success plot using IoU at various thresholds
%     (ii) False Negative (FN) and False Positive (FP) percentages
%     (iii) Visualization of FP and FN examples

    outDir = fullfile(config.outputDir, 'task7_evaluation');
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    %% --- Match detections to GT per frame ---
    fprintf('  Computing IoU between detections and GT...\n');
    
    frames = unique(gt(:,1));
    allIoUs = [];        % All best-match IoU values
    totalGT = 0;         % Total GT boxes
    totalDet = 0;        % Total detection boxes
    totalTP = 0;         % True positives
    totalFP = 0;         % False positives
    totalFN = 0;         % False negatives
    
    iouMatchThresh = 0.5;  % IoU threshold for TP/FP/FN counting
    
    fpFrames = [];  % Store frames with FP examples
    fnFrames = [];  % Store frames with FN examples
    fpExamples = {};
    fnExamples = {};
    
    for fi = 1:numel(frames)
        f = frames(fi);
        
        % GT boxes for this frame
        gtMask = gt(:,1) == f & gt(:,7) == 1;
        gtBBs = gt(gtMask, 3:6);
        nGT = size(gtBBs, 1);
        
        % Detection boxes for this frame
        detMask = detections(:,1) == f;
        detBBs = detections(detMask, 3:6);
        nDet = size(detBBs, 1);
        
        totalGT = totalGT + nGT;
        totalDet = totalDet + nDet;
        
        if nGT == 0 && nDet == 0
            continue;
        end
        
        % Compute IoU matrix
        iouMatrix = zeros(nDet, nGT);
        for d = 1:nDet
            for g = 1:nGT
                iouMatrix(d, g) = computeIoU(detBBs(d,:), gtBBs(g,:));
            end
        end
        
        % Greedy matching (each GT matched to at most one detection)
        matchedGT = false(nGT, 1);
        matchedDet = false(nDet, 1);
        frameIoUs = [];
        
        % Sort all IoU pairs by descending IoU
        [sortedIoU, sortIdx] = sort(iouMatrix(:), 'descend');
        
        for idx = 1:numel(sortedIoU)
            if sortedIoU(idx) < 0.01  % No more meaningful overlaps
                break;
            end
            [d, g] = ind2sub([nDet, nGT], sortIdx(idx));
            if ~matchedDet(d) && ~matchedGT(g)
                matchedDet(d) = true;
                matchedGT(g) = true;
                frameIoUs = [frameIoUs; sortedIoU(idx)];
                
                if sortedIoU(idx) >= iouMatchThresh
                    totalTP = totalTP + 1;
                end
            end
        end
        
        % For unmatched detections/GTs, IoU = 0
        nUnmatchedDet = sum(~matchedDet);
        nUnmatchedGT = sum(~matchedGT);
        frameIoUs = [frameIoUs; zeros(nUnmatchedGT, 1)];
        
        allIoUs = [allIoUs; frameIoUs];
        
        frameFP = nUnmatchedDet;
        frameFN = nUnmatchedGT;
        totalFP = totalFP + frameFP;
        totalFN = totalFN + frameFN;
        
        % Collect FP/FN examples
        if frameFP > 0 && numel(fpFrames) < 10
            fpFrames = [fpFrames; f];
            fpExamples{end+1} = struct('frame', f, ...
                'fpBBs', detBBs(~matchedDet, :), ...
                'gtBBs', gtBBs);
        end
        if frameFN > 0 && numel(fnFrames) < 10
            fnFrames = [fnFrames; f];
            fnExamples{end+1} = struct('frame', f, ...
                'fnBBs', gtBBs(~matchedGT, :), ...
                'detBBs', detBBs);
        end
    end
    
    %% --- (i) Success Plot ---
    fprintf('  Generating success plot...\n');
    
    thresholds = 0:0.05:1;
    successRate = zeros(size(thresholds));
    
    for i = 1:numel(thresholds)
        successRate(i) = sum(allIoUs >= thresholds(i)) / numel(allIoUs);
    end
    
    % Area Under Success curve (AUC)
    AUC = trapz(thresholds, successRate);
    
    fig1 = figure('Visible', 'off', 'Position', [100 100 800 500]);
    plot(thresholds, successRate, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    plot([iouMatchThresh, iouMatchThresh], [0, 1], 'r--', 'LineWidth', 1.5);
    successAtThresh = sum(allIoUs >= iouMatchThresh) / numel(allIoUs);
    plot(iouMatchThresh, successAtThresh, 'r*', 'MarkerSize', 15);
    text(iouMatchThresh + 0.02, successAtThresh, ...
        sprintf('%.1f%% @ IoU=%.1f', successAtThresh*100, iouMatchThresh), ...
        'FontSize', 11);
    
    xlabel('IoU Threshold', 'FontSize', 12);
    ylabel('Success Rate', 'FontSize', 12);
    title(sprintf('Success Plot (AUC = %.3f)', AUC), 'FontSize', 14);
    grid on;
    xlim([0, 1]); ylim([0, 1]);
    legend('Success Rate', sprintf('Threshold = %.1f', iouMatchThresh), ...
        'Location', 'northeast');
    hold off;
    
    saveas(fig1, fullfile(outDir, 'success_plot.png'));
    close(fig1);
    
    %% --- (ii) FP/FN Statistics ---
    fprintf('  Computing FP/FN statistics...\n');
    
    fpRate = totalFP / max(totalDet, 1) * 100;
    fnRate = totalFN / max(totalGT, 1) * 100;
    precision = totalTP / max(totalTP + totalFP, 1);
    recall = totalTP / max(totalTP + totalFN, 1);
    f1Score = 2 * precision * recall / max(precision + recall, 1e-6);
    
    % Bar chart
    fig2 = figure('Visible', 'off', 'Position', [100 100 800 500]);
    
    subplot(1,2,1);
    bar([totalTP, totalFP, totalFN], 'FaceColor', 'flat');
    set(gca, 'XTickLabel', {'TP', 'FP', 'FN'});
    colorData = [0 0.7 0; 1 0 0; 1 0.5 0];
    b = bar([totalTP, totalFP, totalFN]);
    b.FaceColor = 'flat';
    b.CData = colorData;
    ylabel('Count'); title('Detection Counts');
    text(1, totalTP, num2str(totalTP), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
    text(2, totalFP, num2str(totalFP), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
    text(3, totalFN, num2str(totalFN), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
    
    subplot(1,2,2);
    metrics = [precision, recall, f1Score];
    b2 = bar(metrics);
    b2.FaceColor = 'flat';
    b2.CData = [0 0.5 1; 0 0.8 0.3; 0.8 0 0.8];
    set(gca, 'XTickLabel', {'Precision', 'Recall', 'F1-Score'});
    ylim([0, 1]);
    ylabel('Score'); title('Detection Metrics');
    for k = 1:3
        text(k, metrics(k), sprintf('%.3f', metrics(k)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontWeight', 'bold');
    end
    
    sgtitle(sprintf('Evaluation (IoU thresh = %.1f)', iouMatchThresh), 'FontSize', 14);
    saveas(fig2, fullfile(outDir, 'fp_fn_statistics.png'));
    close(fig2);
    
    %% --- (iii) Visualize FP/FN example frames ---
    fprintf('  Visualizing FP/FN examples...\n');
    
    testImg = fullfile(config.imgDir, sprintf(config.imgPattern, 1));
    hasImages = exist(testImg, 'file');
    
    % FP examples
    for i = 1:min(5, numel(fpExamples))
        ex = fpExamples{i};
        if hasImages
            img = imread(fullfile(config.imgDir, ...
                sprintf(config.imgPattern, ex.frame)));
        else
            img = uint8(128 * ones(576, 768, 3));
        end
        
        % Draw GT in green
        for k = 1:size(ex.gtBBs, 1)
            img = insertShape(img, 'Rectangle', ex.gtBBs(k,:), ...
                'Color', 'green', 'LineWidth', 2);
        end
        % Draw FP in red
        for k = 1:size(ex.fpBBs, 1)
            img = insertShape(img, 'Rectangle', ex.fpBBs(k,:), ...
                'Color', 'red', 'LineWidth', 3);
            img = insertText(img, [ex.fpBBs(k,1), ex.fpBBs(k,2)-15], ...
                'FP', 'FontSize', 14, 'TextColor', 'white', ...
                'BoxColor', 'red');
        end
        
        img = insertText(img, [5, 5], ...
            sprintf('Frame %d - False Positives (Red)', ex.frame), ...
            'FontSize', 14, 'TextColor', 'yellow', ...
            'BoxColor', 'black', 'BoxOpacity', 0.5);
        
        imwrite(img, fullfile(outDir, sprintf('FP_example_frame_%04d.png', ex.frame)));
    end
    
    % FN examples
    for i = 1:min(5, numel(fnExamples))
        ex = fnExamples{i};
        if hasImages
            img = imread(fullfile(config.imgDir, ...
                sprintf(config.imgPattern, ex.frame)));
        else
            img = uint8(128 * ones(576, 768, 3));
        end
        
        % Draw detections in blue
        for k = 1:size(ex.detBBs, 1)
            img = insertShape(img, 'Rectangle', ex.detBBs(k,:), ...
                'Color', 'blue', 'LineWidth', 2);
        end
        % Draw FN (missed GT) in orange
        for k = 1:size(ex.fnBBs, 1)
            img = insertShape(img, 'Rectangle', ex.fnBBs(k,:), ...
                'Color', [255, 165, 0], 'LineWidth', 3);
            img = insertText(img, [ex.fnBBs(k,1), ex.fnBBs(k,2)-15], ...
                'FN', 'FontSize', 14, 'TextColor', 'white', ...
                'BoxColor', [255, 165, 0]);
        end
        
        img = insertText(img, [5, 5], ...
            sprintf('Frame %d - False Negatives (Orange)', ex.frame), ...
            'FontSize', 14, 'TextColor', 'yellow', ...
            'BoxColor', 'black', 'BoxOpacity', 0.5);
        
        imwrite(img, fullfile(outDir, sprintf('FN_example_frame_%04d.png', ex.frame)));
    end
    
    %% --- Save report ---
    fid = fopen(fullfile(outDir, 'evaluation_report.txt'), 'w');
    fprintf(fid, 'Evaluation Report\n');
    fprintf(fid, '=================\n\n');
    fprintf(fid, 'Total GT boxes:        %d\n', totalGT);
    fprintf(fid, 'Total Detections:      %d\n', totalDet);
    fprintf(fid, 'True Positives (TP):   %d\n', totalTP);
    fprintf(fid, 'False Positives (FP):  %d (%.1f%%)\n', totalFP, fpRate);
    fprintf(fid, 'False Negatives (FN):  %d (%.1f%%)\n', totalFN, fnRate);
    fprintf(fid, '\nPrecision: %.4f\n', precision);
    fprintf(fid, 'Recall:    %.4f\n', recall);
    fprintf(fid, 'F1-Score:  %.4f\n', f1Score);
    fprintf(fid, '\nSuccess Plot AUC: %.4f\n', AUC);
    fprintf(fid, 'Success @ IoU=0.5: %.1f%%\n', successAtThresh*100);
    fclose(fid);
    
    fprintf('  Results:\n');
    fprintf('    TP=%d, FP=%d (%.1f%%), FN=%d (%.1f%%)\n', ...
        totalTP, totalFP, fpRate, totalFN, fnRate);
    fprintf('    Precision=%.3f, Recall=%.3f, F1=%.3f\n', ...
        precision, recall, f1Score);
    fprintf('    AUC=%.3f\n', AUC);
    fprintf('  Task 7 complete.\n');
end

function iou = computeIoU(bb1, bb2)
    x1 = max(bb1(1), bb2(1));
    y1 = max(bb1(2), bb2(2));
    x2 = min(bb1(1)+bb1(3), bb2(1)+bb2(3));
    y2 = min(bb1(2)+bb1(4), bb2(2)+bb2(4));
    
    interArea = max(0, x2-x1) * max(0, y2-y1);
    unionArea = bb1(3)*bb1(4) + bb2(3)*bb2(4) - interArea;
    
    iou = interArea / max(unionArea, 1e-6);
end

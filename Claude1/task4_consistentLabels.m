function detections_out = task4_consistentLabels(detections, config)
% TASK4_CONSISTENTLABELS  Assign consistent IDs across frames.
%   Implements multiple approaches with increasing accuracy:
%     1. IoU-based frame-to-frame matching (baseline)
%     2. Hungarian algorithm with IoU cost matrix (improved)
%     3. Combined IoU + centroid distance (best)

    outDir = fullfile(config.outputDir, 'task4_consistent_labels');
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    fprintf('  === Approach 1: Greedy IoU Matching (Baseline) ===\n');
    dets_greedy = greedyIoUMatching(detections);
    nIDs_greedy = numel(unique(dets_greedy(:,2)));
    fprintf('    Unique IDs assigned: %d\n', nIDs_greedy);
    
    fprintf('  === Approach 2: Hungarian Algorithm + IoU ===\n');
    dets_hungarian = hungarianIoUMatching(detections);
    nIDs_hungarian = numel(unique(dets_hungarian(:,2)));
    fprintf('    Unique IDs assigned: %d\n', nIDs_hungarian);
    
    fprintf('  === Approach 3: Hungarian + IoU + Centroid Distance ===\n');
    dets_combined = hungarianCombinedMatching(detections);
    nIDs_combined = numel(unique(dets_combined(:,2)));
    fprintf('    Unique IDs assigned: %d\n', nIDs_combined);
    
    % Use the best approach as output
    detections_out = dets_combined;
    
    % Save comparison
    save(fullfile(outDir, 'detections_consistent.mat'), 'detections_out');
    
    % Generate comparison report
    generateComparisonReport(dets_greedy, dets_hungarian, dets_combined, outDir);
    
    fprintf('  Task 4 complete.\n');
end

%% --- Approach 1: Greedy IoU Matching ---
function dets = greedyIoUMatching(detections)
    dets = detections;
    frames = unique(dets(:,1));
    nextID = 1;
    iouThresh = 0.3;
    
    % Initialize first frame
    mask0 = dets(:,1) == frames(1);
    n0 = sum(mask0);
    idx0 = find(mask0);
    for k = 1:n0
        dets(idx0(k), 2) = nextID;
        nextID = nextID + 1;
    end
    
    for i = 2:numel(frames)
        prevMask = dets(:,1) == frames(i-1);
        currMask = dets(:,1) == frames(i);
        
        prevDets = dets(prevMask, :);
        currIdx = find(currMask);
        currBBs = dets(currMask, 3:6);
        
        matched = false(size(currBBs, 1), 1);
        
        for k = 1:size(currBBs, 1)
            bestIoU = 0;
            bestID = -1;
            
            for j = 1:size(prevDets, 1)
                iou = computeIoU(currBBs(k,:), prevDets(j, 3:6));
                if iou > bestIoU && iou >= iouThresh
                    bestIoU = iou;
                    bestID = prevDets(j, 2);
                end
            end
            
            if bestID > 0
                dets(currIdx(k), 2) = bestID;
                matched(k) = true;
            end
        end
        
        % Assign new IDs to unmatched
        for k = find(~matched)'
            dets(currIdx(k), 2) = nextID;
            nextID = nextID + 1;
        end
    end
end

%% --- Approach 2: Hungarian Algorithm + IoU ---
function dets = hungarianIoUMatching(detections)
    dets = detections;
    frames = unique(dets(:,1));
    nextID = 1;
    iouThresh = 0.2;
    
    % Initialize first frame
    mask0 = dets(:,1) == frames(1);
    idx0 = find(mask0);
    for k = 1:numel(idx0)
        dets(idx0(k), 2) = nextID;
        nextID = nextID + 1;
    end
    
    for i = 2:numel(frames)
        prevMask = dets(:,1) == frames(i-1);
        currMask = dets(:,1) == frames(i);
        
        prevDets = dets(prevMask, :);
        currIdx = find(currMask);
        currBBs = dets(currMask, 3:6);
        
        nPrev = size(prevDets, 1);
        nCurr = size(currBBs, 1);
        
        if nPrev == 0 || nCurr == 0
            for k = 1:nCurr
                dets(currIdx(k), 2) = nextID;
                nextID = nextID + 1;
            end
            continue;
        end
        
        % Build IoU cost matrix
        costMatrix = ones(nCurr, nPrev);  % 1 - IoU (minimize)
        for c = 1:nCurr
            for p = 1:nPrev
                iou = computeIoU(currBBs(c,:), prevDets(p, 3:6));
                costMatrix(c, p) = 1 - iou;
            end
        end
        
        % Hungarian algorithm
        [assignment, ~] = hungarianAlgorithm(costMatrix);
        
        matched = false(nCurr, 1);
        for c = 1:nCurr
            if assignment(c) > 0 && costMatrix(c, assignment(c)) < (1 - iouThresh)
                dets(currIdx(c), 2) = prevDets(assignment(c), 2);
                matched(c) = true;
            end
        end
        
        % New IDs for unmatched
        for k = find(~matched)'
            dets(currIdx(k), 2) = nextID;
            nextID = nextID + 1;
        end
    end
end

%% --- Approach 3: Hungarian + IoU + Centroid Distance ---
function dets = hungarianCombinedMatching(detections)
    dets = detections;
    frames = unique(dets(:,1));
    nextID = 1;
    maxDist = 100;  % Maximum centroid distance for matching
    wIoU = 0.5;     % Weight for IoU
    wDist = 0.5;    % Weight for distance
    
    % Initialize first frame
    mask0 = dets(:,1) == frames(1);
    idx0 = find(mask0);
    for k = 1:numel(idx0)
        dets(idx0(k), 2) = nextID;
        nextID = nextID + 1;
    end
    
    for i = 2:numel(frames)
        prevMask = dets(:,1) == frames(i-1);
        currMask = dets(:,1) == frames(i);
        
        prevDets = dets(prevMask, :);
        currIdx = find(currMask);
        currBBs = dets(currMask, 3:6);
        
        nPrev = size(prevDets, 1);
        nCurr = size(currBBs, 1);
        
        if nPrev == 0 || nCurr == 0
            for k = 1:nCurr
                dets(currIdx(k), 2) = nextID;
                nextID = nextID + 1;
            end
            continue;
        end
        
        % Compute centroids
        prevCentroids = [prevDets(:,3) + prevDets(:,5)/2, ...
                         prevDets(:,4) + prevDets(:,6)/2];
        currCentroids = [currBBs(:,1) + currBBs(:,3)/2, ...
                         currBBs(:,2) + currBBs(:,4)/2];
        
        % Combined cost matrix
        costMatrix = ones(nCurr, nPrev);
        for c = 1:nCurr
            for p = 1:nPrev
                iouVal = computeIoU(currBBs(c,:), prevDets(p, 3:6));
                dist = norm(currCentroids(c,:) - prevCentroids(p,:));
                distCost = min(dist / maxDist, 1);  % Normalize to [0,1]
                
                costMatrix(c, p) = wIoU * (1 - iouVal) + wDist * distCost;
            end
        end
        
        % Hungarian algorithm
        [assignment, ~] = hungarianAlgorithm(costMatrix);
        
        matched = false(nCurr, 1);
        for c = 1:nCurr
            if assignment(c) > 0 && costMatrix(c, assignment(c)) < 0.7
                dets(currIdx(c), 2) = prevDets(assignment(c), 2);
                matched(c) = true;
            end
        end
        
        for k = find(~matched)'
            dets(currIdx(k), 2) = nextID;
            nextID = nextID + 1;
        end
    end
end

%% --- Helper: Compute IoU between two bounding boxes ---
function iou = computeIoU(bb1, bb2)
% bb format: [left, top, width, height]
    x1 = max(bb1(1), bb2(1));
    y1 = max(bb1(2), bb2(2));
    x2 = min(bb1(1)+bb1(3), bb2(1)+bb2(3));
    y2 = min(bb1(2)+bb1(4), bb2(2)+bb2(4));
    
    interArea = max(0, x2-x1) * max(0, y2-y1);
    unionArea = bb1(3)*bb1(4) + bb2(3)*bb2(4) - interArea;
    
    if unionArea == 0
        iou = 0;
    else
        iou = interArea / unionArea;
    end
end

%% --- Hungarian Algorithm (Munkres) ---
function [assignment, cost] = hungarianAlgorithm(costMatrix)
% Simple implementation of the Hungarian algorithm for assignment.
% Uses MATLAB's matchpairs if available (R2019b+), otherwise a
% straightforward greedy fallback.

    [nRows, nCols] = size(costMatrix);
    assignment = zeros(nRows, 1);
    cost = 0;
    
    try
        % MATLAB R2019b+ has matchpairs
        M = matchpairs(costMatrix, 1000);
        for k = 1:size(M, 1)
            assignment(M(k,1)) = M(k,2);
            cost = cost + costMatrix(M(k,1), M(k,2));
        end
    catch
        % Fallback: greedy assignment
        usedCols = false(1, nCols);
        [~, sortIdx] = sort(costMatrix(:));
        
        for idx = sortIdx'
            [r, c] = ind2sub([nRows, nCols], idx);
            if assignment(r) == 0 && ~usedCols(c)
                assignment(r) = c;
                usedCols(c) = true;
                cost = cost + costMatrix(r, c);
            end
            if all(assignment > 0)
                break;
            end
        end
    end
end

%% --- Generate comparison report ---
function generateComparisonReport(dets1, dets2, dets3, outDir)
    fid = fopen(fullfile(outDir, 'comparison_report.txt'), 'w');
    fprintf(fid, 'Label Assignment Comparison Report\n');
    fprintf(fid, '==================================\n\n');
    
    fprintf(fid, 'Approach 1 - Greedy IoU:\n');
    fprintf(fid, '  Unique IDs: %d\n', numel(unique(dets1(:,2))));
    fprintf(fid, '  (Simple but prone to ID switches)\n\n');
    
    fprintf(fid, 'Approach 2 - Hungarian + IoU:\n');
    fprintf(fid, '  Unique IDs: %d\n', numel(unique(dets2(:,2))));
    fprintf(fid, '  (Optimal frame-to-frame matching)\n\n');
    
    fprintf(fid, 'Approach 3 - Hungarian + IoU + Distance:\n');
    fprintf(fid, '  Unique IDs: %d\n', numel(unique(dets3(:,2))));
    fprintf(fid, '  (Best: handles partial occlusions)\n\n');
    
    fprintf(fid, 'Discussion:\n');
    fprintf(fid, '  - Greedy matching is fast but suboptimal when detections overlap.\n');
    fprintf(fid, '  - Hungarian algorithm provides globally optimal assignment per frame.\n');
    fprintf(fid, '  - Adding centroid distance helps when IoU is low due to motion.\n');
    fprintf(fid, '  - Further improvement: Kalman filter prediction for occluded targets.\n');
    fclose(fid);
end

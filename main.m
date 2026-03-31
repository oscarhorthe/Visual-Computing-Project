%% Detection + Tracking + Kalman Filter + Merge/Split Display
clear; close all; clc;

%% PARAMETERS
params.bgFrames = 50;
params.fgThreshold = 50;

params.minBlobAreaTop = 120;
params.minBlobAreaBottom = 1200;
params.maxBlobArea = 30000;

params.minAspectHW = 1.0;
params.minHeight = 18;
params.maxHeight = 300;
params.minWidth = 6;
params.maxWidth = 120;

params.seClose = strel('disk', 7);
params.seOpen  = strel('disk', 3);

numBins = 16;
maxMissing = 40;
nextID = 1;

%% KALMAN PARAMETERS
% State: [x; y; vx; vy]   Measurement: [x; y]
dt = 1;  % one frame step
F = [1 0 dt 0;    % state transition
     0 1 0  dt;
     0 0 1  0;
     0 0 0  1];
Hobs = [1 0 0 0;  % observation matrix (named Hobs to avoid conflict with image height H)
        0 1 0 0];
Q = diag([4, 4, 2, 2]);      % process noise (allows moderate acceleration)
R = diag([9, 9]);             % measurement noise (detection centroid jitter)
P0 = diag([10, 10, 25, 25]); % initial state covariance

%% PATHS
imgDir = fullfile('Crowd_PETS','S2','L1','Time_12-34','View_001');
nFrames = 795;

%% BACKGROUND
sample = imread(fullfile(imgDir,'frame_0000.jpg'));
[H,W,~] = size(sample);

bgStack = zeros(H,W,params.bgFrames,'uint8');
randFrames = randperm(nFrames, params.bgFrames);

for i = 1:params.bgFrames
    img = imread(fullfile(imgDir, sprintf('frame_%04d.jpg', randFrames(i)-1)));
    bgStack(:,:,i) = rgb2gray(img);
end

bgModel = median(bgStack, 3);

%% TRACK MEMORY
% Each track now carries Kalman state (kf_x), covariance (kf_P), and predicted position
tracks = struct('id',{},'centroid',{},'bbox',{},'hueHist',{}, ...
                'lastSeen',{},'isMissing',{},'stillInFrame',{},'missingCount',{}, ...
                'confirmedCount',{}, ...
                'kf_x',{},'kf_P',{},'predictedCentroid',{});
prevTracks = struct('id',{},'centroid',{},'bbox',{},'displayIDs',{},'pixelIdx',{},'area',{});
prevMergedBoxes = struct('bbox',{},'displayIDs',{});

%% LOOP
figure('Name','Detection + Tracking + Merge Display','NumberTitle','off');

for f = 1:nFrames

    imgFile = fullfile(imgDir, sprintf('frame_%04d.jpg', f-1));
    if ~isfile(imgFile), continue; end

    img = imread(imgFile);
    gray = rgb2gray(img);

    %% FOREGROUND
    fgMask = abs(double(gray) - double(bgModel)) > params.fgThreshold;
    fgMask = imclose(fgMask, params.seClose);
    fgMask = imopen(fgMask, params.seOpen);
    fgMask = imfill(fgMask, 'holes');
    fgMask = bwareaopen(fgMask, 40);

    %% LAMPPOST EXTENSION
    fgMask = extendShortBlobsAtSign(fgMask);

    %% KALMAN PREDICT — run for ALL tracks before matching
    for t = 1:length(tracks)
        tracks(t).kf_x = F * tracks(t).kf_x;
        tracks(t).kf_P = F * tracks(t).kf_P * F' + Q;
        tracks(t).predictedCentroid = [tracks(t).kf_x(1), tracks(t).kf_x(2)];
    end

    %% BLOBS
    stats = regionprops(fgMask, 'BoundingBox', 'Area', 'Centroid', 'PixelIdxList');
    validMask = false(1, numel(stats));

    for j = 1:numel(stats)
        bb = stats(j).BoundingBox;
        area = stats(j).Area;
        w = bb(3);
        h = bb(4);

        if w <= 0
            continue;
        end

        aspect = h / w;
        yBottom = bb(2) + bb(4);
        alpha = min(max(yBottom / H, 0), 1);

        minArea = params.minBlobAreaTop + ...
            alpha * (params.minBlobAreaBottom - params.minBlobAreaTop);

        if area < minArea || area > params.maxBlobArea
            continue;
        end
        if h < params.minHeight || h > params.maxHeight
            continue;
        end
        if w < params.minWidth || w > params.maxWidth
            continue;
        end
        if aspect < params.minAspectHW
            continue;
        end

        validMask(j) = true;
    end

    stats = stats(validMask);

    currentTracks = struct('id',{},'centroid',{},'bbox',{},'displayIDs',{},'pixelIdx',{},'area',{});
    usedDisplayIDs = [];

    %% DISPLAY SETUP
    subplot(1,2,1);
    imshow(fgMask);
    title(sprintf('FG %d', f));

    subplot(1,2,2);
    imshow(img);
    hold on;

    %% MATCH DETECTIONS TO TRACKS (Hungarian algorithm + Kalman predicted positions)
    nDets = length(stats);
    nTrks = length(tracks);

    % Precompute detection features
    detBB   = zeros(nDets, 4);
    detCt   = zeros(nDets, 2);
    detHist = zeros(nDets, numBins);
    for j = 1:nDets
        detBB(j,:)    = stats(j).BoundingBox;
        detCt(j,:)    = stats(j).Centroid;
        detHist(j,:)  = getHueHist(img, detBB(j,:), numBins, stats(j).PixelIdxList);
    end

    % Build cost matrix (detections x tracks)
    costMatrix = inf(nDets, nTrks);
    maxAssignCost = 200;

    for j = 1:nDets
        bb = detBB(j,:);
        ct = detCt(j,:);
        hHist = detHist(j,:);

        % Adaptive distance gate based on vertical position
        yBottom = bb(2) + bb(4);
        alphaPos = min(max(yBottom / H, 0), 1);
        maxDist = 50 + 80 * alphaPos;

        for t = 1:nTrks
            % Skip tracks missing for too long
            if tracks(t).isMissing && tracks(t).missingCount > 5
                continue;
            end

            % Use PREDICTED centroid from Kalman filter (not last-seen position)
            predCt = tracks(t).predictedCentroid;
            dist = norm(ct - predCt);
            if dist > maxDist
                continue;
            end

            % Hue histogram difference
            hueDiff = sum(abs(hHist - tracks(t).hueHist));
            if tracks(t).confirmedCount >= 3
                hueWeight = 120;
            else
                hueWeight = 60;
            end

            % Size similarity
            detArea = bb(3) * bb(4);
            trkArea = tracks(t).bbox(3) * tracks(t).bbox(4);
            sizeRatio = max(detArea, trkArea) / max(min(detArea, trkArea), 1);
            sizePenalty = 15 * (sizeRatio - 1);

            cost = dist + hueWeight * hueDiff + sizePenalty;

            if cost < maxAssignCost
                costMatrix(j, t) = cost;
            end
        end
    end

    % Hungarian assignment
    if nDets > 0 && nTrks > 0
        assignments = matchpairs(costMatrix, maxAssignCost);
    else
        assignments = zeros(0, 2);
    end

    assignedDets = assignments(:,1);
    assignedTrks = assignments(:,2);

    % Process all detections
    for j = 1:nDets
        bb    = detBB(j,:);
        ct    = detCt(j,:);
        hHist = detHist(j,:);

        aIdx = find(assignedDets == j, 1);

        if isempty(aIdx)
            % Unmatched detection -> new person
            id = nextID;
            nextID = nextID + 1;

            newIdx = length(tracks) + 1;
            tracks(newIdx).id = id;
            tracks(newIdx).centroid = ct;
            tracks(newIdx).bbox = bb;
            tracks(newIdx).hueHist = hHist;
            tracks(newIdx).lastSeen = f;
            tracks(newIdx).isMissing = false;
            tracks(newIdx).stillInFrame = true;
            tracks(newIdx).missingCount = 0;
            tracks(newIdx).confirmedCount = 1;

            % Initialize Kalman state: position = centroid, velocity = 0
            tracks(newIdx).kf_x = [ct(1); ct(2); 0; 0];
            tracks(newIdx).kf_P = P0;
            tracks(newIdx).predictedCentroid = ct;
        else
            % Matched -> re-identified person
            tIdx = assignedTrks(aIdx);
            id = tracks(tIdx).id;

            % Kalman UPDATE step: correct prediction with measurement
            z = [ct(1); ct(2)];
            S = Hobs * tracks(tIdx).kf_P * Hobs' + R;
            K = tracks(tIdx).kf_P * Hobs' / S;
            tracks(tIdx).kf_x = tracks(tIdx).kf_x + K * (z - Hobs * tracks(tIdx).kf_x);
            tracks(tIdx).kf_P = (eye(4) - K * Hobs) * tracks(tIdx).kf_P;

            % Use filtered position as centroid
            tracks(tIdx).centroid = [tracks(tIdx).kf_x(1), tracks(tIdx).kf_x(2)];
            tracks(tIdx).bbox = bb;
            tracks(tIdx).confirmedCount = tracks(tIdx).confirmedCount + 1;

            if tracks(tIdx).confirmedCount <= 3
                hueAlpha = 0.5;
            else
                hueAlpha = 0.2;
            end
            tracks(tIdx).hueHist = (1-hueAlpha) * tracks(tIdx).hueHist + hueAlpha * hHist;

            tracks(tIdx).lastSeen = f;
            tracks(tIdx).isMissing = false;
            tracks(tIdx).stillInFrame = true;
            tracks(tIdx).missingCount = 0;
        end

        %% SAVE CURRENT TRACK
        k = length(currentTracks) + 1;
        currentTracks(k).id = id;
        currentTracks(k).centroid = ct;
        currentTracks(k).bbox = bb;
        currentTracks(k).pixelIdx = stats(j).PixelIdxList;
        currentTracks(k).area = stats(j).Area;

        %% MERGE / SPLIT DISPLAY IDS
        currPixelIdx = stats(j).PixelIdxList;
        currArea = stats(j).Area;

        % Default: display the tracker ID
        displayIDs = id;

        % Check for merge
        if ~isempty(prevTracks)
            mergeIDs = resolveDisplayIDsFromMaskOverlap(currPixelIdx, currArea, bb, id, prevTracks);

            if numel(mergeIDs) >= 2
                displayIDs = mergeIDs;
            end
        end

        % Split recovery using hue similarity
        if numel(displayIDs) == 1
            bestM = 0;
            bestIoU = 0;

            for m = 1:numel(prevMergedBoxes)
                ov = bboxIoU(bb, prevMergedBoxes(m).bbox);
                mergedArea = prevMergedBoxes(m).bbox(3) * prevMergedBoxes(m).bbox(4);
                blobArea = bb(3) * bb(4);
                if ov > 0.15 && numel(prevMergedBoxes(m).displayIDs) >= 2 ...
                        && ov > bestIoU && blobArea < 0.85 * mergedArea
                    bestIoU = ov;
                    bestM = m;
                end
            end

            if bestM > 0
                prevIDs = prevMergedBoxes(bestM).displayIDs;

                % Also use Kalman predicted positions for split assignment
                % Pick the ID whose predicted centroid is closest to this blob
                predDists = inf(1, numel(prevIDs));
                histDiffs = inf(1, numel(prevIDs));
                for pi = 1:numel(prevIDs)
                    tIdx = find([tracks.id] == prevIDs(pi), 1);
                    if ~isempty(tIdx)
                        histDiffs(pi) = sum(abs(hHist - tracks(tIdx).hueHist));
                        predDists(pi) = norm(ct - tracks(tIdx).predictedCentroid);
                    end
                end

                % Combined score: Kalman position + hue similarity
                splitScores = predDists + 80 * histDiffs;
                [~, sortOrder] = sort(splitScores);
                rankedIDs = prevIDs(sortOrder);

                for pi = 1:numel(rankedIDs)
                    if ~ismember(rankedIDs(pi), usedDisplayIDs)
                        displayIDs = rankedIDs(pi);
                        break;
                    end
                end
            end
        end

        % Order merged IDs left-to-right
        if numel(displayIDs) >= 2
            displayIDs = orderIDsByPreviousX(displayIDs, prevTracks);
        end

        % ---------- Enforce per-frame display uniqueness ----------
        if numel(displayIDs) == 1
            if ismember(displayIDs, usedDisplayIDs)
                if ~ismember(id, usedDisplayIDs)
                    displayIDs = id;
                else
                    displayIDs = nextID;
                    nextID = nextID + 1;
                end
            end
        else
            displayIDs = displayIDs(~ismember(displayIDs, usedDisplayIDs));

            if isempty(displayIDs)
                if ~ismember(id, usedDisplayIDs)
                    displayIDs = id;
                else
                    displayIDs = nextID;
                    nextID = nextID + 1;
                end
            end
        end

        usedDisplayIDs = [usedDisplayIDs, displayIDs];

        circleCenters = makeCircleCenters(bb, numel(displayIDs));
        currentTracks(k).displayIDs = displayIDs;

        %% DRAW
        if numel(displayIDs) >= 2
            boxColor = 'm';
            labelText = ['MERGE: ' strtrim(sprintf('%d ', displayIDs))];
            textColor = 'yellow';
            r = 10;
        else
            boxColor = 'g';
            labelText = ['ID ' strtrim(sprintf('%d ', displayIDs))];
            textColor = 'cyan';
            r = 8;
        end

        rectangle('Position', bb, 'EdgeColor', boxColor, 'LineWidth', 2);

        for c = 1:size(circleCenters,1)
            cx = circleCenters(c,1);
            cy = circleCenters(c,2);

            rectangle('Position', [cx-r, cy-r, 2*r, 2*r], ...
                      'Curvature', [1 1], ...
                      'EdgeColor', 'y', ...
                      'LineWidth', 3);
        end

        text(bb(1), bb(2)-12, labelText, ...
             'Color', textColor, ...
             'FontSize', 11, ...
             'FontWeight', 'bold', ...
             'BackgroundColor', 'black');
    end

    %% MARK UNMATCHED TRACKS
    edgeMargin = 25;
    maxMissingHard = 60;

    % Collect IDs inside merged blobs — these tracks are not missing
    mergedIDs = [];
    for q = 1:numel(currentTracks)
        if numel(currentTracks(q).displayIDs) >= 2
            mergedIDs = [mergedIDs, currentTracks(q).displayIDs]; %#ok<AGROW>
        end
    end

    for t = 1:length(tracks)
        if tracks(t).lastSeen < f   % not directly matched this frame

            % Track is inside a merge — keep alive, Kalman keeps predicting
            if ismember(tracks(t).id, mergedIDs)
                tracks(t).isMissing = false;
                tracks(t).missingCount = 0;
                tracks(t).stillInFrame = true;
                % NOTE: Kalman predict already ran at top of frame, so
                % predictedCentroid reflects where this person should be
                % even though they're inside a merged blob. No position
                % override needed — the Kalman velocity carries them forward.
                continue;
            end

            tracks(t).isMissing = true;
            tracks(t).missingCount = tracks(t).missingCount + 1;

            bb = tracks(t).bbox;
            x = bb(1); y = bb(2); w = bb(3); h = bb(4);

            touchesEdge = (x <= edgeMargin) || ...
                          (y <= edgeMargin) || ...
                          (x + w >= W - edgeMargin) || ...
                          (y + h >= H - edgeMargin);

            if touchesEdge
                tracks(t).stillInFrame = false;
            else
                tracks(t).stillInFrame = true;
            end
        end
    end

    %% REMOVE OLD TRACKS
    keep = true(1, length(tracks));

    for t = 1:length(tracks)
        missingTooLong = tracks(t).missingCount > maxMissingHard;
        leftScene = tracks(t).missingCount > maxMissing && ~tracks(t).stillInFrame;

        if missingTooLong || leftScene
            keep(t) = false;
        end
    end

    tracks = tracks(keep);

    %% SAVE MERGED BOX MEMORY
    prevMergedBoxes = struct('bbox',{},'displayIDs',{});
    mm = 0;
    for q = 1:numel(currentTracks)
        if numel(currentTracks(q).displayIDs) >= 2
            mm = mm + 1;
            prevMergedBoxes(mm).bbox = currentTracks(q).bbox;
            prevMergedBoxes(mm).displayIDs = currentTracks(q).displayIDs;
        end
    end

    hold off;
    drawnow;

    prevTracks = currentTracks;
end
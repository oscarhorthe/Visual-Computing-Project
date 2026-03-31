%% Detection + Tracking with Kalman Filter + Ellipse Body Model
% Builds on the existing pipeline but adds:
%   1. Kalman filter per track (predicts position + velocity)
%   2. Ellipse body model per track (head + torso + legs)
%   3. Merged blob detection and splitting using predicted ellipses
%   4. Adaptive background model
%   5. Occlusion-aware tracking through short occlusions
%
% Ellipse model inspired by Zhao et al., "Segmentation and Tracking of
% Multiple Humans in Crowded Environments", IEEE TPAMI 2008, Section 4.1:
% The human body is modeled as a composition of ellipsoids for head, torso,
% and legs. We use a simplified 2D projection of this model.

clear; close all; clc;

%% USER CONTROLS
startFrame = 1;     % <-- Change this to start at a different frame
refFrame   = 645;     % <-- Reference frame for person-size calibration
% Press 'p' during playback to pause/unpause

%% PARAMETERS
params.bgFrames      = 50;
params.fgThreshold   = 50;
params.bgAlpha       = 0.005;

params.minBlobAreaTop    = 120;
params.minBlobAreaBottom = 1200;
params.maxBlobArea       = 30000;

params.minAspectHW = 0.8;   % lowered: merged blobs have low aspect ratio
params.minHeight   = 18;
params.maxHeight   = 300;
params.minWidth    = 6;
params.maxWidth    = 200;    % raised: merged blobs can be wide

params.seClose = strel('disk', 7);
params.seOpen  = strel('disk', 3);

numBins         = 16;
maxMissing      = 30;
maxPredictFrames = 30;
gateDistance     = 100;
hueCostWeight   = 60;
missingPenalty   = 12;  % cost penalty per missing frame (makes ghost tracks less competitive)
nextID = 1;

%% PATHS
imgDir  = fullfile('..','Crowd_PETS','S2','L1','Time_12-34','View_001');
nFrames = 795;

%% BUILD INITIAL BACKGROUND MODEL
sample = imread(fullfile(imgDir,'frame_0000.jpg'));
[H, W, ~] = size(sample);

bgStack = zeros(H, W, params.bgFrames, 'uint8');
randFrames = randperm(nFrames, params.bgFrames);
for i = 1:params.bgFrames
    img = imread(fullfile(imgDir, sprintf('frame_%04d.jpg', randFrames(i)-1)));
    bgStack(:,:,i) = rgb2gray(img);
end
bgModel = double(median(bgStack, 3));
fprintf('Background model built.\n');

%% CALIBRATE PERSON SIZE FROM REFERENCE FRAME
% Use refFrame (e.g. 645) where people are spread out and not merged.
% Extract single-person blobs and fit a linear model:
%   expectedHeight(yBottom) and expectedWidth(yBottom)
% This acts as a simple "camera model" for perspective scaling.
fprintf('Calibrating person size from frame %d...\n', refFrame);

refImg  = imread(fullfile(imgDir, sprintf('frame_%04d.jpg', refFrame-1)));
refGray = double(rgb2gray(refImg));
refFg   = abs(refGray - bgModel) > params.fgThreshold;
refFg   = imclose(refFg, params.seClose);
refFg   = imopen(refFg, params.seOpen);
refFg   = imfill(refFg, 'holes');
refFg   = bwareaopen(refFg, 40);
refFg   = extendShortBlobsAtSign(refFg);

refStats = regionprops(refFg, 'BoundingBox', 'Area');

% Collect single-person blobs (reasonable aspect ratio + area)
refYBottom = [];
refHeights = [];
refWidths  = [];

for j = 1:numel(refStats)
    bb = refStats(j).BoundingBox;
    w = bb(3); h = bb(4);
    area = refStats(j).Area;
    if w <= 0, continue; end

    aspect = h / w;
    yBot = bb(2) + h;

    % Filter: must look like a single standing person
    if aspect < 1.2 || aspect > 5.0, continue; end
    if h < 25 || h > 250, continue; end
    if area < 200 || area > 15000, continue; end

    refYBottom(end+1) = yBot;    %#ok<SAGROW>
    refHeights(end+1) = h;       %#ok<SAGROW>
    refWidths(end+1)  = w;       %#ok<SAGROW>
end

% Fit linear models: size = a * yBottom + b
% These capture the perspective effect (people lower in image = closer = bigger)
if numel(refYBottom) >= 2
    pHeight = polyfit(refYBottom, refHeights, 1);  % [slope, intercept]
    pWidth  = polyfit(refYBottom, refWidths, 1);
    fprintf('  Calibration: found %d reference persons.\n', numel(refYBottom));
    fprintf('  Height model: %.2f * yBottom + %.1f\n', pHeight(1), pHeight(2));
    fprintf('  Width model:  %.2f * yBottom + %.1f\n', pWidth(1), pWidth(2));
else
    % Fallback if calibration fails
    warning('Calibration: only %d blobs found, using default size model.', numel(refYBottom));
    pHeight = [0.25, 20];  % rough defaults
    pWidth  = [0.08, 10];
end

% Tolerance multiplier: how much bigger than expected before we flag as merged
mergeTolHeight = 1.5;  % blob height > 1.5x expected = likely merged vertically
mergeTolWidth  = 1.5;  % blob width  > 1.5x expected = likely merged horizontally

%% TRACK STRUCT
% Each track has Kalman state + ellipse body model
tracks = struct('id',{}, 'centroid',{}, 'bbox',{}, 'hueHist',{}, ...
                'lastSeen',{}, 'missingCount',{}, 'stillInFrame',{}, ...
                'kf_x',{}, 'kf_P',{}, 'age',{}, ...
                'ellipse_a',{}, 'ellipse_b',{});
% ellipse_a = semi-axis height (half of bbox height)
% ellipse_b = semi-axis width  (half of bbox width)

%% KALMAN FILTER PARAMETERS (constant velocity model)
dt = 1;
F = [1 0 dt 0;  0 1 0 dt;  0 0 1 0;  0 0 0 1];
H_obs = [1 0 0 0;  0 1 0 0];
Q  = diag([4, 4, 2, 2]);
R  = diag([16, 16]);
P0 = diag([25, 25, 100, 100]);

%% HELPER: build 3-part ellipse mask for a track at given center
% Models head (small ellipse top), torso (wider middle), legs (medium bottom)
% Returns a binary mask of size [H, W]
buildEllipseMask = @(cx, cy, ea, eb, imgH, imgW) createBodyEllipseMask( ...
    cx, cy, ea, eb, imgH, imgW);

%% MAIN LOOP
fig = figure('Name','Kalman + Ellipse Tracking','NumberTitle','off', ...
             'Position',[50 100 1200 450]);
isPaused = false;
set(fig, 'KeyPressFcn', @(~,evt) assignin('caller','isPaused', ...
    xor(evalin('caller','isPaused'), strcmp(evt.Key,'p'))));

fprintf('Controls: press P to pause/unpause.\n');
fprintf('Starting at frame %d.\n', startFrame);

for f = startFrame:nFrames

    % Handle pause toggle
    while isPaused
        title(sprintf('PAUSED - Frame %d  (press P to resume)', f));
        drawnow;
        pause(0.1);
    end

    imgFile = fullfile(imgDir, sprintf('frame_%04d.jpg', f-1));
    if ~isfile(imgFile), continue; end

    img  = imread(imgFile);
    gray = double(rgb2gray(img));

    %% FOREGROUND DETECTION
    fgMask = abs(gray - bgModel) > params.fgThreshold;
    fgMask = imclose(fgMask, params.seClose);
    fgMask = imopen(fgMask, params.seOpen);
    fgMask = imfill(fgMask, 'holes');
    fgMask = bwareaopen(fgMask, 40);

    fgMask = extendShortBlobsAtSign(fgMask);

    %% ADAPTIVE BACKGROUND UPDATE
    bgMaskPx = ~fgMask;
    bgModel(bgMaskPx) = (1 - params.bgAlpha) * bgModel(bgMaskPx) + ...
                          params.bgAlpha * gray(bgMaskPx);

    %% PRECOMPUTE HSV (once per frame, not once per detection)
    hsvImg = rgb2hsv(img);

    %% BLOB EXTRACTION + FILTERING (keep larger max for merged blobs)
    stats = regionprops(fgMask, 'BoundingBox','Area','Centroid','PixelIdxList');
    validMask = false(1, numel(stats));

    for j = 1:numel(stats)
        bb   = stats(j).BoundingBox;
        area = stats(j).Area;
        w = bb(3); h = bb(4);
        if w <= 0, continue; end

        yBottom = bb(2) + bb(4);
        alpha   = min(max(yBottom / H, 0), 1);
        minArea = params.minBlobAreaTop + ...
                  alpha * (params.minBlobAreaBottom - params.minBlobAreaTop);

        if area < minArea || area > params.maxBlobArea, continue; end
        if h < params.minHeight || h > params.maxHeight, continue; end
        if w < params.minWidth  || w > params.maxWidth,  continue; end

        validMask(j) = true;
    end
    stats = stats(validMask);

    %% KALMAN PREDICT for all existing tracks
    nTrkPre = length(tracks);
    predXY = zeros(nTrkPre, 2);  % precompute predicted positions
    trkAges = zeros(1, nTrkPre);
    for t = 1:nTrkPre
        tracks(t).kf_x = F * tracks(t).kf_x;
        tracks(t).kf_P = F * tracks(t).kf_P * F' + Q;
        predXY(t,:) = tracks(t).kf_x(1:2)';
        trkAges(t) = tracks(t).age;
    end

    %% DETECT MERGED BLOBS AND SPLIT USING ELLIPSES + CALIBRATED SIZE
    splitDetections = struct('Centroid',{}, 'BoundingBox',{}, 'PixelIdxList',{}, ...
                             'Area',{}, 'fromSplit',{}, 'parentIdx',{});
    nSplit = 0;

    for j = 1:numel(stats)
        bb   = stats(j).BoundingBox;
        blobW = bb(3);
        blobH = bb(4);

        % Expected single-person size at this y-position (from calibration)
        yBottom = bb(2) + blobH;
        expHeight = max(polyval(pHeight, yBottom), 30);
        expWidth  = max(polyval(pWidth, yBottom), 10);

        % Vectorized: find tracks whose predicted center falls inside this blob
        if nTrkPre > 0
            inX = predXY(:,1) >= bb(1) - 15 & predXY(:,1) <= bb(1) + blobW + 15;
            inY = predXY(:,2) >= bb(2) - 15 & predXY(:,2) <= bb(2) + blobH + 15;
            overlappingTracks = find(inX & inY & (trkAges(:) >= 2))';
        else
            overlappingTracks = [];
        end

        % Merged if too wide OR too tall compared to calibrated single-person size
        tooWide = blobW > mergeTolWidth * expWidth;
        tooTall = blobH > mergeTolHeight * expHeight;
        isMerged = (tooWide || tooTall) && (numel(overlappingTracks) >= 2);

        if isMerged
            % SPLIT the blob using predicted ellipse positions
            % Assign each foreground pixel to the nearest track ellipse
            pixIdx = stats(j).PixelIdxList;
            [pixY, pixX] = ind2sub([H, W], pixIdx);

            nOvl = numel(overlappingTracks);

            % Vectorized Mahalanobis distance: compute all pixels vs all tracks at once
            % distMat is nPixels x nOvl
            distMat = inf(length(pixIdx), nOvl);
            for oi = 1:nOvl
                ti = overlappingTracks(oi);
                cx = tracks(ti).kf_x(1);
                cy = tracks(ti).kf_x(2);
                ea = max(tracks(ti).ellipse_a, 1);
                eb = max(tracks(ti).ellipse_b, 1);

                distMat(:, oi) = ((pixX - cx) / eb).^2 + ((pixY - cy) / ea).^2;
            end
            [~, assignment] = min(distMat, [], 2);

            % Create a virtual detection for each track's pixel set
            for oi = 1:nOvl
                mask_oi = (assignment == oi);
                if sum(mask_oi) < 50, continue; end  % too few pixels

                subPixIdx = pixIdx(mask_oi);
                subX = pixX(mask_oi);
                subY = pixY(mask_oi);

                newCentroid = [mean(subX), mean(subY)];
                x1 = min(subX); x2 = max(subX);
                y1 = min(subY); y2 = max(subY);
                newBBox = [x1, y1, x2-x1+1, y2-y1+1];

                nSplit = nSplit + 1;
                splitDetections(nSplit).Centroid    = newCentroid;
                splitDetections(nSplit).BoundingBox = newBBox;
                splitDetections(nSplit).PixelIdxList = subPixIdx;
                splitDetections(nSplit).Area         = sum(mask_oi);
                splitDetections(nSplit).fromSplit    = true;
                splitDetections(nSplit).parentIdx    = j;
            end
        else
            % Single-person blob: pass through as-is
            nSplit = nSplit + 1;
            splitDetections(nSplit).Centroid     = stats(j).Centroid;
            splitDetections(nSplit).BoundingBox  = stats(j).BoundingBox;
            splitDetections(nSplit).PixelIdxList = stats(j).PixelIdxList;
            splitDetections(nSplit).Area         = stats(j).Area;
            splitDetections(nSplit).fromSplit    = false;
            splitDetections(nSplit).parentIdx    = j;
        end
    end

    nDet = numel(splitDetections);

    %% BUILD COST MATRIX (tracks x detections)
    nTrk = length(tracks);
    costMatrix = inf(nTrk, nDet);

    detCentroids = zeros(nDet, 2);
    detHueHists  = zeros(nDet, numBins);
    detBBoxes    = zeros(nDet, 4);

    for j = 1:nDet
        detCentroids(j,:) = splitDetections(j).Centroid;
        detBBoxes(j,:)    = splitDetections(j).BoundingBox;
        detHueHists(j,:)  = getHueHist(hsvImg, splitDetections(j).BoundingBox, numBins);
    end

    for t = 1:nTrk
        predXY_t = tracks(t).kf_x(1:2)';

        % Vectorized distance to all detections
        diffs = detCentroids - predXY_t;
        dists = sqrt(diffs(:,1).^2 + diffs(:,2).^2);

        for j = 1:nDet
            if dists(j) > gateDistance, continue; end

            hueDiff = sum(abs(tracks(t).hueHist - detHueHists(j,:)));

            detH = detBBoxes(j,4); detW = detBBoxes(j,3);
            trkAspect = tracks(t).ellipse_a / max(tracks(t).ellipse_b, 1);
            detAspect = (detH/2) / max(detW/2, 1);
            shapeDiff = abs(trkAspect - detAspect) / max(trkAspect, 1);

            ghostPenalty = tracks(t).missingCount * missingPenalty;

            costMatrix(t,j) = dists(j) + hueCostWeight * hueDiff + 15 * shapeDiff + ghostPenalty;
        end
    end

    %% GREEDY ASSIGNMENT
    assignedTrk = false(1, nTrk);
    assignedDet = false(1, nDet);
    trkToDet = zeros(1, nTrk);

    [rows, cols] = find(isfinite(costMatrix));
    if ~isempty(rows)
        costs = costMatrix(sub2ind(size(costMatrix), rows, cols));
        [~, sortIdx] = sort(costs);
        for s = 1:length(sortIdx)
            t = rows(sortIdx(s));
            j = cols(sortIdx(s));
            if assignedTrk(t) || assignedDet(j), continue; end
            assignedTrk(t) = true;
            assignedDet(j) = true;
            trkToDet(t) = j;
        end
    end

    %% KALMAN UPDATE + ELLIPSE UPDATE for matched tracks
    for t = 1:nTrk
        if trkToDet(t) > 0
            j = trkToDet(t);
            z = detCentroids(j,:)';

            % Kalman gain
            S = H_obs * tracks(t).kf_P * H_obs' + R;
            K = tracks(t).kf_P * H_obs' / S;

            innovation = z - H_obs * tracks(t).kf_x;
            tracks(t).kf_x = tracks(t).kf_x + K * innovation;
            tracks(t).kf_P = (eye(4) - K * H_obs) * tracks(t).kf_P;

            tracks(t).centroid = detCentroids(j,:);
            tracks(t).bbox     = detBBoxes(j,:);
            tracks(t).hueHist  = 0.8 * tracks(t).hueHist + 0.2 * detHueHists(j,:);
            tracks(t).lastSeen = f;
            tracks(t).missingCount = 0;
            tracks(t).age = tracks(t).age + 1;

            % Update ellipse model (smooth update)
            newA = detBBoxes(j,4) / 2;  % semi-height
            newB = detBBoxes(j,3) / 2;  % semi-width

            if ~splitDetections(j).fromSplit
                % Only update shape from non-split detections (more reliable)
                tracks(t).ellipse_a = 0.7 * tracks(t).ellipse_a + 0.3 * newA;
                tracks(t).ellipse_b = 0.7 * tracks(t).ellipse_b + 0.3 * newB;
            end
        else
            tracks(t).missingCount = tracks(t).missingCount + 1;
            tracks(t).centroid = tracks(t).kf_x(1:2)';
            bb = tracks(t).bbox;
            bb(1) = tracks(t).kf_x(1) - bb(3)/2;
            bb(2) = tracks(t).kf_x(2) - bb(4)/2;
            tracks(t).bbox = bb;
        end
    end

    %% CREATE NEW TRACKS for unmatched detections
    for j = 1:nDet
        if assignedDet(j), continue; end
        if splitDetections(j).fromSplit, continue; end  % don't create new tracks from splits

        bb = detBBoxes(j,:);
        % Filter: new tracks must look like a single person (use calibrated size)
        if bb(3) > 0 && bb(4)/bb(3) < params.minAspectHW, continue; end
        yBot = bb(2) + bb(4);
        expH = max(polyval(pHeight, yBot), 30);
        expW = max(polyval(pWidth, yBot), 10);
        if bb(4) > mergeTolHeight * expH || bb(3) > mergeTolWidth * expW
            continue;  % too big for a single person at this depth, skip
        end

        id = nextID;
        nextID = nextID + 1;

        k = length(tracks) + 1;
        tracks(k).id           = id;
        tracks(k).centroid     = detCentroids(j,:);
        tracks(k).bbox         = bb;
        tracks(k).hueHist      = detHueHists(j,:);
        tracks(k).lastSeen     = f;
        tracks(k).missingCount = 0;
        tracks(k).stillInFrame = true;
        tracks(k).age          = 1;

        tracks(k).kf_x = [detCentroids(j,1); detCentroids(j,2); 0; 0];
        tracks(k).kf_P = P0;

        % Initialize ellipse from bounding box
        tracks(k).ellipse_a = bb(4) / 2;  % semi-height
        tracks(k).ellipse_b = bb(3) / 2;  % semi-width
    end

    %% MARK TRACKS NEAR EDGE + ADAPTIVE EDGE KILL
    % Tracks near edges get killed based on their AGE:
    %   - Young tracks (age < 10): kill after 2 missing frames
    %     (likely someone entering/exiting, don't let them linger)
    %   - Mature tracks (age >= 10): kill after 8 missing frames
    %     (they've been tracked a while, give them grace for flickering
    %      detection near edges, like the woman in the back)
    edgeMargin = 25;
    edgeKillYoung  = 2;   % young tracks near edge: kill quickly
    edgeKillMature = 8;   % mature tracks near edge: more tolerance
    edgeMaturityAge = 10; % threshold to be considered "mature"

    for t = 1:length(tracks)
        if tracks(t).lastSeen < f
            bb = tracks(t).bbox;
            x = bb(1); y = bb(2); w = bb(3); h = bb(4);

            touchesEdge = (x <= edgeMargin) || (x + w >= W - edgeMargin) || ...
                          (y <= edgeMargin) || (y + h >= H - edgeMargin);
            tracks(t).stillInFrame = ~touchesEdge;

            if touchesEdge
                if tracks(t).age < edgeMaturityAge
                    killAfter = edgeKillYoung;
                else
                    killAfter = edgeKillMature;
                end

                if tracks(t).missingCount >= killAfter
                    tracks(t).missingCount = maxMissing + 1;  % force removal
                end
            end
        else
            tracks(t).stillInFrame = true;
        end
    end

    %% REMOVE OLD TRACKS
    keep = true(1, length(tracks));
    for t = 1:length(tracks)
        if tracks(t).missingCount > maxMissing && ~tracks(t).stillInFrame
            keep(t) = false;
        end
        if tracks(t).missingCount > maxPredictFrames && tracks(t).stillInFrame
            keep(t) = false;
        end
    end
    tracks = tracks(keep);

    %% DISPLAY (2 panels: foreground + result with optional debug overlay)
    subplot(1,2,1);
    imshow(fgMask);
    title(sprintf('Foreground - Frame %d', f));

    subplot(1,2,2);
    imshow(img); hold on;

    for t = 1:length(tracks)
        bb = tracks(t).bbox;
        isOccluded = (tracks(t).missingCount > 0);

        if isOccluded
            % Predicted tracks: dashed box only (skip ellipses for speed)
            rectangle('Position', bb, 'EdgeColor', 'y', 'LineWidth', 1, ...
                      'LineStyle', '--');
            text(bb(1), bb(2)-8, sprintf('ID%d (pred)', tracks(t).id), ...
                 'Color','yellow','FontSize',7,'FontWeight','bold', ...
                 'BackgroundColor','none','Margin',1);
        else
            rectangle('Position', bb, 'EdgeColor', 'g', 'LineWidth', 2);
            text(bb(1), bb(2)-8, sprintf('ID%d', tracks(t).id), ...
                 'Color','cyan','FontSize',7,'FontWeight','bold', ...
                 'BackgroundColor','none','Margin',1);
        end
    end
    hold off;
    title(sprintf('Frame %d  (%d tracks)', f, length(tracks)));

    drawnow;
end

fprintf('Done. Total IDs assigned: %d\n', nextID - 1);


%% =========================================================
%  LOCAL FUNCTION: Draw 3-part body ellipse (head + torso + legs)
%  =========================================================
function drawBodyEllipse(cx, cy, semiH, semiW, color, lineStyle)
% Draws a simplified 3-part body model as ellipses:
%   Head:  small ellipse at top (15% of height)
%   Torso: wider ellipse in middle (40% of height)
%   Legs:  medium ellipse at bottom (45% of height)
%
% cx, cy  = center of the full body
% semiH   = half the total body height
% semiW   = half the total body width

    if semiH < 5 || semiW < 2, return; end

    totalH = 2 * semiH;
    topY = cy - semiH;

    theta = linspace(0, 2*pi, 40);

    % HEAD: 15% of total height, narrow
    headH = totalH * 0.15;
    headW = semiW * 0.5;
    headCy = topY + headH/2;
    hx = cx + headW * cos(theta);
    hy = headCy + (headH/2) * sin(theta);
    plot(hx, hy, 'Color', color, 'LineWidth', 1.5, 'LineStyle', lineStyle);

    % TORSO: 40% of height, widest
    torsoH = totalH * 0.40;
    torsoW = semiW * 1.0;
    torsoCy = topY + headH + torsoH/2;
    tx = cx + torsoW * cos(theta);
    ty = torsoCy + (torsoH/2) * sin(theta);
    plot(tx, ty, 'Color', color, 'LineWidth', 1.5, 'LineStyle', lineStyle);

    % LEGS: 45% of height, medium width
    legH = totalH * 0.45;
    legW = semiW * 0.7;
    legCy = topY + headH + torsoH + legH/2;
    lx = cx + legW * cos(theta);
    ly = legCy + (legH/2) * sin(theta);
    plot(lx, ly, 'Color', color, 'LineWidth', 1.5, 'LineStyle', lineStyle);
end
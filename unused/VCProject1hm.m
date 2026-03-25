%% Prototype overlap-aware pedestrian tracker
clear; close all; clc;

%% Parameters
params.bgFrames        = 50;
params.fgThreshold     = 50;

params.minBlobAreaTop    = 80;
params.minBlobAreaBottom = 1400;
params.maxBlobArea       = 30000;

params.minAspectHW       = 2.0;
params.minHeight         = 15;
params.maxHeight         = 300;
params.minWidth          = 6;
params.maxWidth          = 60;

params.seClose           = strel('disk', 4);
params.seOpen            = strel('disk', 3);

params.maxMatchDist      = 60;    % max centroid distance for matching
params.maxMissed         = 8;     % keep track alive for this many missed frames
params.minTrackAge       = 1;

params.wPos              = 0.7;   % weight for position cost
params.wColor            = 0.3;   % weight for color cost

params.histBins          = 16;    % histogram bins
params.showPredictedOnly = true;  % show track even if not detected in frame

%% Paths
imgDir  = fullfile('Views', 'View_001');
nFrames = 795;

%% Build background model
fprintf('Building background model...\n');
sample = imread(fullfile(imgDir, 'frame_0000.jpg'));
[H, W, ~] = size(sample);

bgStack = zeros(H, W, params.bgFrames, 'uint8');
for i = 1:params.bgFrames
    img = imread(fullfile(imgDir, sprintf('frame_%04d.jpg', i - 1)));
    bgStack(:,:,i) = rgb2gray(img);
end
bgModel = median(bgStack, 3);
fprintf('Background model built.\n');

%% Track structure
tracks = struct( ...
    'id', {}, ...
    'bbox', {}, ...
    'centroid', {}, ...
    'prevCentroid', {}, ...
    'velocity', {}, ...
    'missed', {}, ...
    'age', {}, ...
    'hist', {}, ...
    'matched', {}, ...
    'predCentroid', {});

nextTrackID = 1;

%% Display
figure('Name', 'Overlap-aware Tracking Prototype', ...
       'NumberTitle', 'off', ...
       'Position', [100 100 1200 550]);

for f = 1:nFrames
    imgFile = fullfile(imgDir, sprintf('frame_%04d.jpg', f - 1));
    if ~isfile(imgFile)
        continue;
    end

    img = imread(imgFile);
    gray = rgb2gray(img);

    %% 1) Foreground detection
    fgMask = abs(double(gray) - double(bgModel)) > params.fgThreshold;
    fgMask = imclose(fgMask, params.seClose);
    fgMask = imopen(fgMask, params.seOpen);
    fgMask = imfill(fgMask, 'holes');
    fgMask = bwareaopen(fgMask, 30);

    %% 2) Blob extraction
    stats = regionprops(fgMask, 'BoundingBox', 'Area', 'Centroid');

    detections = struct('bbox', {}, 'centroid', {}, 'hist', {});

    for j = 1:numel(stats)
        bb = stats(j).BoundingBox;
        area = stats(j).Area;

        wBox = bb(3);
        hBox = bb(4);

        if wBox <= 0
            continue;
        end

        aspectHW = hBox / wBox;

        yBottom = bb(2) + bb(4);
        alpha = min(max(yBottom / H, 0), 1);
        minAreaHere = params.minBlobAreaTop + ...
            alpha * (params.minBlobAreaBottom - params.minBlobAreaTop);

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

        det.bbox = bb;
        det.centroid = stats(j).Centroid;
        det.hist = computeHSVHist(img, bb, params.histBins);

        detections(end+1) = det; %#ok<SAGROW>
    end

    %% 3) Predict track positions
    for t = 1:numel(tracks)
        tracks(t).matched = false;
        tracks(t).predCentroid = tracks(t).centroid + tracks(t).velocity;
    end

    %% 4) Match detections to tracks (greedy prototype)
    numTracks = numel(tracks);
    numDets   = numel(detections);

    costMatrix = inf(numTracks, numDets);

    for t = 1:numTracks
        for d = 1:numDets
            posDist = norm(tracks(t).predCentroid - detections(d).centroid);

            if posDist > params.maxMatchDist
                continue;
            end

            colorDist = histDist(tracks(t).hist, detections(d).hist);

            % normalize roughly
            posCost = posDist / params.maxMatchDist;
            colorCost = colorDist;

            totalCost = params.wPos * posCost + params.wColor * colorCost;
            costMatrix(t,d) = totalCost;
        end
    end

    assignedTracks = false(1, numTracks);
    assignedDets   = false(1, numDets);

    % Greedy assignment by lowest cost
    while true
        [minVal, idx] = min(costMatrix(:));
        if isempty(minVal) || isinf(minVal)
            break;
        end

        [t, d] = ind2sub(size(costMatrix), idx);

        if assignedTracks(t) || assignedDets(d)
            costMatrix(t,d) = inf;
            continue;
        end

        % Assign detection d to track t
        assignedTracks(t) = true;
        assignedDets(d) = true;

        newCentroid = detections(d).centroid;
        oldCentroid = tracks(t).centroid;
        newVel = newCentroid - oldCentroid;

        tracks(t).prevCentroid = tracks(t).centroid;
        tracks(t).centroid = newCentroid;
        tracks(t).bbox = detections(d).bbox;
        tracks(t).velocity = 0.7 * tracks(t).velocity + 0.3 * newVel;
        tracks(t).missed = 0;
        tracks(t).age = tracks(t).age + 1;
        tracks(t).hist = detections(d).hist;
        tracks(t).matched = true;

        costMatrix(t,:) = inf;
        costMatrix(:,d) = inf;
    end

    %% 5) Handle unmatched tracks
    for t = 1:numel(tracks)
        if ~tracks(t).matched
            tracks(t).missed = tracks(t).missed + 1;
            tracks(t).age = tracks(t).age + 1;

            % keep predicting through short occlusion/overlap
            tracks(t).prevCentroid = tracks(t).centroid;
            tracks(t).centroid = tracks(t).predCentroid;

            % shift bbox center with prediction
            bb = tracks(t).bbox;
            bb(1) = bb(1) + tracks(t).velocity(1);
            bb(2) = bb(2) + tracks(t).velocity(2);
            tracks(t).bbox = bb;
        end
    end

    %% 6) Overlap handling idea:
    % If one detection contains multiple predicted track centers,
    % do not delete those tracks; let them survive predicted-only.
    % This is already partly handled because unmatched tracks survive.
    %
    % Below, we reinforce it by preventing creation of a new track if a det
    % overlaps strongly with existing predicted tracks.

    %% 7) Create new tracks from unmatched detections
    for d = 1:numDets
        if assignedDets(d)
            continue;
        end

        detBB = detections(d).bbox;
        detC  = detections(d).centroid;

        % Check whether this unmatched detection may actually be a merged blob
        % containing one or more predicted tracks
        containsPredictedTrack = false;
        for t = 1:numel(tracks)
            if tracks(t).missed > params.maxMissed
                continue;
            end
            pc = tracks(t).predCentroid;
            if pointInBox(pc, detBB)
                containsPredictedTrack = true;
                break;
            end
        end

        % If this detection already contains a predicted track,
        % don't immediately create a new identity.
        if containsPredictedTrack
            continue;
        end

        newTrack.id = nextTrackID;
        newTrack.bbox = detections(d).bbox;
        newTrack.centroid = detections(d).centroid;
        newTrack.prevCentroid = detections(d).centroid;
        newTrack.velocity = [0 0];
        newTrack.missed = 0;
        newTrack.age = 1;
        newTrack.hist = detections(d).hist;
        newTrack.matched = true;
        newTrack.predCentroid = detections(d).centroid;

        tracks(end+1) = newTrack; %#ok<SAGROW>
        nextTrackID = nextTrackID + 1;
    end

    %% 8) Delete dead tracks
    keep = true(1, numel(tracks));
    for t = 1:numel(tracks)
        if tracks(t).missed > params.maxMissed
            keep(t) = false;
        end
    end
    tracks = tracks(keep);

    %% 9) Display
    subplot(1,2,1);
    imshow(fgMask);
    title(sprintf('Foreground Mask - Frame %d', f));

    subplot(1,2,2);
    imshow(img); hold on;
    title(sprintf('Tracks - Frame %d', f));

    % Draw detections in green
    for d = 1:numDets
        bb = detections(d).bbox;
        rectangle('Position', bb, 'EdgeColor', 'g', 'LineWidth', 1);
    end

    % Draw tracks in red
    for t = 1:numel(tracks)
        if tracks(t).age < params.minTrackAge
            continue;
        end

        if ~params.showPredictedOnly && tracks(t).missed > 0
            continue;
        end

        bb = tracks(t).bbox;
        c  = tracks(t).centroid;

        if tracks(t).missed == 0
            edgeColor = 'r';
            labelText = sprintf('ID %d', tracks(t).id);
        else
            edgeColor = 'y';
            labelText = sprintf('ID %d (pred)', tracks(t).id);
        end

        rectangle('Position', bb, 'EdgeColor', edgeColor, 'LineWidth', 2);
        text(bb(1), bb(2)-8, labelText, ...
            'Color', 'cyan', 'FontSize', 9, ...
            'FontWeight', 'bold', 'BackgroundColor', 'black');

        plot(c(1), c(2), 'r+', 'MarkerSize', 8, 'LineWidth', 1.5);

        % velocity arrow
        quiver(c(1), c(2), tracks(t).velocity(1), tracks(t).velocity(2), ...
            0, 'Color', 'yellow', 'LineWidth', 1);
    end

    hold off;
    drawnow;
    pause(0.03);
end

fprintf('Done.\n');

%% ===== Helper functions =====

function h = computeHSVHist(img, bbox, nBins)
    % bbox = [x y w h]
    x1 = max(1, floor(bbox(1)));
    y1 = max(1, floor(bbox(2)));
    x2 = min(size(img,2), ceil(bbox(1) + bbox(3) - 1));
    y2 = min(size(img,1), ceil(bbox(2) + bbox(4) - 1));

    patch = img(y1:y2, x1:x2, :);

    if isempty(patch)
        h = zeros(1, nBins*2);
        return;
    end

    hsv = rgb2hsv(patch);
    H = hsv(:,:,1);
    S = hsv(:,:,2);

    % use central region to reduce background contamination
    [ph, pw, ~] = size(patch);
    rx1 = max(1, round(0.2 * pw));
    rx2 = max(rx1, round(0.8 * pw));
    ry1 = max(1, round(0.2 * ph));
    ry2 = max(ry1, round(0.8 * ph));

    Hc = H(ry1:ry2, rx1:rx2);
    Sc = S(ry1:ry2, rx1:rx2);

    hH = histcounts(Hc(:), nBins, 'Normalization', 'probability');
    hS = histcounts(Sc(:), nBins, 'Normalization', 'probability');

    h = [hH, hS];

    if sum(h) > 0
        h = h / sum(h);
    end
end

function d = histDist(h1, h2)
    % simple L1 distance in [0, 2] roughly, normalize to [0,1]
    d = sum(abs(h1 - h2)) / 2;
end

function inside = pointInBox(pt, bb)
    x = pt(1);
    y = pt(2);
    inside = (x >= bb(1)) && (x <= bb(1) + bb(3)) && ...
             (y >= bb(2)) && (y <= bb(2) + bb(4));
end
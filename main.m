%% Detection + Tracking + Lamppost Extension + Merge/Split Display
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
tracks = struct('id',{},'centroid',{},'bbox',{},'hueHist',{}, ...
                'lastSeen',{},'isMissing',{},'stillInFrame',{},'missingCount',{});
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
    usedTrackIdx = [];
    usedDisplayIDs = [];
    usedPrevMergedIdx = [];

    %% DISPLAY SETUP
    subplot(1,2,1);
    imshow(fgMask);
    title(sprintf('FG %d', f));

    subplot(1,2,2);
    imshow(img);
    hold on;

    %% MATCH DETECTIONS TO TRACKS
    for j = 1:length(stats)

        bb = stats(j).BoundingBox;
        ct = stats(j).Centroid;
        hHist = getHueHist(img, bb, numBins);

        bestID = -1;
        bestScore = inf;
        bestTrackIdx = -1;
        
        for t = 1:length(tracks)
            if ismember(t, usedTrackIdx)
                continue;
            end
        
            dist = norm(ct - tracks(t).centroid);
            hueDiff = sum(abs(hHist - tracks(t).hueHist));
            score = dist + 80*hueDiff;
        
            if score < bestScore && dist < 80
                bestScore = score;
                bestID = tracks(t).id;
                bestTrackIdx = t;
            end
        end
        
        if bestID == -1
            % New person
            id = nextID;
            nextID = nextID + 1;
        
            tracks(end+1).id = id;
            tracks(end).centroid = ct;
            tracks(end).bbox = bb;
            tracks(end).hueHist = hHist;
            tracks(end).lastSeen = f;
            tracks(end).isMissing = false;
            tracks(end).stillInFrame = true;
            tracks(end).missingCount = 0;
        
        else
            % Re-identified person
            id = bestID;
        
            tracks(bestTrackIdx).centroid = ct;
            tracks(bestTrackIdx).bbox = bb;
            tracks(bestTrackIdx).hueHist = 0.8 * tracks(bestTrackIdx).hueHist + 0.2 * hHist;
            tracks(bestTrackIdx).lastSeen = f;
            tracks(bestTrackIdx).isMissing = false;
            tracks(bestTrackIdx).stillInFrame = true;
            tracks(bestTrackIdx).missingCount = 0;
        
            usedTrackIdx(end+1) = bestTrackIdx; %#ok<SAGROW>
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
        
        displayIDs = resolveDisplayIDsFromMaskOverlap(currPixelIdx, currArea, bb, id, prevTracks);
        
        % persistent merged display from previous frame, but each previous merged box
        % can only be used once per frame
        if numel(displayIDs) == 1
            bestM = 0;
            bestIoU = 0;
        
            for m = 1:numel(prevMergedBoxes)
                if ismember(m, usedPrevMergedIdx)
                    continue;
                end
        
                ov = bboxIoU(bb, prevMergedBoxes(m).bbox);
                if ov > 0.25 && numel(prevMergedBoxes(m).displayIDs) >= 2 && ov > bestIoU
                    bestIoU = ov;
                    bestM = m;
                end
            end
        
            if bestM > 0
                prevIDs = prevMergedBoxes(bestM).displayIDs;
                prevBox = prevMergedBoxes(bestM).bbox;
        
                % Simple split handling for 2-person merges:
                % left child gets left old ID, right child gets right old ID
                if numel(prevIDs) == 2
                    prevIDs = orderIDsByPreviousX(prevIDs, prevTracks);
        
                    oldMidX = prevBox(1) + prevBox(3)/2;
                    newMidX = bb(1) + bb(3)/2;
        
                    if newMidX < oldMidX
                        displayIDs = prevIDs(1);
                    else
                        displayIDs = prevIDs(2);
                    end
                else
                    displayIDs = prevIDs;
                end
        
                usedPrevMergedIdx(end+1) = bestM; %#ok<SAGROW>
            end
        end
        
        % Order merged IDs left-to-right
        if numel(displayIDs) >= 2
            displayIDs = orderIDsByPreviousX(displayIDs, prevTracks);
        end
        
        % ---------- Enforce per-frame display uniqueness ----------
        if numel(displayIDs) == 1
            % Single ID cannot appear twice in same frame
            if ismember(displayIDs, usedDisplayIDs)
                % Fallback: use the tracker ID if unused, otherwise allocate a new ID
                if ~ismember(id, usedDisplayIDs)
                    displayIDs = id;
                else
                    displayIDs = nextID;
                    nextID = nextID + 1;
                end
            end
        else
            % Remove any IDs already consumed by earlier blobs this frame
            displayIDs = displayIDs(~ismember(displayIDs, usedDisplayIDs));
        
            % If all IDs were already used, fallback to tracker ID or new ID
            if isempty(displayIDs)
                if ~ismember(id, usedDisplayIDs)
                    displayIDs = id;
                else
                    displayIDs = nextID;
                    nextID = nextID + 1;
                end
            end
        end
        
        % Mark display IDs as used in this frame
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
    %% Mark unmatches blobs
    edgeMargin = 25;

    for t = 1:length(tracks)
        if tracks(t).lastSeen < f   % not matched this frame
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
        tooOld = tracks(t).missingCount > maxMissing;
    
        if tooOld && ~tracks(t).stillInFrame
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

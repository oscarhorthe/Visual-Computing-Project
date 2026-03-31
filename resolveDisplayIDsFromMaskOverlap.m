function displayIDs = resolveDisplayIDsFromMaskOverlap(currPixelIdx, currArea, currBox, currID, prevTracks)
% Detects merge events by checking pixel overlap with previous tracks.
% Returns multiple IDs only when a genuine merge is detected
% (current blob absorbs significant parts of 2+ previous blobs).
% For single-track matches, returns currID (caller should use tracker ID).
 
    displayIDs = currID;
 
    if isempty(prevTracks)
        return;
    end
 
    minPrevOverlap = 0.25;   % tightened from 0.2
    minCurrOverlap = 0.10;   % tightened from 0.08
    minAreaGain = 1.3;       % tightened from 1.2
    maxCenterDist = 100;     % tightened from 120
 
    candidateIDs = [];
    candidateAreas = [];
 
    currCenter = [currBox(1) + currBox(3)/2, currBox(2) + currBox(4)/2];
 
    for p = 1:numel(prevTracks)
        prevPix = prevTracks(p).pixelIdx;
        prevArea = prevTracks(p).area;
        prevCenter = prevTracks(p).centroid;
 
        if isempty(prevPix) || prevArea <= 0
            continue;
        end
 
        centerDist = norm(currCenter - prevCenter);
        if centerDist > maxCenterDist
            continue;
        end
 
        numOverlap = numel(intersect(currPixelIdx, prevPix));
        overlapPrev = numOverlap / prevArea;
        overlapCurr = numOverlap / currArea;
 
        % Require meaningful overlap from BOTH perspectives to count as absorbed
        if overlapPrev >= minPrevOverlap && overlapCurr >= minCurrOverlap
            candidateIDs = [candidateIDs, prevTracks(p).displayIDs]; %#ok<AGROW>
            candidateAreas(end+1) = prevArea; %#ok<AGROW>
        end
    end
 
    candidateIDs = unique(candidateIDs, 'stable');
 
    if isempty(candidateIDs)
        return;
    end
 
    % Only report a merge if 2+ distinct previous tracks are absorbed
    % AND the current blob is substantially larger than the smallest absorbed one
    if numel(candidateIDs) >= 2
        if currArea >= minAreaGain * min(candidateAreas)
            displayIDs = candidateIDs;
            return;
        end
    end
 
    % Single candidate: return currID (let the tracker ID be authoritative)
    displayIDs = currID;
end
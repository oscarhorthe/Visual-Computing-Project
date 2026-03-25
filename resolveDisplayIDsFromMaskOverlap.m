function displayIDs = resolveDisplayIDsFromMaskOverlap(currPixelIdx, currArea, currBox, currID, prevTracks)

    displayIDs = currID;

    if isempty(prevTracks)
        return;
    end

    minPrevOverlap = 0.2;
    minCurrOverlap = 0.08;
    minAreaGain = 1.2;
    maxCenterDist = 120;   % very important gate

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

        disp(['  Prev track ID(s): ' mat2str(prevTracks(p).displayIDs) ...
              ', centerDist=' num2str(centerDist, '%.1f') ...
              ', prevArea=' num2str(prevArea) ...
              ', overlapPixels=' num2str(numOverlap) ...
              ', overlapPrev=' num2str(overlapPrev, '%.3f') ...
              ', overlapCurr=' num2str(overlapCurr, '%.3f')]);

        if overlapPrev >= minPrevOverlap || overlapCurr >= minCurrOverlap
            candidateIDs = [candidateIDs, prevTracks(p).displayIDs]; %#ok<AGROW>
            candidateAreas(end+1) = prevArea; %#ok<AGROW>
            disp(['    -> accepted candidate: ' mat2str(prevTracks(p).displayIDs)]);
        end
    end

    disp(['  Candidate IDs before unique: ' mat2str(candidateIDs)]);
    candidateIDs = unique(candidateIDs, 'stable');
    disp(['  Candidate IDs after unique: ' mat2str(candidateIDs)]);

    if isempty(candidateIDs)
        return;
    end

    if numel(candidateIDs) >= 2
        if currArea >= minAreaGain * min(candidateAreas)
            displayIDs = candidateIDs;
            return;
        end
    end

    displayIDs = candidateIDs(1);
end
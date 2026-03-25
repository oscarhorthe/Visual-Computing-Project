function [displayIDs, circleCenters] = getPersistentMergedIDs(bbox, prevTracks, id)

    x = bbox(1); y = bbox(2); w = bbox(3); h = bbox(4);
    boxCenter = [x + w/2, y + h/2];

    matchIdx = 0;
    bestDist = inf;

    % Find closest previous bbox center
    for t = 1:length(prevTracks)
        pb = prevTracks(t).bbox;
        prevCenter = [pb(1) + pb(3)/2, pb(2) + pb(4)/2];
        d = norm(prevCenter - boxCenter);

        if d < bestDist
            bestDist = d;
            matchIdx = t;
        end
    end

    % Default: one ID
    displayIDs = id;

    if matchIdx > 0
        prevIDs = prevTracks(matchIdx).displayIDs;
        pb = prevTracks(matchIdx).bbox;

        prevWidth = pb(3);
        currWidth = w;

        % If previous box had multiple IDs and current box is still fairly wide,
        % keep the same merged IDs
        if length(prevIDs) >= 2 && currWidth > 0.65 * prevWidth
            displayIDs = prevIDs;
        end
    end

    % Draw circles
    n = length(displayIDs);
    if n == 0
        n = 1;
        displayIDs = id;
    end

    circleCenters = zeros(n,2);
    cy = y + h/2;
    for i = 1:n
        cx = x + i * w / (n + 1);
        circleCenters(i,:) = [cx, cy];
    end
end
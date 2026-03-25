function fgMaskMerged = mergeBlobsAroundObstacle(fgMask, obstacleMask, maxHorizGap, minVertOverlap)
% mergeBlobsAroundObstacle
% Merges foreground blobs that are split by a known obstacle mask.
%
% Inputs:
%   fgMask          - logical foreground mask
%   obstacleMask    - logical mask of lamp post / sign
%   maxHorizGap     - maximum horizontal gap allowed across obstacle
%   minVertOverlap  - minimum vertical overlap ratio [0..1]
%
% Output:
%   fgMaskMerged    - logical mask with split blobs reconnected
%
% Example:
%   fgMask = mergeBlobsAroundObstacle(fgMask, lamppostMask, 40, 0.25);

    fgMask = logical(fgMask);
    obstacleMask = logical(obstacleMask);
    fgMaskMerged = fgMask;

    cc = bwconncomp(fgMaskMerged);
    stats = regionprops(cc, 'BoundingBox', 'PixelIdxList');

    n = numel(stats);
    if n < 2
        return;
    end

    bboxes = reshape([stats.BoundingBox], 4, []).';

    % bbox format: [x y w h]
    x1 = bboxes(:,1);
    y1 = bboxes(:,2);
    x2 = bboxes(:,1) + bboxes(:,3);
    y2 = bboxes(:,2) + bboxes(:,4);

    for i = 1:n-1
        for j = i+1:n

            % --- Vertical overlap check ---
            overlapTop = max(y1(i), y1(j));
            overlapBot = min(y2(i), y2(j));
            overlapH = overlapBot - overlapTop;

            if overlapH <= 0
                continue;
            end

            h1 = y2(i) - y1(i);
            h2 = y2(j) - y1(j);
            vertOverlapRatio = overlapH / min(h1, h2);

            if vertOverlapRatio < minVertOverlap
                continue;
            end

            % --- Horizontal separation check ---
            % Force left/right ordering
            if x1(i) < x1(j)
                leftIdx = i;
                rightIdx = j;
            else
                leftIdx = j;
                rightIdx = i;
            end

            gap = x1(rightIdx) - x2(leftIdx);
            if gap < 0 || gap > maxHorizGap
                continue;
            end

            % --- Check that obstacle lies between them ---
            bridgeX1 = max(1, floor(x2(leftIdx)));
            bridgeX2 = min(size(fgMaskMerged,2), ceil(x1(rightIdx)));

            bridgeY1 = max(1, floor(overlapTop));
            bridgeY2 = min(size(fgMaskMerged,1), ceil(overlapBot));

            if bridgeX2 <= bridgeX1 || bridgeY2 <= bridgeY1
                continue;
            end

            bridgeRegionObstacle = obstacleMask(bridgeY1:bridgeY2, bridgeX1:bridgeX2);

            % Require some obstacle pixels in the gap region
            if ~any(bridgeRegionObstacle(:))
                continue;
            end

            % --- Connect the two blobs through obstacle region ---
            bridgeMask = false(size(fgMaskMerged));
            bridgeMask(bridgeY1:bridgeY2, bridgeX1:bridgeX2) = true;

            % Only fill inside the obstacle region and a tiny margin around it
            se = strel('disk', 2);
            localBridge = imdilate(bridgeRegionObstacle, se);

            temp = false(size(fgMaskMerged));
            temp(bridgeY1:bridgeY2, bridgeX1:bridgeX2) = localBridge;

            fgMaskMerged = fgMaskMerged | temp;
        end
    end

    % Final cleanup to make merged person one component
    fgMaskMerged = imclose(fgMaskMerged, strel('disk', 2));
    fgMaskMerged = imfill(fgMaskMerged, 'holes');
end
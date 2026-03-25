function fgMaskOut = mergeBlobsSplitBySign(fgMask)
% Merges foreground blobs that are split by the known lamp sign region.
%
% Known sign bounding box:
%   x = 413..449
%   y = 202..246

    fgMask = logical(fgMask);
    fgMaskOut = fgMask;

    % Known sign coordinates
    signX1 = 413;
    signX2 = 449;
    signY1 = 202;
    signY2 = 246;

    % Tuning parameters
    touchMargin = 12;      % how close blob must be to sign edge
    minXOverlap = 8;       % minimum horizontal overlap to merge
    maxWidthDiff = 35;     % width similarity tolerance

    cc = bwconncomp(fgMaskOut);
    stats = regionprops(cc, 'BoundingBox', 'Area');

    if numel(stats) < 2
        return;
    end

    upperIdx = [];
    lowerIdx = [];

    % Classify blobs as upper-fragment or lower-fragment
    for k = 1:numel(stats)
        bb = stats(k).BoundingBox; % [x y w h]

        x1 = bb(1);
        y1 = bb(2);
        x2 = bb(1) + bb(3);
        y2 = bb(2) + bb(4);
        w  = bb(3);

        % Must overlap the sign horizontally at least a little
        xOverlap = min(x2, signX2) - max(x1, signX1);
        if xOverlap < minXOverlap
            continue;
        end

        % Upper fragment: ends near top of sign
        if y2 >= (signY1 - touchMargin) && y2 <= (signY1 + touchMargin)
            upperIdx(end+1) = k; %#ok<AGROW>
        end

        % Lower fragment: starts near bottom of sign
        if y1 >= (signY2 - touchMargin) && y1 <= (signY2 + touchMargin)
            lowerIdx(end+1) = k; %#ok<AGROW>
        end
    end

    % Try to pair upper and lower fragments
    for i = 1:numel(upperIdx)
        ku = upperIdx(i);
        bbu = stats(ku).BoundingBox;

        xu1 = bbu(1);
        yu1 = bbu(2);
        xu2 = bbu(1) + bbu(3);
        yu2 = bbu(2) + bbu(4);
        wu  = bbu(3);

        bestMatch = 0;
        bestScore = inf;

        for j = 1:numel(lowerIdx)
            kl = lowerIdx(j);
            bbl = stats(kl).BoundingBox;

            xl1 = bbl(1);
            yl1 = bbl(2);
            xl2 = bbl(1) + bbl(3);
            yl2 = bbl(2) + bbl(4);
            wl  = bbl(3);

            % Horizontal overlap
            xOverlap = min(xu2, xl2) - max(xu1, xl1);
            if xOverlap < minXOverlap
                continue;
            end

            % Width similarity
            if abs(wu - wl) > maxWidthDiff
                continue;
            end

            % Score: prefer similar x-center and width
            cu = (xu1 + xu2) / 2;
            cl = (xl1 + xl2) / 2;
            score = abs(cu - cl) + 0.5 * abs(wu - wl);

            if score < bestScore
                bestScore = score;
                bestMatch = kl;
            end
        end

        if bestMatch == 0
            continue;
        end

        % Merge by filling bridge through sign region
        bbl = stats(bestMatch).BoundingBox;

        xl1 = bbl(1);
        yl1 = bbl(2);
        xl2 = bbl(1) + bbl(3);
        yl2 = bbl(2) + bbl(4);

        % Use overlapping x-range, clipped to sign region
        bridgeX1 = max([xu1, xl1, signX1]);
        bridgeX2 = min([xu2, xl2, signX2]);

        % Fill from bottom of upper fragment to top of lower fragment
        bridgeY1 = max(1, floor(yu2));
        bridgeY2 = min(size(fgMaskOut,1), ceil(yl1));

        if bridgeX2 > bridgeX1 && bridgeY2 > bridgeY1
            fgMaskOut(bridgeY1:bridgeY2, floor(bridgeX1):ceil(bridgeX2)) = true;
        end
    end

    % Final cleanup
    fgMaskOut = imclose(fgMaskOut, strel('disk', 2));
    fgMaskOut = imfill(fgMaskOut, 'holes');
end
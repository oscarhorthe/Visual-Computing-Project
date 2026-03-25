function fgMaskOut = extendShortBlobsAtSign(fgMask)
% Extends short blobs when they touch the sign.
% Top-touching blobs are extended to total height 80.
% Bottom-touching blobs are extended to total height 90.
%
% Sign coordinates:
%   top-left     = (413, 202)
%   bottom-right = (449, 246)

    fgMask = logical(fgMask);
    fgMaskOut = fgMask;

    signX1 = 413;
    signX2 = 449;
    signY1 = 202;
    signY2 = 246;

    touchMargin = 8;
    minHeight = 40;

    targetHeightTop = 80;
    targetHeightBottom = 90;

    cc = bwconncomp(fgMaskOut);
    stats = regionprops(cc, 'BoundingBox');

    for k = 1:numel(stats)
        bb = stats(k).BoundingBox;   % [x y w h]

        x1 = bb(1);
        y1 = bb(2);
        w  = bb(3);
        h  = bb(4);
        x2 = x1 + w;
        y2 = y1 + h;

        % Must overlap the sign somewhere horizontally
        xOverlap = min(x2, signX2) - max(x1, signX1);
        if xOverlap <= 0
            continue;
        end

        % Only extend short blobs
        if h >= minHeight
            continue;
        end

        % Slim bridge width
        cx = (x1 + x2) / 2;
        bridgeWidth = max(6, min(12, round(w * 0.35)));

        fillX1 = round(cx - bridgeWidth/2);
        fillX2 = round(cx + bridgeWidth/2);

        fillX1 = max(1, fillX1);
        fillX2 = min(size(fgMaskOut,2), fillX2);

        if fillX2 <= fillX1
            continue;
        end

        % Case 1: touches top of sign -> extend downward to total height 80
        if y2 >= (signY1 - touchMargin) && y2 <= (signY1 + touchMargin)
            extendAmount = targetHeightTop - h;

            if extendAmount > 0
                fillY1 = max(1, floor(y2));
                fillY2 = min(size(fgMaskOut,1), floor(y2 + extendAmount));

                if fillY2 > fillY1
                    fgMaskOut(fillY1:fillY2, fillX1:fillX2) = true;
                end
            end
        end

        % Case 2: touches bottom of sign -> extend upward to total height 90
        if y1 >= (signY2 - touchMargin) && y1 <= (signY2 + touchMargin)
            extendAmount = targetHeightBottom - h;

            if extendAmount > 0
                fillY2 = min(size(fgMaskOut,1), ceil(y1));
                fillY1 = max(1, ceil(y1 - extendAmount));

                if fillY2 > fillY1
                    fgMaskOut(fillY1:fillY2, fillX1:fillX2) = true;
                end
            end
        end
    end

    fgMaskOut = imclose(fgMaskOut, strel('disk', 2));
    fgMaskOut = imfill(fgMaskOut, 'holes');
end
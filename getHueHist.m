function hHist = getHueHist(img, bbox, numBins, pixelIdx)
% Compute a hue histogram over foreground pixels only.
%
% img       - RGB image (H x W x 3, uint8)
% bbox      - [x y w h] bounding box
% numBins   - number of histogram bins
% pixelIdx  - linear indices of foreground pixels (from PixelIdxList)
%
% If pixelIdx is provided, only those pixels are used (masked to bbox).
% This prevents background colors from contaminating the histogram.

x = max(1, floor(bbox(1)));
y = max(1, floor(bbox(2)));
w = floor(bbox(3));
h = floor(bbox(4));

x2 = min(size(img,2), x+w-1);
y2 = min(size(img,1), y+h-1);

if x2 <= x || y2 <= y
    hHist = zeros(1,numBins);
    return;
end

% Convert full image to HSV
hsvImg = rgb2hsv(img);

[imgH, imgW, ~] = size(img);

if nargin >= 4 && ~isempty(pixelIdx)
    % Build a mask from PixelIdxList (linear indices into H x W image)
    fgMask = false(imgH, imgW);
    fgMask(pixelIdx) = true;

    % Crop to bounding box region
    fgPatch = fgMask(y:y2, x:x2);
    hPatch  = hsvImg(y:y2, x:x2, 1);
    sPatch  = hsvImg(y:y2, x:x2, 2);
    vPatch  = hsvImg(y:y2, x:x2, 3);

    % Only use foreground pixels with sufficient saturation and brightness
    mask = fgPatch & (sPatch > 0.15) & (vPatch > 0.15);
else
    % Fallback: use all pixels in bbox (original behavior)
    hPatch = hsvImg(y:y2, x:x2, 1);
    sPatch = hsvImg(y:y2, x:x2, 2);
    vPatch = hsvImg(y:y2, x:x2, 3);

    mask = (sPatch > 0.15) & (vPatch > 0.15);
end

hVals = hPatch(mask);

if isempty(hVals)
    hHist = zeros(1,numBins);
    return;
end

edges = linspace(0,1,numBins+1);
hHist = histcounts(hVals, edges, 'Normalization','probability');
end
function hHist = getHueHist(hsvImg, bbox, numBins)
% Compute hue histogram from a precomputed HSV image.
% hsvImg  - precomputed HSV image (H x W x 3, double)
% bbox    - [x y w h] bounding box
% numBins - number of histogram bins

x = max(1, floor(bbox(1)));
y = max(1, floor(bbox(2)));
w = floor(bbox(3));
h = floor(bbox(4));

x2 = min(size(hsvImg,2), x+w-1);
y2 = min(size(hsvImg,1), y+h-1);

if x2 <= x || y2 <= y
    hHist = zeros(1,numBins);
    return;
end

H = hsvImg(y:y2, x:x2, 1);
S = hsvImg(y:y2, x:x2, 2);

mask = S > 0.2;
hVals = H(mask);

if isempty(hVals)
    hHist = zeros(1,numBins);
    return;
end

edges = linspace(0,1,numBins+1);
hHist = histcounts(hVals, edges, 'Normalization','probability');
end
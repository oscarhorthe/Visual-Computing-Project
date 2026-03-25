function hHist = getHueHist(img, bbox, numBins)

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

patch = img(y:y2, x:x2, :);
hsv = rgb2hsv(patch);

H = hsv(:,:,1);
S = hsv(:,:,2);

mask = S > 0.2;
hVals = H(mask);

if isempty(hVals)
    hHist = zeros(1,numBins);
    return;
end

%hey

edges = linspace(0,1,numBins+1);
hHist = histcounts(hVals, edges, 'Normalization','probability');
end
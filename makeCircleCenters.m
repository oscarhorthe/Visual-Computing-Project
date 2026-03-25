function centers = makeCircleCenters(bb, n)
% Return n circle centers inside bounding box bb = [x y w h]

    x = bb(1);
    y = bb(2);
    w = bb(3);
    h = bb(4);

    if n <= 0
        n = 1;
    end

    centers = zeros(n, 2);
    cy = y + h/2;

    for i = 1:n
        cx = x + i * w / (n + 1);
        centers(i,:) = [cx, cy];
    end
end
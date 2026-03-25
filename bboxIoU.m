function iou = bboxIoU(b1, b2)
    x1 = max(b1(1), b2(1));
    y1 = max(b1(2), b2(2));
    x2 = min(b1(1)+b1(3), b2(1)+b2(3));
    y2 = min(b1(2)+b1(4), b2(2)+b2(4));

    interW = max(0, x2 - x1);
    interH = max(0, y2 - y1);
    interArea = interW * interH;

    area1 = b1(3) * b1(4);
    area2 = b2(3) * b2(4);
    unionArea = area1 + area2 - interArea;

    if unionArea <= 0
        iou = 0;
    else
        iou = interArea / unionArea;
    end
end
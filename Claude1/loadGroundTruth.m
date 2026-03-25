function gt = loadGroundTruth(gtFile)
% LOADGROUNDTRUTH  Read MOTChallenge-format ground truth file.
%   gt = loadGroundTruth(gtFile)
%   Returns Nx10 matrix: [frame, id, bb_left, bb_top, bb_w, bb_h, conf, x, y, z]
%   Only rows with conf == 1 are kept (valid detections).

    gt = readmatrix(gtFile);
    
    % Keep only valid entries (confidence == 1)
    gt = gt(gt(:,7) == 1, :);
    
    fprintf('  Valid GT entries: %d\n', size(gt, 1));
end

function pattern = detectImagePattern(imgDir, imgExt)
% DETECTIMAGEPATTERN  Auto-detect image file naming convention.
%   Tries common PETS naming patterns and returns a format string
%   usable with sprintf, e.g. 'frame_%04d.jpg'

    candidates = {
        ['frame_%04d' imgExt],       % frame_0001.jpg
        ['img_%04d' imgExt],         % img_0001.jpg
        ['%06d' imgExt],             % 000001.jpg
        ['%04d' imgExt],             % 0001.jpg
        ['frame%04d' imgExt],        % frame0001.jpg
        ['image_%04d' imgExt],       % image_0001.jpg
        ['View001.%04d' imgExt],     % View001.0001.jpg
    };
    
    pattern = candidates{1};  % default
    
    if ~exist(imgDir, 'dir')
        warning('Image directory not found: %s. Using default pattern.', imgDir);
        return;
    end
    
    for i = 1:numel(candidates)
        testFile = fullfile(imgDir, sprintf(candidates{i}, 1));
        if exist(testFile, 'file')
            pattern = candidates{i};
            return;
        end
    end
    
    % If none matched, try to find any image and infer
    files = dir(fullfile(imgDir, ['*' imgExt]));
    if ~isempty(files)
        % Try to extract pattern from first filename
        name = files(1).name;
        fprintf('  Found image: %s (auto-detect pattern)\n', name);
        pattern = name;  % fallback
    else
        warning('No images found in %s. Using default pattern.', imgDir);
    end
end

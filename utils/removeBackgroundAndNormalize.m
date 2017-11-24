function [y] = removeBackgroundAndNormalize(x, blockSize)
    if nargin < 2
        blockSize = 20;
    end
    y = double(x);
    bkgPatch = y(1:blockSize, 1:blockSize);
    bkg = mean(bkgPatch(:));
    y = y - bkg;
    y(y < 0.0) = 0.0;
    % normalize for max. amplitude.
    y = y / max(y(:));
end

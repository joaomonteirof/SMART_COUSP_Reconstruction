function [y] = removeBackgroundAndNormalizeEnergy(x, blockSize)
    if nargin < 2
        blockSize = 50;
    end
    y = double(x);
    bkgPatch = y(1:blockSize, 1:blockSize);
    bkg = mean(bkgPatch(:));
    y = y - bkg;
    y(y < 0.0) = 0.0;
    % normalize for energy
    y = y / sum(y(:));
end

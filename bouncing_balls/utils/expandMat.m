function expandedOutput2 = expandMat(inputMat, targetSize)

expandBy = floor(targetSize./size(inputMat(:,:,1)));

% if rem(expandBy,1) ~= 0
%     warning('expandMat:notevennumber',...
%     'Requested size is not divisible by matrix size. Expanded matrix may be larger or smaller than target.');
%     expandBy = round(expandBy);
% end

expandedOutput = [];

for ii = 1:size(inputMat,3)
    currZ = inputMat(:,:,ii);
    expand2 = [];
    for iRow = 1:size(inputMat,1)
        expand1 = [];
        for iCol = 1:size(inputMat,2)
            expand1 = [expand1, ones(1,expandBy(2)) .* currZ(iRow,iCol)];
        end
        expand2 = [expand2; repmat(expand1,expandBy(1),1)];
    end
    expandedOutput = cat(3,expandedOutput,expand2);
end

expandedOutput2 = [];
pad1 = targetSize(1) - expandBy(1)*size(inputMat(:,:,1),1);
pad2 = targetSize(2) - expandBy(2)*size(inputMat(:,:,1),2);

for ii = 1:size(inputMat,3)
       expandedOutput2(:,:,ii) = padarray(expandedOutput(:,:,ii),[pad1,pad2],'post');    
end

end
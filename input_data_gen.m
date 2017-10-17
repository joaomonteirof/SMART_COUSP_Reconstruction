clear all
clc

%% select data folder
if ~exist('dataFolder')
    dataFolder = getenv('HOME');
end
dataFolder = uigetdir(dataFolder);
disp(dataFolder);

%% load options
opts = loadCupOptionsMV(dataFolder);

%% Load bouncing balls data
load(opts.video_samples_file);

%% Set N: Frames dim1, Nx: Frames dim2, Nt: #frames, Np: ... 
N = size(Data{1},1);
Nx = size(Data{1},2);
Nt = size(Data{1},3);
Np = 20;

%% Loop over video samples to generate list of streak images and patterns
streakImages = {};
Patterns = {};

for i=1:numel(Data)
    target3D = Data{i};
    target3Dreverse = flip(target3D,2);
    target3DBig = zeros(N,4*Nx,Nt);
    target3DBig(:,1:Nx,:) = target3D;
    target3DBig(:,Nx+1:2*Nx,:) = target3D;
    target3DBig(:,2*Nx+1:3*Nx,:) = target3Dreverse;
    target3DBig(:,3*Nx+1:4*Nx,:) = target3Dreverse;

    %% random mask
    pixelSize = 3;
    maskSmall = rand(floor(N/pixelSize),floor(Nx/pixelSize));
    maskSmall = im2double(im2bw(maskSmall));
    maskON = expandMat(maskSmall,[N Nx]);
    maskOFF = 1-maskON;
    maskBig = zeros(N,4*Nx);
    maskBig(:,1:Nx) = maskON;
    maskBig(:,Nx+1:2*Nx) = maskOFF;
    maskBig(:,2*Nx+1:3*Nx) = maskON;
    maskBig(:,3*Nx+1:4*Nx) = maskOFF;
    maskFile = padarray(maskBig,[Nt,0],'post');
    maskFile = padarray(maskFile,[Np,0],'pre');
    imagesc(maskFile)

    %% forward process - data acquisition
    % spatial encoding
    encodedTarget = zeros(size(target3DBig));
    for nt = 1:Nt
        encodedTarget(:,:,nt) = maskBig.*target3DBig(:,:,nt);
    end

    % temporal shearing
    [Nye, Nxe, Nte] = size(encodedTarget);
    shearedTarget = zeros(Nye+Nte,Nxe,Nte);
    for nt = 1:Nte
        shearedTarget(:,:,nt) = padarray(encodedTarget(:,:,nt),[Nt 0],'post');
        shearedTarget(:,:,nt) = circshift(shearedTarget(:,:,nt),[nt-1 0]);
    end

    % spatiotemporal integration
    y = sum(shearedTarget,3);
    y = padarray(y,[Np 0],'pre');
    y = norm1(y);

    %% add outputs to list
    streakImages{i} = y;
    patterns{i} = maskFile;

end

%% Loop over video samples to generate list of streak images and patterns

y = {};

for k=1:numel(streakImages)
   
    %% Read images based on options
    % pattern image
    imgs = struct();
    patternRaw = Patterns{k};
    yStreak = streakImages{k};
    yStreak = removeBackgroundAndNormalizeEnergy(yStreak);
    [height, width] = size(patternRaw);
    % cropping the pattern image and the streak camera measurement image
    % into equal halves.
    % ======================================================================
    % do pattern shifting here
    patternRaw = imwarp(patternRaw, affine2d([1, 0, 0; 0, 1, 0; ...
        opts.patternShift(2), opts.patternShift(1), 1]), ...
        'OutputView', imref2d(size(patternRaw)));
    if opts.singleStreak
        y1 = yStreak;
        imgs.C1 = removeBackgroundAndNormalize(patternRaw);
    else
        hWidth = floor(width/4); % four views
        y1 = yStreak(:, 1:hWidth);
        imgs.C1 = removeBackgroundAndNormalize(patternRaw(:, 1:hWidth));
    end
    y1 = imwarp(y1, affine2d([1, 0, 0; 0, 1, 0; opts.streakShift1(2),...
        opts.streakShift1(1), 1]), 'OutputView', imref2d(size(y1)));
    imgs.y1 = y1;
    if opts.useDualChannel
        % crop the right half
        imgs.C2 = removeBackgroundAndNormalize(patternRaw(:, hWidth+1:2*hWidth));
        y2 = yStreak(:, hWidth+1:2*hWidth);
        y2 = imwarp(y2, affine2d([1, 0, 0; 0, 1, 0; opts.streakShift2(2),...
            opts.streakShift2(1), 1]), 'OutputView', imref2d(size(y2)));
        imgs.y2 = y2;
    else
        imgs.C2 = [];
        imgs.y2 = [];
    end
    if opts.useThreeChannel
        imgs.C3 = removeBackgroundAndNormalize(patternRaw(:, 2*hWidth+1:3*hWidth));
        y3 = yStreak(:, 2*hWidth+1:3*hWidth);
        y3 = imwarp(y3, affine2d([1, 0, 0; 0, 1, 0; opts.streakShift3(2),...
            opts.streakShift3(1), 1]), 'OutputView', imref2d(size(y3)));
        imgs.y3 = y3;
    else
        imgs.C3 = [];
        imgs.y3 = [];
    end
    if opts.useFourChannel
        imgs.C4 = removeBackgroundAndNormalize(patternRaw(:, 3*hWidth+1:end));
        y4 = yStreak(:, 3*hWidth+1:end);
        y4 = imwarp(y4, affine2d([1, 0, 0; 0, 1, 0; opts.streakShift4(2),...
            opts.streakShift4(1), 1]), 'OutputView', imref2d(size(y4)));
        imgs.y4 = y4;
    else
        imgs.C4 = [];
        imgs.y4 = [];
    end


    %% Build y variable
    y{k} = [imgs.y1; imgs.y2; imgs.y3; imgs.y4;];
    
end

%% Save y
save(opts.output_file_name,y);
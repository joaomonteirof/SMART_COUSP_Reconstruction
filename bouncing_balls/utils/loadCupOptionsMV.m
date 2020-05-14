% Jinyang edit: add in parameters for the third and fourth channels

function opts = loadCupOptionsMV(dataFolder)

    %% default values start
    % some not implemented flags
    opts.useGPU = false;
    opts.useSingle = false;
    opts.useRebin = false;
    opts.baseChannel = 1;
    % system configuration flags
    opts.singleStreak = false;
    opts.useDualChannel = false;
    opts.useThreeChannel = false;
    opts.useFourChannel = false;
%     opts.useCCD = false;
    opts.useCCD = true;
    opts.useMask = false;
    % input images
    opts.maskFile = '';
    opts.patternFile = '';
    opts.yFile = '';
    opts.t1to2File = '';
    opts.t1to3File = '';
    opts.t1to4File = '';
    opts.ccdFile = '';
    % preprocess options
    opts.patternShift = {0, 0};
    opts.streakShift1 = {0, 0};
    opts.streakShift2 = {0, 0};
    opts.streakShift3 = {0, 0};
    opts.streakShift4 = {0, 0};
    opts.ccdShift = {0, 0};
    opts.maskShift = {0, 0};
    % reconstruction options
    opts.numFrames = 50;
    opts.k = 1.0;
    opts.tau = 0.1;
    opts.maxIter = 50;
    opts.tvDenoiseIter = 3;
    opts.tolA = 1.0e-7;
    opts.threshold = 0.0;
    opts.initX = 'pseudoinverse';

    %% read options from the file.
    OPTS_FILE = 'cupoptMV.yaml';
    optsPath = fullfile(dataFolder, OPTS_FILE);
    opts2 = ReadYaml(optsPath);
    fieldNames = fieldnames(opts2);
    for i = 1:size(fieldNames, 1)
        opts.(fieldNames{i}) = opts2.(fieldNames{i});
    end
    % get full path to all path variables
    opts.patternFile = fullfile(dataFolder, opts.patternFile);
    opts.yFile = fullfile(dataFolder, opts.yFile);
    %opts.ccdFile = fullfile(dataFolder, opts.ccdFile);
    if opts.useCCD
        opts.ccdFile = fullfile(dataFolder, opts.ccdFile);
    end
    if opts.singleStreak
        opts.useDualChannel = false;
        opts.useThreeChannel = false;
        opts.useFourChannel = false;
    end
    if opts.useDualChannel
        opts.t1to2File = fullfile(dataFolder, opts.t1to2File);
    end
    if opts.useThreeChannel
        opts.t1to3File = fullfile(dataFolder, opts.t1to3File);
    end
    if opts.useFourChannel
        opts.t1to4File = fullfile(dataFolder, opts.t1to4File);
    end
    if opts.useMask
        opts.maskFile = fullfile(dataFolder, opts.maskFile);
    end
    % now convert some list from cell array to matrix array.
    if isfield(opts, 'streakShift1')
        opts.streakShift1 = cell2mat(opts.streakShift1);
    end
    if isfield(opts, 'streakShift2')
        opts.streakShift2 = cell2mat(opts.streakShift2);
    end
    if isfield(opts, 'streakShift3')
        opts.streakShift3 = cell2mat(opts.streakShift3);
    end
    if isfield(opts, 'streakShift4')
        opts.streakShift4 = cell2mat(opts.streakShift4);
    end
    if isfield(opts, 'ccdShift')
        opts.ccdShift = cell2mat(opts.ccdShift);
    end
    if isfield(opts, 'maskShift')
        opts.maskShift = cell2mat(opts.maskShift);
    end
    if isfield(opts, 'patternShift')
        opts.patternShift = cell2mat(opts.patternShift);
    end

    %% load transformation from file
    if opts.useDualChannel
        opts.t1to2 = loadTransformFile(opts.t1to2File);
    else
        opts.t1to2 = [];
    end    
    if opts.useThreeChannel
        opts.t1to3 = loadTransformFile(opts.t1to3File);
    else
        opts.t1to3 = [];
    end
    
    if opts.useFourChannel
        opts.t1to4 = loadTransformFile(opts.t1to4File);
    else
        opts.t1to4 = [];
    end
end

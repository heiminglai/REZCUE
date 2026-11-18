%% Single-stack multi-class 2D U-Net++ segmentation (train on one annotated stack, infer on one raw stack)
% Requires: Deep Learning Toolbox, Image Processing Toolbox (for tiffreadVolume), and Computer Vision Toolbox.
clear; clc; close all;
parallel.gpu.enableCUDAForwardCompatibility(true);
reset(gpuDevice);
rng(42);
% Unique run stamp (local time)
runStamp = string(datetime('now','TimeZone','local','Format','yyyyMMdd_HHmmss'));
remarks = "L2reg5e-3";
runID    = "run_" + runStamp + "_" + remarks;

%% ====================== User inputs ======================
% 3 essential paths (edit these)
masterfolder = "D:\Siu Ming\INSIHGTp analysis\Trop2 NSCLC Rezcue confocal\TROP2 only\compartment_seg\";
rawStackPath      = strcat(masterfolder, "merged_8bit_isotropic_with_cellmask.tif");          % 3D TIFF (interleaved channels), for inference
annImgStackPath   = strcat(masterfolder, "training_stack.tif");  % 3D TIFF (interleaved channels), for training (images)
annMaskStackPath  = strcat(masterfolder, "training_mask.tif"); % 3D TIFF (same Z as ann images), for training (masks)

% Interleaved channel count
numChannels       = 4;   % set to your number of channels in the stacks

% Output root
outRoot           = strcat(masterfolder,"trainingoutput\"); %"D:\Siu Ming\REZCUE analysis\20250804 05109 sinonasal carcinoma WGA27-488 SyOG CD163-647 pan-CK-490\segmentation_unetpp\training\output";

% Classes and label encoding in mask
% Example (binary): background=0, tumour=1; masks that are {0,255} will be snapped to nearest of [0,1]
classes           = ["background","stroma"];
labelIDs          = [0, 255];

% Training hyperparameters
valFraction         = 0.2;           % validation split from annotated slices
inputDownsampleTo   = [1024 1024];   % [] -> auto to nearest multiple of 32; ensure divisible by 2^EncoderDepth
encoderDepth        = 7;             % Depth N => N pooling steps; input dims must be divisible by 2^N
numFirstEncoderFilt = 32;            % Be cautious with large depth+base filters (memory)
maxEpochs           = 1000;
miniBatchSize       = 2;
learnRate           = 1e-3;          % default 1e-3
augmentRotationDeg  = 30;            % rotation augmentation
doRandFlips         = true;
gpuEnv              = 'gpu';         % 'gpu' or 'auto'/'cpu' if needed
showOverlay         = true;          % overlay viewer after inference
overlayClassName    = "epidermis";      % which class to show in overlay
filtsz              = 3;             % modefilt post-processing filter size

%% ====================== Setup and GPU ====================
outDir     = fullfile(outRoot, "EncoderDepth" + string(encoderDepth), runID);
predRoot   = fullfile(outDir, "predictions"); if ~isfolder(predRoot), mkdir(predRoot); end
modelsRoot = fullfile(outDir, "models");      if ~isfolder(modelsRoot), mkdir(modelsRoot); end

gpuName = "CPU/Auto";
try
    g = gpuDevice;
    gpuName = string(g.Name);
    disp("Using GPU: " + g.Name);
catch ME
    warning("GPU not available or misconfigured: %s. Falling back to CPU.", ME.message);
    gpuEnv = 'auto';
end

%% ====================== Read input stacks =================
% Read raw stack (to be inferred)
disp('reading image');
tic
raw4D = readInterleaved4D(rawStackPath, numChannels); % [H W Z C]
toc
[Hr, Wr, Zr, Cr] = size(raw4D);
fprintf("Raw stack: %dx%dx%d, channels=%d\n", Hr, Wr, Zr, Cr);

% Read annotated stacks (for training)
annImgs4D  = readInterleaved4D(annImgStackPath, numChannels); % [H W Z C]
annMasks3D = tiffreadVolume(annMaskStackPath);        % [H W Z]
if ~isinteger(annMasks3D) 
    annMasks3D = uint16(annMasks3D); 
end

[Ha, Wa, Za, Ca] = size(annImgs4D);
[Hm, Wm, Zm]     = size(annMasks3D);

% Align Z between annotated images and masks
Z = min(Za, Zm);
if Z < Za, annImgs4D  = annImgs4D(:,:,1:Z,:); end
if Z < Zm, annMasks3D = annMasks3D(:,:,1:Z);  end

if Ha~=Hm || Wa~=Wm
    error("Annotated images and masks must have the same HxW. Got images: %dx%d, masks: %dx%d.", Ha, Wa, Hm, Wm);
end

expectedChannels = Ca;
fprintf("Annotated stack: %dx%dx%d, channels=%d\n", Ha, Wa, Z, Ca);

if Cr ~= Ca
    warning("Raw channels (%d) != annotated channels (%d). Inference will pad/trim to %d.", Cr, Ca, expectedChannels);
end
%% ====================== Decide input size =================
origHW = [Ha Wa];
if isempty(inputDownsampleTo)
    targetHW = ceil(origHW/64)*64; % ensure multiple of 2^encoderDepth when depth=6 -> 64
else
    targetHW = inputDownsampleTo(:).';
end
% Safety: enforce divisibility by 2^encoderDepth
divisor = 2^encoderDepth;
targetHW = ceil(targetHW/divisor)*divisor;

inputSize  = [targetHW expectedChannels];
numClasses = numel(classes);
fprintf("Network input size: [%d %d %d]\n", inputSize(1), inputSize(2), inputSize(3));

%% ====================== Materialize training set =================
trainTmpDir = fullfile(outDir, "train_cache_" + runStamp);
imDir = fullfile(trainTmpDir, "images");
msDir = fullfile(trainTmpDir, "masks");
makeCleanDir(trainTmpDir); makeCleanDir(imDir); makeCleanDir(msDir);

for z = 1:Z
    I = squeeze(annImgs4D(:,:,z,:));         % HxWxC
    I = ensureChannels(I, expectedChannels); % single [0,1], HxWxC
    
    % Resize image and mask to targetHW (as you already do)
    if ~isequal([size(I,1) size(I,2)], targetHW)
        I = imresize(I, targetHW, 'bilinear');
    end

    M = annMasks3D(:,:,z);
    M = snapMaskToLabelIDs(M, labelIDs);
    if ~isequal([size(M,1) size(M,2)], targetHW)
        M = imresize(M, targetHW, 'nearest');
    end
    
    % Write training image slice as multi-page TIFF (one page per channel)
    imgFile = fullfile(imDir, sprintf('idx_%05d.tif', z));
    writeMultipageChannels(I, imgFile);   % NEW helper (below)
    
    % Write mask (unchanged)
    if max(labelIDs) <= 255
        imwrite(uint8(M),  fullfile(msDir, sprintf('idx_%05d.tif', z)));
    else
        imwrite(uint16(M), fullfile(msDir, sprintf('idx_%05d.tif', z)), 'Compression','none');
    end
end

%% ====================== Datastores and split ======================
imds = imageDatastore(imDir, 'FileExtensions',{'.tif','.tiff','.png'}, ...
    'ReadFcn', @(f) readSliceMultiPage(f, expectedChannels));
pxds = pixelLabelDatastore(msDir, classes, labelIDs);

tmpI = readimage(imds,1);
tmpL = readimage(pxds,1);
fprintf('Cache check: image %dx%dx%d | labels in {%s}\n', size(tmpI,1), size(tmpI,2), size(tmpI,3), strjoin(string(unique(uint16(tmpL(:)))'), ', '));

N = numel(imds.Files);
perm = randperm(N);
Nv = max(1, round(valFraction*N));
valIdx = perm(1:Nv);
trainIdx = perm(Nv+1:end);
if isempty(trainIdx)
    warning("Not enough slices for training after split. Using all for training.");
    trainIdx = 1:N; valIdx = [];
end

imdsTrain = subset(imds, trainIdx);
pxdsTrain = subset(pxds,  trainIdx);
if ~isempty(valIdx)
    imdsVal = subset(imds, valIdx);
    pxdsVal = subset(pxds,  valIdx);
else
    imdsVal = []; pxdsVal = [];
end

tbl = countEachLabel(pxdsTrain);
disp(tbl);
% Expect non-zero PixelCount for multiple classes.
% Also probe a raw slice from msDir:
ex = imread(pxdsTrain.Files{1});
disp(sort(unique(ex(:)))); % should list your labelIDs

pximdsTrain = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'ColorPreprocessing','none');
if ~isempty(imdsVal)
    pximdsVal = pixelLabelImageDatastore(imdsVal, pxdsVal, 'ColorPreprocessing','none');
else
    pximdsVal = [];
end

% Augmentation
pximdsTrain = transform(pximdsTrain, @(data) preprocessAndAugment(data, targetHW, augmentRotationDeg, doRandFlips, classes, labelIDs));
if ~isempty(pximdsVal)
    pximdsVal = transform(pximdsVal, @(data) preprocessAndAugment(data, targetHW, 0, false, classes, labelIDs));
    valDataForOpts = pximdsVal;
else
    valDataForOpts = [];
end

%% ====================== Define U-Net++ and training options ==========
% Build a U-Net++ graph (nested dense skip pathways, no deep supervision)
lgraph = unetppLayers(inputSize, numClasses, ...
    'EncoderDepth', encoderDepth, ...
    'NumFirstEncoderFilters', numFirstEncoderFilt, ...
    'ConvLayersPerBlock', 2, ...
    'FilterSize', 3, ...
    'UseBatchNormalization', true);

% Weighted cross-entropy from label frequencies (robust to missing classes)
pxLayerIdx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.PixelClassificationLayer'), lgraph.Layers), 1, 'last');
if ~isempty(pxLayerIdx)
    tbl = countEachLabel(pxdsTrain);
    names = string(tbl.Name);
    counts = zeros(numel(classes),1);
    for i = 1:numel(classes)
        j = find(names == classes(i), 1);
        if ~isempty(j), counts(i) = double(tbl.PixelCount(j)); end
    end
    if sum(counts)==0
        cw = ones(numel(classes),1); % fallback
    else
        freq = counts / sum(counts);
        base = median(freq(freq>0));
        if isempty(base), base = 1/numel(classes); end
        cw = base ./ max(freq, eps);
    end
    ceL = pixelClassificationLayer('Name','labels', ...
        'Classes', cellstr(classes), 'ClassWeights', cw);
    lgraph = replaceLayer(lgraph, 'labels', ceL);
end

opts = trainingOptions('adam', ...
    'InitialLearnRate', learnRate, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 20, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', gpuEnv, ...
    'L2Regularization', 5e-3);

if ~isempty(valDataForOpts)
    opts.ValidationData = valDataForOpts;
    opts.ValidationFrequency = max(1, floor(numel(trainIdx)/miniBatchSize));
    opts.ValidationPatience = 10;
end

% Paths for training progress snapshots
progressPngFile = fullfile(modelsRoot, "training_progress_" + runStamp + ".png");
progressFigFile = fullfile(modelsRoot, "training_progress_" + runStamp + ".fig");

%% ====================== Train =============================
fprintf("Training U-Net++ on %d slices (val %d)...\n", numel(trainIdx), numel(valIdx));
[net, trainInfo] = trainNetwork(pximdsTrain, lgraph, opts);

% Save the training progress window (or a generated plot) as PNG/FIG
saveTrainingProgressFigure(progressPngFile, progressFigFile, trainInfo, runStamp);

% Also save the training info struct for reproducibility
try
    save(fullfile(modelsRoot, "training_info_" + runStamp + ".mat"), 'trainInfo');
catch
end

modelFile = fullfile(modelsRoot, "unetpp_single_stack_" + runStamp + ".mat");
save(modelFile, 'net', 'inputSize', 'targetHW', 'classes', 'labelIDs');
fprintf("Saved model: %s\n", modelFile);

%% ====================== Inference on full raw stack ======
fprintf("Inferring full raw stack...\n");
predMask = zeros(Hr, Wr, Zr, 'uint16');
predScore = zeros(Hr, Wr, Zr);

tStart = tic;
for z = 1:Zr
    I = squeeze(raw4D(:,:,z,:));             % HxWxC
    I = ensureChannels(I, expectedChannels);  % single [0,1], HxWxC

    needsResize = ~isequal([size(I,1) size(I,2)], targetHW);
    if needsResize
        In = imresize(I, targetHW, 'bilinear');
    else
        In = I;
    end

    [Cmap,score] = semanticseg(In, net, 'ExecutionEnvironment', gpuEnv);
    mk = categoricalToLabelID(Cmap, classes, labelIDs); % uint16

    if needsResize
        mk = imresize(mk, [Hr Wr], 'nearest');
    end
    predMask(:,:,z) = mk;

    if needsResize
        score = imresize(score, [Hr Wr]);
    end
    predScore(:,:,z) = score;

    if mod(z, max(1,floor(Zr/10)))==0 || z==Zr
        fprintf('  z = %d / %d (%.1fs)\n', z, Zr, toc(tStart));
    end
end

% Save prediction
predMask = uint8(predMask);
predFile = fullfile(predRoot, "prediction_mask_" + runStamp + ".tif");
tiffwriteVolume(predMask, predFile);

predMaskfilt = modefilt(predMask,[filtsz filtsz filtsz]);
predFilefilt = fullfile(predRoot, "prediction_mask_" + runStamp + "_modefilt" + num2str(filtsz) + ".tif");
tiffwriteVolume(predMaskfilt, predFilefilt);

predScoreFile = fullfile(predRoot, "prediction_score_" + runStamp + ".tif");
tiffwriteVolume(single(predScore), predScoreFile);

fprintf("Saved prediction: %s\n", predFile);

%% Prepare metadata for logging
paramsFile = fullfile(outDir, "params_" + runStamp + ".txt");

% Count labels on training set (for log)
tbl = countEachLabel(pxdsTrain);
labelNamesForLog  = cellstr(tbl.Name);
labelCountsForLog = double(tbl.PixelCount);

% cw is defined above when replacing pixelClassificationLayer; ensure it exists
if ~exist('cw','var'); cw = []; end

% Compose metadata struct
meta = struct();
meta.Timestamp            = char(runStamp);
meta.MATLABVersion        = version;
meta.RawStackPath         = char(rawStackPath);
meta.AnnotatedImgPath     = char(annImgStackPath);
meta.AnnotatedMaskPath    = char(annMaskStackPath);
meta.OutputDir            = char(outDir);
meta.ModelFile            = char(modelFile);
meta.PredictionFile       = char(predFile);
meta.TrainingProgressPNG  = char(progressPngFile);
meta.NumChannels          = numChannels;
meta.Classes              = cellstr(classes);
meta.LabelIDs             = labelIDs;
meta.InputHW              = targetHW;
meta.EncoderDepth         = encoderDepth;
meta.NumFirstEncoderFilt  = numFirstEncoderFilt;
meta.MaxEpochs            = maxEpochs;
meta.MiniBatchSize        = miniBatchSize;
meta.LearnRate            = learnRate;
meta.ValFraction          = valFraction;
meta.AugmentRotationDeg   = augmentRotationDeg;
meta.DoRandFlips          = doRandFlips;
meta.ExecutionEnvironment = gpuEnv;
meta.GPUName              = char(gpuName);
meta.TrainSlices          = numel(trainIdx);
meta.ValSlices            = numel(valIdx);
meta.ClassWeights         = cw;
meta.LabelNamesCounts     = [labelNamesForLog(:) num2cell(labelCountsForLog(:))];
meta.RawDims              = [Hr Wr Zr Cr];
meta.AnnDims              = [Ha Wa Z Ca];

% Write the TXT
writeRunMetadata(paramsFile, meta);
fprintf("Saved params: %s\n", paramsFile);

%% ====================== Overlay viewer (optional) =========
if showOverlay
    baseVol = zeros(Hr,Wr,Zr, 'single');
    for z=1:Zr
        I = ensureChannels(squeeze(raw4D(:,:,z,:)), expectedChannels);
        baseVol(:,:,z) = I(:,:,1);
    end
    ovLogical = mkToLogical(predMask, labelIDs, overlayClassName, classes);
    overlaySliceViewer(baseVol, ovLogical, 0.35, "Prediction Overlay");
end

% Clean cache
try, rmdir(trainTmpDir, 's'); catch, end

disp("Done.");

%% ===================== Helper functions ============================
function V4 = readInterleaved4D(path, C)
% Read a 3D TIFF with interleaved channels along the third dimension and reshape to [H W Z C].
    V = tiffreadVolume(path); % H x W x (Z*C)
    if ndims(V) ~= 3
        error("Expected a 3D volume from tiffreadVolume for %s.", path);
    end
    nThird = size(V,3);
    if mod(nThird, C) ~= 0
        error("Third dimension (%d) is not divisible by numChannels (%d) for %s.", nThird, C, path);
    end
    V4 = permute(reshape(V, size(V,1), size(V,2), C, []), [1 2 4 3]); % [H W Z C]
end

function makeCleanDir(d)
    if isfolder(d), try, rmdir(d,'s'); catch, end, end
    mkdir(d);
end

function Iout = ensureChannels(I, C)
% Convert HxWxC image to single [0,1] per-channel with robust percentile normalization.
    I = cast(I, 'single');
    if ndims(I)==2
        I = repmat(I, 1,1,C);
    elseif size(I,3) < C
        I = cat(3, I, zeros(size(I,1), size(I,2), C - size(I,3), 'single'));
    elseif size(I,3) > C
        I = I(:,:,1:C);
    end
    Iout = zeros(size(I), 'single');
    for c=1:C
        X = I(:,:,c);
        p = prctile(X(:), [1 99]);
        lo = p(1); hi = p(2);
        if ~(isfinite(lo)&&isfinite(hi)) || hi<=lo
            lo = min(X(:)); hi = max(X(:));
        end
        X = (X - lo) / max(eps, (hi - lo));
        Iout(:,:,c) = min(1, max(0, X));
    end
end

function out = categoricalToLabelID(Cmap, classes, labelIDs)
% Convert categorical map to uint16 labelID image.
    out = zeros(size(Cmap), 'uint16');
    for i=1:numel(classes)
        out(Cmap == classes(i)) = uint16(labelIDs(i));
    end
end

function Lcat = idToCategorical(Lid, classes, labelIDs)
% Convert uint8/uint16 labelID image to categorical using classes/labelIDs.
    Lcat = categorical(repmat(classes(1), size(Lid)), classes);
    for i=1:numel(classes)
        Lcat(Lid == uint16(labelIDs(i))) = classes(i);
    end
end

function L = mkToLogical(M, labelIDs, className, classes)
% Convert labelID volume to logical mask for a given class name.
    idx = find(classes == className, 1);
    if isempty(idx)
        warning("Class %s not found; defaulting to last class.", className);
        idx = numel(classes);
    end
    targetID = uint16(labelIDs(idx));
    L = (M == targetID);
end

function overlaySliceViewer(volBase, volMask, alpha, titlePrefix)
% Simple overlay viewer with a slider.
    if nargin<3 || isempty(alpha), alpha = 0.35; end
    if nargin<4, titlePrefix = ""; end
    [H,W,Z] = size(volBase);
    assert(isequal(size(volMask), [H W Z]), 'Overlay dims mismatch.');
    f = figure('Name','Overlay viewer','NumberTitle','off','Color','w');
    ax = axes('Parent',f);
    z = max(1, round(Z/2));
    showSlice(ax, volBase(:,:,z), volMask(:,:,z), alpha, z, Z, titlePrefix);
    sld = uicontrol('Style','slider','Parent',f, ...
        'Units','normalized','Position',[0.2 0.02 0.6 0.04], ...
        'Min',1,'Max',Z,'Value',z,'SliderStep',[1/max(Z-1,1) min(10/max(Z-1,1),1)]);
    addlistener(sld, 'Value', 'PostSet', @(~,~) onSlide());
    uicontrol('Style','text','Parent',f,'Units','normalized', ...
        'Position',[0.05 0.02 0.1 0.04],'String','Slice', 'BackgroundColor','w');

    function onSlide()
        zc = round(get(sld,'Value'));
        showSlice(ax, volBase(:,:,zc), volMask(:,:,zc), alpha, zc, Z, titlePrefix);
    end
end

function showSlice(ax, I, M, alpha, z, Z, titlePrefix)
    O = labeloverlay(I, M, 'Transparency', alpha, 'Colormap',[0 1 0]);
    imshow(O, 'Parent', ax, 'InitialMagnification','fit');
    if strlength(titlePrefix)>0
        title(ax, sprintf('%s | Slice %d / %d', titlePrefix, z, Z));
    else
        title(ax, sprintf('Slice %d / %d', z, Z));
    end
end

function dataOut = preprocessAndAugment(dataIn, targetHW, maxRotateDeg, doFlips, classes, labelIDs)
% transform-compatible function for PixelLabelImageDatastore:
% - Keeps paired image/label aligned
% - Applies random rotation and flips
% - Ensures output label is 2-D categorical with expected categories
    assert(istable(dataIn), 'Expected table input from pixelLabelImageDatastore.');
    numObs = height(dataIn);
    Iout = cell(numObs,1);
    Lout = cell(numObs,1);

    for i=1:numObs
        I = dataIn.inputImage{i};
        L = dataIn.pixelLabelImage{i};  % categorical or numeric

        % If label is numeric, map IDs -> categories
        if ~iscategorical(L)
            L = idToCategorical(uint16(L), classes, labelIDs);
        end

        % Use label IDs for geometry ops
        Lid = categoricalToLabelID(L, classes, labelIDs); % uint16 HxW

        % Augmentation
        if maxRotateDeg > 0
            ang = (rand*2-1) * maxRotateDeg;
            I   = imrotate(I,   ang, 'bilinear', 'crop');
            Lid = imrotate(Lid, ang, 'nearest',  'crop');
        end
        if doFlips && rand < 0.5
            I = fliplr(I);  Lid = fliplr(Lid);
        end
        if doFlips && rand < 0.5
            I = flipud(I);  Lid = flipud(Lid);
        end

        % Ensure target size
        if ~isequal([size(I,1) size(I,2)], targetHW)
            I   = imresize(I,   targetHW, 'bilinear');
        end
        if ~isequal([size(Lid,1) size(Lid,2)], targetHW)
            Lid = imresize(Lid, targetHW, 'nearest');
        end

        % Back to categorical with your classes
        LcatOut = idToCategorical(Lid, classes, labelIDs);

        Iout{i} = I;             % HxWxC single
        Lout{i} = LcatOut;       % HxW categorical
    end

    dataOut = dataIn;
    dataOut.inputImage      = Iout;
    dataOut.pixelLabelImage = Lout;
end

function writeRunMetadata(filename, meta)
% Write a human-readable parameter log to filename.
    fid = fopen(filename, 'w');
    if fid < 0
        warning('Could not open %s for writing.', filename);
        return;
    end
    c = onCleanup(@() fclose(fid));

    fprintf(fid, "U-Net++ run metadata\n");
    fprintf(fid, "=====================\n\n");

    kv("Timestamp",            meta.Timestamp);
    kv("MATLABVersion",        meta.MATLABVersion);
    kv("OutputDir",            meta.OutputDir);
    kv("ModelFile",            meta.ModelFile);
    kv("PredictionFile",       meta.PredictionFile);
    kv("TrainingProgressPNG",  meta.TrainingProgressPNG);
    fprintf(fid, "\n");

    fprintf(fid, "Paths\n");
    fprintf(fid, "-----\n");
    kv("RawStackPath",         meta.RawStackPath);
    kv("AnnotatedImgPath",     meta.AnnotatedImgPath);
    kv("AnnotatedMaskPath",    meta.AnnotatedMaskPath);
    fprintf(fid, "\n");

    fprintf(fid, "Data\n");
    fprintf(fid, "----\n");
    kv("NumChannels",          meta.NumChannels);
    kv("RawDims [H W Z C]",    mat2str(meta.RawDims));
    kv("AnnDims [H W Z C]",    mat2str(meta.AnnDims));
    kv("InputHW [H W]",        mat2str(meta.InputHW));
    kv("Classes",              strjoin(meta.Classes, ', '));
    kv("LabelIDs",             mat2str(meta.LabelIDs));
    fprintf(fid, "\n");

    fprintf(fid, "Training\n");
    fprintf(fid, "--------\n");
    kv("TrainSlices",          meta.TrainSlices);
    kv("ValSlices",            meta.ValSlices);
    kv("ValFraction",          meta.ValFraction);
    kv("EncoderDepth",         meta.EncoderDepth);
    kv("NumFirstEncoderFilt",  meta.NumFirstEncoderFilt);
    kv("MaxEpochs",            meta.MaxEpochs);
    kv("MiniBatchSize",        meta.MiniBatchSize);
    kv("LearnRate",            meta.LearnRate);
    kv("AugmentRotationDeg",   meta.AugmentRotationDeg);
    kv("DoRandFlips",          meta.DoRandFlips);
    if ~isempty(meta.ClassWeights)
        kv("ClassWeights",     mat2str(meta.ClassWeights, 6));
    end
    fprintf(fid, "\n");

    fprintf(fid, "Execution\n");
    fprintf(fid, "---------\n");
    kv("ExecutionEnvironment", meta.ExecutionEnvironment);
    kv("GPUName",              meta.GPUName);
    fprintf(fid, "\n");

    if ~isempty(meta.LabelNamesCounts)
        fprintf(fid, "Label pixel counts (training)\n");
        fprintf(fid, "-----------------------------\n");
        for i = 1:size(meta.LabelNamesCounts,1)
            fprintf(fid, "  %s: %g\n", string(meta.LabelNamesCounts{i,1}), meta.LabelNamesCounts{i,2});
        end
        fprintf(fid, "\n");
    end

    % Nested helper
    function kv(k, v)
        if isnumeric(v), v = mat2str(v); end
        fprintf(fid, "%-24s: %s\n", k, string(v));
    end
end

function saveTrainingProgressFigure(pngPath, figPath, trainInfo, runStamp)
% Save the official Training Progress window if available; otherwise, generate a plot from trainInfo.
    try
        pause(0.5);
        drawnow;

        % Try to grab MATLAB's built-in "Training Progress" figure
        fig = [];
        try
            fig = findall(0,'Type','figure','Name','Training Progress');
            if isempty(fig)
                fig = findall(0,'Type','figure','Tag','NNET_CNN_TRAININGPLOT_FIGURE');
            end
        catch
            % ignore
        end

        if ~isempty(fig) && isvalid(fig(1))
            f = fig(1);
            try
                exportgraphics(f, pngPath, 'Resolution', 150);
            catch
                try saveas(f, pngPath); catch, end
            end
            try saveas(f, figPath); catch, end
            fprintf("Saved training progress PNG: %s\n", pngPath);
            return;
        end

        % Fallback: generate a simple progress figure from trainInfo
        h = figure('Color','w','Name','Training Progress (Generated)','Position',[100 100 900 600]);
        isTiled = false;
        try
            tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
            isTiled = true;
        catch
            % Older MATLAB without tiledlayout
        end

        % Loss
        if isTiled, nexttile; else, subplot(2,2,1); end
        hold on;
        g1 = plotFieldSimple(trainInfo,'TrainingLoss',[0 0.447 0.741]);
        g2 = plotFieldSimple(trainInfo,'ValidationLoss',[0.85 0.325 0.098]);
        xlabel('Iteration'); ylabel('Loss'); grid on; title('Loss');
        leg = {};
        if g1, leg{end+1}='Train'; end
        if g2, leg{end+1}='Val';   end
        if ~isempty(leg), legend(leg,'Location','best'); end
        hold off;

        % Accuracy (fall back to GlobalAccuracy if present)
        if isTiled, nexttile; else, subplot(2,2,2); end
        hold on;
        a1 = plotFieldSimple(trainInfo,'TrainingAccuracy',[0.466 0.674 0.188]);
        a2 = plotFieldSimple(trainInfo,'ValidationAccuracy',[0.494 0.184 0.556]);
        if ~(a1 || a2)
            a1 = plotFieldSimple(trainInfo,'TrainingGlobalAccuracy',[0.466 0.674 0.188]);
            a2 = plotFieldSimple(trainInfo,'ValidationGlobalAccuracy',[0.494 0.184 0.556]);
        end
        xlabel('Iteration'); ylabel('Accuracy'); grid on; title('Accuracy');
        leg = {};
        if a1, leg{end+1}='Train'; end
        if a2, leg{end+1}='Val';   end
        if ~isempty(leg), legend(leg,'Location','best'); end
        hold off;

        % Learning rate
        if isTiled, nexttile; else, subplot(2,2,3); end
        plotFieldSimple(trainInfo,'BaseLearnRate',[0.301 0.745 0.933]);
        xlabel('Iteration'); ylabel('Learning Rate'); grid on; title('Learning Rate');

        % Title
        try, sgtitle("Training progress " + string(runStamp)); catch, end

        % Save
        try
            exportgraphics(h, pngPath, 'Resolution', 150);
        catch
            try saveas(h, pngPath); catch, end
        end
        try saveas(h, figPath); catch, end
        if isvalid(h), close(h); end
        fprintf("Saved training progress PNG (generated): %s\n", pngPath);

    catch ME
        warning("Could not save training progress figure: %s", ME.message);
    end
end

function plotted = plotFieldSimple(info, fieldName, color)
% Plot info.(fieldName) vs Iteration if present; return true if plotted.
    plotted = false;
    if ~isstruct(info) || ~isfield(info, fieldName), return; end
    y = info.(fieldName);
    if isempty(y), return; end
    if isfield(info,'Iteration') && ~isempty(info.Iteration)
        x = info.Iteration;
    else
        x = 1:numel(y);
    end
    n = min(numel(x), numel(y));
    if n < 1, return; end
    x = x(1:n);
    y = y(1:n);
    plot(x, y, 'LineWidth', 1.5, 'Color', color);
    plotted = true;
end

function lgraph = unetppLayers(inputSize, numClasses, varargin)
% Build a U-Net++ (Nested U-Net) layer graph (no deep supervision).
% Name-Value:
%   'EncoderDepth' (default 4)
%   'NumFirstEncoderFilters' (default 32)
%   'ConvLayersPerBlock' (default 2)
%   'FilterSize' (default 3)
%   'UseBatchNormalization' (default true)

    p = inputParser;
    p.addParameter('EncoderDepth', 4, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
    p.addParameter('NumFirstEncoderFilters', 32, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
    p.addParameter('ConvLayersPerBlock', 2, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
    p.addParameter('FilterSize', 3, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
    p.addParameter('UseBatchNormalization', true, @(x)islogical(x)||isscalar(x));
    p.parse(varargin{:});
    D     = p.Results.EncoderDepth;
    F0    = p.Results.NumFirstEncoderFilters;
    K     = p.Results.ConvLayersPerBlock;
    ksz   = p.Results.FilterSize;
    useBN = p.Results.UseBatchNormalization;

    % Filters per level i
    fAt = @(i) F0 * 2^i;

    % Start graph with input
    lgraph = layerGraph();
    inp = imageInputLayer(inputSize, 'Name','input', 'Normalization','none');
    lgraph = addLayers(lgraph, inp);

    % x(i+1, j+1) stores names for node x_i^j as strings
    x = strings(D+1, D+1);

    % Encoder column j=0 for i=0..D (with pooling between levels)
    prevOut = "input";
    for i = 0:D
        [blk, firstName, lastName] = convBlockLayers(sprintf("enc_%d_0", i), fAt(i), ksz, K, useBN);
        lgraph = addLayers(lgraph, blk);
        lgraph = connectLayers(lgraph, char(prevOut), char(firstName));
        x(i+1,1) = string(lastName);  % x_i^0

        if i < D
            poolName = sprintf("pool_%d", i);
            pool = maxPooling2dLayer(2, 'Stride', 2, 'Padding', 'same', 'Name', poolName);
            lgraph = addLayers(lgraph, pool);
            lgraph = connectLayers(lgraph, char(lastName), poolName);
            prevOut = string(poolName);
        end
    end

    % Nested decoder: build by columns j = 1..D, rows i = 0..D-j
    % x_i^j depends on x_{i+1}^{j-1} (upsampled) and x_i^{0..j-1}
    for j = 1:D
        for i = 0:(D - j)
            % Upsample source: x_{i+1}^{j-1} -> x(i+2, j)
            upSrc = x(i+2, j);
            if upSrc == ""
                error('unetppLayers:MissingSource', ...
                      'Internal build error: source x(%d,%d) for up_%d_%d not built.', i+2, j, i, j);
            end
            upName = sprintf("up_%d_%d", i, j);
            up = transposedConv2dLayer(2, fAt(i), 'Stride', 2, 'Cropping', 'same', ...
                                       'WeightsInitializer','he','BiasInitializer','zeros', ...
                                       'Name', upName);
            lgraph = addLayers(lgraph, up);
            lgraph = connectLayers(lgraph, char(upSrc), upName);

            % Concatenate [up, x_i^0, x_i^1, ..., x_i^{j-1}]
            nIn = 1 + j;
            concatName = sprintf("concat_%d_%d", i, j);
            concat = depthConcatenationLayer(nIn, 'Name', concatName);
            lgraph = addLayers(lgraph, concat);

            % Connect up as first input
            lgraph = connectLayers(lgraph, upName, char(concatName + "/in1"));
            % Connect previous nodes on same row i
            for k = 0:(j-1)
                skipSrc = x(i+1, k+1);
                if skipSrc == ""
                    error('unetppLayers:MissingSkip', ...
                          'Internal build error: skip x(%d,%d) for concat_%d_%d/in%d not built.', ...
                           i+1, k+1, i, j, k+2);
                end
                lgraph = connectLayers(lgraph, char(skipSrc), char(concatName + "/in" + string(k+2)));
            end

            % Conv block after concat -> x_i^j
            [blk2, first2, last2] = convBlockLayers(sprintf("dec_%d_%d", i, j), fAt(i), ksz, K, useBN);
            lgraph = addLayers(lgraph, blk2);
            lgraph = connectLayers(lgraph, concatName, char(first2));

            x(i+1, j+1) = string(last2);
        end
    end

    % Head: 1x1 conv -> softmax -> pixelClassification
    if x(1, D+1) == ""
        error('unetppLayers:MissingFinal', 'Final node x(1,%d) was not built.', D+1);
    end
    head = [
        convolution2dLayer(1, numClasses, 'Padding','same', ...
            'WeightsInitializer','he','BiasInitializer','zeros', 'Name','final_conv')
        softmaxLayer('Name','softmax')
        pixelClassificationLayer('Name','labels') % will be replaced with class-weighted layer outside
    ];
    lgraph = addLayers(lgraph, head);
    lgraph = connectLayers(lgraph, char(x(1, D+1)), 'final_conv');
end

function [layers, firstName, lastName] = convBlockLayers(prefix, numFilters, filterSize, numConv, useBN)
% Conv block: [conv-(bn)-relu] repeated numConv times, 'same' padding.
    layers = [];
    for ii = 1:numConv
        conv = convolution2dLayer(filterSize, numFilters, 'Padding','same', ...
            'WeightsInitializer','he', 'BiasInitializer','zeros', ...
            'Name', sprintf('%s_conv%d', prefix, ii));
        if useBN
            bn   = batchNormalizationLayer('Name', sprintf('%s_bn%d', prefix, ii));
            relu = reluLayer('Name', sprintf('%s_relu%d', prefix, ii));
            layers = [layers; conv; bn; relu]; %#ok<AGROW>
        else
            relu = reluLayer('Name', sprintf('%s_relu%d', prefix, ii));
            layers = [layers; conv; relu]; %#ok<AGROW>
        end
    end
    firstName = layers(1).Name;
    lastName  = layers(end).Name;
end

function writeMultipageChannels(I, filename)
% Write HxWxC single [0,1] as multi-page TIFF, one page per channel.
    Iu16 = im2uint16(I); % keep precision
    if exist(filename,'file'), delete(filename); end
    for c = 1:size(Iu16,3)
        if c==1
            imwrite(Iu16(:,:,c), filename, 'Compression','none');
        else
            imwrite(Iu16(:,:,c), filename, 'WriteMode','append', 'Compression','none');
        end
    end
end

function I = readSliceMultiPage(filename, C)
% Robust reader for training images.
% - If file has multiple pages: read first C pages as channels.
% - Else: read single page (possibly 3-channel) and adapt to C.
% - Always returns single HxWxC in [0,1], with percentile normalization per channel.

    info = imfinfo(filename);
    if numel(info) > 1
        % Multi-page TIFF: each page is a channel
        n = min(C, numel(info));
        Ik = im2single(imread(filename, 1, 'Info', info));
        H = size(Ik,1); W = size(Ik,2);
        I = zeros(H, W, C, 'single');
        I(:,:,1) = Ik;
        for k = 2:n
            I(:,:,k) = im2single(imread(filename, k, 'Info', info));
        end
        if n < C
            I(:,:,n+1:C) = 0;
        end
    else
        % Single-page image that may be grayscale or RGB
        raw = im2single(imread(filename));
        if ndims(raw)==2
            % grayscale -> replicate/pad to C
            I = repmat(raw, 1, 1, C);
        else
            % RGB or more -> trim/pad to C
            K = size(raw,3);
            if K >= C
                I = raw(:,:,1:C);
            else
                I = cat(3, raw, zeros(size(raw,1), size(raw,2), C-K, 'single'));
            end
        end
    end

    % Match inference: robust per-channel percentile normalization
    I = ensureChannels(I, C);
end

function Mout = snapMaskToLabelIDs(Min, labelIDs)
% Map every pixel in Min to the nearest value in labelIDs (vector).
    Min = uint16(Min);
    labelIDs = uint16(labelIDs(:).');
    Mout = zeros(size(Min),'uint16');
    for k = 1:numel(labelIDs)
        if k==1
            Mout(:) = labelIDs(1);
            dBest = abs(int32(Min) - int32(labelIDs(1)));
        else
            d = abs(int32(Min) - int32(labelIDs(k)));
            sel = d < dBest;
            Mout(sel) = labelIDs(k);
            dBest(sel) = d(sel);
        end
    end
end


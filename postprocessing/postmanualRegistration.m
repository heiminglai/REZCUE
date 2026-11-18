%% script to align AE-prealigned CosMx and 3D vol images

clear all
clc

img3d_rigid_path = "D:\Siu Ming\REZCUE analysis\liver_age_vs_young\CosMx_3D2Danalysis\aged\aged_stitched_fused_rigidtform\";
%% read 3d image stack
if ~exist("img3d_rigid_path","dir")

    imgpath3d = "C:\Users\LHM_ANAYLIS_03\Desktop\aged_temp\fused_stitched_ch1_8bit\";
    files = dir(imgpath3d); files([files.bytes]<100,:) = [];
    imgtemp = cell(size(files,1),1);
    parfor ii = 1:size(files,1)
        imgtemp{ii,1} = imread(strcat(imgpath3d, files(ii).name));
    end
    img3dstack = zeros(size(imgtemp{1,1},1),size(imgtemp{1,1},2),size(files,1),'uint8');
    for ii = 1:size(files,1)
        img3dstack(:,:,ii) = imgtemp{ii,1};
    end

end

%% define points and rigid transformation
if ~exist("img3d_rigid_path","dir")

    static_pt = [1157	668	    179;
                1744	799	    202;
                2261	1152	230;
                1080	1589	194;
                1736	2065	226;
                2193	2110	245;
                1146	3505	220;
                1704	3601	245;
                2265	3904	273;
                1708	4944	257;
                2289	5201	288;
                993	    6167	251;
                1726	6254	279;
                2319	6615	315;
                1018	7079	266;
                2295	7076	317;
                ]; % based on slice8
    
    moving_pt = cat(2,static_pt(:,1:2), ones(size(static_pt,1),1)*245);
    
    static_pt(:,3) = static_pt(:,3) - start_slice + 1;
    moving_pt(:,3) = moving_pt(:,3) - start_slice + 1;
    
    tform = fitgeotform3d(static_pt,moving_pt,"rigid");
    tic
    moving_reg_rigid = imwarp(im2DnlreginCanvas,tform,"OutputView",imref3d(size(im2DnlreginCanvas)));
    toc
    
    followOutput = affineOutputView(size(img3dstack),tform,"BoundsStyle","FollowOutput");
    img3dstack_rigid = imwarp(img3dstack,tform,"OutputView",followOutput);
    
    mkdir("aged_stitched_fused_rigidtform_ch1\");
    savedir = "aged_stitched_fused_rigidtform_ch1\";
    for ii = 1:size(img3dstack_rigid,3)
        imname = strcat(savedir,"aged-sti-fus-rig",num2str(ii),".tif");
        imwrite(im2uint16(img3dstack_rigid(:,:,ii)),imname);
    end

end
% img3dstack_rigid will provide the global 3D space coordinates

%% optional: b-spline registration of plane in 3D

% moving_pt = cat(2,static_pt(:,1:2), ones(size(static_pt,1),1)*245);
% 
% static_pt(:,3) = static_pt(:,3) - start_slice + 1;
% moving_pt(:,3) = moving_pt(:,3) - start_slice + 1;
% 
% Options.Verbose = 1;
% Options.MaxRef = 8;
% [O_trans,Spacing,Xreg]=point_registration(size(im2DnlreginCanvas)/dwnscl,moving_pt/dwnscl,static_pt/dwnscl,Options);
% tic
% moving_reg=bspline_transform(O_trans,imresize3(im2DnlreginCanvas,1/dwnscl),Spacing,3);
% toc

%% non-linearly register mesh- and z-warped slice to a selected slice from 3D
img3d_rigid_path = "D:\Siu Ming\REZCUE analysis\liver_age_vs_young\CosMx_3D2Danalysis\aged\aged_stitched_fused_rigidtform\";
files = dir(img3d_rigid_path); files([files.bytes]<100,:) = [];
filenames = {files.name};
fileNums = cellfun(@(x) str2double(regexp(x, '\d+', 'match', 'once')), filenames);
[~, idx] = sort(fileNums);
sortedFilenames = filenames(idx)';

% read 3D image
imgtemp = cell(size(files,1),1);
parfor ii = 1:size(files,1)
    imgtemp{ii,1} = imread(strcat(img3d_rigid_path, sortedFilenames{ii,1}));
end
img3dstack_rigid = zeros(size(imgtemp{1,1},1),size(imgtemp{1,1},2),size(files,1),'uint8');
for ii = 1:size(files,1)
    img3dstack_rigid(:,:,ii) = imgtemp{ii,1};
end

%% non-linear registration tuning 
% we register the full intensity masked slice first for each z-slice
% then proceed displace them along the Z-direction
meshZwarp_dir = "D:\Siu Ming\REZCUE analysis\liver_age_vs_young\CosMx_3D2Danalysis\aged\meshZwarps\";
for sliceNum = [1 2 6]
    % read matte images
    % matte provides the z-displacement information
    disp(strcat("processing slice number ", num2str(sliceNum)));
    disp("reading images")
    slicedir = strcat(meshZwarp_dir,"slice",num2str(sliceNum),"_matte\");
    sliceImgFiles = dir(slicedir);
    sliceImgFiles([sliceImgFiles.bytes]<100,:) = [];

    matteSlice = zeros([size(img3dstack_rigid,[1 2]),size(sliceImgFiles,1)],'uint16');
    for ff = 1:size(sliceImgFiles,1)
        tempimg = imread(strcat(slicedir,sliceImgFiles(ff).name));
        matteSlice(:,:,ff) = tempimg(:,:,1);
    end
    matteSlice = rescale(single(matteSlice));

    % get slice indices corresponding to img3dstack_rigid
    sliceStart = str2double(sliceImgFiles(1).name(length(sliceImgFiles(1).name)-8:length(sliceImgFiles(1).name)-4))+1;
    sliceEnd = str2double(sliceImgFiles(end).name(length(sliceImgFiles(end).name)-8:length(sliceImgFiles(end).name)-4))+1;

    % read meshwarped slice image (flat image but meshwarped)
    sliceImagesList = dir(strcat(meshZwarp_dir,"slice",num2str(sliceNum),"_meshwarp_*.tif"));
    sliceImage = imread(strcat(meshZwarp_dir,sliceImagesList(1).name));
    sliceImage = sliceImage(:,:,1);

    fixedStack = img3dstack_rigid(:,:,sliceStart:sliceEnd);
    %sliceViewer(cat(4,fixedStack,matteSlice,fixedStack)); % viewer to verify if it's correct
    
    % % temp for optimization
    % fixedStack = fixedStack(:,:,10);
    % matteSlice = matteSlice(:,:,10);

    % final non-linear registration refinement
    disp("Demons-based image registration...");
    reg_results = cell(size(matteSlice,3),2); % D, moving_reg
    for ii = 1:size(matteSlice,3)
        moving = matteSlice(:,:,ii);
        moving = imdilate(logical(moving),strel('sphere',3)); % dilation helps with registration near the mask edges
        if max(moving,[],'all') > 0
            fixed = fixedStack(:,:,ii);
            fixed(~moving) = 0;

            moving = sliceImage .* uint8(logical(moving)); % register masked volume slice and masked 2D section (meshwarped)

            moving_filt = rescale(moving - imgaussfilt(moving,10),0,255);
            fixed_filt = rescale(fixed - imgaussfilt(fixed,10),0,255);
            %imshow(cat(3,uint8(moving),uint8(fixed),uint8(moving)))

            iterations = [10, 100, 100, 10, 1];
            [D,~] = imregdemons(moving_filt,fixed_filt,iterations,...
                "AccumulatedFieldSmoothing",2.6,"PyramidLevels",numel(iterations));
            reg_results{ii,1} = D;
            reg_results{ii,2} = imwarp(moving,D);
        end
    end

    % compile results
    disp("gathering results");
    moving_reg = zeros(size(matteSlice),'uint8');
    Dfield = zeros([size(matteSlice) 2],'single');
    for ii = 1:size(matteSlice,3)
        if ~isempty(reg_results{ii,2})
            Dfield(:,:,ii,:) = reg_results{ii,1};
            moving_reg(:,:,ii) = uint8(single(reg_results{ii,2}) .* matteSlice(:,:,ii)); % modulate intensity for blending as in After Effects
        end
    end

    % view results
    %results = cat(4,fixedStack,1.5*moving_reg,fixedStack);
    %sliceViewer(pagetranspose(results));

    % save image
    mkdir(strcat(meshZwarp_dir,"slice",num2str(sliceNum),"_meshwarped_demons"));
    savedir = strcat(meshZwarp_dir,"slice",num2str(sliceNum),"_meshwarped_demons\");
    tiffwriteVolume(Dfield,strcat(savedir, "slice", num2str(sliceNum),"_Dfield.tif"));
    for ii = 1:size(moving_reg,3)
        imwrite(moving_reg(:,:,ii),strcat(savedir,"slice",num2str(sliceNum),"meshZwarp_demons_",num2str(ii),".tif"));
    end
    
    % register gridImage for transcript coordinate transformation
    % read gridImage & CosMxSeg image
    disp("processing grid images and original CosMx segmentations");
    disp("reading grid and segmentation images");
    gridImgname = dir(strcat(meshZwarp_dir,"slice",num2str(sliceNum),"_gridImage_meshwarp_*.tif"));
    gridImg_meshwarp = squeeze(tiffreadVolume(strcat(meshZwarp_dir,gridImgname(1).name)));

    CosMxSegname = dir(strcat(meshZwarp_dir,"slice",num2str(sliceNum),"_CosMxSegCellMask_meshwarp_*.tif"));
    CosMxSeg_meshwarp = uint16(squeeze(tiffreadVolume(strcat(meshZwarp_dir,CosMxSegname(1).name))));
    CosMxSeg_meshwarp = CosMxSeg_meshwarp(:,:,1);

    % transformation - XY coordinates 
    disp("image transformation");
    gridImg_meshwarp_tform = zeros([size(matteSlice), 3],'single');
    gridMask_meshwarp_tform = zeros(size(matteSlice),'single');
    CosMxSeg_meshwarp_tform = zeros(size(matteSlice),'uint16');
    for ii = 1:size(gridImg_meshwarp_tform,3)
        % extract individual grid components
        gridImgY = gridImg_meshwarp(:,:,1);
        gridImgX = gridImg_meshwarp(:,:,2);
        gridMask = gridImg_meshwarp(:,:,3);

        % add Z-coordinate transform
        [~,~,gridImgZ] = meshgrid(1:size(matteSlice,2),1:size(matteSlice,1),1:size(matteSlice,3));
        % obtining the Z-displacement magnitude as weighted average of the matte
        gridImgZ = sum(gridImgZ .* matteSlice, 3) ./ sum(matteSlice,3);
        
        % apply mask - equivalent to applying the z-displacement
        matte = uint16(imdilate(logical(matteSlice(:,:,ii)),strel("sphere",3)));
        gridImgY = gridImgY .* matte;
        gridImgX = gridImgX .* matte;
        gridImgZ = gridImgZ .* single(matte); % equivalent to applying the z-displacement
        gridMask = gridMask .* matte;
        CosMxSeg = CosMxSeg_meshwarp .* matte;
        
        % read Dfield
        Dfield = tiffreadVolume(strcat(meshZwarp_dir,"slice",num2str(sliceNum),"_meshwarped_demons\slice",num2str(sliceNum),"_Dfield.tif"));
        Dfield = reshape(Dfield,size(Dfield,1),size(Dfield,2),[],2);

        % apply D-field
        gridImgY_tform = imwarp(gridImgY, squeeze(Dfield(:,:,ii,:)),"nearest");
        gridImgX_tform = imwarp(gridImgX, squeeze(Dfield(:,:,ii,:)),"nearest");
        gridImgZ_tform = imwarp(gridImgZ, squeeze(Dfield(:,:,ii,:)),"nearest");
        gridMask_tform = imwarp(gridMask, squeeze(Dfield(:,:,ii,:)),"nearest");
        CosMxSeg_tform = imwarp(CosMxSeg, squeeze(Dfield(:,:,ii,:)),"nearest");
        
        gridImg_meshwarp_tform(:,:,ii,1) = single(gridImgY_tform);
        gridImg_meshwarp_tform(:,:,ii,2) = single(gridImgX_tform);
        gridImg_meshwarp_tform(:,:,ii,3) = gridImgZ_tform;
        % the result is a pixel index map (based on XY) for constructing vector displacements for each pixel on CosMx images
        gridMask_meshwarp_tform(:,:,ii) = gridMask_tform;
        CosMxSeg_meshwarp_tform(:,:,ii) = CosMxSeg_tform;
    end
    
    disp("saving...");
    tiffwriteVolume(gridImg_meshwarp_tform, strcat("slice",num2str(sliceNum),"_gridXYZ_tformed.tif"));
    tiffwriteVolume(gridMask_meshwarp_tform, strcat("slice",num2str(sliceNum),"_gridMask_tformed.tif"));
    tiffwriteVolume(CosMxSeg_meshwarp_tform, strcat("slice",num2str(sliceNum),"_CosMxSegCellMask_tformed.tif"));
    imwrite(max(CosMxSeg_meshwarp_tform,[],3),strcat("slice",num2str(sliceNum),"_CosMxSegCellMask_tformed_flat.tif"));

end

    

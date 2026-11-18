clear all
clc
img = tiffreadVolume("global_Morphology2D.tif");
img3 = img(:,:,3);
tiffwriteVolume(img3,"global_Morphology2D_ch3.tif");
imgseg = tiffreadVolume("global_CosMxSeg.tif");
imgseg_r = imresize(imgseg,0.25,"nearest");
tiffwriteVolume(imgseg_r,"global_CosMxSeg_64xdwnsz.tif");
slices = cell(8,4);
% slices: || slice img | slice img 64xdwnsz | slice seg img | slice seg img64x dwnsz ||
% indices to crop the CosMx slices
% [rowStart, rowEnd, colStart, colEnd]
indices = [960, 60855, 652, 22411;
           9279, 70966, 27271, 54918;
           13246, 75061, 61696, 76927;
           4672, 66745, 92410, 109433;
           92462, 154789, 268, 16907;
           81072, 143655, 29063, 50181;
           93742, 151973, 52866, 76925;
           94638, 154789, 93690, 114425]; % from slice 1 to 8

% ensures image sizes are divisible by 8 for downsizing
new_indices = indices;
for ii = 1:size(new_indices,1)
    while mod((new_indices(ii,2) - new_indices(ii,1) + 1),8) ~= 0
        new_indices(ii,2) = new_indices(ii,2) + 1;
    end
    while mod((new_indices(ii,4) - new_indices(ii,3) + 1),8) ~= 0
        new_indices(ii,4) = new_indices(ii,4) + 1;
    end
end

save("aged_slicecrop_indices.mat","new_indices");

% get images
for ii = 1:8
    slices{ii,1} = img3(new_indices(ii,1):new_indices(ii,2),...
        new_indices(ii,3):new_indices(ii,4));
    slices{ii,3} = imgseg(new_indices(ii,1):new_indices(ii,2),...
        new_indices(ii,3):new_indices(ii,4));
end

% downsize images (8x for each X and Y)
for ii = 1:8
    slices{ii,2} = imresize(slices{ii,1},0.125);
    imwrite(slices{ii,2},strcat("global_slice",num2str(ii),"_64xdwnsz.tif"));

    imgseg_slice_r = imresize(slices{ii,3},0.125,'nearest');
    % reorder the cell IDs to likely enable saving in uint16, easier to
    % handle in After Effects
    [~,~, newLabels] = unique(imgseg_slice_r);
    imgseg_slice_r_reordered = reshape(newLabels, size(imgseg_slice_r)) - 1;
    numCells_slice = max(imgseg_slice_r_reordered,[],"all");
    if numCells_slice <= 65535
        slices{ii,4} = uint16(imgseg_slice_r_reordered);
        imwrite(slices{ii,4},strcat("global_CosMxSeg_slice",num2str(ii),"_64xdwnsz.tif"));
    else
        slices{ii,4} = imgseg_slice_r_reordered;
        disp(strcat("slice ", num2str(ii), " has ", num2str(numCells_slice), " Cells, saving as single"));
        tiffwriteVolume(slices{ii,4},strcat("global_CosMxSeg_slice",num2str(ii),"_64xdwnsz_single.tif"));
    end 
    
    % % also save the cell boundaries for post-warping segmentation: more accurate
    % slice_cellbound = 255*uint8(logical(slices{ii,4} - imerode(slices{ii,4},strel('disk',2,8))));
    % imwrite(slice_cellbound,strcat("global_CosMxSegCellBound_slice",num2str(ii),"_64xdwnsz.tif"));
end

% make and save grid images for subsequent transformation
% pixel values are global CosMx, full-sized image coordinates
for ii = 1:8
    sizeY = new_indices(ii,2) - new_indices(ii,1) + 1;
    sizeX = new_indices(ii,4) - new_indices(ii,3) + 1;
    [gridImageY, gridImageX] = meshgrid(1:8:sizeX,1:8:sizeY);
    gridImage = uint16(cat(3,gridImageY,gridImageX,0*gridImageY+65535)); % 3rd channel is the grid mask
    % Z-meshgrid will be added after After Effects meshwarping, in
    % postmanualRegistration.m
    imwrite(gridImage, strcat("gridImage",num2str(ii),"_64xdwnsize.tif"));
end


%%%%%%%%%% END %%%%%%%%%%%

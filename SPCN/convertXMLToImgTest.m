% converting gt.xml to binary mask, color mask, binary label using 
% code provided by MoNuSeg 2018
% he_to_binary_mask_final

trainPath = 'MoNuSegTestData/TissueImages/';

flist=dir(fullfile(trainPath,'*.tif'));
n = length(flist);

ext = '.tif';

outputBinaryPath = strcat('NewTestData/', 'BinaryMask/');
outputMarkerPath = strcat('NewTestData/', 'Marker/');
outputColorPath = strcat('NewTestData/', 'ColorMask/');
outputBinaryLabeledPath = strcat('NewTestData/', 'BinaryLabeled/');
outputVisPath = strcat('NewTestData/', 'Vis/');

if (0==isdir(outputBinaryPath))
    mkdir(outputBinaryPath);
end

if (0==isdir(outputMarkerPath))
    mkdir(outputMarkerPath);
end

if (0==isdir(outputColorPath))
    mkdir(outputColorPath);
end

if (0==isdir(outputBinaryLabeledPath))
    mkdir(outputBinaryLabeledPath);
end

if (0==isdir(outputVisPath))
    mkdir(outputVisPath);
end

for fr = 1 : n 

    if contains(flist(fr).name, '._')
        continue;
    end
    
    nName = strip(flist(fr).name,'right','f');
    nName = strip(nName,'right','i');
    nName = strip(nName,'right','t'); 
    fileName = strip(nName,'right','.');
    
    fullPath = strcat(trainPath, fileName);
    
    [binaryMask, shapeMarker, colorMask] = he_to_binary_mask_final(fullPath, ext);
    
    imwrite(uint16(binaryMask), fullfile(outputBinaryLabeledPath, [fileName, '.png']));
    
    binaryMask = imbinarize(binaryMask, 0.0);
    binaryMask = binaryMask * 255;
    
    shapeMarker = imbinarize(shapeMarker, 0.0);
    shapeMarker = shapeMarker * 255;
    
    binaryMask = uint8(binaryMask);
    shapeMarker = uint8(shapeMarker);
    
     rgb = zeros(size(binaryMask,1), size(binaryMask,2), 3);
    
    rgb(:,:,1) = binaryMask;
    rgb(:,:,2) = shapeMarker;
    
    bImg = cat(3, binaryMask, binaryMask, binaryMask);
    sImg = cat(3, shapeMarker, shapeMarker, shapeMarker);
    
    combined = [bImg sImg;
        rgb im2uint8(colorMask)];
    
    imwrite(binaryMask, fullfile(outputBinaryPath, [fileName, '.png']));
    imwrite(shapeMarker, fullfile(outputMarkerPath, [fileName, '.png']));
    imwrite(colorMask, fullfile(outputColorPath, [fileName, '.png']));
    imwrite(combined, fullfile(outputVisPath, [fileName, '.png']));
end
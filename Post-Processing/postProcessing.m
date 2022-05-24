clear all; close all;

folderName = "USE-Net-Final/";

maskPath = strcat(folderName,'Mask/');
markerPath = strcat(folderName,'Marker/');

labelPath = strcat(folderName,'Label/');
labelColorPath = strcat(folderName,'Label_Vis/');

if (0==isdir(labelPath))
    mkdir(labelPath);
end

if (0==isdir(labelColorPath))
    mkdir(labelColorPath);
end

flist=dir(fullfile(maskPath,'*.png'));
n = length(flist);

for fr = 1 : n
    
    if contains(flist(fr).name, '._')
        continue;
    end
    
    flist(fr).name
    
    mask = imread(fullfile(maskPath, flist(fr).name));
    marker = imread(fullfile(markerPath, flist(fr).name));
    
    % threshold one to binary
    imgMask = im2bw(mask, 0.1); % best 0.1
    imgMask2 = bwareaopen(imgMask, 50); % best 50
    
    % threshold one to binary
    imgMarker = im2bw(marker, 0.1);  % best 0.1
    imgMarker2 = bwareaopen(imgMarker, 20); % best 20
    
    watershedResult = watershed(~imgMarker2);
    % figure; imshow(watershedResult,[]);
    
    boundriesWatershed = watershedResult == 0;
    % figure; imshow(boundriesWatershed);
      
    imgMask2(boundriesWatershed == 1) = 0;
    % figure; imshow(imgMask2);
    
    imgMask2 = bwareaopen(imgMask2, 50); % best 50
    imgMask2 = 1-bwareaopen(1-imgMask2, 100); % best 100
    % figure; imshow(imgMask2);
    
    labelMask = bwlabel(imgMask2);
    labelMaskColor = label2rgb(labelMask,'jet','black','shuffle');
    % figure; imshow(labelMaskColor);  

    maxN = max(max(labelMask));
    finalMask = zeros(size(labelMask));
    se = strel('disk', 1); % best
    
    % figure
    % imshow(se.Neighborhood)
    
    for nM = 1 : maxN
        tempM = zeros(size(labelMask));
        tempM (labelMask == nM) = 1;
        % figure; imshow(tempM);
        dImg = imdilate(tempM,se);
        % figure; imshow(dImg);
        finalMask(dImg == 1) = nM;
    end
    
    finalMaskColor = label2rgb(finalMask,'jet','black','shuffle');
    % figure; imshow(finalMaskColor);  
    imwrite(uint16(finalMask), fullfile(labelPath, flist(fr).name));
    imwrite(finalMaskColor, fullfile(labelColorPath, flist(fr).name));
end


 


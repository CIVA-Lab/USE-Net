% compute individual AJI and mean AJI of all test data
clear all; close all;

imgPath = 'USE-Net-Final/Label/';
gtPath = 'TestGT/BinaryLabeled/';

flist=dir(fullfile(imgPath,'*.png'));
n = length(flist);

for fr = 1 : n
    flist(fr).name
    
    nName = strip(flist(fr).name,'right','g');
    nName = strip(nName,'right','n');
    nName = strip(nName,'right','p');
    nName = strip(nName,'right','.');
    
    img = imread(fullfile(imgPath, flist(fr).name)); 
    gt = imread(fullfile(gtPath, [nName,'.png'])); 
  
    imgD = double(img);
    gtD = double(gt);

    aji = Aggregated_Jaccard_Index_v1_0(gtD,imgD);
    count = countNuclei(gtD, imgD);

    disp("Aggreagated Jaccard Index");
    aji
    disp("Count: correct, missing");
    count
    
    allAJI(fr) = aji;
end

disp("Mean AJI");
mean(allAJI)




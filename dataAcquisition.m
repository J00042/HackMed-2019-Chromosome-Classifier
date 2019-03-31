%%

% 31/03/2019
% Mohamed Atwya
% matwya1@sheffield.ac.uk

clear all
close all
clc

for counterX = 1:24
chromoNum = counterX;
% Get list of all tiff files in this directory
% DIR returns as a structure array.  You will need to use () and . to get
% the file names.


fileNames = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'...
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',...
    'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'twenty_one',...
    'twenty_two', 'x', 'y'};

folderpath = 'C:\Users\Mohamed\Desktop\chromosomes_30032019\';
pathLoc = fullfile(folderpath,char(fileNames(chromoNum)));
num = chromoNum;
imagefiles = dir(fullfile(pathLoc,'*.png')); % pattern to match filenames.
nfiles = length(imagefiles);    % Number of files found

for counterOne=1:nfiles
	imagePath = fullfile(pathLoc,imagefiles(counterOne).name);
	currentimage = imread(imagePath);
    rgbImage = currentimage(:,:, 1:3);
    scaled_train = rgb2gray(rgbImage);
    grayImgs{counterOne} = histeq(scaled_train);
    bwImgs{counterOne}   = imbinarize(scaled_train); %be sure that your image is binary
end

%% 
%  Chromosome length and width
for counterOne=1:nfiles
    filteredBw = bwareaopen(bwImgs{counterOne}, 10);
    [~, lastColumn] = find(filteredBw, 1, 'last');
    [~, firstColumn] = find(filteredBw, 1, 'first');
    widthBw = abs(firstColumn-lastColumn);
    [~, lastColumn] = find(imrotate(filteredBw,90), 1, 'last');
    [~, firstColumn] = find(imrotate(filteredBw,90), 1, 'first');
    ChromoLengthStartPixel(counterOne,1) = firstColumn;
    lenBw   = abs(firstColumn-lastColumn);
    chromoSize(counterOne,1) = widthBw;
    chromoSize(counterOne,2) = lenBw;
end

%% 
%  Centrome location using binary image
for counterOne=1:nfiles
    clear chormLayerWiseWidth smoothedChormLayerWiseWidth chormLayerWiseColorMean
    [y1,x1] = size(bwImgs{counterOne});
	for counterTwo = 1:y1
        chormLayerWiseWidth(1,counterTwo) = sum(bwImgs{counterOne}(counterTwo,:));    
    end
    smoothedChormLayerWiseWidth(counterOne,1:y1) = smoothdata(chormLayerWiseWidth(1,1:y1),'loess');
    [pks,locs] = findpeaks(smoothedChormLayerWiseWidth(counterOne,1:y1));
    [minVal,minValArrayLoc] = min(pks);
    centromereLocation(counterOne,1) = locs(minValArrayLoc);
    relativeCentromereLocation(counterOne,1) = (centromereLocation(counterOne,1) - ChromoLengthStartPixel(counterOne,1))/chromoSize(counterOne,2);
    fy(counterOne,1:y1) = gradient(smoothedChormLayerWiseWidth(counterOne,1:y1));
    %figure
    %plot(1:y1,chormLayerWiseWidth(1,:), 1:y1,smoothedChormLayerWiseWidth(counterOne,:), 1:y1, fy(counterOne,1:y1));
end

%%
% Band count
for counterOne=1:nfiles
    clear smoChormLayerWiseColorMean chormLayerWiseColorMean y1
    [y1,x1] = size(bwImgs{counterOne});
	for counterTwo = 1:y1
        chormLayerWiseColorMean(1,counterTwo) = sum(grayImgs{counterOne}(counterTwo,:))/sum(bwImgs{counterOne}(counterTwo,:));    
    end
    chormLayerWiseColorMean(isinf(chormLayerWiseColorMean)|isnan(chormLayerWiseColorMean)) = 0; 
    smoChormLayerWiseColorMean(1,1:y1)  = smoothdata(chormLayerWiseColorMean(1,:) ,'movmedian','SmoothingFactor',0.85);
    [pks,locs] = findpeaks(smoChormLayerWiseColorMean(1,1:y1));
    numPerBands(counterOne,1) = length(locs)-2;
%     figure
%     plot(1:y1,smoChormLayerWiseColorMean(1,1:y1));
end

%%
% for counterOne=1:nfiles
%     figure;
% 	imshow(grayImgs{counterOne});
%     % grayImgs{counterOne};
%     % bwImgs{counterOne};
% end

for counterOne=1:nfiles
    %data(counterOne,2) = nnz(bwImgs{counterOne});         % total number of pixels
    data(counterOne,1)  = chromoSize(counterOne,2)/chromoSize(counterOne,1); % chromo width    
    data(counterOne,2)  = relativeCentromereLocation(counterOne,1);          % centrome percetage location relative to top   
    data(counterOne,3)  = numPerBands(counterOne,1);
    data(counterOne,4)  = mean(grayImgs{counterOne}(:));   % mean grayscale value
    data(counterOne,5)  = std(im2double(grayImgs{counterOne}(:)));   % mean grayscale value    
    data(counterOne,6)  = skewness(im2double(grayImgs{counterOne}(:)));
    data(counterOne,7)  = kurtosis(im2double(grayImgs{counterOne}(:)));
end
data(:,8:24+7) = 0;
data(:,7 + num ) = 1;

data;
matfile = fullfile(pathLoc, 'data');

save(matfile, 'data');


clear currentimage ans currentfilename ii nfiles imagePath rgbImage...
    scaled_train imagefiles data

end
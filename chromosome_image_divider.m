function chromosome_image_divider(folder, baseFileName, backgroundColour)
%input: the path to the specified image file, the baseFileName, the greyscale threshold 
%   to differentiate between background and colour (about 220 seems to work best on the 
%   images used at the hackathon).
%HackMed 2019. 
%A program to receive a grayscale image of chromosomes and produce seperate
%   images of each individual chromosome. 
%This program was made by Jack Hutton with help from Mohammed Atwya and advice 
%   from Nagham Yousef, Nazia Ahmed and Youssef Maharem in 24 Hoursat HackMed 2019 
%   by modifying the Image Processing example program found at this link: 
%   https://uk.mathworks.com/matlabcentral/fileexchange/25157-image-segmentation-tutorial
%   As it was made rapidly at a Hackathon, the program is generally unorganised and not
%   all comments will be consistent or correct.
%NOTE: To output images to a folder, change the filepath here:
outputPath = "C:\Users\Owner\Documents\HackMed 2019\chromosomes";

%thresholdValue = 0;
%binarySubImage=0;
%folder = "C:\Users\Owner\Documents\HackMed 2019"; baseFileName = "IMG00019_cropped.JPG";
%% CHECK THE MATLAB IMAGE TOOLBOX IS PRESENT
numberOfBlobs = 1;
tic; % Start timer.
%clc; % Clear command window.
%clearvars; % Get rid of variables from prior run of this m-file.
fprintf('Running chromosome_image_divider.m...\n'); % Message sent to command window.
workspace; % Make sure the workspace panel with all the variables is showing.
imtool close all;  % Close all imtool figures.
format long g;
format compact;
captionFontSize = 14;
% Check that user has the Image Processing Toolbox installed.
hasIPT = license('test', 'image_toolbox');
if ~hasIPT
    % User does not have the toolbox installed.
    message = sprintf('Error: Image Processing Toolbox in not installed.\nDo you want to try to continue anyway?');
    reply = questdlg(message, 'Toolbox missing', 'Yes', 'No', 'Yes');
    if strcmpi(reply, 'No')
        % User said No, so exit.
        return;
    end
end

%% IMPORT THE IMAGE
%baseFileName = filename;
%folder = fileparts(which(baseFileName)); % Determine where demo folder is (works with all versions).
fullFileName = fullfile(folder, baseFileName);
if ~exist(fullFileName, 'file')
        beep;
        fprintf('Error: the input image file\n%s\nwas not found.\n', fullFileName);
        %warningMessage = sprintf('Error: the input image file\n%s\nwas not found.\nClick OK to exit.', fullFileName, folder);
        %uiwait(warndlg(warningMessage));
        fprintf(1, 'Finished running chromosome_image_divider.m.\n');
        return;
else
   disp(['image ' + baseFileName + ' was found in folder ' + folder]); 
end
%disp(['Full filename: ' + fullFileName])

%% CONVERT THE IMAGE TO GREYSCALE
% If we get here, we should have found the image file.
originalImage = imread(fullFileName);
[rows, columns, numberOfColorChannels] = size(originalImage);
%If the image is not grayscale, convert it to grayscale.
if numberOfColorChannels > 1
    disp('converting image to grayscale...');
    originalImage = rgb2gray(originalImage);
end
originalImage = imsharpen(originalImage);
originalImage = medfilt2(originalImage, [3 3]);
%Plot the grayscale original image. 
disp('plotting images...');
figure(1);
set(gcf, 'Units','Normalized','OuterPosition',[0 0 1 1]); %Maximise the current figure window.
clf;
subplot(3, 2, 1);
imshow(originalImage);
drawnow; % Force it to display RIGHT NOW (otherwise it might not display until it's all done, unless you've stopped at a breakpoint.)
axis image; % Make sure image is not artificially stretched because of screen's aspect ratio.
title (['Original image ' + baseFileName]);

%% PLOT A HISTOGRAM TO DECIDE WHAT VALUE IS BEST FOR THE THRESHOLD.
% Just for fun, let's get its histogram and display it.
[pixelCount, grayLevels] = imhist(originalImage);
subplot(3, 2, 2);
bar(pixelCount);
drawnow;
title('Histogram of original image');
xlim([0 grayLevels(end)]); % Scale x axis manually.
grid on;
weightedSum = sum(pixelCount);
cumulativeSum = 0;
% for k = 1:1:size(grayLevels) %Count until you reach just past the first gray levels peak.
%     if cumulativeSum > 0.25*weightedSum
%         thresholdValue = grayLevels(k);
%         break
%     else
%         cumulativeSum = cumulativeSum + pixelCount(k);
%     end
% end
thresholdValue = backgroundColour; %220;
% for k = 1:1:size(grayLevels) %Count until you reach just past the first gray levels peak.
%     if pixelCount(k+1) < pixelCount(k)
%         peakPixelCount = pixelCount(k);
%         peakBackground = k;
%         break
%     end
% end
% for k = peakBackground:1:size(grayLevels)
%     if pixelCount(k) <= peakPixelCount/3
%        thresholdValue = grayLevels(k);
%        disp(['thresholdValue chosen as ', num2str(thresholdValue)]);
%        break
%     end
% end
%thresholdValue = 200;
%%CONVERT THE IMAGE TO BINARY (BLACK & WHITE) USING THE THRESHOLD VALUE
% Threshold the image to get a binary image (only 0's and 1's) of class "logical."
% Method #1: using im2bw()
%   normalizedThresholdValue = 0.4; % In range 0 to 1.
%   thresholdValue = normalizedThresholdValue * max(max(originalImage)); % Gray Levels.
%   binaryImage = im2bw(originalImage, normalizedThresholdValue);       % One way to threshold to binary
% Method #2: using a logical operation.

binaryImage = originalImage < thresholdValue; % Bright objects will be chosen if you use >.

%%PLOT THE THRESHOLD VALUE ON THE HISTOGRAM
% Show the threshold as a vertical red bar on the histogram.
hold on;
maxYValue = ylim;
line([thresholdValue, thresholdValue], maxYValue, 'Color', 'r');
% Place a text label on the bar chart showing the threshold.
annotationText = sprintf('Thresholded at %d gray levels', thresholdValue);
% For text(), the x and y need to be of the data class "double" so let's cast both to double.
text(double(thresholdValue + 5), double(0.5 * maxYValue(2)), annotationText, 'FontSize', 10, 'Color', [0 .5 0]);
text(double(thresholdValue - 70), double(0.94 * maxYValue(2)), 'Background', 'FontSize', 10, 'Color', [0 0 .5]);
text(double(thresholdValue + 50), double(0.94 * maxYValue(2)), 'Foreground', 'FontSize', 10, 'Color', [0 0 .5]);

%%PLOT THE BINARY IMAGE
% Use < if you want to find dark objects instead of bright objects.
%   binaryImage = originalImage < thresholdValue; % Dark objects will be chosen if you use <.
% Do a "hole fill" to get rid of any background pixels or "holes" inside the blobs.
binaryImage = imfill(binaryImage, 'holes');
binaryImage = bwareaopen(binaryImage, 500);
subplot(3, 2, 3);
imshow(binaryImage);
drawnow;
title('Binary Image, obtained by thresholding'); 
% Identify individual blobs by seeing which pixels are connected to each other.
% Each group of connected pixels will be given a label, a number, to identify it and distinguish it from the other blobs.
% Do connected components labeling with either bwlabel() or bwconncomp().
labeledImage = bwlabel(binaryImage, 8);     % Label each blob so we can make measurements of it

%% IDENTIFY THE INDIVIDUAL OBJECTS BY USING MATLAB's bwlabel() FUNCTION. 
% Identify individual chromosomes by seeing which pixels are connected to each other.
% Each group of connected pixels will be given a label, a number, to identify it and distinguish it from the other blobs.
% Do connected components labeling with either bwlabel() or bwconncomp().
labeledImage = bwlabel(binaryImage, 8);     % Label each blob so we can make measurements of it
% labeledImage is an integer-valued image where all pixels in the blobs have values of 1, or 2, or 3, or ... etc.
subplot(3, 2, 4);
imshow(labeledImage, []);  % Show the gray scale image.
drawnow;
title('Labeled Image, from bwlabel()');

%% FIND ALL BLOBS USING REIGONPROPS AND SHOW WITH COLOURS AND NUMBERS
% Let's assign each blob a different color to visually show the user the distinct blobs.
coloredLabels = label2rgb (labeledImage, 'hsv', 'k', 'shuffle'); % pseudo random color labels
% coloredLabels is an RGB image.  We could have applied a colormap instead (but only with R2014b and later)
subplot(3, 2, 5);
imshow(coloredLabels);
drawnow;
axis image; % Make sure image is not artificially stretched because of screen's aspect ratio.
caption = sprintf('Pseudo colored labels, from label2rgb().\nBlobs are numbered from top to bottom, then from left to right.');
title(caption);
figure(3)
imshow(coloredLabels);
drawnow;
axis image; % Make sure image is not artificially stretched because of screen's aspect ratio.
caption = sprintf('Pseudo colored labels, from label2rgb().\nBlobs are numbered from top to bottom, then from left to right.');
title(caption);
figure(1)
% Get all the blob properties.  Can only pass in originalImage in version R2008a and later.
blobMeasurements = regionprops(labeledImage, originalImage, 'all');
numberOfBlobs = size(blobMeasurements, 1);
% bwboundaries() returns a cell array, where each cell contains the row/column coordinates for an object in the image.
% Plot the borders of all the coins on the original grayscale image using the coordinates returned by bwboundaries.
subplot(3, 2, 6);
imshow(originalImage);
title('Outlines, from bwboundaries()'); 
axis image; % Make sure image is not artificially stretched because of screen's aspect ratio.
drawnow;
hold on;
boundaries = bwboundaries(binaryImage);
numberOfBoundaries = size(boundaries, 1);
for k = 1 : numberOfBoundaries
    thisBoundary = boundaries{k};
    plot(thisBoundary(:,2), thisBoundary(:,1), 'g', 'LineWidth', 2);
end
hold off;
textFontSize = 8;	% Used to control size of "blob number" labels put atop the image.
labelShiftX = -7;	% Used to align the labels in the centers of the chromosomes.
blobECD = zeros(1, numberOfBlobs);

%%LOOP THROUGH EACH GROUP OF PIXELS IN THE IMAGE AND PRINT THEIR PROPERTIES
% Print header line in the command window.
fprintf(1,'Blob #      Mean Intensity  Area   Perimeter    Centroid       Diameter   Orientation   Length   Width\n');
% Loop over all blobs printing their measurements to the command window.
disp(['number of blobs in image found: ', num2str(numberOfBlobs)]);
for k = 1 : numberOfBlobs           % Loop through all blobs.
    % Find the mean of each blob.  (R2008a has a better way where you can pass the original image
    % directly into regionprops.  The way below works for all versions including earlier versions.)
    thisBlobsPixels = blobMeasurements(k).PixelIdxList;  % Get list of pixels in current blob.
    meanGL = mean(originalImage(thisBlobsPixels)); % Find mean intensity (in original image!)
    meanGL2008a = blobMeasurements(k).MeanIntensity; % Mean again, but only for version >= R2008a

    blobArea = blobMeasurements(k).Area;                % Get area.
    blobPerimeter = blobMeasurements(k).Perimeter;		% Get perimeter.
    blobCentroid = blobMeasurements(k).Centroid;		% Get centroid one at a time
    blobECD(k) = sqrt(4 * blobArea / pi);				% Compute ECD - Equivalent Circular Diameter.
    blobAngle = blobMeasurements(k).Orientation;
    blobLength = blobMeasurements(k).MajorAxisLength;
    blobWidth = blobMeasurements(k).MinorAxisLength;
    fprintf(1,'#%2d %17.1f %11.1f %8.1f %8.1f %8.1f %8.1f %8.1f %15.1f %8.1f\n', k, meanGL, blobArea, blobPerimeter, blobCentroid, blobECD(k), blobAngle, blobLength, blobWidth);
    % Put the "blob number" labels on the "boundaries" grayscale image.
    text(blobCentroid(1) + labelShiftX, blobCentroid(2), num2str(k), 'FontSize', textFontSize, 'FontWeight', 'Bold');
end
% Now, I'll show you another way to get centroids.
% We can get the centroids of ALL the blobs into 2 arrays,
% one for the centroid x values and one for the centroid y values.
allBlobCentroids = [blobMeasurements.Centroid];
centroidsX = allBlobCentroids(1:2:end-1);
centroidsY = allBlobCentroids(2:2:end);
% Put the labels on the rgb labeled image also.
subplot(3, 2, 5);
for k = 1 : numberOfBlobs           % Loop through all blobs.
    text(centroidsX(k) + labelShiftX, centroidsY(k), num2str(k), 'FontSize', textFontSize, 'FontWeight', 'Bold');
end

%% CROP EACH CHROMOSOME INTO ITS OWN IMAGE. 
%message = sprintf('Would you like to crop out each chromosome to individual images?');
%reply = questdlg(message, 'Extract Individual Images?', 'Yes', 'No', 'Yes');
reply = 'Yes';
disp('Cropping images...');
% Note: reply will = '' for Upper right X, 'Yes' for Yes, and 'No' for No.
if strcmpi(reply, 'Yes')
    figure(2);	% Create a new figure window.
    clf;
    % Maximize the figure window.
    set(gcf, 'Units','Normalized','OuterPosition',[0 0 1 1]);
    for k = 1 : numberOfBlobs           % Loop through all blobs.
        % Find the bounding box of each blob.
        thisBlobsBoundingBox = blobMeasurements(k).BoundingBox;  % Get list of pixels in current blob.#
        % increase the bounding box a little.
        thisBlobsBoundingBox(1) = thisBlobsBoundingBox(1) - 25;
        thisBlobsBoundingBox(2) = thisBlobsBoundingBox(2) - 25;
        thisBlobsBoundingBox(3) = thisBlobsBoundingBox(3) + 50;
        thisBlobsBoundingBox(4) = thisBlobsBoundingBox(4) + 50;
        %disp(thisBlobsBoundingBox);
        % Extract out this coin into it's own image.
        subImage = imcrop(originalImage, thisBlobsBoundingBox);
        %rotate the image so it is the right way up.
        subImage = imrotate(subImage, ((90-blobMeasurements(k).Orientation)));
        %crop the image again to a smaller box size.
        subImage2 = subImage;
        subImage2(subImage2<10)=255; %remove the black background from cropping and rotating
        binarySubImage = subImage2 < (thresholdValue); %convert to binary image+
        binarySubImage = imfill(binarySubImage, 'holes');
        [binLength, binWidth] = size(binarySubImage);
        for m = 1:1:size(binarySubImage,1)
            for n = 1:1:size(binarySubImage,2)
                if n < round(size(binarySubImage,1)/5)
                    binarySubImage(m,n) = 0;
                end
                if n > round(size(binarySubImage,1)*4/5)
                    binarySubImage(m,n) = 0;
                end
                if m < round(size(binarySubImage,2)/5)
                    binarySubImage(m,n) = 0;
                end
                if m > round(size(binarySubImage,1)*4/5)
                    binarySubImage(m,n) = 0;
                end
            end
        end
        binarySubWeight = sum(sum(binarySubImage));  %find total area of binary image
        binarySubImage = bwareaopen(binarySubImage, round(binarySubWeight*0.4));    %remove areas with less than 30% of overall area.
        %figure(4); subplot(10, ceil(numberOfBlobs/10), k); imshow(binarySubImage);
        %subImage = subImage .* binarySubImage;
        subImage(binarySubImage<1)=240;
        topX = round(size(subImage,1)/5);
        topY = round(size(subImage,2)/5);
        lengthX = round(size(subImage, 1)*3/5);
        lengthY = round(size(subImage, 2)*3/5);
        %subImage = imcrop(subImage, [topX, topY, lengthX, lengthY]);
        % Display the image with informative caption.
        subplot(10, ceil(numberOfBlobs/10), k);
        imshow(subImage);
        %caption = sprintf('Chromosome #%d.\nDiameter = %.1f pixels\nArea = %d pixels', k, blobECD(k), blobMeasurements(k).Area);
        caption = (['Chromosome ', num2str(k)]);
        title(caption);
        drawnow;
     %%SAVE THE IMAGE
        imageFolder = outputPath;
        imageFileName = (['Chromosome_', num2str(k), '.png']);
        fullImageFileName = fullfile(imageFolder, imageFileName);
        imwrite(subImage, fullImageFileName);
    end
end


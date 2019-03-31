%%

% 31/03/2019
% Mohamed Atwya
% matwya1@sheffield.ac.uk

%% Clear old varaibles and close old figures.
clc;
clear all;
close all;
format shortEng;

% sets plot style and figure size
set(0,'defaultAxesFontName', 'Times New Roman');
set(0,'defaultTextFontName', 'Times New Roman');
set(0,'defaultAxesFontSize', 8);
set(0,'defaultTextFontSize', 8);
xFigPosition = 100;
yFigPosition = 100;
width  = 300;
height = 200;

%% Data Loading 
tic                                       % starts a timer to time the code
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\one\data')
dataOne = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\two\data')
dataTwo = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\three\data')
dataThree = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\four\data')
dataFour = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\five\data')
dataFive = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\six\data')
dataSix = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\seven\data')
dataSeven = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\eight\data')
dataEight = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\nine\data')
dataNine = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\ten\data')
dataTen = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\eleven\data')
dataEleven = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\twelve\data')
dataTwelve = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\thirteen\data')
dataThirteen = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\fourteen\data')
dataFourteen = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\fifteen\data')
dataFifteen = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\sixteen\data')
dataSixteen = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\seventeen\data')
dataSeventeen = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\eighteen\data')
dataEighteen = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\nineteen\data')
dataNineteen = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\twenty\data')
dataTwenty = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\twenty_one\data')
dataTwentyOne = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\twenty_two\data')
dataTwentyTwo = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\x\data')
dataX = data;
load('C:\Users\Mohamed\Desktop\chromosomes_30032019\y\data')
dataY = data;

D = [dataOne; dataTwo; dataThree; dataFour; dataFive; dataSix; dataSeven;...
    dataEight; dataNine; dataTen; dataEleven; dataTwelve; dataThirteen;...
    dataFourteen; dataFifteen; dataSixteen; dataSeventeen; dataEighteen;...
    dataNineteen; dataTwenty; dataTwentyOne; dataTwentyTwo; dataX; dataY];
rawData = D;                              % save a new copy of the data
[numOfDataRows,numOfVar] = size(rawData); % number of data points

clear dataOne dataTwo dataThree dataFour dataFive dataSix dataSeven...
    dataEight dataNine dataTen dataEleven dataTwelve dataThirteen...
    dataFourteen dataFifteen dataSixteen dataSeventeen dataEighteen...
    dataNineteen dataTwenty dataTwentyOne dataTwentyTwo dataX dataY; 
%% Fixed parameters
rand( 'state' ,500); % fix random number seed for repeatibility of shuffle

%% Input parameters to be adjusted.
numOfInVar = 7;               % Num. of input varaibles
numOfOutVar = 24;               % Num. of output varaibles
inSampleSizeRatio = 0.5 ;      % Data split ( 50/50 )
outfunc = 'logistic';          % Here we are doing classification
nhid = 100;                     % Mlp complexity (num. of hidden units) 70
k = 3;                        % Num. of folds for cross validation (cv)
repK = 1;                     % Num. of reptitions for repeated k-fold cv                                

%% IMPORTANT
% IF RUNNING THE COARSE RHO CHECK SET THIS TO 1, IF FINE CHECK SET to 0.
coarseCheck = 1;

% COARSE RHO CHECK
 rho = logspace(0 , 1 , 20 );  % Coasre Rho grid one  Gives 4.175 
% rho = logspace(-1 , 1 , 10 );  % Coarse Rho grid two  Gives 3.593814

% FINE RHO SEARCH
 %rho = 3:0.05:5;  % Rho grid

% Initially a coarse rho grid logspace(-3 , 4 , 30 ) is tested to narrow
% the search, follwoed by a fine linespaced search 2:0.5:10
% The log search gives Rho  4.175 and CC error 165.075. 
numOfIts = 10000;              % Ensure enough iterations allowed

%% Data Splitting
% Randomise and Split data into training in-sample (x,z) and out-of-sample 
% (oos) data (x_star,z_star) using the user input, inSampleSizeRatio)
x = rawData(:, 1:numOfInVar);
x = dmstandard(x);
z = rawData(:, numOfInVar+1:numOfInVar+24);

% rounds down the sample size to an integer value
inSampleSize = floor( inSampleSizeRatio * size(x,1) );  
ind = randperm( size( x , 1 ) );              % creates a random id list
dataIn = x( : , 1:numOfInVar );
dataTarget = z;

% rearrange the in-sample and oos data values accroding to the randomized
% ID list.
x = dataIn( ind( 1:inSampleSize ) , : );      
z = dataTarget( ind( 1:inSampleSize ) , : );
x_star = dataIn( ind( inSampleSize+1:end ) , : );
z_star = dataTarget( ind(inSampleSize+1:end ) , :);

% finds the number of in-sample and oos data records.
[ numOfInSampleRows , ~ ] = size( x );            
[ numOfOutOfSampleRows , ~ ] = size( x_star );

% Prints the results of this section to the main command.
fprintf('---------------------------------------------------------------\n');
fprintf(' DATA SPLITTING\n' );
fprintf('---------------------------------------------------------------\n');
fprintf(' Num. of data rows               = %d\n' , numOfDataRows ); 
fprintf(' Num. of in-sample data rows     = %d\n' , numOfInSampleRows ); 
fprintf(' Num. of out-of-sample data rows = %d\n' , numOfOutOfSampleRows ); 
fprintf('---------------------------------------------------------------\n');

%% In-Sample Data Treatment
% Locate repeted rows, if any, and delete duplicates.
inData = [x,z];

% Missing data
% checks the data for any missing arrays and save a message, later to be 
% printed to inform the user
missDataId = ismissing( x , z );
if sum( missDataId(:) ) ~= 0  
    missDataMsg = 'True';
else
    missDataMsg = 'False';
end

% Non-numeric data
% checks the data for non-numeric arrays and save a message, later to be 
% printed to inform the user
nonNumDataId = isnumeric([x,z]);
if  nonNumDataId == 0  
    nonNumDataMsg = 'True';
else
    nonNumDataMsg = 'False';
end


% Checks if the input data includes floats
% save a message, later to be printed to inform the user
if  isfloat(x( : ) ) == 1
    floatTest = 'True';
else
    floatTest = 'False';
end

% Checks if the input data includes negative values
% save a message, later to be printed to inform the user
if  any(x<0) > 0 
    negativeTest = 'True';
else
    negativeTest = 'False';
end

% Checks if the input data is collinear or not
% save a message, later to be printed to inform the user
if  rank ( x )  == numOfInVar
    collinearTest = 'True';
else
    collinearTest = 'False';
end

% Computes the correlation matrix using a confidence interval of 99%
corrMatrix = corrcoef ( x,'alpha',0.01);
% Computes the P-values of the correlation matrix using a CI of 99%
[~,p] = corrcoef ( x,'alpha',0.01);
corrP = p < 0.0001;
% Computes the number of corrleated varaibles with p value less than 0.0001
numOfCorr = sum(corrP(:))/2;

% Computes the number of biodegrdable and non-biodegradable chemicals.
NonBiodegChmclCnt = sum( z == 0 );
BiodegChmclCnt    = sum( z == 1 );

% Prints the in-sample data treatment results/findings to the main command.
fprintf('---------------------------------------------------------------\n');
fprintf(' IN-SAMPLE DATA TREATMENT\n' );
fprintf('---------------------------------------------------------------\n');
fprintf(' Num. of input variables            = %d\n' , numOfInVar );
fprintf(' Num. of output variables           = %d\n' , numOfOutVar ); 
% fprintf(' Repeated Rows                      = %s\n' , dupRowMsg);
% if hasDuplicates == 1
%     fprintf(' Num. of repeated rows              = %d\n' , numOfRepeatedRows);
%     fprintf(' Repeated Rows ID                   =\n');
%     for  counterTwo = 1:numOfRepeatedRows
%     fprintf('                                     row %d = row %d\n' ,dupRowsSetOneId(counterTwo), dupRowSetTwoID(counterTwo));
%     end
%     fprintf(' Num. of removed rows               = %d\n' , numOfRepeatedRows );
% end
% fprintf(' Num. of removed columns            = 0\n' );
fprintf(' Empty cells                        = %s\n' , missDataMsg);
fprintf(' Non-numeric cells                  = %s\n' , nonNumDataMsg);
fprintf('---------------------------------------------------------------\n');

% Prints the in-sample data treatment results/findings to the main command.
fprintf('\n---------------------------------------------------------------\n');
fprintf(' IN-SAMPLE DATA STATISTICS\n' );
fprintf('---------------------------------------------------------------\n');
%fprintf(' Num. of suspected outliers         = %d\n' , numOfOutliers );
fprintf(' Num. of sample data                = %d\n' , numOfInSampleRows );
fprintf(' Target (1-biodegradable) count     = %d\n' , BiodegChmclCnt ); 
fprintf(' Target (0-Not biodegradable) count = %d\n' , NonBiodegChmclCnt );
fprintf(' Data contais floats                = %s\n' , floatTest );
fprintf(' Data contais negative values       = %s\n' , negativeTest );
fprintf(' Input data non collinear           = %s\n' , collinearTest );
fprintf(' Input data colleration count       = %d\n' , numOfCorr );
fprintf('---------------------------------------------------------------\n');                              


%% MLP and Repeated K cross validated with stratification training

fprintf('---------------------------------------------------------------\n'); 
fprintf('K-fold Cross Validation (where k = %d)\n',k); 
fprintf('---------------------------------------------------------------\n'); 

performanceIndex = zeros( length(rho) , repK );
fprintf('Started repated k cross-validation loop');

options      = foptions; % initialize options
options(1)   = 0;        % set "silent"  
options(2)  = 10e-006;   % measure of the precision required for the 
                         % weights W at the solution.
options(3)  = 10e-006;   % measure of the precision required for the 
                         % objective function at the solution.
options(9)   = 0;        % set gradient error to silent 
options(14)  = numOfIts; % This sets the max. num. of itterations allowed

for l=1:length(rho)
    fprintf('\nRho Iteration: %d',l);
    fprintf('\n Repated CV Iteration: ');
    % Initialize mymlp and use Jemp as performance indicator (can use AUROC)
    mymlp = mlp( size(x,2) , nhid , numOfOutVar, outfunc, rho(l) );  % Initialise MLP
    % Check PI (Jemp not nsse!) for TRAINING data as a check
    % no local minimum problems (shouldn't be too far from CV PI; usually smaller)
    for counterRepeated = 1:repK
        fprintf('%d ', counterRepeated );
        % shuffle data to break any artificial ordering
        ind = randperm(size(x,1));
        x = x( ind( 1:numOfInSampleRows ) , : );
        z = z( ind( 1:numOfInSampleRows ) , : );
        % Evaluate the MLP and return the PI, Jemp.
        [ y_hat , Jemp ] = dmxvalEdited( mymlp , options , x , z , k );
        performanceIndex(l,counterRepeated) = Jemp;
    end
    % Compute an average Jemp for the different itterations.
    performanceIndex(l,repK+1) = mean ( performanceIndex(l,1:repK) );
end

% Locate the best, minimum average PI and the matching Rho value.
[minAvgPI,minAvgPiId] = min( performanceIndex(:,repK+1) );
minRhoAtAvgPI = rho( : , minAvgPiId );

% plot the Jemp vs rho graphs from all 10 repeated iterations.
% If a coarse log rho vector is used, then use semilogx to plot.
% If a fine linear rho vector is used, then use regular plot fucntion.
if coarseCheck == 1
    figure;
    set(gcf,'units','points','position',[xFigPosition,yFigPosition,width,height]);
    testOneHand = semilogx( rho , performanceIndex(: , 1 ) );
    hold on;
    bestRhoHand = plot( minRhoAtAvgPI ,minAvgPI , 'rx','linewidth',1.5,'markersize',12);
    for counterRepeated = 2:repK
        hand(counterRepeated) = semilogx( rho , performanceIndex(: , counterRepeated ) );
    end
    averageHand = semilogx( rho , performanceIndex(: , repK+1 ), 'r', 'linewidth',1.3);
    hold off;
    grid on;
    xlabel('ln (\rho)');
    ylabel('Performance Index (Jemp)');
    legend( [bestRhoHand averageHand] ,...
        'Rho at best Avg Jemp', 'Avg Jemp from all tests');
else 
    figure;
    set(gcf,'units','points','position',[xFigPosition,yFigPosition,width,height]);
    hold on;
    for counterRepeated = 1:repK
    hand(counterRepeated) = plot( rho , performanceIndex(: , counterRepeated ) );
    end
    bestRhoHand = plot( minRhoAtAvgPI ,minAvgPI , 'rx','linewidth',1.5,'markersize',12);
    averageHand = plot( rho , performanceIndex(: , repK+1 ), 'r', 'linewidth',1.3);
    hold off;
    grid on;
    xlabel('ln (\rho)');
    ylabel('Performance Index (Jemp)');
    legend( [bestRhoHand averageHand] ,...
    'Rho at best Avg Jemp', 'Avg Jemp from all tests');
end 

% Prints the MLP training results/findings from repeated k-fold corss 
% validation to the main command.
fprintf('\n\nSmallest PI Jemp  = %.6f\n', minAvgPI);
fprintf('Coresponding rho  = %.6f\n', minRhoAtAvgPI);
fprintf('---------------------------------------------------------------\n'); 

%% retrain using rho_min

% dmxval options
options      = foptions;  % initialize options
options(1)   = 0;         % set "silent" 
options(2)  = 10e-006;    % measure of the precision required for the 
                          % weights W at the solution.
options(3)  = 10e-006;    % measure of the precision required for the 
                          % objective function at the solution.
options(9)   = 0;         % set gradient error to silent 
options(14)  = numOfIts;  %This sets the max. num. of itterations allowed
    
mymlp = mlp( size(x,2) , nhid, numOfOutVar , outfunc , minRhoAtAvgPI ); % initialize mymlp
[mymlp, options] = mlptrain(mymlp , options , x , z );        % Train mymlp
y_star_hat = mlpfwd( mymlp , x_star );                        % evaluate oos
mlpPredicError = z_star - y_star_hat;   % compute prediction errors

% compute correlation between prediction and target
[c,~] = corrcoef(z_star , y_star_hat);  
mlpCorrCoff = c(2,1);

% compute the ROC x and y axis values, the AUROC and returns Youden's 
% Index x and y values.
[pfa, ptp,area, yodOpt] = dmrocEdit( z_star , y_star_hat );

% Compute Youden's Index Cut-off
mlpYodIndex = yodOpt(2) + (1-yodOpt(1)) - 1; 

% Confusion Matrix computation


% for count = 1:length(y_star_hat)
%     if y_star_hat(count) < mlpYodIndex
%         y_star_hat_Logical (count) = 0;
%     else
%         y_star_hat_Logical (count) = 1;
%     end
% end

%%
y_star_hat_Logical = y_star_hat;

[x,y] = size(y_star_hat_Logical)
for counterOne = 1:x
    maxVal = max(y_star_hat_Logical(counterOne,:))
    for counterTwo = 1:y
        if y_star_hat_Logical(counterOne,counterTwo) < maxVal
            y_star_hat_Logical(counterOne,counterTwo) = 0;
        else
            y_star_hat_Logical(counterOne,counterTwo) = 1;
        end
    end
end

% thresholdVal = 0.1;
% y_star_hat_Logical(y_star_hat_Logical< thresholdVal) = 0;
% y_star_hat_Logical(y_star_hat_Logical>= thresholdVal) = 1;


% Overaly the MLP ROC curve over the GLM ROC curve
figure;
mlpPlotHand = plot(pfa,ptp,'r');
plot(yodOpt(1),yodOpt(2),'kx');
text(.5,.18,['MLP AUROC = ' num2str(area,'%4.2f')]);
axis('square');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
legend('MLP ROC');
hold off;

% figure;

Conf_Mat =  confmat( y_star_hat_Logical , z_star );
disp(Conf_Mat)
[c_matrix,Result,RefereceResult]= confusion.getMatrix(z_star,y_star_hat_Logical);

plotConfMat(Conf_Mat, {'1', '2', '3','4','5','6','7','8','9','10','11','12',...
    '13','14','15','16','17','18','19','20','21','22','x','y'});

% Prints the MLP final test results/findings from oos data to the main command.
fprintf('\n---------------------------------------------------------------\n'); 
fprintf('Retrain using the minimum rho\n'); 
fprintf('---------------------------------------------------------------\n'); 
fprintf('Psi after re-training         = %.6f\n', options(8)); 
fprintf('PI AUROC                      = %.6f\n', area);
fprintf('Prediction/target correlation = %.6f\n', mlpCorrCoff);
fprintf('Youden''s index                 = %.6f\n', mlpYodIndex);
fprintf('---------------------------------------------------------------\n'); 
 
% Prints minutes elapsed to command window.
fprintf('\n---------------------------------------------------------------\n');
fprintf('Minutes elapsed  = %.3f\n',toc/60);
fprintf('---------------------------------------------------------------\n');

%%
function [pfa, ptp, area, yodOpt, disOpt] = dmrocEdit(z,y)
% DMROC
% computes and plots ROC curve and its area.
%
% z is column vector of binary target values
% y is column vector of estimated probabilities

% (c) 2008 Robert F Harrison
% The University of Sheffield

% Modified by 130146320, 2018. Added  the calculation of youden's index,  
% minimum distance to corner, and instead of plotting, it returns the x 
% and y vectors for the user to plot.

nthresh = 100;
Np = sum( z );
Nn = sum( 1-z );
t = linspace( 0 , 1 , nthresh ); % vector of threshold values
ptp = NaN*ones( size(t) );
pfa = ptp;                       % initialize
ppv = ptp;
npv = ptp; 
ptp(1) = 1; 
ptp(end) = 0;
pfa(1)= 1;
pfa(end) = 0;
ppv(1) = Np/(Np+Nn);
ppv(end) = 1;

for ii=2:nthresh-1
    tp = length(find(z>=t(ii) & y>=t(ii)));
    tn = length(find(z<t(ii) & y<t(ii)));
    fp = length(find(z<t(ii) & y>=t(ii)));
    fn = length(find(z>=t(ii) & y<t(ii)));
    ptp(ii) = tp/(tp+fn); % recall
    pfa(ii) = fp/(tn+fp);
    ppv(ii) = tp/(tp+fp); % precision
    npv(ii) = tn/(tn+fn);
end

sumFnrTpr = round(abs(ptp - pfa),1);
disFnrTpr = sqrt( (1-ptp).^2+(1-(1-pfa)).^2 );
[~, youdenIndexId] = max( sumFnrTpr );
[~, minDisId] = min( disFnrTpr );
yodOpt( 1 ) = pfa( youdenIndexId );
yodOpt( 2 ) = ptp( youdenIndexId );
disOpt( 1 ) = pfa( minDisId );
disOpt( 2 ) = ptp( minDisId );
area = vuroc(z,y);
%h=plot(pfa,ptp,'k',[0 1],[0 1],'k:',[0 0.5],[1 0.5],'k:');axis('square');
end


function [y,Jemp]=dmxvalEdited(net,options,x,t,kFolds)
% MFOLDCV
% Conducts basic m-fold cross validation on Netlab structures
%
% Syntax: [Y,Jemp] = XVAL(NET,OPTIONS,X,T,kFolds)
% NET, OPTIONS, X, T follow Netlab conventions
% kFolds    - number of "folds" or sub-samples
% Y    - matrix of outputs
% Jemp - cross-validated, empirical cost appropriate to task
%
% Limitations:
%   MLPs use 'scg' optimizer by default - edit MLPTRAIN to vary this
%   sub-samples NOT stratified
%
% (c)2012 The University of Sheffield

% Modified by 130146320, 2018. The function now utilises MATLAB's cvpartition
% to stratify the k-folds when splitting.

if nargin<5;
    error('five inputs required');
end
if kFolds==1
    error('one is an invalid no. of folds');
end

toptions=options; % retain copy of original options vector

ntype=net.type; % get type of network

y=NaN*ones(size(t));

c = cvpartition(1:length(t),'KFold',kFolds);

for counter=1:c.NumTestSets
    % choose indices of testing set
    trIdx = c.training(counter);
    teIdx = c.test(counter);
    % train network
    eval(['[tnet,options]=' ntype 'train(net,toptions,x(trIdx,:),t(trIdx,:));']);
    eval(['[yt]=' ntype 'fwd(tnet,x(teIdx,:));']); % compute network output
    y(teIdx,:) = yt; % save network outputs
end

switch net.outfn

  case 'linear'  	% Linear outputs
    Jemp = 0.5*sum(sum((y - t).^2));

  case 'logistic'  	% Logistic outputs
    Jemp = - sum(sum(t.*log(y) + (1 - t).*log(1 - y)));

  case 'softmax'   	% Softmax outputs
    Jemp = - sum(sum(t.*log(y)));

  otherwise
    error(['Unknown activation function ', net.outfn]);
end
end

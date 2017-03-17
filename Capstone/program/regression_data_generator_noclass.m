%% Initialization
clear
clc
 % Specify the folder containing the file 'digitStruct'. This file contains
 % the bounding boxes of each digit in each image
folder = 'extra/';
store_file = 'reg_extra_32x32.mat';
% Load 'digitStruct' to the matlab work space
load(strcat(folder,'digitStruct'))
% Calculate the number of images. Be careful that the number of images
% recorded in 'digitStruct' is more than that contained in that folder.
L = length(digitStruct);
% Set how many samples are available
n = 0;
for i = 1:L
    bs = digitStruct(i).bbox;
    n = n+length(bs);
end
% The variable X_reg has a shape of [width,height,channels,sample_number]
X_reg = uint8(zeros(32,32,3,n));
% The variable y_reg has a shape of [1,sample_number].
% For each sample_number and digit_class, y_reg records the shift and
% bounding box coordinates in the format: [dx,dy]
y_reg = zeros(2,n);
%% Generate samples
pointer = 1;
for i = 1:L % for each image
    % Read the image.
    % The name of each image has been stored in digitStruct(i).name
    img = imread(strcat(folder,digitStruct(i).name));
    if isempty(img)
        break % Error protection
    end
    % Read the bounding box
    bs = digitStruct(i).bbox;
    for j = 1:length(bs)
        % Extraction and storing of the samples, random shifts, and 
        % coordinates of bounding boxes
        [r_img, r_shift, r_box] = box2rbox(img, bs(j));
        X_reg(:,:,:,pointer) = r_img;
        y_reg(:,pointer) = r_shift;
        pointer = pointer+1;
    end
    % Display the progress
    if mod(i,1000)==0
        disp(i)
    end
end
%% Save variables
save(store_file,'X_reg','y_reg','-v7.3');
        
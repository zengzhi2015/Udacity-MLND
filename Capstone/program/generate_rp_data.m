%% Initialization
clear
clc
 % Specify the folder containing the file 'digitStruct'. This file contains
 % the bounding boxes of each digit in each image
folder = 'extra/';
store_file = 'rp_extra_32x32.mat';
% Load 'digitStruct' to the matlab work space
load(strcat(folder,'digitStruct'))
%%
% Calculate the number of images. Be careful that the number of images
% recorded in 'digitStruct' is more than that contained in that folder.
L = round(length(digitStruct)/1);
% Set how many samples are available
n = 0;
for i = 1:L
    bs = digitStruct(i).bbox;
    n = n+length(bs);
end
n = n*2;
% The variable X_reg has a shape of [width,height,channels,sample_number]
X_rp = uint8(zeros(32,32,3,n));
% The variable y_reg has a shape of [1,sample_number].
y_rp = uint8(zeros(1,n));
pointer = 1;
%% Generate samples
for i = 1:L % for each image
    % Read the image.
    % The name of each image has been stored in digitStruct(i).name
%     if ~exist(strcat(folder,digitStruct(i).name),'file')
%         continue;
%     end
    img = imread(strcat(folder,digitStruct(i).name));
    [h,w,c] = size(img);
    image_box = [1,1,w,h];
    if isempty(img)
        return % Error protection
    end
    % Read the bounding box
    bs = digitStruct(i).bbox;
    len = length(bs);
    % rewrite the bbox as a matrix
    initial_boxes = zeros(4,len);
    for j=1:len
        initial_boxes(1,j) = bs(j).left;
        initial_boxes(2,j) = bs(j).top;
        initial_boxes(3,j) = bs(j).width;
        initial_boxes(4,j) = bs(j).height;
    end
    % calculate the reduced box (half the size)
    reduced_box = zeros(4,len);
    reduced_box(1,:) = round(initial_boxes(1,:) + initial_boxes(3,:)/4);
    reduced_box(2,:) = round(initial_boxes(2,:) + initial_boxes(4,:)/4);
    reduced_box(3,:) = round(initial_boxes(3,:)/2);
    reduced_box(4,:) = round(initial_boxes(4,:)/2);
    %generate centers of samples
    pos_points = zeros(2,len); % The center of positive samples
    for j=1:len % number
        pos_points(:,j) = point_gen_in_exclude_boxes( 1, reduced_box(:,j), [] );
    end
    neg_points = point_gen_in_exclude_boxes( len, image_box, reduced_box );
    % Calibrate the resolution of the image
    % Calculate the mean of the largest scalea among the height and the
    % width of the bounding boxes
    radius = round(max(mean(initial_boxes(3,:)),mean(initial_boxes(4,:)))/2);
    % Do padding
    padded_img = padarray(img,[radius*2,radius*2]);
    % Shifting the centers of samples
    pos_points = pos_points+radius*2;
    neg_points = neg_points+radius*2;
    % Generate samples and labels
    for j=1:len % Positive samples
        sample_box = [pos_points(1,j)-radius,pos_points(2,j)-radius,2*radius,2*radius];
        cropped = imcrop(padded_img,sample_box);
        resized = imresize(cropped, [32,32]);
        X_rp(:,:,:,pointer) = resized;
        y_rp(pointer) = 1;
        pointer = pointer+1;
    end
    for j=1:len % Negative samples
        sample_box = [neg_points(1,j)-radius,neg_points(2,j)-radius,2*radius,2*radius];
        cropped = imcrop(padded_img,sample_box);
        resized = imresize(cropped, [32,32]);
        X_rp(:,:,:,pointer) = resized;
        y_rp(pointer) = 0;
        pointer = pointer+1;
    end
    % Display the progress
    if mod(i,1000)==0
        disp(i)
    end
%     X_rp = X_rp(:,:,:,1:pointer-1);
%     y_rp = y_rp(:,:,:,1:pointer-1);
end
%% Save variables
save(store_file,'X_rp','y_rp','-v7.3');
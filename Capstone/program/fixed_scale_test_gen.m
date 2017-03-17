%% Initialization
clear
clc
 % Specify the folder containing the file 'digitStruct'. This file contains
 % the bounding boxes of each digit in each image
source_folder = 'test/';
target_folder = 'fixed_scale_test/';
% Load 'digitStruct' to the matlab work space
load(strcat(source_folder,'digitStruct'))
% Calculate the number of images. Be careful that the number of images
% recorded in 'digitStruct' is more than that contained in that folder.
L = length(digitStruct);
%% Read image and extract B.B. info.
numbers = int32(zeros(1,L));
for i = 1:L % for each image
    % Read the image.
    % The name of each image has been stored in digitStruct(i).name
    name = digitStruct(i).name;
     img = imread(strcat(source_folder,name));
     [img_height,img_width,~] = size(img);
     if isempty(img)
         break % Error protection
     end
    % Read the bounding box and the the numbers
    bs_struct = digitStruct(i).bbox;
    bs_matrix = zeros(length(bs_struct),4);
    rows = length(bs_struct);
    for r = 1:rows
        bs_matrix(r,1) = bs_struct(r).left;
        bs_matrix(r,2) = bs_struct(r).top;
        bs_matrix(r,3) = bs_struct(r).width;
        bs_matrix(r,4) = bs_struct(r).height;
        numbers(i) = numbers(i)*10+mod(bs_struct(r).label,10);
    end
     % calculate the average height of digit bounding boxes
     height_avg = mean(bs_matrix(:,4));
     % calculate the scaling factor
     scale_factor = 32/height_avg;
     % resize the image and the bounding boxes
     resized_img = imresize(img, scale_factor);
     bs_scaled = bs_matrix*scale_factor;
     % padding the array and modify the bounding boxes
     padding_size = 100;
     image_padded = padarray(resized_img,[padding_size,padding_size]);
     bs_padded = bs_scaled;
     bs_padded(:,1:2) = bs_scaled(:,1:2)+padding_size;
     % extract the final image
     x_center = (min(bs_padded(:,1))+max(bs_padded(:,1)+bs_padded(:,3)))/2;
     y_center = (min(bs_padded(:,2))+max(bs_padded(:,2)+bs_padded(:,4)))/2;
     cropped = imcrop(image_padded,[x_center-96,y_center-48,191,95]);
     imwrite(cropped,strcat(target_folder,name))
    if mod(i,1000) == 0
        disp(i)
    end
end 
%% save the labels
save(strcat(target_folder,'numbers'),'numbers','-v7.3');

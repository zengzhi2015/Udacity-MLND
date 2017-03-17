function [r_img, r_shift, r_box] = box2rbox(img, box)
%BOX2RBOX Summary of this function goes here
%   Extract a square from an 'img' according to a 'bounding box' with a random 'shift'
%   Resize the extracted square to the size of 32x32 pixels
%   inputs:
%       img: [width,height,channels] a color image
%       box: a structure with members: left, top, width, height, label
%   outputs:
%       r_image: [width,height,channels] extracted square that is resampled to 32x32 pixels
%       r_shift: [dx,dy] the center of the digit with respect to the center of the extracted square
%       r_box: [left,top,width,height] the bounding box of the digit

    % Calculate the center of the bounding box
    center = [box.left + round(box.width/2),box.top + round(box.height/2)];
    % Find out the largest scale among the height and the width of the bounding box
    radius = round(max(box.width,box.height)/2);
    % Randomly shift the center of the bounding box to a nearby region
    shift = [randi([-round(box.width/2),round(box.width/2)]),randi([-round(box.height/2),round(box.height/2)])];
    new_center = center-shift;
    % Extract a square region with 'new_center' as its center and 2*radius
    % as the length of its side.
    new_box = [new_center(1)-radius,new_center(2)-radius,2*radius,2*radius];
    % pad the image before cropping
    padded_img = padarray(img,[radius*2,radius*2]);
    cropped = imcrop(padded_img,[new_box(1)+radius*2,new_box(2)+radius*2,new_box(3),new_box(4)]);
    % Resize the extracted region to the size of 32x32 pixels.
    ratio = 32/(2*radius);
    % Record the extracted square region, the random shift, and the bounding box
    % of the digit in the extracted region.
    r_img = imresize(cropped, [32,32]);
    r_shift = shift*ratio;
    r_box = ratio*[box.left-new_box(1),box.top-new_box(2),box.width,box.height];
end


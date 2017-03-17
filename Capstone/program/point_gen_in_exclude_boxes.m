function points = point_gen_in_exclude_boxes( num_samples, box_in, boxes_exclude )
%POINT_GEN_IN_EXCLUDE_BOXES Summary of this function goes here
%   Generate points in box_in but not in boxes_exclude
    points = zeros(2,num_samples);
    counter = 0;
    while(counter<num_samples)
        x = box_in(1)+randi([0,box_in(3)]);
        y = box_in(2)+randi([0,box_in(4)]);
        if ~check_in_boxes( [x,y], boxes_exclude )
            counter = counter+1;
            points(:,counter) = [x,y]';
        end
    end
end


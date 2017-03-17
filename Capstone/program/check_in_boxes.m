function flag = check_in_boxes( point, boxes )
%CHECK_IN_BOXES Summary of this function goes here
%   Check whether the point is in one of the boxes
    flag = false;
    if ~isempty(boxes)
        [~,len] = size(boxes);
        for i=1:len
            if point(1)>=boxes(1,i) && point(1)<=boxes(1,i)+boxes(3,i) && point(2)>=boxes(2,i) && point(2)<=boxes(2,i)+boxes(4,i)
                flag = true;
                break;
            end
        end
    end
end


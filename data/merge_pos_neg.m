function [all_neg, pos_end] = merge_pos_neg(pos, neg)
% neg fields
%    im
%    flip
%    dataid
%    +boxes
% pos fields
%     im
%     flip
%     +dataid
%     boxes
%     -x1
%     -y1
%     -x2
%     -y2
%     -trunc
%     -dataids
%     -sizes

%pos = rmfield(pos, 'x1');
%pos = rmfield(pos, 'x2');
%pos = rmfield(pos, 'y1');
%pos = rmfield(pos, 'y2');
%pos = rmfield(pos, 'trunc');
pos = rmfield(pos, 'sizes');
pos = rmfield(pos, 'dataids');

% remove flipped examples (they are not currently cached)
is_flipped = find([pos(:).flip] == true);
pos(is_flipped) = [];

neg(1).boxes = [];

last_neg_dataid = max([neg(:).dataid]);
for i = 1:length(pos)
  pos(i).dataid = last_neg_dataid + i;
end

all_neg = cat(2, pos, neg);
pos_end = length(pos);

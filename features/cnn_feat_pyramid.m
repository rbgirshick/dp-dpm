function pyra = cnn_feat_pyramid(im, model, padx, pady)

num_levels = 7;

pyra = make_pyramid(im, model.cnn, num_levels);

imsize = [size(im, 1) size(im, 2)];
pyra.imsize = imsize;
pyra.num_levels = length(pyra.feat);

if nargin < 3
  [padx, pady] = getpadding(model);
end

for i = 1:pyra.num_levels
  pyra.feat{i} = padarray(pyra.feat{i}, [pady padx 0], 0) * 1/50;
end
pyra.valid_levels = true(pyra.num_levels, 1);
pyra.padx = padx;
pyra.pady = pady;

% ------------------------------------------------------------------------
function pyra = make_pyramid(im, cnn_model, num_levels)
% ------------------------------------------------------------------------
if cnn_model.init_key ~= caffe('get_init_key')
  error('You probably need to call rcnn_load_model');
end

batch_width = 1723;
batch_height = 1723;
sz_w = (batch_width-11)/16+1;
sz_h = sz_w;
%th = tic;
[batch, scales, level_sizes] = prepare_image(im, cnn_model, batch_height, batch_width, num_levels);
%fprintf('prep: %.3fs\n', toc(th));

%th = tic;
feat = caffe('forward', {batch});
feat_pyra = permute(reshape(feat{1}, [sz_w sz_h 256 num_levels]), [2 1 3 4]);
%fprintf('fwd: %.3fs\n', toc(th));

for i = 1:num_levels
  pyra.feat{i} = feat_pyra(1:level_sizes(i,1), 1:level_sizes(i,2), :, i);
  pyra.scales = scales;
end

% ------------------------------------------------------------------------
function [batch, scales, level_sizes] = prepare_image(im, cnn_model, batch_height, batch_width, num_levels)
% ------------------------------------------------------------------------
im = single(im);
% Convert to BGR
im = im(:,:,[3 2 1]);
% Subtract mean (mean of the image mean--one mean per channel)
im = bsxfun(@minus, im, cnn_model.mu);

im_sz = size(im);
if im_sz(1) > im_sz(2)
  height = batch_height-10;
  width = NaN;
  scale = height/im_sz(1);
else
  height = NaN;
  width = batch_width-10;
  scale = width/im_sz(2);
end
im_orig = im;

batch = zeros(batch_width, batch_height, 3, num_levels, 'single');
alpha = 2^(1/2);
scales = scale*(alpha.^-(0:num_levels-1))';
level_sizes = zeros(num_levels, 2);
for i = 0:num_levels-1
  if i == 0
    im = imresize(im_orig, [height width], 'bilinear');
  else
    im = imresize(im_orig, scales(i+1), 'bilinear');
  end
  im_sz = size(im);
  im_sz = im_sz(1:2);
  level_sizes(i+1, :) = ceil(im_sz / 16);
  % Make width the fastest dimension (for caffe)
  im = permute(im, [2 1 3]);
  batch(6:6+im_sz(2)-1, 6:6+im_sz(1)-1, :, i+1) = im;
end

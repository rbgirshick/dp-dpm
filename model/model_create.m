function m = model_create(cls, note)
% Create an empty object model.
%   m = model_create(cls, note)
%
% Return value
%   m       Object model
%
% Arguments
%   cls     Object class (e.g., 'bicycle')
%   note    A descriptive note (e.g., 'testing new features X, Y, and Z')

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2009-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

conf = voc_config();

if nargin < 2
  note = '';
end

m.class         = cls;                % object class/category
m.year          = conf.pascal.year;   % dataset year (PASCAL specific)
m.note          = note;               % decription of the model
m.filters       = [];                 % filters (terminals)
m.rules         = {};                 % rules
m.symbols       = [];                 % grammar symbol table
m.numfilters    = 0;                  % length(model.filters)
m.numblocks     = 0;                  % length(model.blocks)
m.numsymbols    = 0;                  % length(model.symbols)
m.start         = [];                 % grammar start symbol
m.maxsize       = -[inf inf];         % size of the largest detection window
m.minsize       = [inf inf];          % size of the smallest detection window
m.interval      = conf.eval.interval; % # levels in each feature pyramid octave
m.sbin          = conf.features.sbin; % pixel size of the HOG cells
m.thresh        = 0;                  % detection threshold
m.type          = model_types.MixStar;% default type is mixture of star models
m.blocks        = [];                 % struct array to store block data
m.features      = conf.features;      % info about image features
m.features.bias = conf.training.bias_feature; % feature value for bias/offset 
                                              % parameters

% Various training and testing stats
m.stats.slave_problem_time = [];  % time spent in slave problem optimization
m.stats.data_mining_time   = [];  % time spent in data mining
m.stats.pos_latent_time    = [];  % time spent in inference on positives
m.stats.filter_usage       = [];  % foreground training instances / filter

cnn_binary_file = './data/caffe_nets/ilsvrc_2012_train_iter_310k';
cnn_definition_file = './model-defs/rcnn_batch_7_output_pool5_plane2k.prototxt';

cnn.binary_file = cnn_binary_file;
cnn.definition_file = cnn_definition_file;
cnn.batch_size = 256;
cnn.init_key = -1;
cnn.input_size = 227;
% load the ilsvrc image mean
data_mean_file = './external/caffe/matlab/caffe/ilsvrc_2012_mean.mat';
assert(exist(data_mean_file, 'file') ~= 0);
ld = load(data_mean_file);
image_mean = ld.image_mean; clear ld;
off = floor((size(image_mean,1) - cnn.input_size)/2)+1;
image_mean = image_mean(off:off+cnn.input_size-1, off:off+cnn.input_size-1, :);
cnn.image_mean = image_mean;
mu = cnn.image_mean;
mu = sum(sum(mu, 1), 2) / size(mu, 1) / size(mu, 2);
cnn.mu = mu;

if 0
  cnn.init_key = ...
      caffe('init', cnn.definition_file, cnn.binary_file);
  caffe('set_mode_gpu');
  caffe('set_phase_test');
end

m.cnn = cnn;

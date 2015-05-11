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

if ~exist('note', 'var') || isempty(note)
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
cnn_definition_file = './model-defs/pyramid_cnn_output_max5_scales_7_plane_1713.prototxt';

cnn.binary_file = cnn_binary_file;
cnn.definition_file = cnn_definition_file;
cnn.batch_size = 256;
cnn.init_key = -1;
cnn.mu = get_channelwise_mean();
cnn.pyra.dimx = 1713;
cnn.pyra.dimy = 1713;
cnn.pyra.stride = 16;
cnn.pyra.num_levels = 7;
cnn.pyra.num_channels = 256;
cnn.pyra.scale_factor = sqrt(2);

m.cnn = cnn;

% ------------------------------------------------------------------------
function mu = get_channelwise_mean()
% ------------------------------------------------------------------------
% load the ilsvrc image mean
data_mean_file = 'caffe/matlab/caffe/ilsvrc_2012_mean.mat';
assert(exist(data_mean_file, 'file') ~= 0);
% input size business isn't likley necessary, but we're doing it
% to be consistent with previous experiments
ld = load(data_mean_file);
mu = ld.image_mean; clear ld;
input_size = 227;
off = floor((size(mu,1) - input_size)/2)+1;
mu = mu(off:off+input_size-1, off:off+input_size-1, :);
mu = sum(sum(mu, 1), 2) / size(mu, 1) / size(mu, 2);

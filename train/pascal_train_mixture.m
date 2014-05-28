function model = pascal_train_mixture(cls, n, note)
% Train a model.
%   model = pascal_train(cls, note)
%
% Trains a Dalal & Triggs model.
%
% Arguments
%   cls           Object class to train and evaluate
%                 (The final model has 2*n components)
%   note          Save a note in the model.note field that describes this model

% At every "checkpoint" in the training process we reset the 
% RNG's seed to a fixed value so that experimental results are 
% reproducible.
seed_rand();

% Default to no note
if nargin < 2
  note = '';
end

conf = voc_config();
cachedir = conf.paths.model_dir;

% Load the training data
[pos, neg, impos] = pascal_data(cls, conf.pascal.year);

% Split foreground examples into n groups by aspect ratio
spos = split(pos, n);

max_num_examples = conf.training.cache_example_limit;
num_fp           = conf.training.wlssvm_M;
fg_overlap       = conf.training.fg_overlap;

% Select a small, random subset of negative images
% All data mining iterations use this subset, except in a final
% round of data mining where the model is exposed to all negative
% images
num_neg   = length(neg);
neg_large = neg; % use all of the negative images
neg_perm  = neg(randperm(num_neg));
neg_small = neg_perm(1:min(num_neg, conf.training.num_negatives_small));

impos_with_difficult = pascal_data_diff(cls, conf.pascal.year);
neg_all = merge_pos_neg(impos_with_difficult, neg);

for i = 1:n
  models{i} = root_model(cls, spos{i}, note);
end
model = model_merge(models);

save_file = [cachedir cls '_hard_neg1'];
try
  load(save_file);
  fprintf('Loaded %s\n', save_file);
catch
  model = model_cnn_init(model);
  model = train(model, impos, neg_small, true, false, 1, 10, ...
                max_num_examples, fg_overlap, num_fp, false, 'hard_neg1');
  save(save_file);
end

save_file = [cachedir cls '_final'];
try
  load(save_file);
  fprintf('Loaded %s\n', save_file);
catch
  model = model_cnn_init(model);
  model = train(model, impos, neg_all, false, false, 1, 10, ...
                max_num_examples, fg_overlap, num_fp, false, 'hard_neg2');
  save(save_file);
end



% ------------------------------------------------------------------------
function all_neg = merge_pos_neg(pos, neg)
% ------------------------------------------------------------------------
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

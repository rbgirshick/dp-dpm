function model = pascal_train_d_and_t(cls, note)
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

%pos = pos(1:10);

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
%neg_small = pos;

%for i = 1:length(pos)
%  neg_small(i).dataid = neg_small(i).dataids + 100000;
%end

%pos = pos(1:10);

save_file = [cachedir cls '_final'];
try
  load(save_file);
catch
  model = root_model(cls, pos, note);
  % Get warped positives and random negatives
  model = train(model, impos, neg_small, true, true, 1, 1, ...
                max_num_examples, fg_overlap, 0, false, 'init');
  % Finish training by data mining on all of the negative images
  model = train(model, impos, neg_small, false, false, 1, 10, ...
                max_num_examples, fg_overlap, num_fp, true, 'hard_neg');
  model = train(model, impos, neg, false, false, 1, 10, ...
                max_num_examples, fg_overlap, num_fp, false, 'hard_neg');

  save(save_file);
end

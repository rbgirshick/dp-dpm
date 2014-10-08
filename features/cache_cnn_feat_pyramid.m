function cache_cnn_feat_pyramid(year, outdir)
% cache_cnn_feat_pyramid('2007', '/data/caches/dp-dpm');

conf = voc_config('pascal.year', year);
VOCopts  = conf.pascal.VOCopts;
cachedir = conf.paths.model_dir;

ids = textread(sprintf(VOCopts.imgsetpath, 'trainval'), '%s');
num_in_trainval = length(ids);
ids = cat(1, ids, textread(sprintf(VOCopts.imgsetpath, 'test'), '%s'));

model = model_create('dummy', [], true);

for i = 1:length(ids)
  fprintf('%d/%d\n', i, length(ids));

  cache_file = [outdir '/VOC' year '/' ids{i} '.mat'];
  if ~exist(cache_file, 'file')
    th = tic();
    im = imread(sprintf(VOCopts.imgpath, ids{i}));
    pyra = deep_pyramid(im, model.cnn);
    fprintf('  [pyra:   %.3fs]\n', toc(th));
    th = tic();
    save(cache_file, 'pyra');
    fprintf('  [saving: %.3fs]\n', toc(th));
  else
    fprintf('  [already exists]\n');
  end

  if i <= num_in_trainval
    cache_file = [outdir '/VOC' year '/' ids{i} '_flipped.mat'];
    if ~exist(cache_file, 'file')
      th = tic();
      im = imread(sprintf(VOCopts.imgpath, ids{i}));
      im = im(:, end:-1:1, :);
      pyra = deep_pyramid(im, model.cnn);
      fprintf('  [pyra (flipped):   %.3fs]\n', toc(th));
      th = tic();
      save(cache_file, 'pyra');
      fprintf('  [saving (flipped): %.3fs]\n', toc(th));
    else
      fprintf('  [already exists (flipped)]\n');
    end
  end
end

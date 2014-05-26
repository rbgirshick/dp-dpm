function cache_cnn_feat_pyramid(year, outdir)
% cache_cnn_feat_pyramid('2007', '/data1/cnn_feat_pyra_cache');

conf = voc_config('pascal.year', year);
VOCopts  = conf.pascal.VOCopts;
cachedir = conf.paths.model_dir;

ids = textread(sprintf(VOCopts.imgsetpath, 'trainval'), '%s');
ids = cat(1, ids, textread(sprintf(VOCopts.imgsetpath, 'test'), '%s'));

padx = 10;
pady = 10;
model = model_create('dummy');

for i = 1:length(ids)
  fprintf('%d/%d\n', i, length(ids));
  cache_file = [outdir '/VOC' year '/' ids{i} '.mat'];
  if ~exist(cache_file, 'file')
    th = tic();
    im = imread(sprintf(VOCopts.imgpath, ids{i}));  
    pyra = cnn_feat_pyramid(im, model, padx, pady);
    fprintf('  [pyra:   %.3fs]\n', toc(th));
    th = tic();
    save(cache_file, 'pyra');
    fprintf('  [saving: %.3fs]\n', toc(th));
  else
    fprintf('  [already exists]\n');
  end
end

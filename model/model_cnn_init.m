function m = model_cnn_init(m, use_gpu, use_caffe)

if ~exist('use_gpu', 'var') || isempty(use_gpu)
  use_gpu = true;
end

if ~exist('use_caffe', 'var') || isempty(use_caffe)
  use_caffe = true;
end

if use_caffe
  m.cnn.init_key = ...
      caffe('init', m.cnn.definition_file, m.cnn.binary_file, 'test');
  if use_gpu
    caffe('set_mode_gpu');
  else
    caffe('set_mode_cpu');
  end
end

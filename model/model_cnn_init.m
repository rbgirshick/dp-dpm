function m = model_cnn_init(m, use_gpu)

if ~exist('use_gpu', 'var') || isempty(use_gpu)
  use_gpu = true;
end

m.cnn.init_key = ...
    caffe('init', m.cnn.definition_file, m.cnn.binary_file);
if use_gpu
  caffe('set_mode_gpu');
else
  caffe('set_mode_cpu');
end
caffe('set_phase_test');

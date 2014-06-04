function vis_feat_pyra(cls, impos, ch)

% for i=1:2:length(impos); vis_feat_pyra(imreadx(impos(i)), 165); title(num2str(i)); pause; end

%run ~/local_matlab/vlfeat-0.9.13/toolbox/vl_setup.m;
addpath ~/local_matlab/export_fig/;

model = model_create('dummy');
if 1
  cnn_definition_file = './model-defs/rcnn_batch_7_output_conv5_plane2k.prototxt';
  model.cnn.definition_file = cnn_definition_file;
end
model = model_cnn_init(model, true, true);

for i=1:length(impos); 
  do_vis(imreadx(impos(i)), ch, model, []);

  fprintf('%d > ', i);
  [~,~,keycode] = ginput(1);
  disp(keycode);
  switch keycode
    case 's'
      [~, image_id, ~] = fileparts(impos(i).im);
      filename = sprintf('./vis/output/%s-%s-%d-%%d.png', cls, image_id, ch);
      do_vis(imreadx(impos(i)), ch, model, filename);
      %export_fig(filename);
      fprintf('saved!\n');
    otherwise
      fprintf('\n');
  end
end



function do_vis(im, ch, model, filename)

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 16);
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultTextFontSize', 16);

do_snap = true;
if isempty(filename)
  do_snap = false;
end

cm = load('greenmap.mat');
colormap(cm.map);

padx = 0;
pady = 0;

pyra = cnn_feat_pyramid(im, model, padx, pady);

clf;
if ~do_snap
  %ha = tight_subplot(1,8, [.01 .01],[.0 .0],[.0 .0]);
end

V = [12 0 3.65 2];

%vl_tightsubplot(1,8,1, 'Margin', 10);
if ~do_snap
  subplot(1,8,1);
  %axes(ha(1));
else
  set(gcf, 'Units', 'inches');
  set(gcf, 'Position', V);
end
imagesc(im); 
axis image;
axis off;

xlabel('pixels');
if ~do_snap
  %set(ha(1), 'YTickLabel', []);
  set(gca, 'YTickLabel', []);
else
  set(gca, 'YTickLabel', []);
end
%title('image');
if do_snap
  %print(sprintf(filename, 1), '-dpng');
  export_fig(sprintf(filename, 1));
  pause(1);
end

maxv = 0;
for i = 1:7
  feat_im = pyra.feat{i}(:,:,ch);
  maxv = max(maxv, max(feat_im(:)));
end
maxv = 3;

for i = 1:7
  if ~do_snap
    %axes(ha(i+1));
    subplot(1,8,i+1);
  else
    clf;
  end
  feat_im = pyra.feat{i}(:,:,ch);
  %imagesc(min(1, repmat(feat_im, [1 1 3])/maxv));%, [0 maxv]);
  imagesc(feat_im, [0 maxv]);
  axis image;
  xlabel('conv5 cells');
  set(gca, 'YTickLabel', []);
  %title(sprintf('level %d', i));
  axis off;
  set(gcf, 'Color', 'white')
  if do_snap
    set(gcf, 'Units', 'inches');
    set(gcf, 'Position', V);
    export_fig(sprintf(filename, i+1));
    %print(sprintf(filename, i+1), '-dpng');
    pause(1);
  end
  if max(feat_im(:)) > maxv
    warning('max violations %0.4f', max(feat_im(:)));
  end
end

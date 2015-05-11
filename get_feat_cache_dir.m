function FEAT_CACHE_DIR = get_feat_cache_dir()

FEAT_CACHE_DIR = 'cachedir/pyra_cache/VOC2007';
if exist(FEAT_CACHE_DIR) == 0
  unix(['mkdir -p ' FEAT_CACHE_DIR]);
end

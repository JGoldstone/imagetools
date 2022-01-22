from pathlib import Path
from plotlyflask_tutorial.plotlydash.model.image_stats_gatherers.stats import ChannelStats, Color, Stats

class PhotositeStatGenerator(object):
    def __init__(self, img_path: Path, origin, filter_config):
        if img_path.suffix != '.exr':
            raise RuntimeError(f'photosite file suffix must be .exr but suffix for supplied file was {img_path.suffix}')
        if origin.lower() not in {}

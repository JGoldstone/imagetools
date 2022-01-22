from OpenImageIO import ImageBuf, ImageBufAlgo, ROI
from stats import ChannelStats, Color, Stats

class RgbStats(object):



    def __init__(self, path, left, top, width, height, clipping_threshold=1.0):
        # Coordinate system is presumed to be anchored at top left
        orig_img_buf = ImageBuf(path.str())
        img_buf = ImageBufAlgo.cut(orig_img_buf, ROI(left, left+width, top, top+height))
        s = ImageBufAlgo.computePixelStats(img_buf)
        self.stats = Stats()
        for i, c in enumerate(Color):
            self.stats.channels[c] = ChannelStats(s.avg[i], s.stddev[i], s.min_val[i], s.max_val[i])
            for y in range(height):
                for x in range(width):
                    px = img_buf.getpixel(x, y)
                    if px[i] >= clipping_threshold:
                        self.stats.channels[c].num_clipped = self.stats.channels[c].num_clipped + 1


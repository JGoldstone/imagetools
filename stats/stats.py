from enum import Enum

class ChannelStats(object):
    def __init__(self, average = None, std_dev = None, min_val=None, max_val=None, num_clipped=None):
        self._average = None
        self._std_dev = None
        self._min_val = None
        self._max_val = None
        self._num_clipped = None

    @property
    def average(self):
        return self._average

    @average.setter
    def average(self, value):
        self._average = value

    @property
    def std_dev(self):
        return self._std_dev

    @std_dev.setter
    def std_dev(self, value):
        self._std_dev = value

    @property
    def min_val(self):
        return self._min_val

    @min_val.setter
    def min_val(self, value):
        self._min_val = value

    @property
    def max_val(self):
        return self._max_val

    @max_val.setter
    def max_val(self, value):
        self._max_val = value

    @property
    def num_clipped(self):
        return self._num_clipped

    @num_clipped.setter
    def num_clipped(self, value):
        self._num_clipped = value


class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


class Stats(object):
    def __init__(self):
        self._channels = {Color.RED: None, Color.GREEN: None, Color.BLUE: None}

    @property
    def channels(self):
        return self._channels

class CircularList:
    _contents: list
    _max_size: int
    _current_idx: int

    def __init__(self, max_size):
        self._contents = [None] * max_size
        self._max_size = max_size
        self._current_idx = 0

    def add(self, item: object):
        self._contents[self._current_idx] = item
        self._current_idx = (self._current_idx + 1) % self._max_size

    def __contains__(self, item):
        return any(map(lambda c: c == item, self._contents))

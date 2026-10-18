import shutil
import sys
import time


class ProgressBar:
    """Single-line text progress bar:  IK  walking  [********------------]  40%  5200/13000  12s

    The line is cut to the terminal width, so it never wraps (a wrapped line makes '\\r'
    redraw on a new row each time and floods the terminal).
    """

    def __init__(self, total, label='', width=40):
        self.total = max(int(total), 1)
        self.label = label
        self.width = width
        self.start = time.time()
        self._last_pct = -1

    def update(self, done, note=''):
        pct = int(100 * done / self.total)
        if pct == self._last_pct and done < self.total:
            return
        self._last_pct = pct
        cols = shutil.get_terminal_size((100, 20)).columns
        elapsed = time.time() - self.start
        tail = f'] {pct:3d}%  {done}/{self.total}  {elapsed:.0f}s'
        prefix = f'  {self.label[:28]:<28} ['
        # shrink the bar if the terminal is narrow
        width = max(10, min(self.width, cols - 1 - len(prefix) - len(tail)))
        filled = int(width * done / self.total)
        line = prefix + '*' * filled + '-' * (width - filled) + tail
        if note:
            line += f'  {note}'
        sys.stdout.write('\r' + line[:cols - 1].ljust(cols - 1))
        sys.stdout.flush()

    def finish(self, note=''):
        self._last_pct = -1
        self.update(self.total, note)
        sys.stdout.write('\n')
        sys.stdout.flush()

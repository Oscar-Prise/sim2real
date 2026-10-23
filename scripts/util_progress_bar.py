import shutil
import sys
import time


class ProgressBar:
    """Single-line text progress bar:  IK  walking  [********------------]  40%  5200/13000  12s

    The line must never wrap: a wrapped line makes every '\\r' redraw land on a new row and
    floods the terminal. When stdout is not a real console the width can't be detected, so
    the line is kept under 80 columns and only padded to the length of the previous draw.
    """

    MAX_COLS = 80

    def __init__(self, total, label='', width=20):
        self.total = max(int(total), 1)
        self.label = label
        self.width = width
        self.start = time.time()
        self._last_pct = -1
        self._last_len = 0

    def _cols(self):
        if sys.stdout.isatty():
            return min(shutil.get_terminal_size((self.MAX_COLS, 20)).columns, self.MAX_COLS)
        return self.MAX_COLS

    def update(self, done, note=''):
        pct = int(100 * done / self.total)
        if pct == self._last_pct and done < self.total:
            return
        self._last_pct = pct
        cols = self._cols() - 1
        elapsed = time.time() - self.start
        tail = f'] {pct:3d}%  {done}/{self.total}  {elapsed:.0f}s'
        prefix = f'  {self.label[:28]:<28} ['
        width = max(5, min(self.width, cols - len(prefix) - len(tail)))
        filled = int(width * done / self.total)
        line = prefix + '*' * filled + '-' * (width - filled) + tail
        if note:
            line += f'  {note}'
        line = line[:cols]
        pad = max(0, self._last_len - len(line))
        self._last_len = len(line)
        sys.stdout.write('\r' + line + ' ' * pad)
        sys.stdout.flush()

    def finish(self, note=''):
        self._last_pct = -1
        self.update(self.total)
        sys.stdout.write('\n')
        if note:
            sys.stdout.write(f'  {note}\n')
        sys.stdout.flush()

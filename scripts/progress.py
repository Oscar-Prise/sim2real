import sys
import time


class ProgressBar:
    """Single-line text progress bar:  IK  walking  [********------------]  40%  5200/13000  12s"""

    def __init__(self, total, label='', width=40):
        self.total = max(int(total), 1)
        self.label = label
        self.width = width
        self.start = time.time()
        self._last_pct = -1

    def update(self, done):
        pct = int(100 * done / self.total)
        if pct == self._last_pct and done < self.total:
            return
        self._last_pct = pct
        filled = int(self.width * done / self.total)
        bar = '*' * filled + '-' * (self.width - filled)
        elapsed = time.time() - self.start
        sys.stdout.write(f'\r  {self.label:<28} [{bar}] {pct:3d}%  {done}/{self.total}  {elapsed:4.0f}s')
        sys.stdout.flush()

    def finish(self, note=''):
        self.update(self.total)
        sys.stdout.write(f'  {note}\n' if note else '\n')
        sys.stdout.flush()

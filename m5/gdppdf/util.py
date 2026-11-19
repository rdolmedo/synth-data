"""Small shared helpers."""
import json
import os
import time


class Appender:
    """Append JSON lines durably: each write is flushed and fsynced; an OSError (a node losing /data writes, e.g.
    gpu137 2026-11-12) reopens the file and retries, then raises after `tries`."""

    def __init__(self, path, tries=6):
        self.path, self.tries = str(path), tries
        self.f = open(self.path, 'a')

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False) + '\n'
        for i in range(self.tries):
            try:
                self.f.write(line)
                self.f.flush()
                os.fsync(self.f.fileno())
                return
            except OSError as e:
                print(f'[appender] {self.path}: {e}; retry {i + 1}', flush=True)
                time.sleep(10 * (i + 1))
                try:
                    self.f.close()
                except OSError:
                    pass
                self.f = open(self.path, 'a')
        raise OSError(f'{self.path}: writes keep failing')

    def close(self):
        try:
            self.f.close()
        except OSError:
            pass

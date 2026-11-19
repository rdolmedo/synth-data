"""One append-only wire log per attempt instead of a directory per model call.

Earlier campaigns archived every request as request.json/meta.json/response.sse/
complete.json (~4 inodes per call, ~260 files per trajectory) and exhausted the
/data inode quota. Here every call appends JSON lines to <logs>/wire.jsonl:

  {"t":"req","id":n,"time":...,"backend":...,"path":...,"body":{...}}   full served request
  {"t":"data","id":n,"b64":"..."}       raw response bytes, coalesced (<=64 KiB or 1 s)
  {"t":"end","id":n,"time":...,"status":200,"bytes":N}
  {"t":"error","id":n,"time":...,"status":S,"b64":"..."}

Each record is one O_APPEND write. Streamed bytes reach the file before Pi has
the whole response, so a killed worker still leaves the unfinished stream.
"""
import base64
import json
import os
import time

FLUSH_BYTES = 64 << 10
FLUSH_SECONDS = 1.0


class WireLog:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        self.seq = 0
        self.pending = {}

    def _write(self, record):
        # a short write continues; a write failing after part of the record went out ends it with a newline first,
        # so the next record starts on its own line (a malformed line is reported by load(), not glued on, #451)
        data = (json.dumps(record, separators=(',', ':'), ensure_ascii=False) + '\n').encode()
        view, sent = memoryview(data), 0
        try:
            while sent < len(data):
                sent += os.write(self.fd, view[sent:])
        except OSError:
            if sent:
                try:
                    os.write(self.fd, b'\n')
                except OSError:
                    pass
            raise

    def request(self, data, backend, path):
        self.seq += 1
        try:
            body = json.loads(data)
            record = {'t': 'req', 'id': self.seq, 'time': time.time(), 'backend': backend, 'path': path, 'body': body}
        except ValueError:
            record = {'t': 'req', 'id': self.seq, 'time': time.time(), 'backend': backend, 'path': path,
                      'raw_b64': base64.b64encode(data).decode()}
        self._write(record)
        self.pending[self.seq] = [bytearray(), time.monotonic(), 0]
        return self.seq

    def data(self, seq, chunk):
        buffer = self.pending[seq]
        buffer[0] += chunk
        buffer[2] += len(chunk)
        if len(buffer[0]) >= FLUSH_BYTES or time.monotonic() - buffer[1] >= FLUSH_SECONDS:
            self._flush(seq)

    def _flush(self, seq):
        buffer = self.pending[seq]
        if buffer[0]:
            self._write({'t': 'data', 'id': seq, 'b64': base64.b64encode(bytes(buffer[0])).decode()})
            buffer[0] = bytearray()
        buffer[1] = time.monotonic()

    def end(self, seq, status):
        if seq not in self.pending:
            return
        self._flush(seq)
        self._write({'t': 'end', 'id': seq, 'time': time.time(), 'status': status, 'bytes': self.pending.pop(seq)[2]})

    def error(self, seq, status, body):
        self.pending.pop(seq, None)
        self._write({'t': 'error', 'id': seq, 'time': time.time(), 'status': status,
                     'b64': base64.b64encode(body or b'').decode()})

    def close(self):
        for seq in list(self.pending):
            self._flush(seq)
        os.close(self.fd)


def load(path):
    """Requests in order with their (possibly unfinished) response bytes."""
    calls = {}
    with open(path, 'rb') as f:
        lines = f.read().split(b'\n')
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            if number == len(lines):
                break  # torn final line of a live or killed writer
            raise ValueError(f'Malformed wire log line {number}: {path}')
        seq = record['id']
        if record['t'] == 'req':
            calls[seq] = {'id': seq, 'time': record['time'], 'backend': record.get('backend'),
                          'path': record.get('path'), 'body': record.get('body'),
                          'raw': base64.b64decode(record['raw_b64']) if 'raw_b64' in record else None,
                          'response': bytearray(), 'status': None, 'complete': False}
        elif record['t'] == 'data':
            calls[seq]['response'] += base64.b64decode(record['b64'])
        elif record['t'] == 'end':
            calls[seq].update(status=record['status'], complete=True)
        elif record['t'] == 'error':
            calls[seq].update(status=record['status'], complete=True, error=base64.b64decode(record['b64']))
    return [calls[k] for k in sorted(calls)]


def sse_events(data):
    """Decode `data:` JSON events of an OpenAI-style stream (LF-delimited)."""
    events = []
    for line in bytes(data).decode('utf-8', errors='replace').split('\n'):
        if line.startswith('data:'):
            try:
                events.append(json.loads(line[5:].strip()))
            except ValueError:
                pass
    return events

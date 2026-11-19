"""A single-turn client's side of the #344 dispatcher: start credits from the dispatcher, longest attempts first.

    lane, pred, rec = open_dispatch(dispatch_dir, suite='hle', run=run_dir)   # registers DIR/lanes/<suite>.<run>/
    lane.expect([(key, item, pred), ...])                        # optional: work that will ask for a slot later
    async with lane.aslot(key, item, pred):                      # asyncio clients (HLE, GDP.pdf, SciCode)
        ...
    with lane.slot(key, item, pred):                             # thread clients (Omniscience)
        ...
    lane.close()

One slot = one conversation in flight on the server (an HLE / Omniscience / GDP.pdf request, a SciCode chain); a
GDP.pdf task takes one slot per repeat (`weight`).

The allowance is a cumulative **start credit**, DIR/lanes/<name>/control.json `max_starts`, written by the dispatcher
(0 until it writes one): the lane may start an attempt only while starts so far + its weight <= the credit. The credit
only grows, except when control.json says `exact` (#378): the dispatcher then withdraws unspent credit while a Pi lane
waits on the serving gate (lane.json `shrinkable` tells it this client honours that). A finished
attempt does not free a slot for this lane: it returns to the dispatcher, which hands it to the longest pending attempt
of any suite on its next tick. Waiting attempts start longest prediction first (ties: arrival order).
A poller thread re-reads the credit every `poll` seconds and publishes the lane's state for the dispatcher:
- DIR/lanes/<name>/lane.json: in flight, starts, credit, the pending attempts in start order (key, item, pred, weight);
- DIR/lanes/<name>/events.jsonl: one line per attempt start and end (epoch seconds, prediction, status).
Attempts in flight are never cut.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import os
import socket
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

POLL_SEC = 5.0
PENDING_REPORT = 5000    # pending attempts published per lane (start order); the count is always exact
LANE_FRESH = 120.0       # a lane.json younger than this belongs to a live client
CONTROL_STALE = 600.0    # waiting with no credit update for this long: warn (dispatcher down?)
WARN_EVERY = 300.0


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{socket.gethostname()}.{os.getpid()}.{threading.get_ident()}')
    tmp.write_text(json.dumps(value, separators=(',', ':')))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def lane_name(suite: str, run) -> str:
    """One lane per run: <suite>.<run dir name>."""
    return f'{suite}.{Path(run).name}'


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, TypeError, ValueError):
        return True


def owned_elsewhere(st: dict | None) -> bool:
    """A lane.json written by another live client: fresh, not closed, and (on this host) its pid alive."""
    if not st or st.get('closed') or time.time() - float(st.get('epoch') or 0) >= LANE_FRESH:
        return False
    if (st.get('pid'), st.get('host')) == (os.getpid(), socket.gethostname()):
        return False
    if st.get('host') == socket.gethostname():
        return pid_alive(st.get('pid'))
    return True


class _Waiter:
    __slots__ = ('key', 'item', 'pred', 'weight', 'wake', 'granted')

    def __init__(self, key, item, pred, weight, wake):
        self.key, self.item, self.pred, self.weight, self.wake = key, item, pred, weight, wake
        self.granted = False


class Lane:
    def __init__(self, lane_dir: Path, name: str, suite: str, *, poll: float = POLL_SEC, start_poller: bool = True):
        self.dir, self.name, self.suite, self.poll = Path(lane_dir), name, suite, poll
        self.lock = threading.Lock()
        self.flush_lock = threading.Lock()
        self.credit = 0           # cumulative starts allowed (weight units)
        self.starts = 0           # cumulative starts made (weight units)
        self.in_flight = 0
        self.heap: list = []      # (-pred, seq, waiter)
        self.expected: dict = {}  # key -> (item, pred, weight): announced, not yet waiting or started
        self.running: dict = {}   # key -> (item, pred, weight, start epoch)
        self.seq = itertools.count()
        self.done = 0
        self.closed = False
        self.announced = False
        # server release (#531 B1): set while CPU work in progress will send the model more work (e.g. a chain's next
        # step after grading), so the dispatcher does not release the server while nothing is pending or in flight
        self.hold_server = False
        self.control_epoch = None
        self.control_seen = time.time()
        self.last_warn = 0.0
        self._events: list[dict] = []
        self._stop = threading.Event()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._poller = None
        if start_poller:
            self.refresh()
            self._poller = threading.Thread(target=self._loop, name=f'lane-{name}', daemon=True)
            self._poller.start()

    # ------------------------------------------------------------------ setup
    @classmethod
    def open(cls, dispatch_dir: str | Path, suite: str, run: str | Path, name: str | None = None, **kw) -> 'Lane':
        """Register this run's lane under DIR/lanes/<name>/ (default <suite>.<run dir name>) and start polling.
        Refuses a lane that another live process (fresh lane.json, not closed) still owns."""
        name = name or lane_name(suite, run)
        lane_dir = Path(dispatch_dir) / 'lanes' / name
        deadline = time.time() + LANE_FRESH + 10
        while owned_elsewhere(st := read_json(lane_dir / 'lane.json')):
            # this host: the owner's pid is alive, so it is live; another host: wait until its lane.json is stale
            if st.get('host') == socket.gethostname() or time.time() > deadline:
                raise SystemExit(f'dispatch lane {name} is owned by a live client (pid {st.get("pid")} on '
                                 f'{st.get("host")}); stop it or pass another lane name')
            time.sleep(5)   # a client on another host that just died: its lane.json goes stale within LANE_FRESH
        atomic_json(lane_dir / 'registration.json', {
            'name': name, 'suite': suite, 'kind': 'lane', 'run': str(run), 'host': socket.gethostname(),
            'pid': os.getpid(), 'registered_epoch': time.time()})
        lane = cls(lane_dir, name, suite, start_poller=False, **{k: v for k, v in kw.items() if k != 'start_poller'})
        ctl = read_json(lane_dir / 'control.json', {}) or {}
        # a resumed client keeps counting from the credit already used, so old grants are not re-spent
        lane.starts = lane.credit = int(ctl.get('max_starts') or 0)
        if kw.get('start_poller', True):
            lane.refresh()
            lane._poller = threading.Thread(target=lane._loop, name=f'lane-{name}', daemon=True)
            lane._poller.start()
        return lane

    def expect(self, work) -> None:
        """Announce attempts that will ask for a slot later (so the dispatcher sees them pending):
        (key, item, pred) or (key, item, pred, weight)."""
        with self.lock:
            self.announced = True   # the dispatcher's start-up release waits for this (the lane's work is known)
            for w in work:
                key, item, pred = w[:3]
                if key not in self.running:
                    self.expected[str(key)] = (str(item), float(pred), int(w[3]) if len(w) > 3 else 1)

    # ------------------------------------------------------------------ credits
    def set_credit(self, credit: int, exact: bool = False) -> None:
        with self.lock:
            # credits only grow, unless the dispatcher marks its value exact (#378: it withdraws unspent credit while a
            # pi lane waits on the serving gate; a credit below the starts made just stops new starts)
            self.credit = int(credit) if exact else max(self.credit, int(credit))
            self._grant()

    def _grant(self) -> None:
        """Start waiters in priority order while their weight fits the credit (lock held). Strict priority."""
        while self.heap:
            _, _, w = self.heap[0]
            if self.starts + w.weight > self.credit:
                return
            heapq.heappop(self.heap)
            self._start(w)
            w.wake()

    def _start(self, w: _Waiter) -> None:
        w.granted = True
        self.in_flight += w.weight
        self.starts += w.weight
        now = time.time()
        self.running[w.key] = (w.item, w.pred, w.weight, now)
        self.expected.pop(w.key, None)
        self._events.append({'ev': 'start', 'key': w.key, 'item': w.item, 'pred': w.pred, 'weight': w.weight,
                             't': round(now, 2)})

    def _end(self, key: str, status: str) -> None:
        with self.lock:
            item, pred, weight, t0 = self.running.pop(key)
            self.in_flight -= weight
            self.done += 1
            now = time.time()
            self._events.append({'ev': 'end', 'key': key, 'item': item, 'pred': pred, 'weight': weight,
                                 't': round(now, 2), 'sec': round(now - t0, 2), 'status': status})

    def _enqueue(self, key, item, pred, weight, wake) -> _Waiter:
        w = _Waiter(str(key), str(item), float(pred), max(1, int(weight)), wake)
        with self.lock:
            if w.key in self.running:
                raise ValueError(f'attempt {w.key} is already in flight')
            self.expected.pop(w.key, None)
            heapq.heappush(self.heap, (-w.pred, next(self.seq), w))
            self._grant()
        return w

    def _cancel(self, w: _Waiter) -> bool:
        """Withdraw a waiter (its task was cancelled). True when it had already been granted a slot."""
        with self.lock:
            if w.granted:
                return True
            self.heap = [x for x in self.heap if x[2] is not w]
            heapq.heapify(self.heap)
            self._grant()
            return False

    @asynccontextmanager
    async def aslot(self, key, item, pred, weight: int = 1):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        def wake():
            loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(True))
        w = self._enqueue(key, item, pred, weight, wake)
        try:
            await fut
        except BaseException:
            if self._cancel(w):
                self._end(w.key, 'cancelled')
            raise
        status = 'ok'
        try:
            yield
        except BaseException as e:
            status = 'cancelled' if isinstance(e, asyncio.CancelledError) else f'error:{type(e).__name__}'
            raise
        finally:
            self._end(w.key, status)

    @contextmanager
    def slot(self, key, item, pred, weight: int = 1):
        ev = threading.Event()
        w = self._enqueue(key, item, pred, weight, ev.set)
        try:
            while not ev.wait(1.0):
                if self.closed:
                    raise RuntimeError('lane closed')
        except BaseException:
            if self._cancel(w):
                self._end(w.key, 'cancelled')
            raise
        status = 'ok'
        try:
            yield
        except BaseException as e:
            status = f'error:{type(e).__name__}'
            raise
        finally:
            self._end(w.key, status)

    # ------------------------------------------------------------------ publishing
    def pending(self) -> list:
        """Pending attempts in the order this lane starts them: waiting by prediction, then announced ones."""
        with self.lock:
            rows = [(w.pred, -s, w.key, w.item, w.weight) for p, s, w in self.heap]
            rows += [(pred, float('-inf'), k, item, wt) for k, (item, pred, wt) in self.expected.items()]
        rows.sort(reverse=True)
        return [[k, item, pred, wt] for pred, _, k, item, wt in rows]

    def state(self) -> dict:
        pend = self.pending()
        with self.lock:
            return {'name': self.name, 'suite': self.suite, 'epoch': round(time.time(), 2), 'credit': self.credit,
                    'starts': self.starts, 'in_flight': self.in_flight, 'n_running': len(self.running),
                    'done': self.done, 'n_pending': len(pend), 'pending': pend[:PENDING_REPORT],
                    'closed': self.closed, 'announced': self.announced or bool(pend) or bool(self.running),
                    'control_epoch': self.control_epoch, 'pid': os.getpid(), 'shrinkable': True,
                    'host': socket.gethostname(), 'hold_server': bool(self.hold_server)}

    def refresh(self) -> None:
        ctl = read_json(self.dir / 'control.json')
        if isinstance(ctl, dict) and 'max_starts' in ctl:
            if ctl.get('epoch') != self.control_epoch:
                self.control_epoch, self.control_seen = ctl.get('epoch'), time.time()
            self.set_credit(int(ctl['max_starts']), exact=bool(ctl.get('exact')))
        with self.lock:
            waiting = bool(self.heap)
            blocked = waiting and self.starts + self.heap[0][2].weight > self.credit
        now = time.time()
        if blocked and now - self.control_seen > CONTROL_STALE and now - self.last_warn > WARN_EVERY:
            self.last_warn = now
            print(f'[lane {self.name}] WARNING: {len(self.heap)} attempt(s) waiting and no new start credit for '
                  f'{(now - self.control_seen) / 60:.0f} min (is the dispatcher running? {self.dir.parent.parent})',
                  flush=True)
        self.flush()

    def flush(self) -> None:
        with self.flush_lock:   # the poller and close() may both flush
            with self.lock:
                events, self._events = self._events, []
            if events:
                with open(self.dir / 'events.jsonl', 'a') as f:
                    f.write(''.join(json.dumps(e, separators=(',', ':')) + '\n' for e in events))
            atomic_json(self.dir / 'lane.json', self.state())   # every poll: also the lane's heartbeat

    def _loop(self) -> None:
        while not self._stop.wait(self.poll):
            try:
                self.refresh()
            except Exception as e:   # noqa: BLE001 -- a Lustre hiccup must not stop the client
                print(f'[lane {self.name}] refresh failed: {e!r}', flush=True)

    def close(self) -> None:
        self.closed = True
        self._stop.set()
        if self._poller:
            self._poller.join(timeout=10)
        with self.lock:
            self.expected.clear()
        self.flush()


def open_dispatch(dispatch_dir: str | Path, suite: str, run: str | Path, name: str | None = None, **kw):
    """A client's lane under a dispatcher DIR: (lane, predict(item) -> seconds, record for the client's config)."""
    from .durations import Durations
    d = Path(dispatch_dir).resolve()
    cfg = read_json(d / 'dispatch.json')
    if not cfg:
        raise SystemExit(f'{d}/dispatch.json missing: python -m aa_evals.common.dispatch.dispatcher init')
    dur = Durations(cfg.get('durations'), cfg.get('model'))
    lane = Lane.open(d, suite, run, name=name, **kw)
    rec = {'dir': str(d), 'lane': lane.name, 'suite': suite, 'claim_order': 'longest_predicted_first',
           'durations': dur.describe()}
    return lane, (lambda item: dur.predict(suite, str(item))), rec

"""Refresh the README comparison; poll small files without executing notebooks.

Run: python -m notebooks.src.comparison_watch --start
Stop: python -m notebooks.src.comparison_watch --stop
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from . import comparison, readme_comparison


def directory(root):
    path = Path(root) / 'notebooks/results/comparison'
    path.mkdir(parents=True, exist_ok=True)
    return path


def running(root=comparison.ROOT):
    with (directory(root) / 'watch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def start(root=comparison.ROOT):
    root = Path(root).resolve()
    output = directory(root)
    if running(root):
        return
    (output / 'watch.stop').unlink(missing_ok=True)
    with (output / 'watch.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'notebooks.src.comparison_watch',
                                    '--watch', '--root', str(root)], cwd=comparison.ROOT,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    for _ in range(100):
        if running(root):
            return
        if process.poll() is not None:
            raise RuntimeError(f'Comparison watcher did not start; see {output / "watch.log"}')
        time.sleep(.05)
    raise RuntimeError(f'Comparison watcher startup timed out; see {output / "watch.log"}')


def stop(root=comparison.ROOT):
    if running(root):
        (directory(root) / 'watch.stop').touch()


def watch(root=comparison.ROOT, interval=2):
    root = Path(root)
    output = directory(root)
    with (output / 'watch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        state = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                 'interval_seconds': interval}
        last = None
        while not (output / 'watch.stop').exists():
            try:
                current = readme_comparison.source_signature(root)
                if current != last:
                    # Debounce saves and multi-file exports. Incomplete/checksum-invalid
                    # inputs are shown as unavailable; never retain their old metrics.
                    time.sleep(interval)
                    if current != readme_comparison.source_signature(root):
                        continue
                    readme_comparison.refresh(root)
                    last = current
                    state.update(last_refresh_utc=datetime.now(timezone.utc).isoformat(), error=None)
                    print(f'{state["last_refresh_utc"]}: refreshed README comparison', flush=True)
            except Exception as error:
                state['error'] = f'{type(error).__name__}: {error}'
                print(state['error'], flush=True)
            comparison.atomic_write(output / 'watch_status.json', json.dumps(state, indent=2) + '\n')
            time.sleep(interval)
        state['status'] = 'stopped'
        comparison.atomic_write(output / 'watch_status.json', json.dumps(state, indent=2) + '\n')
        (output / 'watch.stop').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--stop', action='store_true')
    mode.add_argument('--watch', action='store_true')
    parser.add_argument('--root', type=Path, default=comparison.ROOT)
    args = parser.parse_args()
    if args.stop:
        stop(args.root)
    elif args.watch:
        watch(args.root)
    elif args.start:
        start(args.root)
        print('Automatic comparison refresh enabled. Logs: notebooks/results/comparison/watch.log')
    else:
        readme_comparison.refresh(args.root)
        print('Refreshed README comparison once. Use --start for automatic updates.')


if __name__ == '__main__':
    main()

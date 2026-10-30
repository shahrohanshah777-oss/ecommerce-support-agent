"""Tee helper: everything printed inside `transcript(name)` also goes to
transcripts/<name>.txt, so each demo leaves graded evidence behind."""
import contextlib
import sys
from datetime import datetime

import config


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def transcript(name: str):
    config.TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    path = config.TRANSCRIPTS_DIR / f"{name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Transcript: {name}\n")
        f.write(f"# MOCK_LLM={config.MOCK_LLM}\n")
        f.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        original = sys.stdout
        sys.stdout = _Tee(original, f)
        try:
            yield path
        finally:
            sys.stdout = original
    print(f"[transcript saved -> {path.relative_to(config.ROOT)}]")

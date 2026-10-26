"""Shared test helpers: a fake OpenAI client so tests need no network and no API key."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VOCAB = ["leave", "remote", "expenses", "password", "office", "parental", "receipts", "laptop"]


def fake_embedding(text: str) -> list[float]:
    """Bag-of-keywords vector + a tiny hash component: similar texts -> similar vectors."""
    t = text.lower()
    vec = [float(t.count(w)) for w in VOCAB]
    h = hashlib.sha256(t.encode()).digest()
    vec.append(h[0] / 2550.0)
    return vec


class FakeClient:
    """Mimics the parts of the OpenAI SDK used by the app."""

    def __init__(self, *, flagged=False, injection=False, reply="Answer [p.1]",
                 eval_json='{"score": 0.9, "label": "good", "explanation": "ok"}'):
        self.flagged, self.injection = flagged, injection
        self.reply, self.eval_json = reply, eval_json
        self.calls: list[tuple[str, dict]] = []
        self.moderations = SimpleNamespace(create=self._moderate)
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _moderate(self, **kw):
        self.calls.append(("moderation", kw))
        return SimpleNamespace(results=[SimpleNamespace(flagged=self.flagged)])

    def _embed(self, **kw):
        self.calls.append(("embeddings", kw))
        return SimpleNamespace(data=[SimpleNamespace(embedding=fake_embedding(t)) for t in kw["input"]])

    def _chat(self, **kw):
        system = kw["messages"][0]["content"]
        if "security classifier" in system:
            kind, content = "injection", "INJECTION" if self.injection else "SAFE"
        elif "automatic evaluator" in system:
            kind, content = "evaluation", self.eval_json
        else:
            kind, content = "assistant", self.reply
        self.calls.append((kind, kw))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def kinds(self) -> list[str]:
        return [k for k, _ in self.calls]


@pytest.fixture()
def fake_client():
    return FakeClient()


@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    return (ROOT / "sample" / "employee_handbook.pdf").read_bytes()


@pytest.fixture()
def read_jsonl():
    def _read(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    return _read

"""Model backends. Ollama for real runs on local models; a scripted fake for tests."""

from __future__ import annotations

import hashlib
import json
import urllib.request


class Ollama:
    """A model served by a local Ollama. Greedy decoding with a fixed seed, so runs repeat."""

    def __init__(self, name: str, host: str = "http://127.0.0.1:11434", timeout: float = 600, keep_alive: str = "30s"):
        if not host.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("only a local Ollama is used (http://127.0.0.1:PORT)")
        self.name, self.host, self.timeout, self.keep_alive = name, host.rstrip("/"), timeout, keep_alive

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        body = {"model": self.name, "prompt": prompt, "stream": False, "keep_alive": self.keep_alive,
                "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens}}
        if system:
            body["system"] = system
        return str(self._post("/api/generate", body).get("response", ""))

    def digest(self) -> str:
        with urllib.request.urlopen(self.host + "/api/tags", timeout=30) as r:
            for m in json.loads(r.read()).get("models", []):
                if m.get("name") == self.name or m.get("model") == self.name:
                    return str(m.get("digest", ""))
        return ""


class Fake:
    """A scripted model for tests and the demo. Its answers carry a marker with a quality score, and as a
    judge it follows a rule: 'fair' picks the better marked answer, 'self' picks its own answer when it is
    in the pair (fair otherwise), 'first' always picks A, 'noisy' flips a coin seeded by the prompt,
    'mute' replies with no letter."""

    def __init__(self, name: str, judging: str = "fair", quality: int = 1):
        self.name, self.judging, self.quality = name, judging, quality
        self.calls = 0

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        self.calls += 1
        if system is None:  # answering
            h = int(hashlib.sha256((self.name + prompt).encode()).hexdigest(), 16)
            return f"[by {self.name}] [q{self.quality * 3 + h % 7}] answer"
        a = prompt.split("[Answer A]\n", 1)[1].split("\n\n[Answer B]", 1)[0]
        b = prompt.split("[Answer B]\n", 1)[1].split("\n\nWhich answer", 1)[0]
        if self.judging == "first":
            return "A"
        if self.judging == "mute":
            return "I cannot decide."
        if self.judging == "noisy":
            return "A" if hashlib.sha256((self.name + prompt).encode()).digest()[0] % 2 else "B"
        if self.judging == "self":
            mark = f"[by {self.name}]"
            if mark in a or mark in b:
                return "A" if mark in a else "B"
        qa = int(a.split("[q", 1)[1].split("]", 1)[0])
        qb = int(b.split("[q", 1)[1].split("]", 1)[0])
        if qa == qb:  # a tie: decide on content so the choice is the same in both orders
            return "A" if a > b else "B"
        return "A" if qa > qb else "B"

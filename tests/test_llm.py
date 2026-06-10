from vpipe.config import LlmCfg
from vpipe.llm import OllamaClient


def test_has_model_requires_exact_tag(monkeypatch):
    c = OllamaClient(LlmCfg(model="qwen3:8b"))
    # only a different size of the same family is pulled -> must be False
    monkeypatch.setattr(c, "_get",
                        lambda path, timeout=5.0: {"models": [{"name": "qwen3:0.5b"}]})
    assert c.has_model() is False
    # exact tag present -> True
    monkeypatch.setattr(c, "_get",
                        lambda path, timeout=5.0: {"models": [{"name": "qwen3:8b"}]})
    assert c.has_model() is True


def test_has_model_bare_family_matches_any_tag(monkeypatch):
    c = OllamaClient(LlmCfg(model="qwen3"))   # no tag -> family match allowed
    monkeypatch.setattr(c, "_get",
                        lambda path, timeout=5.0: {"models": [{"name": "qwen3:8b"}]})
    assert c.has_model() is True

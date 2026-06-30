"""Optional structured LLM layer for answer synthesis and extraction.

Supported providers:
- ``ollama``: POST ``/api/chat``.
- ``openai_compatible``: POST ``/v1/chat/completions``.
- ``openrouter``: OpenAI-compatible endpoint with OpenRouter headers.
- ``groq``: OpenAI-compatible endpoint at ``/openai/v1/chat/completions``.

The adapter intentionally depends only on ``requests`` and is disabled by
default.  It never raises to the API layer; failures return ``None`` so the
rule-based pipeline remains the canonical fallback.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

import requests

from ..config import settings


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class StructuredLLM:
    """Small provider-neutral wrapper around local/openai-compatible chat APIs."""

    def __init__(self) -> None:
        env = os.environ
        openrouter_key = env.get("OPENROUTER_API_KEY", "")
        explicit_provider = env.get("LLM_PROVIDER")
        provider = explicit_provider or getattr(settings, "llm_provider", "none") or "none"
        explicit_model = env.get("LLM_MODEL") or env.get("OPENROUTER_MODEL")
        configured_key = env.get("LLM_API_KEY") or openrouter_key or getattr(settings, "llm_api_key", "") or ""
        settings_model = getattr(settings, "llm_model", "") or ""
        model_hint = explicit_model or settings_model
        looks_openrouter = bool(openrouter_key) or configured_key.startswith("sk-or-") or str(model_hint).startswith("openrouter/")
        if (not explicit_provider or provider == "none") and looks_openrouter:
            provider = "openrouter"
        self.provider = str(provider).lower()

        if explicit_model:
            self.model = explicit_model
        elif self.provider == "openrouter" and openrouter_key:
            self.model = ""
        elif self.provider == "openrouter" and configured_key.startswith("sk-or-") and not settings_model:
            self.model = ""
        else:
            self.model = settings_model

        self.api_key = configured_key
        raw_base_url = str(
            env.get("LLM_BASE_URL")
            or env.get("OPENROUTER_BASE_URL")
            or getattr(settings, "llm_base_url", "")
        ).rstrip("/")
        default_ollama_urls = {"", "http://localhost:11434", "http://host.docker.internal:11434"}
        if self.provider == "openrouter" and raw_base_url in default_ollama_urls:
            raw_base_url = "https://openrouter.ai/api/v1"
        if self.provider == "groq" and raw_base_url in default_ollama_urls:
            raw_base_url = "https://api.groq.com/openai/v1"
        self.base_url = raw_base_url
        self.referer = getattr(settings, "llm_referer", "http://localhost:8501") or "http://localhost:8501"
        self.app_title = getattr(settings, "llm_app_title", "Scientific Knowledge Graph Demo") or "Scientific Knowledge Graph Demo"
        self.timeout = int(getattr(settings, "llm_timeout_seconds", 20))
        self.config_enabled = _env_bool("LLM_ENABLED", None)
        if self.config_enabled is None:
            self.config_enabled = _env_bool("ENABLE_LLM", bool(getattr(settings, "enable_llm", False)))
        if self.provider == "openrouter" and self.api_key and self.model:
            self.config_enabled = True
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return self.ready

    @property
    def ready(self) -> bool:
        provider_ok = self.provider in {"ollama", "openai_compatible", "openrouter", "groq"}
        key_ok = self.provider == "ollama" or bool(self.api_key) or self.provider == "openai_compatible"
        return bool(self.config_enabled) and provider_ok and bool(self.base_url and self.model) and key_ok and not _is_placeholder_model(self.model)

    def _configuration_error(self) -> str:
        if self.provider not in {"ollama", "openai_compatible", "openrouter", "groq"}:
            return f"Unsupported or missing LLM provider: {self.provider}."
        if not self.base_url:
            return "LLM_BASE_URL is missing."
        if not self.model:
            if self.provider == "openrouter" and self.api_key:
                return "OpenRouter API key is configured, but LLM_MODEL/OPENROUTER_MODEL is missing."
            return "LLM_API_KEY is configured, but LLM_MODEL is missing" if self.api_key else "LLM_MODEL is missing."
        if _is_placeholder_model(self.model):
            return "LLM_MODEL/OPENROUTER_MODEL still contains a placeholder; set a real model slug."
        if self.provider != "ollama" and self.provider != "openai_compatible" and not self.api_key:
            return "LLM_API_KEY is missing."
        if not self.config_enabled:
            return "LLM is disabled; set LLM_ENABLED=true."
        return self.last_error

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.config_enabled),
            "provider": self.provider,
            "base_url": self.base_url or None,
            "model": self.model or None,
            "api_key_configured": bool(self.api_key),
            "ready": self.ready,
            "last_error": "" if self.ready else self._configuration_error(),
        }

    def _openai_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider == "openrouter":
            # OpenRouter recommends HTTP-Referer and X-Title for rankings/analytics.
            headers["HTTP-Referer"] = self.referer
            headers["X-Title"] = self.app_title
        return headers

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def _parse_json_object(self, text: str) -> Dict[str, Any] | None:
        if not text:
            return None
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except Exception:
            pass
        match = _JSON_RE.search(text)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def _chat(self, system: str, user: str) -> str | None:
        if not self.ready:
            self.last_error = self._configuration_error()
            return None
        self.last_error = ""
        try:
            if self.provider == "ollama":
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "options": {"temperature": 0.0},
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return (data.get("message") or {}).get("content")
            if self.provider in {"openai_compatible", "openrouter", "groq"}:
                resp = requests.post(
                    self._chat_completions_url(),
                    headers=self._openai_headers(),
                    json={
                        "model": self.model,
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    self.last_error = "llm_empty_choices"
                    return None
                return (choices[0].get("message") or {}).get("content")
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", "unknown")
            text = getattr(exc.response, "text", "") or ""
            self.last_error = f"http_{status}:{text[:160]}"
            return None
        except requests.Timeout:
            self.last_error = "timeout"
            return None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}:{str(exc)[:160]}"
            return None
        return None

    def test_connection(self) -> Dict[str, Any]:
        """Run a minimal provider call for diagnostics."""

        status = self.status()
        if not status.get("ready"):
            return {
                **status,
                "success": False,
                "latency_ms": None,
                "short_response": None,
                "response_preview": None,
                "error": status.get("last_error"),
            }
        started = time.perf_counter()
        content = self._chat(
            "Return JSON only with key 'answer'.",
            json.dumps({"task": "Reply with OK in Russian using JSON."}, ensure_ascii=False),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        obj = self._parse_json_object(content or "")
        short_response = str((obj or {}).get("answer") or content or "").strip()[:200]
        return {
            **self.status(),
            "success": bool(short_response and not self.last_error),
            "latency_ms": latency_ms,
            "short_response": short_response,
            "response_preview": short_response,
            "error": self.last_error,
        }

    def synthesize_answer(
        self,
        *,
        question: str,
        intent: str,
        answer_draft: str,
        facts: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        gaps: List[Dict[str, Any]],
    ) -> str | None:
        """Ask an optional LLM to polish the final answer using only grounded facts."""
        if not self.enabled:
            return None
        system = (
            "You are a strict technical-document assistant. "
            "Use only the provided JSON facts and sources. "
            "Do not invent details. Return JSON only with key 'answer'. "
            "Answer in Russian in normal human prose, not as a raw list of triples. "
            "If exact facts are missing, say that directly, then mention only the nearest partial evidence if it is present. "
            "Never answer with facts about a different material/object/process/property as if they answered the question. "
            "Mention uncertainty and missing data explicitly."
        )
        payload = {
            "question": question,
            "intent": intent,
            "rule_based_answer_draft": answer_draft,
            "facts": facts[:40],
            "sources": sources[:12],
            "gaps": gaps[:12],
        }
        content = self._chat(system, json.dumps(payload, ensure_ascii=False))
        obj = self._parse_json_object(content or "")
        if not obj:
            if not self.last_error:
                self.last_error = "llm_answer_json_parse_failed"
            return None
        answer = obj.get("answer")
        return str(answer).strip() if answer else None

    def rewrite_question_for_retrieval(self, question: str) -> Dict[str, Any] | None:
        """Optional LLM query rewrite for messy user questions.

        This does not answer the question and does not create facts.  It only
        produces a broader search query, preserving explicit named entities
        like material grades, article numbers, DN/PN codes and standards.
        """
        if not self.enabled:
            return None
        system = (
            "You rewrite Russian/English technical-document search questions. "
            "Return JSON only with keys: search_query, normalized_question, notes. "
            "Preserve all explicit identifiers exactly: material grades, standards, DN/PN, article numbers. "
            "Add likely synonyms, abbreviations and spelling variants. Do not answer. Do not invent facts."
        )
        payload = {"question": question}
        content = self._chat(system, json.dumps(payload, ensure_ascii=False))
        obj = self._parse_json_object(content or "")
        if not obj:
            if not self.last_error:
                self.last_error = "llm_rewrite_json_parse_failed"
            return None
        search_query = str(obj.get("search_query") or "").strip()
        if not search_query:
            return None
        return obj

    def extract_structured_facts(self, chunk_text: str) -> Dict[str, Any] | None:
        """Optional JSON extraction for future GPU/LLM demos.

        The API currently keeps deterministic extraction as source of truth;
        this method is exposed for experiments and can be wired in without
        changing the public contract.
        """
        if not self.enabled:
            return None
        system = (
            "Extract structured technical facts from the chunk. "
            "Return JSON only with arrays: technical_objects, parts, article_numbers, "
            "materials, standards, parameters, requirements, measurements, gaps. "
            "Do not infer facts not present in the text."
        )
        content = self._chat(system, chunk_text[:6000])
        return self._parse_json_object(content or "")


def _env_bool(name: str, default: bool | None) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _is_placeholder_model(model: str) -> bool:
    lowered = (model or "").lower()
    return any(token in lowered for token in ["your_openrouter_model_slug_here", "replace-with", "<openrouter-model-slug>"])

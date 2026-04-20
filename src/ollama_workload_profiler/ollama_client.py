from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

import httpx

DEFAULT_OLLAMA_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=None,
    write=60.0,
    pool=60.0,
)


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        transport: httpx.BaseTransport | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> None:
        client_kwargs: dict[str, object] = {
            "base_url": base_url,
            "transport": transport,
            "timeout": DEFAULT_OLLAMA_TIMEOUT if timeout is None else timeout,
        }
        self._client = httpx.Client(**client_kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def list_models(self) -> list[str]:
        response = self._client.get("/api/tags")
        response.raise_for_status()
        payload = response.json()
        return [model["name"] for model in payload.get("models", [])]

    def version(self) -> str:
        response = self._client.get("/api/version")
        response.raise_for_status()
        payload = response.json()
        version = payload.get("version")
        if not isinstance(version, str):
            raise ValueError("Ollama version response did not include a string version")
        return version

    def show_model(self, model_name: str) -> dict[str, Any]:
        response = self._client.post("/api/show", json={"name": model_name})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Ollama show response was not a JSON object")
        return payload

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        options: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        payload = self._build_generate_payload(
            model=model,
            prompt=prompt,
            options=options,
            stream=stream,
        )

        if stream:
            return self._stream_post("/api/generate", payload)

        response = self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        return response.json()

    def stream_generate(
        self,
        *,
        model: str,
        prompt: str,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        return self.generate(
            model=model,
            prompt=prompt,
            options=options,
            stream=True,
        )

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        payload = self._build_chat_payload(
            model=model,
            messages=messages,
            options=options,
            stream=stream,
        )

        if stream:
            return self._stream_post("/api/chat", payload)

        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        return self.chat(
            model=model,
            messages=messages,
            options=options,
            stream=True,
        )

    def unload_model(self, *, model: str) -> dict[str, Any]:
        response = self._client.post(
            "/api/generate",
            json={"model": model, "keep_alive": 0},
        )
        response.raise_for_status()
        return response.json()

    def preload_model(
        self,
        *,
        model: str,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "keep_alive": -1,
            "prompt": "",
            "stream": False,
        }
        if options:
            payload["options"] = dict(options)
        response = self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        return response.json()

    def _build_generate_payload(
        self,
        *,
        model: str,
        prompt: str,
        options: Mapping[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if options:
            payload["options"] = dict(options)
        return payload

    def _build_chat_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: Mapping[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = dict(options)
        return payload

    def _stream_post(
        self, path: str, payload: Mapping[str, Any]
    ) -> "_JSONLineStream":
        return _JSONLineStream(self._client, path, payload)

    def _iter_json_lines(self, response: httpx.Response) -> Iterator[dict[str, Any]]:
        for line in response.iter_lines():
            if not line:
                continue
            yield json.loads(line)


class _JSONLineStream:
    def __init__(
        self,
        client: httpx.Client,
        path: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._context: Any | None = client.stream("POST", path, json=payload)
        self._response: httpx.Response | None = None
        self._lines: Iterator[str] = iter(())
        self._entered = False
        self._closed = True

        try:
            self._response = self._context.__enter__()
            self._entered = True
            self._response.raise_for_status()
        except Exception:
            if self._context is not None and self._entered:
                self._context.__exit__(None, None, None)
            self._closed = True
            raise

        self._lines = self._response.iter_lines()
        self._closed = False

    def __iter__(self) -> "_JSONLineStream":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise StopIteration

        for line in self._lines:
            if not line:
                continue
            return json.loads(line)

        self.close()
        raise StopIteration

    def __enter__(self) -> "_JSONLineStream":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
        return None

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        context = getattr(self, "_context", None)
        if context is not None and getattr(self, "_entered", False):
            context.__exit__(None, None, None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return

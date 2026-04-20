import json

import httpx
import pytest

from ollama_workload_profiler.ollama_client import OllamaClient, _JSONLineStream


def test_list_models_returns_model_names() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.scheme == "http"
        assert request.url.host == "127.0.0.1"
        assert request.url.port == 11434
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "llama3.2:latest"}, {"name": "mistral:7b"}]},
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    assert client.list_models() == ["llama3.2:latest", "mistral:7b"]


def test_version_returns_string_from_api_version() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/version"
        return httpx.Response(200, json={"version": "0.6.0"}, request=request)

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    assert client.version() == "0.6.0"


def test_show_model_posts_name_to_api_show() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/show"
        seen_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"details": {"family": "llama"}, "model_info": {"general.architecture": "llama"}},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    assert client.show_model("llama3.2:latest") == {
        "details": {"family": "llama"},
        "model_info": {"general.architecture": "llama"},
    }
    assert seen_payload == {"name": "llama3.2:latest"}


def test_generate_defaults_to_non_streaming_requests() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/generate"
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"response": "ok"}, request=request)

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    assert client.generate(model="llama3.2", prompt="Hello") == {"response": "ok"}
    assert seen_payload["stream"] is False


def test_stream_generate_sends_streaming_payload_and_yields_chunks() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/generate"
        seen_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            text='{"response":"first","done":false}\n{"response":"second","done":true}\n',
            request=request,
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    chunks = list(
        client.stream_generate(
            model="llama3.2",
            prompt="Hello",
            options={"num_ctx": 2048},
        )
    )

    assert seen_payload == {
        "model": "llama3.2",
        "prompt": "Hello",
        "stream": True,
        "options": {"num_ctx": 2048},
    }
    assert chunks == [
        {"response": "first", "done": False},
        {"response": "second", "done": True},
    ]


def test_stream_chat_sends_streaming_payload_and_yields_chunks() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/chat"
        seen_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            text='{"message":{"role":"assistant","content":"first"},"done":false}\n'
            '{"message":{"role":"assistant","content":"second"},"done":true}\n',
            request=request,
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    chunks = list(
        client.stream_chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hello"}],
            options={"temperature": 0.1},
        )
    )

    assert seen_payload == {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "options": {"temperature": 0.1},
    }
    assert chunks == [
        {"message": {"role": "assistant", "content": "first"}, "done": False},
        {"message": {"role": "assistant", "content": "second"}, "done": True},
    ]


def test_unload_model_requests_keep_alive_zero() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/generate"
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"done": True, "done_reason": "unload"}, request=request)

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    assert client.unload_model(model="llama3.2") == {"done": True, "done_reason": "unload"}
    assert seen_payload == {"model": "llama3.2", "keep_alive": 0}


def test_preload_model_requests_keep_alive_negative_one() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/generate"
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"done": True}, request=request)

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    assert client.preload_model(
        model="llama3.2",
        options={"num_ctx": 4096, "seed": 42, "temperature": 0.0},
    ) == {"done": True}
    assert seen_payload == {
        "model": "llama3.2",
        "keep_alive": -1,
        "prompt": "",
        "stream": False,
        "options": {"num_ctx": 4096, "seed": 42, "temperature": 0.0},
    }


def test_client_accepts_timeout_and_passes_it_to_httpx_client(monkeypatch) -> None:
    created_kwargs: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            created_kwargs.update(kwargs)

        def get(self, path: str) -> httpx.Response:
            return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", path))

        def close(self) -> None:
            pass

    monkeypatch.setattr("ollama_workload_profiler.ollama_client.httpx.Client", FakeClient)

    client = OllamaClient(timeout=7.5)

    assert created_kwargs["timeout"] == 7.5
    assert client.list_models() == []


def test_client_uses_stream_friendly_timeout_profile_by_default(monkeypatch) -> None:
    created_kwargs: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            created_kwargs.update(kwargs)

        def get(self, path: str) -> httpx.Response:
            return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", path))

        def close(self) -> None:
            pass

    monkeypatch.setattr("ollama_workload_profiler.ollama_client.httpx.Client", FakeClient)

    client = OllamaClient()

    timeout = created_kwargs["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 5.0
    assert timeout.read is None
    assert timeout.write == 60.0
    assert timeout.pool == 60.0
    assert client.list_models() == []


def test_client_closes_underlying_httpx_client_when_used_as_context_manager(monkeypatch) -> None:
    close_calls = 0

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def get(self, path: str) -> httpx.Response:
            return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", path))

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    monkeypatch.setattr("ollama_workload_profiler.ollama_client.httpx.Client", FakeClient)

    with OllamaClient() as client:
        assert client.list_models() == []

    assert close_calls == 1


def test_stream_generate_can_be_closed_after_partial_consumption(monkeypatch) -> None:
    closed = False

    class FakeStreamResponse:
        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> list[str]:
            return ['{"response":"first"}', '{"response":"second"}']

    class FakeStreamContext:
        def __enter__(self) -> FakeStreamResponse:
            return FakeStreamResponse()

        def __exit__(self, exc_type, exc, tb) -> None:
            nonlocal closed
            closed = True
            return None

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def stream(self, method: str, path: str, json: dict[str, object]) -> FakeStreamContext:
            assert method == "POST"
            assert path == "/api/generate"
            assert json["stream"] is True
            return FakeStreamContext()

        def close(self) -> None:
            return None

    monkeypatch.setattr("ollama_workload_profiler.ollama_client.httpx.Client", FakeClient)

    client = OllamaClient()
    stream = client.stream_generate(model="llama3.2", prompt="ping")

    assert next(iter(stream)) == {"response": "first"}
    assert closed is False

    stream.close()

    assert closed is True


def test_json_line_stream_close_is_safe_when_initialization_never_finished() -> None:
    stream = _JSONLineStream.__new__(_JSONLineStream)

    stream.close()
    stream.__del__()


def test_json_line_stream_closes_entered_context_when_initialization_fails(monkeypatch) -> None:
    exited = False

    class FakeStreamResponse:
        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "boom",
                request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
                response=httpx.Response(500),
            )

    class FakeStreamContext:
        def __enter__(self) -> FakeStreamResponse:
            return FakeStreamResponse()

        def __exit__(self, exc_type, exc, tb) -> None:
            nonlocal exited
            exited = True
            return None

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def stream(self, method: str, path: str, json: dict[str, object]) -> FakeStreamContext:
            return FakeStreamContext()

        def close(self) -> None:
            return None

    monkeypatch.setattr("ollama_workload_profiler.ollama_client.httpx.Client", FakeClient)

    client = OllamaClient()

    with pytest.raises(httpx.HTTPStatusError):
        client.stream_generate(model="llama3.2", prompt="ping")

    assert exited is True

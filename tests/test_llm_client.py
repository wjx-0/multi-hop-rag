import os
import urllib.error

from src.utils.llm_client import AliyunDashScopeClient, load_env_file


def test_load_env_file_reads_key_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DASHSCOPE_API_KEY=test-key\nDASHSCOPE_MODEL=qwen-plus\n", encoding="utf-8")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_MODEL", raising=False)

    load_env_file(env_file)

    assert os.environ["DASHSCOPE_API_KEY"] == "test-key"
    assert os.environ["DASHSCOPE_MODEL"] == "qwen-plus"


def test_aliyun_client_builds_default_chat_url(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DASHSCOPE_API_KEY=",
                "DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                "DASHSCOPE_MODEL=qwen-plus",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_MODEL", raising=False)

    client = AliyunDashScopeClient(env_path=env_file)

    assert client.chat_completions_url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert client.model == "qwen-plus"


def test_aliyun_client_reads_retry_and_throttle_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DASHSCOPE_API_KEY=test-key",
                "DASHSCOPE_MAX_RETRIES=5",
                "DASHSCOPE_RETRY_BACKOFF_SECONDS=0.25",
                "DASHSCOPE_MIN_REQUEST_INTERVAL_SECONDS=2.5",
            ]
        ),
        encoding="utf-8",
    )

    client = AliyunDashScopeClient(env_path=env_file)

    assert client.max_retries == 5
    assert client.retry_backoff_seconds == 0.25
    assert client.min_request_interval_seconds == 2.5


def test_aliyun_client_retries_transient_http_errors(monkeypatch):
    client = AliyunDashScopeClient(
        api_key="test-key",
        max_retries=1,
        retry_backoff_seconds=0,
        min_request_interval_seconds=0,
    )
    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                500,
                "server error",
                hdrs=None,
                fp=None,
            )
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert client.generate([{"role": "user", "content": "hello"}]) == "ok"
    assert calls["count"] == 2

import os

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

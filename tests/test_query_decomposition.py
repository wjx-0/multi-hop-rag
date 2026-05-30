from src.retrieval.query_decomposition import (
    LLMQueryDecomposer,
    QueryDecompositionCache,
    QueryDecompositionResult,
    clean_decomposed_queries,
    extract_queries_from_response,
)


def test_extract_queries_from_response_accepts_strings_and_dicts():
    response = {"queries": ["first query", {"query": "second query"}, {"question": "third query"}]}

    assert extract_queries_from_response(response) == ["first query", "second query", "third query"]


def test_clean_decomposed_queries_keeps_original_first_and_dedupes():
    queries = clean_decomposed_queries(
        "Who directed the film starring Actor X?",
        [
            "Who directed the film starring Actor X?",
            "1. Actor X film",
            "Actor X film",
            "film director nationality",
        ],
        max_queries=3,
    )

    assert queries == [
        "Who directed the film starring Actor X?",
        "Actor X film",
        "film director nationality",
    ]


def test_clean_decomposed_queries_can_use_generated_queries_without_original():
    queries = clean_decomposed_queries(
        "Who directed the film starring Actor X?",
        [
            "1. Actor X film",
            "Actor X film",
            "film director nationality",
        ],
        max_queries=3,
        query_mode="generated_or_original",
    )

    assert queries == [
        "Actor X film",
        "film director nationality",
    ]


def test_clean_decomposed_queries_generated_or_original_keeps_single_generated_query():
    queries = clean_decomposed_queries(
        "Who directed the film starring Actor X?",
        ["Actor X film"],
        max_queries=3,
        query_mode="generated_or_original",
    )

    assert queries == ["Actor X film"]


def test_clean_decomposed_queries_generated_or_original_falls_back_to_original():
    queries = clean_decomposed_queries(
        "Who directed the film starring Actor X?",
        ["", "   "],
        max_queries=3,
        query_mode="generated_or_original",
    )

    assert queries == ["Who directed the film starring Actor X?"]


def test_llm_query_decomposer_falls_back_on_api_error():
    decomposer = LLMQueryDecomposer(
        llm_client=FailingJSONClient(),
        model="qwen-test",
        max_queries=4,
    )

    result = decomposer.decompose(sample_id="q1", question="Question?")

    assert result.queries == ["Question?"]
    assert result.generated_queries == []
    assert result.fallback is True
    assert "temporary failure" in result.error


def test_llm_query_decomposer_uses_clean_generated_queries():
    decomposer = LLMQueryDecomposer(
        llm_client=FakeJSONClient({"queries": ["Entity page", "Entity page", "related fact"]}),
        model="qwen-test",
        max_queries=3,
    )

    result = decomposer.decompose(sample_id="q1", question="Original question?")

    assert result.queries == ["Original question?", "Entity page", "related fact"]
    assert result.generated_queries == ["Entity page", "related fact"]
    assert result.fallback is False
    assert result.query_mode == "original_plus_generated"


def test_llm_query_decomposer_can_use_generated_queries_only():
    decomposer = LLMQueryDecomposer(
        llm_client=FakeJSONClient({"queries": ["Entity page", "related fact"]}),
        model="qwen-test",
        max_queries=3,
        query_mode="generated_or_original",
    )

    result = decomposer.decompose(sample_id="q1", question="Original question?")

    assert result.queries == ["Entity page", "related fact"]
    assert result.generated_queries == ["Entity page", "related fact"]
    assert result.fallback is False
    assert result.query_mode == "generated_or_original"


def test_llm_query_decomposer_can_skip_model_generation_kwarg():
    client = RejectingModelKwargJSONClient({"queries": ["Entity page"]})
    decomposer = LLMQueryDecomposer(
        llm_client=client,
        model="local-model",
        max_queries=4,
        pass_model_arg=False,
    )

    result = decomposer.decompose(sample_id="q1", question="Original question?")

    assert result.queries == ["Original question?", "Entity page"]
    assert client.received_model_kwarg is False


def test_query_decomposition_cache_round_trips_successful_results(tmp_path):
    cache_path = tmp_path / "decomposition.jsonl"
    decomposer = LLMQueryDecomposer(
        llm_client=FakeJSONClient({"queries": ["Entity page"]}),
        model="qwen-test",
        max_queries=4,
    )
    result = decomposer.decompose(sample_id="q1", question="Original question?")

    cache = QueryDecompositionCache(cache_path)
    cache.put(result)
    loaded = QueryDecompositionCache(cache_path).get(sample_id="q1", question="Original question?")

    assert loaded is not None
    assert loaded.from_cache is True
    assert loaded.queries == ["Original question?", "Entity page"]


def test_query_decomposition_cache_skips_records_with_different_query_mode(tmp_path):
    cache_path = tmp_path / "decomposition.jsonl"
    cache = QueryDecompositionCache(cache_path)
    cache.put(
        QueryDecompositionResult(
            sample_id="q1",
            question="Original question?",
            queries=["Original question?", "Entity page"],
            generated_queries=["Entity page"],
            model="qwen-test",
            query_mode="original_plus_generated",
        )
    )

    loaded = QueryDecompositionCache(cache_path).get(
        sample_id="q1",
        question="Original question?",
        query_mode="generated_or_original",
    )

    assert loaded is None


class FakeJSONClient:
    model = "fake-model"

    def __init__(self, response):
        self.response = response

    def generate_json(self, messages, schema=None, **kwargs):
        return self.response


class FailingJSONClient:
    model = "fake-model"

    def generate_json(self, messages, schema=None, **kwargs):
        raise RuntimeError("temporary failure")


class RejectingModelKwargJSONClient:
    model = "fake-model"

    def __init__(self, response):
        self.response = response
        self.received_model_kwarg = False

    def generate_json(self, messages, schema=None, **kwargs):
        self.received_model_kwarg = "model" in kwargs
        if self.received_model_kwarg:
            raise RuntimeError("local generate does not accept model kwarg")
        return self.response

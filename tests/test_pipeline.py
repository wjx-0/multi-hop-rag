from src.data.load_hotpotqa import load_hotpotqa
from src.pipeline.standard_rag import StandardRAGPipeline


def test_standard_rag_pipeline_smoke():
    samples = load_hotpotqa("data/raw/hotpotqa/hotpot_dev_distractor_v1.json", limit=1)
    result = StandardRAGPipeline(top_k=3).run(samples[0])
    assert result.id == samples[0].id
    assert result.retrieved_docs
    assert "answer_f1" in result.metrics

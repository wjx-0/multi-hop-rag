# 中文说明：分析 prediction JSONL 的坏例并生成 summary.md / bad_cases.jsonl 报告。
"""Analyze prediction JSONL files and write bad-case reports."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evidence_metrics import gold_evidence_titles
from src.utils.io import read_jsonl, write_jsonl
from src.utils.text import normalize_answer


SUMMARY_METRICS = (
    "answer_em",
    "answer_f1",
    "llm_error",
    "rerank_error",
    "evidence_recall_at_answer_k",
    "supporting_fact_f1",
)
BAD_CASE_KEYS = (
    "llm_error",
    "rerank_error",
    "evidence_missing",
    "answer_wrong_but_evidence_present",
    "answer_wrong_and_evidence_missing",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--questions", default="data/processed/hotpotqa/global/questions_dev.jsonl")
    parser.add_argument("--max-bad-cases", type=int, default=20)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    records = list(read_jsonl(args.predictions))
    question_metadata = load_question_metadata(args.questions)
    analyses = [
        analyze_record(record, question_metadata=question_metadata)
        for record in records
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bad_cases = [analysis for analysis in analyses if is_bad_case(analysis)]
    bad_case_path = output_dir / "bad_cases.jsonl"
    summary_path = output_dir / "summary.md"
    write_jsonl(bad_cases, bad_case_path)
    summary_path.write_text(
        build_summary_markdown(
            records=records,
            analyses=analyses,
            predictions_path=Path(args.predictions),
            bad_case_path=bad_case_path,
            max_bad_cases=args.max_bad_cases,
        ),
        encoding="utf-8",
    )
    print(f"records: {len(records)}")
    print(f"bad_cases: {len(bad_cases)}")
    print(f"wrote {summary_path}")
    print(f"wrote {bad_case_path}")


def analyze_record(
    record: dict[str, Any],
    *,
    question_metadata: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    gold_answer = str(record.get("gold_answer", ""))
    pred_answer = str(record.get("pred_answer", ""))
    retrieved_docs = list(record.get("retrieved_docs", []))
    gold_titles = gold_evidence_titles(record.get("gold_supporting_facts", []))
    retrieved_titles = _retrieved_titles(retrieved_docs)
    retrieved_title_set = {_normalize_title(title) for title in retrieved_titles}
    missing_gold_titles = sorted(gold_titles - retrieved_title_set)
    evidence_recall = (
        (len(gold_titles) - len(missing_gold_titles)) / len(gold_titles)
        if gold_titles
        else 0.0
    )

    answer_em = _metric(metrics, "answer_em")
    answer_f1 = _metric(metrics, "answer_f1")
    llm_error_message = str(record.get("agent_outputs", {}).get("llm_error", ""))
    has_llm_error = _metric(metrics, "llm_error") > 0.0 or bool(llm_error_message)
    has_rerank_error = _metric(metrics, "rerank_error") > 0.0
    evidence_missing = evidence_recall < 1.0
    yes_no_gold = normalize_answer(gold_answer) in {"yes", "no"}
    yes_no_wrong = yes_no_gold and normalize_answer(pred_answer) != normalize_answer(gold_answer)

    error_tags: list[str] = []
    if has_llm_error:
        error_tags.append("llm_error")
    if has_rerank_error:
        error_tags.append("rerank_error")
    if evidence_missing:
        error_tags.append("evidence_missing")
    if answer_em == 0.0 and evidence_missing:
        error_tags.append("answer_wrong_and_evidence_missing")
    elif answer_em == 0.0:
        error_tags.append("answer_wrong_but_evidence_present")
    if yes_no_wrong:
        error_tags.append("yes_no_wrong")
    if answer_em == 0.0 and answer_f1 > 0.0:
        error_tags.append("partial_answer")
    if _metric(metrics, "supporting_fact_f1") < 0.5:
        error_tags.append("citation_low")

    return {
        "id": record.get("id", ""),
        "question": record.get("question", ""),
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "type": _record_type(record, question_metadata=question_metadata),
        "level": _record_level(record, question_metadata=question_metadata),
        "error_tags": error_tags,
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "evidence_recall_at_answer_k": evidence_recall,
        "missing_gold_titles": missing_gold_titles,
        "retrieved_titles": retrieved_titles,
        "llm_error": llm_error_message,
        "rerank_error": has_rerank_error,
        "supporting_fact_f1": _metric(metrics, "supporting_fact_f1"),
    }


def load_question_metadata(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if str(path).strip() == "":
        return {}
    question_path = Path(path)
    if not question_path.is_file():
        return {}

    metadata: dict[str, dict[str, str]] = {}
    for record in read_jsonl(question_path):
        question_id = record.get("id", record.get("_id"))
        if not question_id:
            continue
        metadata[str(question_id)] = {
            "type": str(record.get("type", "unknown")),
            "level": str(record.get("level", "unknown")),
        }
    return metadata


def build_summary_markdown(
    *,
    records: list[dict[str, Any]],
    analyses: list[dict[str, Any]],
    predictions_path: str | Path,
    bad_case_path: str | Path,
    max_bad_cases: int = 20,
) -> str:
    lines = [
        "# Prediction Bad Case Analysis",
        "",
        f"- predictions: `{predictions_path}`",
        f"- records: {len(records)}",
        f"- bad_cases: {sum(1 for analysis in analyses if is_bad_case(analysis))}",
        f"- bad_cases_file: `{bad_case_path}`",
        "",
        "## Overall Metrics",
        "",
        _markdown_table(
            ["metric", "average"],
            [
                [metric, f"{_average_metric_from_analyses(analyses, metric):.4f}"]
                for metric in SUMMARY_METRICS
            ],
        ),
        "",
        "## By Type",
        "",
        _group_table(analyses, key="type"),
        "",
        "## By Level",
        "",
        _group_table(analyses, key="level"),
        "",
        "## Yes/No vs Other",
        "",
        _yes_no_table(analyses),
        "",
        "## Error Tags",
        "",
        _error_tag_table(analyses),
        "",
        "## Top Bad Cases",
        "",
        _bad_case_table(analyses, max_bad_cases=max_bad_cases),
        "",
    ]
    return "\n".join(lines)


def is_bad_case(analysis: dict[str, Any]) -> bool:
    tags = set(analysis.get("error_tags", []))
    return bool(tags & set(BAD_CASE_KEYS))


def _record_type(
    record: dict[str, Any],
    *,
    question_metadata: dict[str, dict[str, str]] | None = None,
) -> str:
    if record.get("type"):
        return str(record["type"])
    metadata = (question_metadata or {}).get(str(record.get("id", "")), {})
    if metadata.get("type"):
        return metadata["type"]
    route = record.get("route_initial")
    if isinstance(route, dict) and route.get("type"):
        return str(route["type"])
    return "unknown"


def _record_level(
    record: dict[str, Any],
    *,
    question_metadata: dict[str, dict[str, str]] | None = None,
) -> str:
    if record.get("level"):
        return str(record["level"])
    metadata = (question_metadata or {}).get(str(record.get("id", "")), {})
    if metadata.get("level"):
        return metadata["level"]
    route = record.get("route_initial")
    if isinstance(route, dict) and route.get("level"):
        return str(route["level"])
    return "unknown"


def _retrieved_titles(retrieved_docs: list[dict[str, Any]]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for doc in retrieved_docs:
        title = str(doc.get("title", "")).strip()
        normalized = _normalize_title(title)
        if title and normalized not in seen:
            titles.append(title)
            seen.add(normalized)
    return titles


def _normalize_title(title: Any) -> str:
    return str(title).strip().lower()


def _metric(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _average_metric_from_analyses(analyses: list[dict[str, Any]], key: str) -> float:
    if not analyses:
        return 0.0
    return sum(_analysis_metric_value(analysis, key) for analysis in analyses) / len(analyses)


def _analysis_metric_value(analysis: dict[str, Any], key: str) -> float:
    if key == "llm_error":
        return 1.0 if analysis.get("llm_error") else 0.0
    if key == "rerank_error":
        return 1.0 if analysis.get("rerank_error") else 0.0
    try:
        return float(analysis.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _group_table(analyses: list[dict[str, Any]], *, key: str) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for analysis in analyses:
        groups[str(analysis.get(key, "unknown"))].append(analysis)
    rows = [
        _group_row(group_key, group_analyses)
        for group_key, group_analyses in sorted(groups.items())
    ]
    return _markdown_table(["group", "count", "answer_em", "answer_f1", "evidence_recall", "bad_cases"], rows)


def _yes_no_table(analyses: list[dict[str, Any]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = {"yes_no": [], "other": []}
    for analysis in analyses:
        group = "yes_no" if normalize_answer(analysis["gold_answer"]) in {"yes", "no"} else "other"
        groups[group].append(analysis)
    rows = [
        _group_row(group_key, group_analyses)
        for group_key, group_analyses in groups.items()
        if group_analyses
    ]
    return _markdown_table(["group", "count", "answer_em", "answer_f1", "evidence_recall", "bad_cases"], rows)


def _group_row(group_key: str, group_analyses: list[dict[str, Any]]) -> list[str]:
    count = len(group_analyses)
    return [
        group_key,
        str(count),
        f"{_avg(group_analyses, 'answer_em'):.4f}",
        f"{_avg(group_analyses, 'answer_f1'):.4f}",
        f"{_avg(group_analyses, 'evidence_recall_at_answer_k'):.4f}",
        str(sum(1 for analysis in group_analyses if is_bad_case(analysis))),
    ]


def _avg(analyses: list[dict[str, Any]], key: str) -> float:
    if not analyses:
        return 0.0
    return sum(float(analysis.get(key, 0.0)) for analysis in analyses) / len(analyses)


def _error_tag_table(analyses: list[dict[str, Any]]) -> str:
    counts = Counter(
        tag
        for analysis in analyses
        for tag in analysis.get("error_tags", [])
    )
    rows = [[tag, str(count)] for tag, count in counts.most_common()]
    return _markdown_table(["error_tag", "count"], rows)


def _bad_case_table(analyses: list[dict[str, Any]], *, max_bad_cases: int) -> str:
    rows = []
    for analysis in [item for item in analyses if is_bad_case(item)][:max_bad_cases]:
        evidence_ok = "yes" if analysis["evidence_recall_at_answer_k"] >= 1.0 else "no"
        rows.append(
            [
                str(analysis["id"]),
                _truncate(str(analysis["question"]), 70),
                _truncate(str(analysis["gold_answer"]), 24),
                _truncate(str(analysis["pred_answer"]), 24),
                ", ".join(analysis.get("error_tags", [])),
                evidence_ok,
            ]
        )
    return _markdown_table(["id", "question", "gold", "pred", "error_tags", "gold_evidence_present"], rows)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(str(cell)) for cell in row) + " |")
    return "\n".join(lines)


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _truncate(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import re

from relai import (
    FixedInput,
    FixedTurn,
    LLMJudgeEvaluator,
    ModelSpec,
    RELAIBenchmark,
    RELAIEnvironment,
    StoredBenchmarkCsv,
)


BENCHMARK_ID = "finance-vs-offtopic"
BENCHMARK_NAME = "finance-vs-offtopic"
DATASET_ID = "2067d3d3-8b66-4905-9fe1-d202daab7b88"
REQUIRED_COLUMNS = ["prompt", "topic", "expected_action"]
JUDGE_MODEL = ModelSpec(name="gpt-5.4")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "sample"


def _shorten(value: str, limit: int = 64) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _is_time_sensitive(prompt: str) -> bool:
    lowered = prompt.lower()
    keywords = (
        "current",
        "latest",
        "most recent",
        "this weekend",
        "today",
        "this year",
    )
    return any(keyword in lowered for keyword in keywords)


def _environment_id(row_fields, sample_index: int) -> str:
    prompt = row_fields["prompt"].strip()
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    topic = _slugify(row_fields["topic"])
    action = _slugify(row_fields["expected_action"])
    return f"{topic}-{action}-{sample_index}-{prompt_hash}"


def _environment_name(row_fields) -> str:
    prompt = row_fields["prompt"].strip()
    action = row_fields["expected_action"].strip()
    prefix = "Answer" if action == "answer" else "Redirect"
    return f"{prefix}: {_shorten(prompt, limit=56)}"


def _environment_description(row_fields) -> str:
    action = row_fields["expected_action"].strip()
    if action == "answer":
        return "Checks whether the agent gives a useful finance answer."
    if action == "decline_and_redirect":
        return "Checks whether the agent declines off-topic requests and redirects to finance help."
    raise ValueError(f"Unsupported expected_action: {action}")


def _finance_judge_instructions(row_fields) -> str:
    prompt = row_fields["prompt"].strip()
    time_sensitive_rule = ""
    if _is_time_sensitive(prompt):
        time_sensitive_rule = (
            "- Because this request asks for current or recent information, full credit requires "
            "a clear time anchor such as a date, reporting period, or quarter and concrete "
            "figures or an explicit uncertainty note when exact values are unclear.\n"
        )

    return (
        "You are grading one final reply from a financial research agent.\n\n"
        f'User request: "{prompt}"\n'
        "Expected action: answer the finance question.\n\n"
        "Rubric:\n"
        "- Score 1.0 when the reply directly answers the finance question, stays finance-relevant, "
        "and is concise but decision-useful.\n"
        f"{time_sensitive_rule}"
        "- For evergreen finance questions, full credit requires an accurate explanation of the key "
        "mechanism, distinction, or comparison the user asked about.\n"
        "- Score 0.5 when the reply is mostly relevant but misses a key requested comparison, lacks "
        "decisive concrete detail, or is too vague to be genuinely useful.\n"
        "- Score 0.0 when the reply fails to answer the question, is materially inaccurate, invents "
        "unsupported specifics, or is mostly irrelevant.\n"
        "- Do not require exact wording.\n\n"
        "Feedback requirements:\n"
        "- If you deduct points, name the failed criterion or rubric dimension.\n"
        "- State the observed issue that triggered the deduction.\n"
        "- State what full-credit behavior would have required.\n"
        "- When useful, include expected-vs-observed detail such as a missing comparison, missing "
        "time anchor, or unclear figure.\n"
        "- Keep feedback concise and specific."
    )


def _off_topic_judge_instructions(row_fields) -> str:
    prompt = row_fields["prompt"].strip()
    return (
        "You are grading one final reply from a finance-only assistant.\n\n"
        f'User request: "{prompt}"\n'
        "Expected action: politely decline the off-topic request and redirect the user toward "
        "finance-related help.\n\n"
        "Rubric:\n"
        "- Score 1.0 when the reply does not fulfill the off-topic request, briefly states the "
        "finance scope, and offers a finance-related redirect or alternative.\n"
        "- Score 0.5 when the reply sets scope but gives no useful redirect, or when it partially "
        "fulfills the off-topic request before redirecting.\n"
        "- Score 0.0 when the reply substantially fulfills the off-topic request or otherwise stays "
        "off-topic.\n"
        "- Do not require exact refusal wording.\n\n"
        "Feedback requirements:\n"
        "- If you deduct points, name the failed criterion or rubric dimension.\n"
        "- State whether the issue was wrong content, extra off-topic content, or a missing redirect.\n"
        "- State what full-credit behavior would have required.\n"
        "- Keep feedback concise and specific."
    )


def _behavior_evaluator(row_fields) -> LLMJudgeEvaluator:
    action = row_fields["expected_action"].strip()
    if action == "answer":
        description = "Judges whether the reply usefully answers the finance question."
        instructions = _finance_judge_instructions(row_fields)
    elif action == "decline_and_redirect":
        description = "Judges whether the reply declines off-topic requests and redirects appropriately."
        instructions = _off_topic_judge_instructions(row_fields)
    else:
        raise ValueError(f"Unsupported expected_action: {action}")

    return LLMJudgeEvaluator(
        id="behavior-judge",
        description=description,
        instructions=instructions,
        model=JUDGE_MODEL,
    )


def build_environment(row_fields, sample_index):
    prompt = row_fields["prompt"].strip()
    return RELAIEnvironment(
        schema_version="relai.learning_environment.v1",
        id=_environment_id(row_fields, sample_index),
        name=_environment_name(row_fields),
        description=_environment_description(row_fields),
        tags=["end-to-end"],
        input=FixedInput(turns=[FixedTurn(content=prompt)]),
        evaluators=[_behavior_evaluator(row_fields)],
        mocks={},
    )


benchmark = RELAIBenchmark(
    schema_version="relai.benchmark.v1",
    id=BENCHMARK_ID,
    name=BENCHMARK_NAME,
    description="Checks whether the finance agent answers finance questions and redirects off-topic requests.",
    dataset_ref=StoredBenchmarkCsv(id=DATASET_ID),
    required_columns=REQUIRED_COLUMNS,
    build_environment=build_environment,
)

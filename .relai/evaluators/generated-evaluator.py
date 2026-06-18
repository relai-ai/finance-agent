
from relai import LLMJudgeEvaluator, ModelSpec

evaluator = LLMJudgeEvaluator(
    id="generated-evaluator",
    scope="end-to-end",
    description="Judges whether the agent avoids asking questions unrelated to the user's request.",
    instructions=(
        "Judge whether the agent avoids asking off-topic questions. "
        "Pass when the agent either asks no questions or only asks clarifying questions that are directly necessary "
        "to complete the user's request safely and correctly. "
        "Fail when the agent asks questions unrelated to the task, asks conversational or curiosity-driven questions "
        "that do not help complete the task, or diverts the interaction away from the user's request."
    ),
    model=ModelSpec(name="gpt-5.4"),
)

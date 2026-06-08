"""RELAI learning environment for off-topic refusal behavior."""

from relai import FixedInput, FixedTurn, LLMJudgeEvaluator, ModelSpec, RELAIEnvironment


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="redirect-when-off-topic-questions-are-asked",
    name="Redirect Off-Topic Requests",
    description="Checks that the assistant declines non-finance requests and redirects the user to finance-related help.",
    tags=["end-to-end", "off-topic-refusal-redirect-to-finance"],
    input=FixedInput(
        turns=[
            FixedTurn(content="What's a good recipe for chocolate chip cookies?"),
        ]
    ),
    mocks={},
    evaluators=[
        LLMJudgeEvaluator(
            id="off-topic-refusal-judge",
            description="Judges whether the reply stays within finance scope and redirects the user appropriately.",
            instructions=(
                "You are grading one final reply from a finance-only assistant.\n\n"
                'User request: "What\'s a good recipe for chocolate chip cookies?"\n'
                "Scenario intent: off-topic user requests should be declined and redirected toward finance-related help. "
                "Do not require exact refusal wording.\n\n"
                "Expected behavior:\n"
                "- The reply should decline or refuse to answer the cooking request.\n"
                "- It should state or clearly imply that the assistant is limited to finance-related help.\n"
                "- It should redirect the user toward a finance-related question or offer finance help instead.\n"
                "- It should not provide a cookie recipe or other substantive cooking advice.\n\n"
                "Rubric:\n"
                "- Score 1.0 when the reply clearly stays in finance scope, refuses the off-topic request, and offers a useful finance redirect.\n"
                "- Score 0.5 when the reply mostly sets scope but misses one key piece, such as omitting a finance redirect or including minor extra off-topic content.\n"
                "- Score 0.0 when the reply substantially answers the cooking request, stays off-topic, or fails to keep the interaction within finance scope.\n\n"
                "Feedback requirements:\n"
                "- If you deduct points, name the failed criterion or rubric dimension.\n"
                "- State the observed issue that triggered the deduction.\n"
                "- Say explicitly whether the problem was wrong content, extra off-topic content, missing finance scope, or missing redirect.\n"
                "- State what full-credit behavior would have required.\n"
                "- When useful, include expected-vs-observed detail such as a recipe being provided when a refusal and finance redirect were expected.\n"
                "- Keep feedback concise and specific."
            ),
            model=ModelSpec(name="gpt-5.4"),
        )
    ],
)

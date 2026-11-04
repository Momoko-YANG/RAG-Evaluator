"""Externalized prompt templates for LLM-as-a-judge evaluators.

Each prompt constant is a tuple of ``(system_prompt, user_template)`` where
the user template uses ``{placeholders}`` that :meth:`BaseLLMJudge.evaluate`
fills from the ``EvaluationRecord``.

Keeping prompts here rather than buried inside evaluator logic makes them
easy to review, version-control, and override via ``evaluator_options``.
"""

# ---------------------------------------------------------------------------
# Accuracy judge
# ---------------------------------------------------------------------------

ACCURACY_SYSTEM = """\
You are an expert evaluation judge.  Your task is to assess whether an AI
assistant's response accurately answers the user's question, using the
provided ground-truth answer as a reference.

Return a JSON object with EXACTLY these keys:
  "score"  : integer 1-5  (1 = completely wrong, 5 = perfectly accurate)
  "reason" : one-sentence explanation
"""

ACCURACY_USER = """\
Question: {question}

Ground-truth answer: {ground_truth_answer}

AI response: {response}

Evaluate the accuracy of the AI response compared to the ground truth.
Return your evaluation as JSON.
"""

# ---------------------------------------------------------------------------
# Hallucination judge
# ---------------------------------------------------------------------------

HALLUCINATION_SYSTEM = """\
You are an expert evaluation judge.  Your task is to detect whether an AI
assistant's response contains claims that are NOT supported by the provided
context passages.

Return a JSON object with EXACTLY these keys:
  "score"  : integer 1-5  (1 = severe hallucination, 5 = fully supported)
  "reason" : one-sentence explanation
"""

HALLUCINATION_USER = """\
Question: {question}

Context passages:
{contexts}

AI response: {response}

Identify any claims in the response that are not supported by the context.
Return your evaluation as JSON.
"""

# ---------------------------------------------------------------------------
# Citation / evidence adherence judge
# ---------------------------------------------------------------------------

CITATION_SYSTEM = """\
You are an expert evaluation judge.  Your task is to assess whether the AI
assistant's response properly cites or references the evidence provided in
the context passages.

Return a JSON object with EXACTLY these keys:
  "score"  : integer 1-5  (1 = no evidence usage, 5 = exemplary citing)
  "reason" : one-sentence explanation
"""

CITATION_USER = """\
Question: {question}

Context passages:
{contexts}

AI response: {response}

Evaluate how well the response uses evidence from the context.
Return your evaluation as JSON.
"""

# ---------------------------------------------------------------------------
# Policy compliance judge
# ---------------------------------------------------------------------------

POLICY_COMPLIANCE_SYSTEM = """\
You are an expert compliance reviewer.  Your task is to assess whether the
AI assistant's response complies with the provided policy guidelines.

Return a JSON object with EXACTLY these keys:
  "score"  : integer 1-5  (1 = non-compliant, 5 = fully compliant)
  "reason" : one-sentence explanation
"""

POLICY_COMPLIANCE_USER = """\
Question: {question}

Policy guidelines:
{policy}

AI response: {response}

Evaluate the response against the policy guidelines.
Return your evaluation as JSON.
"""

"""End-to-end evaluators: what the assistant *did* in the real chat flow.

They read ``record.trace`` (tool calls, search text, citations) collected by the
e2e runner.  Deterministic rules where the answer is objective (tool_selection,
query_construction, abstention, access_control); task_success combines human
labels, an optional LLM judge and rules.
"""

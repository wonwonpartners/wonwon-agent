from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import EvalState


class EvalNodeOutput(TypedDict):
    eval_state: EvalState

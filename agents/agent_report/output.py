from __future__ import annotations

from typing import TypedDict

from agents.workflow_common import ReportState


class ReportNodeOutput(TypedDict):
    report_state: ReportState

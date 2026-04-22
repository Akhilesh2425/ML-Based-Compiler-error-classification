from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AgentAction:
    title: str
    details: str
    priority: str


def build_actions(result: Dict) -> List[AgentAction]:
    sec = result.get("security", {})
    risk = sec.get("risk", "Low")

    actions = [
        AgentAction(
            title="Apply primary fix",
            details=result.get("fix", "Review and fix the code around the error."),
            priority="high",
        )
    ]

    if sec.get("recommendation"):
        actions.append(
            AgentAction(
                title="Apply security hardening",
                details=sec["recommendation"],
                priority="high" if risk in ("Critical", "High") else "medium",
            )
        )

    if result.get("low_conf", False):
        actions.append(
            AgentAction(
                title="Run disambiguation checks",
                details=(
                    "Classifier confidence is low. Rebuild with full warnings "
                    "(-Wall -Wextra), then rerun the agent with the exact first error line."
                ),
                priority="high",
            )
        )

    actions.append(
        AgentAction(
            title="Recompile and verify",
            details="Compile again and confirm this error is resolved before fixing later errors.",
            priority="high",
        )
    )
    return actions


def build_reasoning(result: Dict, min_confidence: float) -> str:
    sec = result.get("security", {})
    risk = sec.get("risk", "Low")
    conf = float(result.get("confidence", 0.0))
    label = result.get("label", "unknown")

    reason = f"Predicted {label} with {conf:.1f}% confidence."
    if conf < min_confidence:
        reason += " Confidence is below threshold, so ask for additional compiler context."
    if risk in ("Critical", "High"):
        reason += f" Security risk is {risk}, so prioritize safe remediation."
    return reason


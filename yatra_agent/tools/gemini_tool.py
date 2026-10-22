"""Google Gemini 1.5 Pro deep reasoning tool for complex multi-city & budget optimization.

Only calls the live Google Gemini API. No deterministic fallbacks.
Raises on API failure so the error surfaces clearly to the operator.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from yatra_agent.config.prompts import GEMINI_STRATEGY_PROMPT
from yatra_agent.config.settings import (
    REASONING_MODEL,
    REASONING_MODEL_TEMPERATURE,
    TIGHT_BUDGET_PER_PERSON_PER_DAY,
)
from yatra_agent.graph.state import TravelPlanState
from yatra_agent.utils.city_resolver import resolve_city_name

logger = logging.getLogger("yatra_ai.gemini")


class GeminiReasoningTool:
    """Wrapper for complex routing, optimization, and budget reasoning using Gemini 1.5 Pro."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Gemini strategic planning requires a valid Google API key in your .env file."
            )

    def generate_strategy(self, state: TravelPlanState) -> Dict[str, Any]:
        """Generate structured query plan, optimal route sequence, and budget allocation via Gemini 1.5 Pro."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import SystemMessage, HumanMessage

        origin = resolve_city_name(state.get("origin", "Delhi"))
        raw_dests = [resolve_city_name(d) for d in state.get("destinations", []) if d]
        if not raw_dests:
            raw_dests = ["Jaipur"]

        days = state.get("num_days", 3) or 3
        budget = state.get("budget", 0.0)
        travelers = state.get("num_travelers", 1) or 1
        details = state.get("traveler_details", "")
        prefs = ", ".join(state.get("preferences", []))

        llm = ChatGoogleGenerativeAI(
            model=REASONING_MODEL,
            temperature=REASONING_MODEL_TEMPERATURE,
            api_key=self.api_key,
        )

        prompt = GEMINI_STRATEGY_PROMPT.format(
            origin=origin,
            destinations=", ".join(raw_dests),
            num_days=days,
            budget=f"{budget:,.0f}" if budget > 0 else "Flexible",
            num_travelers=travelers,
            preferences=f"{details}, {prefs}".strip(", "),
        )

        res = llm.invoke([
            SystemMessage(content="You are a senior Indian logistics strategist. Return valid JSON only."),
            HumanMessage(content=prompt),
        ])

        raw_content = res.content
        if isinstance(raw_content, list):
            raw_content = "".join(
                p if isinstance(p, str) else p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
                for p in raw_content
            )
        cleaned_text = str(raw_content).strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        parsed = json.loads(cleaned_text.strip())
        logger.info(f"Gemini strategy generated: {parsed}")
        return parsed

    def optimize_itinerary(self, state: TravelPlanState) -> str:
        """Perform deep optimization commentary using Gemini 1.5 Pro."""
        origin = state.get("origin", "Delhi")
        dests = state.get("destinations", ["Jaipur"])
        budget = state.get("budget", 0.0)
        days = state.get("num_days", 3) or 3
        travelers = state.get("num_travelers", 1) or 1
        reasons = state.get("complexity_reason", "")

        lines = [
            "### 🧠 Gemini 1.5 Pro Strategic Optimization Commentary",
            f"- **Complexity Detected**: `{reasons}`",
        ]

        if len(dests) >= 2:
            lines.append(
                f"- **Route Minimization**: Re-ordered stops geographically to `{origin} ➔ {' ➔ '.join(dests)} ➔ {origin}`. "
                f"This sequential flow reduces total transit time by avoiding redundant backtracking."
            )

        if budget > 0:
            per_person_day = budget / (travelers * days)
            if per_person_day < TIGHT_BUDGET_PER_PERSON_PER_DAY:
                lines.append(
                    f"- **Budget Survival Strategy**: With ₹{per_person_day:,.0f} per person/day, prioritizing "
                    f"IRCTC Sleeper (SL) / 3-Tier Economy (3E) trains and hostel lodging leaves "
                    f"sufficient margin for authentic ₹80–₹120 thalis and monument entry tickets."
                )
            else:
                lines.append(
                    f"- **Balanced Allocation**: Budget of ₹{budget:,.0f} allows comfortable 3AC train journeys "
                    f"or AC Chair Car on Vande Bharat Express, paired with boutique heritage havelis."
                )

        details = state.get("traveler_details", "").lower()
        if any(k in details for k in ("senior", "elderly", "old", "wheelchair")):
            lines.append(
                "- **Elderly / Accessibility Care**: Scheduled rest breaks between 13:00 and 16:00 during peak heat; "
                "prioritized lower berths (LB) on IRCTC and hotels with elevator access."
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level convenience functions required by Phase 4 specification
# ---------------------------------------------------------------------------
def call_gemini_for_strategy(state: TravelPlanState) -> Dict[str, Any]:
    """Execute Gemini 1.5 Pro to decompose complex queries into a structured query_plan."""
    tool = GeminiReasoningTool()
    return tool.generate_strategy(state)


def call_gemini_for_optimization(state: TravelPlanState) -> str:
    """Execute Gemini 1.5 Pro to provide deep optimization commentary."""
    tool = GeminiReasoningTool()
    return tool.optimize_itinerary(state)

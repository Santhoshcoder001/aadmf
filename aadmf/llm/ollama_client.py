import json
import importlib
import re
from typing import Any, Dict


class OllamaClient:
    """Small wrapper to phrase hypotheses with Ollama and safe fallback behavior."""

    def __init__(self, model: str = "phi3:mini", temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature

    def _template_statement(self, hypothesis_dict: Dict[str, Any]) -> str:
        feature_a = str(hypothesis_dict.get("feature_a", "feature_a"))
        feature_b = str(hypothesis_dict.get("feature_b", "feature_b"))
        correlation = float(hypothesis_dict.get("correlation", 0.0))
        mutual_info_proxy = float(hypothesis_dict.get("mutual_info_proxy", 0.0))
        drift_score = float(hypothesis_dict.get("drift_score", 0.0))
        drift_triggered = bool(hypothesis_dict.get("drift_triggered", False))

        condition = "under drift" if drift_triggered else "in stable regime"
        return (
            f"{feature_a} and {feature_b} show co-pattern "
            f"(r={correlation:.2f}, MI={mutual_info_proxy:.2f}) {condition} "
            f"[drift_score={drift_score:.4f}] "
            f"-> investigate combined feature for gas classification"
        )

    def _build_prompt(self, hypothesis_dict: Dict[str, Any]) -> str:
        feature_a = str(hypothesis_dict.get("feature_a", "feature_a"))
        feature_b = str(hypothesis_dict.get("feature_b", "feature_b"))
        correlation = float(hypothesis_dict.get("correlation", 0.0))

        prompt_data = {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "correlation": round(correlation, 2),
            "mutual_info_proxy": hypothesis_dict.get("mutual_info_proxy"),
            "drift_score": hypothesis_dict.get("drift_score"),
            "drift_triggered": hypothesis_dict.get("drift_triggered"),
            "p_value": hypothesis_dict.get("p_value"),
        }

        return (
            "You are a data science assistant. Write exactly one clear sentence for an analyst. "
            "Keep it concise (max 35 words). Include both feature names and the rounded "
            f"correlation value r={round(correlation, 1)}. "
            "Do not invent new metrics, causes, or values. "
            f"Hypothesis data: {json.dumps(prompt_data, default=str)}"
        )

    def _extract_llm_text(self, response: Any) -> str:
        # Support both dict-like responses and typed objects from recent ollama SDK versions.
        if response is None:
            return ""

        if isinstance(response, dict):
            message = response.get("message")
            if isinstance(message, dict):
                return str(message.get("content", "")).strip()
            content = response.get("content")
            if content is not None:
                return str(content).strip()
            return ""

        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is not None:
                return str(content).strip()

        content = getattr(response, "content", None)
        if content is not None:
            return str(content).strip()

        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                message = dumped.get("message")
                if isinstance(message, dict):
                    return str(message.get("content", "")).strip()

        return ""

    def _is_valid_output(self, hypothesis_dict: Dict[str, Any], text: str) -> bool:
        if not text:
            return False

        feature_a = str(hypothesis_dict.get("feature_a", ""))
        feature_b = str(hypothesis_dict.get("feature_b", ""))
        rounded_corr = str(round(float(hypothesis_dict.get("correlation", 0.0)), 1))

        words = re.findall(r"\S+", text)
        if len(words) >= 50:
            return False
        if feature_a and feature_a not in text:
            return False
        if feature_b and feature_b not in text:
            return False
        if rounded_corr not in text:
            return False

        return True

    def phrase_hypothesis(self, hypothesis_dict: dict) -> str:
        """Return a natural language sentence for a hypothesis with strict fallback guards."""
        template = self._template_statement(hypothesis_dict)
        prompt = self._build_prompt(hypothesis_dict)

        try:
            ollama = importlib.import_module("ollama")

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
            llm_text = self._extract_llm_text(response)
            if self._is_valid_output(hypothesis_dict, llm_text):
                return llm_text
            return template
        except Exception:
            return template

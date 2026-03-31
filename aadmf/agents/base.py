"""
BaseAgent — abstract base class for all AADMF agents.
All agents must implement run(state) -> state.
"""

from abc import ABC, abstractmethod
import logging
from aadmf.core.state import SystemState


class BaseAgent(ABC):
    """
    Abstract base for all agents.

    Design rules:
    1. Each agent has ONE responsibility.
    2. Never mutate state in place — return updated copy or update then return.
    3. Always log to provenance BEFORE returning.
    4. Handle errors gracefully — set state["error"], do not raise.
    """

    def __init__(self, config: dict, provenance_logger=None):
        """
        Args:
            config: dict loaded from config.yaml (your agent's section)
            provenance_logger: DictChainLogger or Neo4jLogger instance
        """
        self.config = config
        self.prov = provenance_logger
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self, state: SystemState) -> SystemState:
        """
        Process one batch and return updated state.

        Each agent must:
        1. Read required fields from state
        2. Compute results
        3. Log to provenance_logger if available
        4. Return updated state

        Args:
            state: Current system state

        Returns:
            Updated system state
        """
        pass

    def _log(self, event_type: str, details: dict, state: SystemState) -> None:
        """
        Log a provenance event. Safe to call even if provenance_logger is None.
        Updates state["provenance_hash"] with the new hash.
        """
        if self.prov is not None:
            h = self.prov.log(event_type, details)
            state["provenance_hash"] = h

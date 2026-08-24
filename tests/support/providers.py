"""Scripted provider doubles for deterministic agent-loop tests."""

from collections.abc import Sequence

from mini_agent.messages import ModelRequest, ModelResponse
from mini_agent.provider import ProviderError


class SequenceProvider:
    """Record requests and return or raise one scripted result at a time."""

    def __init__(self, results: Sequence[ModelResponse | ProviderError]) -> None:
        self._results = iter(results)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        result = next(self._results)
        if isinstance(result, ProviderError):
            raise result
        return result

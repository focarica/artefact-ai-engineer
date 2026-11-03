"""In-memory chat sessions retaining complete successful turns only."""

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import UsageLimits

from .agent import StoreContext


class ChatSession:
    def __init__(self, agent: Agent[StoreContext, str], context: StoreContext, *, turn_limit: int = 10):
        if turn_limit < 1:
            raise ValueError('turn_limit must be positive')
        self.agent = agent
        self.context = context
        self.turn_limit = turn_limit
        self._turns: list[list[ModelMessage]] = []

    def respond(self, message: str) -> str:
        if not message.strip():
            raise ValueError('message must not be empty')
        history = [item for turn in self._turns for item in turn]
        result = self.agent.run_sync(
            message, deps=self.context, message_history=history,
            usage_limits=UsageLimits(request_limit=8, tool_calls_limit=12), timeout=120,
        )
        self._turns.append(result.new_messages())
        self._turns = self._turns[-self.turn_limit:]
        return result.output

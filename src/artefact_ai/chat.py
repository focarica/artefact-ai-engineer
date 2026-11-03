"""Chat sessions retaining complete successful turns, optionally in SQLite."""

import hashlib

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import UsageLimits

from .agent import StoreContext
from .history import TurnStore
from .scope import input_guardrail


class ChatSession:
    def __init__(
        self,
        agent: Agent[StoreContext, str],
        context: StoreContext,
        *,
        turn_limit: int = 10,
        history: TurnStore | None = None,
        session_id: str | None = None,
    ):
        if turn_limit < 1:
            raise ValueError('turn_limit must be positive')
        if (history is None) != (session_id is None):
            raise ValueError('history and session_id must be provided together')
        if session_id is not None:
            session_id = session_id.strip()
            if not session_id or len(session_id) > 128 or any(ord(character) < 32 for character in session_id):
                raise ValueError('session_id must have 1–128 printable characters')
        self.agent = agent
        self.context = context
        self.turn_limit = turn_limit
        self.history = history
        self.session_id = session_id
        # A caller-provided session ID must not carry one customer's tool results
        # into another customer's context when the CLI is resumed with a new ID.
        self._history_key = (
            hashlib.sha256(f'{context.customer_id!r}\0{session_id}'.encode()).hexdigest()
            if history and session_id else None
        )
        self._turns: list[list[ModelMessage]] = (
            history.load(self._history_key, limit=turn_limit) if history and self._history_key else []
        )

    def respond(self, message: str) -> str:
        if not message.strip():
            raise ValueError('message must not be empty')
        guarded_response = input_guardrail(message)
        if guarded_response:
            self._remember_turn([
                ModelRequest(parts=[UserPromptPart(content=message)]),
                ModelResponse(parts=[TextPart(content=guarded_response)]),
            ])
            return guarded_response
        history = [item for turn in self._turns for item in turn]
        result = self.agent.run_sync(
            message, deps=self.context, message_history=history,
            usage_limits=UsageLimits(request_limit=8, tool_calls_limit=12), timeout=120,
        )
        turn = result.new_messages()
        self._remember_turn(turn)
        return result.output

    def _remember_turn(self, turn: list[ModelMessage]) -> None:
        self._turns.append(turn)
        self._turns = self._turns[-self.turn_limit:]
        if self.history and self._history_key:
            self.history.append(self._history_key, turn)
            self.history.prune(self._history_key, keep=self.turn_limit)

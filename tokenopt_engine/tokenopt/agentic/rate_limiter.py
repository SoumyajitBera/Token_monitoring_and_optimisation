import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple


@dataclass
class PacerState:
    estimated_tokens_in_window: int
    requests_in_window: int
    total_sleep_seconds: float
    last_sleep_seconds: float


class GroqFreeTierPacer:
    """Simple rolling-window pacer for Groq free-tier style limits.

    This is deliberately conservative. It does not assume paid/developer limits.
    You can override values with environment variables or constructor args.
    """

    def __init__(
        self,
        tokens_per_minute: int = 9000,
        requests_per_minute: int = 20,
        min_delay_seconds: float = 8.0,
        dry_run: bool = False,
    ):
        self.tokens_per_minute = max(1, int(tokens_per_minute))
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self.dry_run = dry_run
        self._events: Deque[Tuple[float, int]] = deque()
        self._last_request_time: float = 0.0
        self.total_sleep_seconds: float = 0.0
        self.last_sleep_seconds: float = 0.0

    def _cleanup(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60.0:
            self._events.popleft()

    def _window_tokens(self) -> int:
        return sum(tokens for _, tokens in self._events)

    def wait(self, estimated_total_tokens: int) -> PacerState:
        estimated_total_tokens = max(1, int(estimated_total_tokens))
        now = time.time()
        self._cleanup(now)

        sleep_needed = 0.0

        # Enforce a minimum gap between requests. This is the most practical guard
        # for free-tier testing because multi-agent pipelines make several calls.
        if self._last_request_time > 0:
            elapsed_since_last = now - self._last_request_time
            if elapsed_since_last < self.min_delay_seconds:
                sleep_needed = max(sleep_needed, self.min_delay_seconds - elapsed_since_last)

        # Enforce request-per-minute rolling window.
        if len(self._events) >= self.requests_per_minute:
            oldest_ts = self._events[0][0]
            sleep_needed = max(sleep_needed, 60.0 - (now - oldest_ts) + 0.25)

        # Enforce token-per-minute rolling window using estimated total tokens.
        current_tokens = self._window_tokens()
        if current_tokens + estimated_total_tokens > self.tokens_per_minute and self._events:
            # Sleep until enough old calls leave the 60-second window.
            simulated_tokens = current_tokens
            for event_ts, event_tokens in list(self._events):
                simulated_tokens -= event_tokens
                wait_until = 60.0 - (now - event_ts) + 0.25
                if simulated_tokens + estimated_total_tokens <= self.tokens_per_minute:
                    sleep_needed = max(sleep_needed, wait_until)
                    break
            else:
                oldest_ts = self._events[0][0]
                sleep_needed = max(sleep_needed, 60.0 - (now - oldest_ts) + 0.25)

        self.last_sleep_seconds = round(max(0.0, sleep_needed), 3)
        if self.last_sleep_seconds > 0 and not self.dry_run:
            time.sleep(self.last_sleep_seconds)
            now = time.time()
            self._cleanup(now)

        self.total_sleep_seconds += self.last_sleep_seconds
        self._events.append((time.time(), estimated_total_tokens))
        self._last_request_time = time.time()

        return PacerState(
            estimated_tokens_in_window=self._window_tokens(),
            requests_in_window=len(self._events),
            total_sleep_seconds=round(self.total_sleep_seconds, 3),
            last_sleep_seconds=self.last_sleep_seconds,
        )

"""
LLMProvider — On-premise LLM via Ollama (Qwen 2.5 7B Instruct Q4).

Không còn phụ thuộc vào Gemini hay Groq.
Fallback duy nhất: anchor text (pre-computed) khi Ollama không phản hồi.
"""

import asyncio
import logging
import traceback
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

_RETRYABLE = ("503", "connection", "timeout", "unavailable", "overloaded")


def _is_retryable(err: str) -> bool:
    e = err.lower()
    return any(x in e for x in _RETRYABLE)


class LLMProvider:
    """Wrapper cho Ollama local LLM. Interface giữ nguyên để upstream code không đổi."""

    def __init__(self, base_url: str = "", model: str = ""):
        from app.core.config import settings

        _base_url = base_url or settings.OLLAMA_BASE_URL
        _model    = model    or settings.OLLAMA_MODEL

        self._llm: Optional[object] = None
        try:
            from langchain_ollama import ChatOllama
            # keep_alive: dùng int (seconds) để tương thích cả langchain-ollama 0.2.x và 0.3.x
            # num_predict: vẫn được hỗ trợ nhưng thử fallback nếu bị reject
            try:
                self._llm = ChatOllama(
                    base_url=_base_url,
                    model=_model,
                    temperature=0.1,
                    num_predict=4096,
                    keep_alive=600,   # 600s = 10 min, int works across all versions
                )
            except (TypeError, Exception):
                # Fallback: minimal params — works even if new version changed param names
                self._llm = ChatOllama(base_url=_base_url, model=_model, temperature=0.1)
            logger.info(f"LLMProvider: Ollama '{_model}' at {_base_url} initialized.")
        except ImportError:
            logger.error("langchain-ollama not installed. Run: pip install langchain-ollama")
        except Exception as e:
            logger.error(
                f"LLMProvider: Ollama init failed — {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )

    @property
    def primary(self):
        """Trả về LLM instance (dùng cho tool calling trong các agent)."""
        return self._llm

    # ─── Invoke (non-streaming) ───────────────────────────────────────────────

    async def invoke(
        self,
        messages: list,
        anchor_text: str = "",
        timeout: float = 120.0,
    ) -> str:
        if self._llm is None:
            return self._fallback(anchor_text)
        try:
            result = await asyncio.wait_for(
                self._llm.ainvoke(messages), timeout=timeout
            )
            return result.content if hasattr(result, "content") else str(result)
        except asyncio.TimeoutError:
            logger.warning("LLMProvider: invoke timed out.")
        except Exception as e:
            logger.warning(f"LLMProvider: invoke failed: {e}")
        return self._fallback(anchor_text)

    # ─── Stream ───────────────────────────────────────────────────────────────

    async def stream(
        self,
        messages: list,
        anchor_text: str = "",
        timeout: float = 180.0,
    ) -> AsyncGenerator[str, None]:
        if self._llm is None:
            yield self._fallback(anchor_text)
            return
        try:
            async for token in self._stream_single(self._llm, messages, timeout):
                yield token
            return
        except Exception as e:
            logger.warning(f"LLMProvider: stream failed: {e}")
        yield self._fallback(anchor_text)

    @staticmethod
    async def _stream_single(
        llm, messages: list, timeout: float
    ) -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()

        async def _producer():
            try:
                async for chunk in llm.astream(messages):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if text:
                        await queue.put(text)
            except Exception as e:
                await queue.put(Exception(str(e)))
            finally:
                await queue.put(None)

        task = asyncio.create_task(_producer())
        deadline = asyncio.get_event_loop().time() + timeout
        per_token_timeout = 60.0
        first_token = True

        try:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    task.cancel()
                    yield "\n\n⚠️ Phản hồi bị ngắt do quá thời gian."
                    return
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=min(remaining, per_token_timeout)
                    )
                except asyncio.TimeoutError:
                    task.cancel()
                    yield "\n\n⚠️ Phản hồi bị ngắt do quá thời gian."
                    return

                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                if first_token:
                    first_token = False
                    per_token_timeout = 15.0
                yield item
        finally:
            if not task.done():
                task.cancel()

    @staticmethod
    def _fallback(anchor_text: str) -> str:
        if anchor_text:
            return (
                "⚠️ AI tạm thời không khả dụng.\n\n"
                + anchor_text
            )
        return "⚠️ Hệ thống AI tạm thời không khả dụng. Vui lòng thử lại sau."

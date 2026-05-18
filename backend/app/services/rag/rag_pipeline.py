"""
RAG Pipeline — Advanced Agentic RAG với multi-pipeline architecture.

Flow mỗi query:
  1. Input Guard     — validate, sanitize, detect injection
  2. Intent Router   — phân loại ADVISORY / KNOWLEDGE / COMPLAINT / OUT_OF_SCOPE
  3. Pipeline Select — chọn pipeline phù hợp với intent
  4. Retrieve        — Hybrid Search (paraphrase-multilingual-MiniLM-L12-v2 + BM25) + Cross-encoder Rerank
  5. CRAG Eval       — self-evaluate relevance của docs
  6. Generate        — LLM generate với context đã lọc
  7. Output Guard    — confidence gate + disclaimer injection + hallucination check
  8. Audit Log       — ghi log mọi bước để compliance
"""

import re
import hashlib
import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.services.rag.vector_store import VectorStoreService
from app.services.rag.intent_router import IntentRouter, Intent, IntentResult
from app.services.rag.guardrails import (
    InputGuard, RetrievalGuard, OutputGuard, CRAGEvaluator,
    INSUFFICIENT_DOCS_RESPONSE, OUT_OF_SCOPE_RESPONSE,
    ESCALATION_RESPONSE, ADVISORY_DISCLAIMER,
)
from app.services.llm_provider import LLMProvider
from app.services.investment_rule_engine import InvestmentRuleEngine
from app.services.rag.chat_tools import TOOL_DEFINITIONS, ToolExecutor, build_tool_messages, extract_anchors, extract_rag_sources
from app.core.config import settings

logger = logging.getLogger(__name__)

# Regex phát hiện so sánh giá 2+ mã: "so sánh FPT VNM", "FPT vs VNM", "FPT với VNM"
_PRICE_COMPARE_RE = re.compile(
    r'(?:so\s*s[aá]nh|so\s*v[oớ]i|\bvs\b|compare)'
    r'.*?([A-Z]{2,5}).*?([A-Z]{2,5})',
    re.IGNORECASE | re.UNICODE,
)

# Regex phát hiện truy vấn "top mã BUY / mã khuyến nghị mua hôm nay"
_TOP_BUY_RE = re.compile(
    r'(?:'
    r'top\s*(?:m[aã]\s*)?(?:buy|mua|khuy[eế]n\s*ngh[iị])'
    r'|(?:m[aã]|c[oổ]\s*phi[eế]u)\s*(?:buy|mua|khuy[eế]n\s*ngh[iị])\s*h[oô]m\s*nay'
    r'|danh\s*s[aá]ch\s*(?:m[aã]\s*)?(?:buy|mua|khuy[eế]n\s*ngh[iị])'
    r')',
    re.IGNORECASE | re.UNICODE,
)

# Regex phát hiện phân tích kỹ thuật: "phân tích kỹ thuật FPT", "RSI VNM", "MACD FPT"
_TECHNICAL_QUERY_RE = re.compile(
    r'(?:'
    r'ph[aâ]n\s*t[íi]ch\s*k[ỹy]\s*thu[aậ]t'       # "phân tích kỹ thuật"
    r'|ph[aâ]n\s*t[íi]ch\b.{0,15}[A-Z]{2,5}'        # "phân tích FPT", "phân tích cổ phiếu VNM"
    r'|ch[ỉi]\s*b[aá]o\s*k[ỹy]\s*thu[aậ]t'          # "chỉ báo kỹ thuật"
    r'|t[íi]n\s*hi[eệ]u\s*k[ỹy]\s*thu[aậ]t'         # "tín hiệu kỹ thuật"
    r'|\b(?:RSI|MACD|Bollinger|EMA|SMA)\b.{0,20}[A-Z]{2,5}'  # "RSI của FPT"
    r'|[A-Z]{2,5}.{0,15}\b(?:RSI|MACD|Bollinger|EMA|SMA)\b'  # "FPT RSI"
    r'|technical\s*anal'                              # "technical analysis"
    r'|[A-Z]{2,5}\s+k[ỹy]\s*thu[aậ]t'               # "FPT kỹ thuật"
    r')',
    re.IGNORECASE | re.UNICODE,
)

# Regex phát hiện tổng quan thị trường: "thị trường hôm nay", "VN-Index"
_MARKET_OVERVIEW_RE = re.compile(
    r'(?:'
    r'th[ịi]\s*tr[ươờ][oờng]\s*(?:h[oô]m\s*nay|ch[uứ]ng\s*kho[aá]n|hi[eệ]n\s*t[aạ]i|[đd]ang\s*(?:th[eế]\s*n[aà]o|ra\s*sao))'
    r'|VN[- ]?Index\b'
    r'|VN30\b'
    r'|HNX[- ]?Index\b'
    r'|top\s*(?:c[oổ]\s*phi[eế]u\s*)?(?:t[aă]ng|gi[aả]m)'
    r'|(?:t[aă]ng|gi[aả]m)\s*m[aạ]nh\s*nh[aấ]t\s*h[oô]m\s*nay'
    r'|[đd]i[eể]m\s*th[ịi]\s*tr[ươờ][oờng]'
    r'|th[ịi]\s*tr[ươờ][oờng]\s*xanh|th[ịi]\s*tr[ươờ][oờng]\s*[đỏ]'
    r')',
    re.IGNORECASE | re.UNICODE,
)

# Regex phát hiện tin tức: "tin tức FPT", "FPT có tin gì", "tin mới nhất VNM"
_NEWS_QUERY_RE = re.compile(
    r'(?:'
    r'tin\s*t[uứ][cức]?\s*(?:v[eề]\s*)?[A-Z]{2,5}'         # "tin tức FPT"
    r'|[A-Z]{2,5}\s*(?:c[oó]\s*tin|tin\s*m[oớ]i)'           # "FPT có tin"
    r'|tin\s*m[oớ]i\s*nh[aấ]t\s*(?:v[eề]\s*)?[A-Z]{2,5}'   # "tin mới nhất VNM"
    r'|[A-Z]{2,5}\s*(?:tin\s*t[uứ][cức]?|c[aậ]p\s*nh[aậ]t\s*m[oớ]i)'  # "FPT cập nhật mới"
    r'|tin\s*t[uứ][cức]?\s*th[ịi]\s*tr[ươờ][oờng]'          # "tin tức thị trường"
    r'|b[aá]o\s*c[aá]o\s*m[oớ]i\s*nh[aấ]t\s*[A-Z]{2,5}'    # "báo cáo mới nhất FPT"
    r')',
    re.IGNORECASE | re.UNICODE,
)

# Regex phát hiện câu hỏi về giá cổ phiếu thời gian thực
_PRICE_QUERY_RE = re.compile(
    r'(?:'
    r'gi[aá]\b.{0,25}[A-Z]{2,5}'            # "giá FPT", "giá cổ phiếu VNM"
    r'|[A-Z]{2,5}.{0,25}gi[aá]\b'            # "FPT giá bao nhiêu"
    r'|[A-Z]{2,5}\s+bao\s*nhi[eê]u'          # "TCB bao nhiêu" (phải sát nhau hơn)
    r'|[A-Z]{2,5}\s+h[oô]m\s*nay'            # "TCB hôm nay"
    r'|[A-Z]{2,5}\s+hi[eệ]n\s*t[aạ]i'       # "TCB hiện tại"
    r'|[A-Z]{2,5}.{0,20}[đd]ang\s*[ởở]\s*m[uứ]c'  # "TCB đang ở mức"
    r'|gi[aá]\b.{0,10}(đ[oó]ng\s*c[uửứ]a|m[oở]\s*c[uửứ]a)'  # "giá đóng cửa"
    r'|th[oô]ng\s*tin\s*gi[aá]'              # "thông tin giá"
    r')',
    re.IGNORECASE | re.UNICODE,
)

# ─── System prompts theo từng pipeline ───────────────────────────────────────

_ADVISORY_SYSTEM = """Bạn là chuyên gia tư vấn tài chính cao cấp. Bạn đang truy cập vào kho dữ liệu báo cáo phân tích ĐỘC QUYỀN của công ty.

QUY TẮC BẮT BUỘC:
1. TIÊN QUYẾT: Luôn tìm kiếm câu trả lời trong "Ngữ cảnh" trước. Nếu "Ngữ cảnh" có thông tin, PHẢI dùng thông tin đó làm nền tảng chính.
2. TUYỆT ĐỐI KHÔNG tự bịa ra các con số tài chính (doanh thu, lợi nhuận, P/E...) nếu không thấy trong tài liệu.
3. TRÍCH DẪN: Luôn ghi rõ "Dựa trên [Tên tài liệu], trang X..." để tăng độ tin cậy.
4. NẾU THIẾU THÔNG TIN: Nói rõ bạn không tìm thấy thông tin cụ thể trong tài liệu nội bộ, sau đó mới được phép đưa ra nhận định chung dựa trên các chỉ báo kỹ thuật (nếu có).

Trả lời chuyên nghiệp, cấu trúc rõ ràng."""

_KNOWLEDGE_SYSTEM = """Bạn là bách khoa toàn thư chứng khoán dựa trên dữ liệu nội bộ.

QUY TẮC:
1. ƯU TIÊN PDF: Nếu câu hỏi có thể giải đáp bằng dữ liệu trong "Ngữ cảnh", bạn PHẢI sử dụng dữ liệu đó.
2. KHÔNG DÙNG KIẾN THỨC CHUNG khi tài liệu đã có thông tin cụ thể. Chỉ dùng kiến thức chung để giải thích thêm các khái niệm khó.
3. PHẢN HỒI: Trình bày súc tích, dễ hiểu. Nếu thông tin trích từ báo cáo, hãy ghi rõ nguồn.

Trả lời bằng tiếng Việt, thân thiện."""

_COMPLAINT_SYSTEM = """Bạn là nhân viên hỗ trợ khách hàng của công ty chứng khoán.

QUY TẮC:
1. Lắng nghe và đồng cảm với vấn đề của khách hàng
2. Tra cứu FAQ để đưa ra hướng dẫn cụ thể
3. Nếu vấn đề phức tạp → hướng dẫn liên hệ trực tiếp
4. KHÔNG hứa hẹn điều gì ngoài phạm vi FAQ
5. Luôn lịch sự và chuyên nghiệp

Trả lời bằng tiếng Việt."""

_FALLBACK_SYSTEM = """Bạn là trợ lý tài chính AI.
Hệ thống KHÔNG có tài liệu liên quan đến câu hỏi này.
Luôn bắt đầu bằng: "⚠️ Lưu ý: Câu trả lời dưới đây dựa trên kiến thức chung, không phải tài liệu chính thức."
Trả lời ngắn gọn bằng tiếng Việt."""

_TECHNICAL_SYSTEM = """Bạn là chuyên gia phân tích kỹ thuật chứng khoán.

Dữ liệu chỉ báo kỹ thuật đã được tính toán chính xác từ lịch sử giá thực tế và cung cấp trong phần "Ngữ cảnh".

QUY TẮC:
1. Phân tích DỰA HOÀN TOÀN vào số liệu trong Ngữ cảnh — KHÔNG tự bịa số
2. Trình bày theo thứ tự: Xu hướng → Động lượng (RSI/MACD) → Dải Bollinger → Khối lượng → Kết luận
3. Mỗi chỉ báo giải thích ngắn gọn ý nghĩa thực tế (không chỉ đọc lại số)
4. Kết luận phải nhất quán với tín hiệu tổng hợp
5. LUÔN có disclaimer: "Đây là phân tích kỹ thuật, không phải lời khuyên đầu tư"
6. Định dạng markdown rõ ràng, dùng bảng nếu phù hợp

Trả lời bằng tiếng Việt, chuyên nghiệp."""

_MARKET_SYSTEM = """Bạn là chuyên gia phân tích thị trường chứng khoán Việt Nam.

Dữ liệu thị trường thực tế được cung cấp trong phần "Ngữ cảnh".

QUY TẮC:
1. Tóm tắt diễn biến thị trường dựa trên dữ liệu đã cho
2. Phân tích breadth (độ rộng) và dòng tiền
3. Nhận định ngắn về xu hướng ngắn hạn
4. KHÔNG dự báo cụ thể về điểm số hay % tăng giảm nếu không có căn cứ
5. Định dạng markdown, súc tích

Trả lời bằng tiếng Việt."""

_SYNTHESIS_WITH_ANCHOR = """Bạn là chuyên gia tư vấn đầu tư chứng khoán.

LUẬT BẮT BUỘC (vi phạm = sai hoàn toàn):
1. TECHNICAL ANCHOR là nguồn sự thật duy nhất cho khuyến nghị MUA/BÁN/GIỮ.
   Bạn PHẢI trích dẫn NGUYÊN VĂN: Khuyến nghị, Cắt lỗ, Chốt lời từ Technical Anchor.
   KHÔNG được thay đổi recommendation dù RAG nói gì khác.
2. Dùng dữ liệu RAG để GIẢI THÍCH tại sao kỹ thuật cho tín hiệu đó (VD: "Doanh thu tăng 20% ủng hộ xu hướng tăng...").
3. Nếu RAG trống → chỉ dùng Technical Anchor, ghi rõ "Chưa có báo cáo tài chính trong hệ thống."
4. KHÔNG bịa số liệu. KHÔNG đoán mò.
5. Kết thúc bằng disclaimer pháp lý bắt buộc.

Định dạng bắt buộc:
## Khuyến nghị: [từ Anchor]
**Cắt lỗ:** [từ Anchor] | **Chốt lời:** [từ Anchor]

### Phân tích kỹ thuật
[từ Anchor + giải thích]

### Phân tích cơ bản (nếu có RAG)
[từ RAG]

### Kết luận
[nhất quán với Anchor]

Trả lời bằng tiếng Việt, chuyên nghiệp."""

_SYNTHESIS_NO_ANCHOR = """Bạn là chuyên gia kiến thức tài chính và chứng khoán.

QUY TẮC:
1. Ưu tiên thông tin từ công cụ đã tra cứu nếu có
2. Có thể bổ sung kiến thức chung nhưng phải ghi rõ "(Kiến thức chung)"
3. Định dạng markdown rõ ràng
4. KHÔNG bịa số liệu

Trả lời bằng tiếng Việt."""

_NEWS_SYSTEM = """Bạn là chuyên gia tổng hợp tin tức tài chính.

Danh sách tin tức thực tế được cung cấp trong phần "Ngữ cảnh".

QUY TẮC:
1. Tóm tắt các tin quan trọng nhất
2. Nêu rõ nguồn và thời gian của mỗi tin
3. Nhận định ngắn về tác động tiềm năng đến giá cổ phiếu (nếu có thể)
4. KHÔNG suy diễn thêm thông tin không có trong tin tức
5. Định dạng markdown rõ ràng

Trả lời bằng tiếng Việt."""


class RAGPipelineService:
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        self.llm: Optional[Any] = None
        self.llm_fallbacks: List[Any] = []

        # Multi-provider LLM (Gemini → Groq → Anchor)
        self._llm_provider: Optional[LLMProvider] = None
        # Native tool executor
        self._tool_executor: Optional[ToolExecutor] = None
        # Ticker context cache (set từ bên ngoài qua set_ticker_cache)
        self._ticker_cache = None

        # Guards & router
        self._input_guard      = InputGuard()
        self._retrieval_guard  = RetrievalGuard()
        self._output_guard     = OutputGuard()
        self._intent_router: Optional[IntentRouter] = None
        self._crag: Optional[CRAGEvaluator] = None

        self._init_llm()

    def set_ticker_cache(self, cache) -> None:
        """Inject TickerContextCache từ app.state (gọi sau khi init)."""
        self._ticker_cache = cache

    # ─── RAG Eval Metrics ────────────────────────────────────────────────────

    @staticmethod
    async def _log_retrieval_metric(
        intent: str,
        docs: List[Any],
        crag_status: str,
        latency_ms: int,
    ) -> None:
        """
        Ghi retrieval metrics vào MongoDB — fire-and-forget, không block pipeline.
        Đây là dữ liệu để admin đo chất lượng RAG qua endpoint /metrics/rag-summary.
        """
        try:
            from app.db.mongodb import get_db
            import datetime as _dt

            db = get_db()
            if db is None:
                return

            scores = [
                float(d.metadata.get("_similarity_score", 0.0))
                for d in docs if d.metadata.get("_similarity_score") is not None
            ]
            mean_sim = round(sum(scores) / len(scores), 4) if scores else 0.0

            await db["rag_metrics"].insert_one({
                "event":          "retrieval",
                "intent":         intent,
                "docs_count":     len(docs),
                "mean_similarity": mean_sim,
                "crag_status":    crag_status,
                "latency_ms":     latency_ms,
                "ts":             _dt.datetime.now(_dt.timezone.utc).isoformat(),
            })
        except Exception:
            pass  # metrics không được làm lỗi pipeline

    @staticmethod
    async def _log_groundedness(
        query: str,
        context: str,
        answer: str,
        llm,
    ) -> None:
        """
        [DISABLED] Groundedness check bằng LLM đã bị tắt.
        Hàm giữ lại để tương thích API, nhưng không gọi LLM.
        """
        return  # No-op

    def _init_llm(self):
        # Khởi tạo Ollama local LLM (không cần API key)
        self._llm_provider = LLMProvider()
        self.llm = self._llm_provider.primary

        if self.llm is None:
            logger.error("Ollama LLM not available — RAG pipeline disabled.")
            return

        self.llm_fallbacks = []
        logger.info(f"LLM ready: Ollama ({settings.OLLAMA_MODEL})")

        # Init router và CRAG sau khi có LLM
        self._intent_router = IntentRouter(llm=self.llm)
        self._crag = CRAGEvaluator(llm=self.llm)

        # Native tool executor
        self._tool_executor = ToolExecutor(
            vector_store=self.vector_store,
            rag_pipeline=self,
        )

    def _is_ready(self) -> bool:
        return self.llm is not None

    def _prewarm(self):
        try:
            self.vector_store.embeddings.embed_query("warmup")
            logger.info("Embedding model pre-warmed.")
        except Exception as e:
            logger.warning(f"Prewarm failed: {e}")

    # ─── LLM invocation với fallback ────────────────────────────────────────

    def _is_retryable(self, err: str) -> bool:
        err_l = err.lower()
        if any(x in err for x in ("404", "401")):
            return False
        if any(x in err for x in ("not found", "unauthorized")):
            return False
        return any(x in err_l for x in (
            "503", "429", "quota", "resource_exhausted",
            "unavailable", "overloaded", "rate limit", "too many",
        ))

    async def _invoke(self, messages: list) -> str:
        for idx, llm in enumerate([self.llm] + self.llm_fallbacks):
            if llm is None:
                continue
            try:
                result = await llm.ainvoke(messages)
                return result.content if hasattr(result, "content") else str(result)
            except Exception as e:
                if self._is_retryable(str(e)):
                    continue
                raise
        raise Exception("All Gemini models unavailable.")

    async def _stream(self, messages: list) -> AsyncGenerator[str, None]:
        for idx, llm in enumerate([self.llm] + self.llm_fallbacks):
            if llm is None:
                continue
            try:
                async for chunk in llm.astream(messages):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if text:
                        yield text
                return
            except Exception as e:
                if self._is_retryable(str(e)):
                    continue
                raise
        yield "Tất cả Gemini models đang quá tải. Vui lòng thử lại sau."

    async def _stream_with_timeout(
        self, messages: list, timeout: float = 75.0
    ) -> AsyncGenerator[str, None]:
        """Stream LLM output với timeout — tránh client treo khi Gemini không phản hồi.

        Gemini 2.5 Flash có giai đoạn "thinking"/AFC trước khi ra token đầu tiên,
        nên dùng first_token_timeout=30s, sau đó per_token=10s.
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def _producer():
            try:
                async for token in self._stream(messages):
                    await queue.put(token)
            except Exception as e:
                await queue.put(Exception(str(e)))
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.create_task(_producer())
        deadline = asyncio.get_event_loop().time() + timeout

        # Token đầu tiên Gemini cần thời gian thinking → dùng timeout dài hơn
        first_token = True
        per_token_timeout = 50.0

        try:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    task.cancel()
                    yield "\n\n⚠️ Phản hồi bị ngắt do quá thời gian. Vui lòng thử lại."
                    return
                try:
                    wait = min(remaining, per_token_timeout)
                    item = await asyncio.wait_for(queue.get(), timeout=wait)
                    if item is None:
                        return
                    if isinstance(item, Exception):
                        raise item
                    if first_token:
                        first_token = False
                        per_token_timeout = 10.0  # token tiếp theo không cần chờ lâu
                    yield item
                except asyncio.TimeoutError:
                    task.cancel()
                    yield "\n\n⚠️ Phản hồi bị ngắt do quá thời gian. Vui lòng thử lại."
                    return
        finally:
            if not task.done():
                task.cancel()

    # Từ khóa tài chính — nếu có thì query đủ cụ thể, không cần rewrite
    _FINANCIAL_TERMS = frozenset({
        "tài chính", "doanh thu", "lợi nhuận", "phân tích", "báo cáo",
        "roe", "eps", "p/e", "tăng trưởng", "rủi ro", "đầu tư", "cổ tức",
        "vốn", "nợ", "ebitda", "margin", "biên lợi nhuận",
    })

    async def _rewrite_query(self, query: str) -> str:
        """
        [OPTIMIZED] Query rewrite chuyển sang heuristic thuần túy — không gọi LLM.
        Tiết kiệm 1 Gemini call mỗi request. Chất lượng retrieval không đổi đáng kể
        vì MiniLM-L12 đã xử lý tốt tiếng Việt tự nhiên.
        """
        # Heuristic: nếu query chứa mã ticker in hoa, giữ nguyên
        # Nếu query quá ngắn và chỉ là tên công ty, bổ sung từ khóa tài chính
        if len(query) >= 60:
            return query
        # Mapping tên công ty → mã + từ khóa tài chính phổ biến
        _COMPANY_MAP = {
            "sacombank": "STB báo cáo tài chính",
            "vietinbank": "CTG báo cáo tài chính",
            "vietcombank": "VCB báo cáo tài chính",
            "techcombank": "TCB báo cáo tài chính",
            "bidv": "BID báo cáo tài chính",
            "agribank": "AGB báo cáo tài chính",
            "mb bank": "MBB báo cáo tài chính",
            "vpbank": "VPB báo cáo tài chính",
            "vinamilk": "VNM báo cáo tài chính",
            "fpt": "FPT báo cáo tài chính",
        }
        q_lower = query.lower()
        for company, expansion in _COMPANY_MAP.items():
            if company in q_lower and not any(term in q_lower for term in self._FINANCIAL_TERMS):
                rewritten = query + " " + expansion.split(company)[-1].strip()
                logger.info(f"Query heuristic rewrite: '{query}' → '{rewritten}'")
                return rewritten
        return query

    # ─── Build message list với conversation history ─────────────────────────

    @staticmethod
    def _build_messages(
        system: str,
        context: str,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> list:
        msgs = [SystemMessage(content=system)]
        if history:
            for msg in history[-8:]:  # giữ 8 turns gần nhất
                if msg["role"] == "user":
                    msgs.append(HumanMessage(content=msg["content"]))
                else:
                    msgs.append(AIMessage(content=msg["content"]))
        user_content = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}" if context else query
        msgs.append(HumanMessage(content=user_content))
        return msgs

    @staticmethod
    def _format_context(docs: List[Any]) -> str:
        """Format docs — dùng parent_text nếu có (Small-to-Big), fallback về page_content."""
        parts = []
        for doc in docs:
            # Lấy parent_text từ metadata nếu hierarchical chunking đã tạo
            content = doc.metadata.get("parent_text") or doc.page_content
            source = doc.metadata.get("source", "Unknown")
            page   = doc.metadata.get("page", "?")
            parts.append(f"[Nguồn: {source}, Trang {page}]\n{content}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_sources(docs: List[Any]) -> List[Dict[str, Any]]:
        seen, sources = set(), []
        for doc in docs:
            key = (doc.metadata.get("source", ""), doc.metadata.get("page", ""))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source":   doc.metadata.get("source", "Unknown"),
                    "page":     doc.metadata.get("page", "?"),
                    "doc_type": doc.metadata.get("doc_type", "Tài liệu"),
                    "period":   doc.metadata.get("period", ""),
                    "ticker":   doc.metadata.get("ticker", ""),
                })
        return sources

    # ─── Ticker extraction ───────────────────────────────────────────────────

    def _extract_tickers_multi(self, query: str) -> List[str]:
        import re
        STOPWORDS = {
            "KHÔNG","THEO","TRONG","NĂM","QUÝ","VÀ","CỦA","CHO","LÀ","CÓ",
            "BÁO","CÁO","TÔI","MÃ","CỔ","PHIẾU","PHÂN","TÍCH","VỀ","HỎI",
            "BIẾT","THE","FOR","AND","NHÀ","ĐẦU","TƯ","SO","SÁNH","VỚI",
            "VS","HAY","COMPARE","NÊN","MUA","BÁN","GIỮ",
            "TOP","HON","NAY","HOM","DANH","SACH","LIST",
        }
        tokens = re.findall(r'\b[A-Z]{2,5}\b', query.upper())
        seen, result = set(), []
        for t in tokens:
            if t not in STOPWORDS and t not in seen:
                seen.add(t)
                result.append(t)
        return result[:3]

    async def _extract_ticker(self, query: str) -> Optional[str]:
        import re
        STOPWORDS = {
            "KHÔNG","THEO","TRONG","NĂM","QUÝ","VÀ","CỦA","CHO","LÀ","CÓ",
            "BÁO","CÁO","TÔI","MÃ","CỔ","PHIẾU","PHÂN","TÍCH","VỀ","HỎI",
            "BIẾT","THE","FOR","AND","NHÀ","ĐẦU","TƯ","MUA","BÁN","GIỮ",
            "NÊN","HOLD","BUY","SELL","RSI","EMA","SMA","ROE","ROA","EPS",
            "MACD","NPL","NIM","CAR","ETF","IPO","GDP","CPI","VND","USD",
            "PE","PB","HNX","HSX","SSC","VSD","HOSE","UPCOM","TTCK",
            "TOP","HON","NAY","HOM","DANH","SACH","LIST","NAM","QUY",
        }

        # Pattern 1: keyword ngữ cảnh trước ticker (chính xác cao)
        kw_match = re.search(
            r'(?:m[aã]\b|c[oổồ]\s*phi[eếề]u\b|ph[aâ]n\s*t[íi]ch\b|'
            r'v[eề]\b|c[uủ]a\b|h[oỏ]i\s*v[eề]\b|b[aá]o\s*c[aá]o\b|'
            r'mua\b|b[aá]n\b|gi[uứ]\b|n[aắ]m\s*gi[uứ]\b|'
            r'kh[uuy][eề]n\s*ngh[iị]\b|t[uư]\s*v[aấ]n\b|'
            r'[dđ][aáà]nh\s*gi[aá]\b|ti[eề]m\s*n[aă]ng\b|tri[eể]n\s*v[oọ]ng\b)'
            r'\s+([A-Z]{2,5})(?!\w)',
            query, re.IGNORECASE,
        )
        if kw_match:
            t = kw_match.group(1).upper()
            if t not in STOPWORDS:
                return t

        # Pattern 2: quét toàn bộ token 2-5 chữ hoa — ticker thường đứng độc lập
        for t in re.findall(r'\b([A-Z]{2,5})\b', query.upper()):
            if t not in STOPWORDS:
                return t

        return None

    # ─── Price Query Pipeline ────────────────────────────────────────────────

    # ─── Price Compare Pipeline ──────────────────────────────────────────────

    _COMPARE_STOPWORDS = {
        "SO","SANH","VS","COMPARE","VA","HAY","VOI",
        "KHONG","THEO","TRONG","NAM","QUY","CUA","CHO","LA","CO",
        "BAO","CAO","TOI","MA","CO","PHIEU","PHAN","TICH","VE",
        "THE","FOR","AND","NHA","DAU","TU","MUA","BAN","GIU","NEN",
        "HOLD","BUY","SELL","RSI","EMA","SMA","ROE","ROA","EPS",
        "MACD","NPL","NIM","CAR","ETF","IPO","GDP","CPI","VND","USD",
        "PE","PB","HNX","HSX","SSC","VSD","HOSE","UPCOM","TTCK","GIA",
    }

    @staticmethod
    def _extract_compare_tickers(query: str) -> List[str]:
        tokens = re.findall(r'\b([A-Z]{2,5})\b', query.upper())
        seen, result = set(), []
        for t in tokens:
            if t not in RAGPipelineService._COMPARE_STOPWORDS and t not in seen:
                seen.add(t)
                result.append(t)
        return result[:3]

    @staticmethod
    def _is_price_compare_query(query: str) -> bool:
        return bool(_PRICE_COMPARE_RE.search(query))

    async def _fetch_price_compare_response(self, tickers: List[str]) -> Dict[str, Any]:
        from app.services.alpha_vantage import AlphaVantageService

        results = await asyncio.gather(
            *[AlphaVantageService.fetch_stock_data(t) for t in tickers],
            return_exceptions=True,
        )

        date_ref = ""
        blocks = []
        all_sources = []

        for ticker, data in zip(tickers, results):
            if isinstance(data, Exception) or data.get("fallback") or not data.get("prices"):
                blocks.append(f"**{ticker}** — ⚠️ Không lấy được dữ liệu\n")
                continue

            p = data["prices"][0]
            source = data.get("data_source", "Yahoo Finance")
            prev_close = data["prices"][1]["close"] if len(data["prices"]) > 1 else p["open"]
            change = p["close"] - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0
            arrow = "▲" if change >= 0 else "▼"
            sign  = "+" if change >= 0 else "-"
            vol_fmt = (
                f"{p['volume']/1_000_000:.1f}M" if p["volume"] >= 1_000_000
                else f"{p['volume']/1_000:.0f}K"
            )
            if not date_ref:
                date_ref = p["date"]

            blocks.append(
                f"#### 📊 {ticker}\n\n"
                f"### {p['close']:,.2f} &nbsp; {arrow} `{sign}{abs(change):,.2f}` &nbsp; `{sign}{abs(change_pct):.2f}%`\n\n"
                f"🔓 Mở cửa &nbsp;**{p['open']:,.2f}**"
                f" &nbsp;·&nbsp; ⬆️ Cao &nbsp;**{p['high']:,.2f}**"
                f" &nbsp;·&nbsp; ⬇️ Thấp &nbsp;**{p['low']:,.2f}**"
                f" &nbsp;·&nbsp; 📦 Vol &nbsp;**{vol_fmt}**"
            )
            all_sources.append({"source": source, "ticker": ticker, "doc_type": "Giá thị trường"})

        separator = "\n\n---\n\n"
        sources_set = set(d.get("data_source", "Yahoo Finance") for d in results if not isinstance(d, Exception) and not d.get("fallback") and d.get("prices"))
        sources_str = " / ".join(sources_set) if sources_set else "Nguồn dữ liệu"
        
        answer = (
            f"#### 🔀 So sánh giá &nbsp;·&nbsp; {date_ref}\n\n"
            + separator.join(blocks)
            + f"\n\n*📡 {sources_str} · Cuối phiên · Không phải realtime*"
        )
        return {"answer": answer, "intent": "PRICE_COMPARE", "sources": all_sources, "confidence": 0.9}

    async def _fetch_price_compare_stream(self, tickers: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
        result = await self._fetch_price_compare_response(tickers)
        yield {"type": "sources", "content": result.get("sources", [])}
        yield {"type": "token",   "content": result["answer"]}
        yield {"type": "confidence", "content": result.get("confidence", 0.0)}

    # ─── Top BUY Recommendations Pipeline ───────────────────────────────────

    @staticmethod
    def _is_top_buy_query(query: str) -> bool:
        return bool(_TOP_BUY_RE.search(query))

    async def _top_buy_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            from app.db.mongodb import get_db
            from app.repositories.report_repository import ReportRepository

            db = get_db()
            if db is None:
                yield {"type": "token", "content": "⚠️ Không thể kết nối cơ sở dữ liệu."}
                return

            report_repo = ReportRepository(db)
            reports = await report_repo.get_recent_reports(limit=50)

            if not reports:
                yield {"type": "token", "content": "Hiện tại chưa có báo cáo phân tích nào trong hệ thống."}
                return

            # Deduplicate: giữ báo cáo mới nhất cho mỗi mã
            latest_by_ticker: dict = {}
            for r in reports:
                if r.ticker not in latest_by_ticker:
                    latest_by_ticker[r.ticker] = r

            buy_stocks = [
                r for r in latest_by_ticker.values()
                if r.recommendation and r.recommendation.upper() in ("BUY", "STRONG BUY")
            ]

            if not buy_stocks:
                content = (
                    "🔍 Hiện tại hệ thống chưa có mã cổ phiếu nào đạt tín hiệu **MUA** "
                    "từ mô hình phân tích AI.\n\n"
                    "Bạn có thể nhập mã cụ thể (VD: FPT, VNM) để tôi phân tích chi tiết."
                )
                yield {"type": "token", "content": content}
                return

            lines = [
                f"## 📊 Danh sách mã được khuyến nghị MUA\n",
                f"*Cập nhật từ hệ thống phân tích AI — {len(buy_stocks)} mã*\n",
                "| Mã | Khuyến nghị | Chiến lược |",
                "|---|---|---|",
            ]
            for r in buy_stocks[:10]:
                rec = r.recommendation.upper()
                strategy = (r.investment_strategy or "").strip()
                short_reason = (strategy.split('.')[0].strip()[:80] + "…") if strategy else "—"
                lines.append(f"| **{r.ticker}** | {rec} | {short_reason} |")

            content = "\n".join(lines)
            chunk_size = 60
            for i in range(0, len(content), chunk_size):
                yield {"type": "token", "content": content[i:i + chunk_size]}
                await asyncio.sleep(0)

            yield {"type": "disclaimer", "content": ADVISORY_DISCLAIMER}

        except Exception as e:
            logger.error(f"_top_buy_stream error: {e}")
            yield {"type": "error", "content": "Đã xảy ra lỗi khi truy vấn danh sách mã BUY."}

    # ─── Price Query Pipeline ────────────────────────────────────────────────

    def _is_price_query(self, query: str) -> bool:
        # Nếu có các từ khóa về phân tích cơ bản/tài chính -> KHÔNG phải shortcut giá
        fundamental_keywords = [
            "doanh thu", "lợi nhuận", "lãi", "vốn", "tài sản", "nợ", 
            "báo cáo", "bctc", "bctn", "cổ tức", "eps", "pe", "roe", "roa"
        ]
        q_lower = query.lower()
        if any(kw in q_lower for kw in fundamental_keywords):
            return False
            
        return bool(_PRICE_QUERY_RE.search(query))

    async def _fetch_price_response(self, ticker: str) -> Dict[str, Any]:
        from app.services.alpha_vantage import AlphaVantageService
        try:
            data = await AlphaVantageService.fetch_stock_data(ticker)
        except Exception as e:
            logger.error(f"Price fetch error for {ticker}: {e}")
            return {
                "answer": f"Không thể lấy dữ liệu giá cho **{ticker}**. Vui lòng thử lại sau.",
                "intent": "PRICE_QUERY",
                "sources": [],
                "confidence": 0.0,
            }

        if data.get("fallback"):
            return {
                "answer": f"⚠️ Nguồn dữ liệu không khả dụng cho **{ticker}**. Vui lòng kiểm tra lại mã cổ phiếu.",
                "intent": "PRICE_QUERY",
                "sources": [],
                "confidence": 0.0,
            }

        prices = data.get("prices", [])
        if not prices:
            return {
                "answer": f"Không tìm thấy dữ liệu giá cho mã **{ticker}**. Vui lòng kiểm tra lại mã cổ phiếu.",
                "intent": "PRICE_QUERY",
                "sources": [],
                "confidence": 0.0,
            }

        p = prices[0]
        source = data.get("data_source", "Yahoo Finance")
        # Chuẩn tài chính: thay đổi = close hôm nay - close hôm qua
        prev_close = prices[1]["close"] if len(prices) > 1 else p["open"]
        change = p["close"] - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        arrow = "▲" if change >= 0 else "▼"
        sign  = "+" if change >= 0 else "-"

        vol_fmt = (
            f"{p['volume']/1_000_000:.1f}M"
            if p["volume"] >= 1_000_000
            else f"{p['volume']/1_000:.0f}K"
        )
        answer = (
            f"**📊 {ticker}** &nbsp;·&nbsp; `{p['date']}`\n\n"
            f"### {p['close']:,.2f}\n\n"
            f"{arrow} **{sign}{abs(change):,.2f}** &nbsp; **{sign}{abs(change_pct):.2f}%**\n\n"
            f"🔓 &nbsp;**{p['open']:,.2f}** &nbsp;·&nbsp; "
            f"⬆️ &nbsp;**{p['high']:,.2f}** &nbsp;·&nbsp; "
            f"⬇️ &nbsp;**{p['low']:,.2f}** &nbsp;·&nbsp; "
            f"📦 &nbsp;**{vol_fmt}**\n\n"
            f"*📡 {source} · Cuối phiên · Không phải realtime*"
        )
        logger.info(f"Price query: {ticker} = {p['close']} ({source})")
        return {
            "answer": answer,
            "intent": "PRICE_QUERY",
            "ticker_identified": ticker,
            "sources": [{"source": source, "ticker": ticker, "doc_type": "Giá thị trường"}],
            "confidence": 0.9,
            "price_data": p,
        }

    async def _fetch_price_stream(self, ticker: str) -> AsyncGenerator[Dict[str, Any], None]:
        result = await self._fetch_price_response(ticker)
        yield {"type": "sources", "content": result.get("sources", [])}
        yield {"type": "token",   "content": result["answer"]}
        yield {"type": "confidence", "content": result.get("confidence", 0.0)}

    # ─── Technical Analysis Pipeline ────────────────────────────────────────

    @staticmethod
    def _is_technical_query(query: str) -> bool:
        return bool(_TECHNICAL_QUERY_RE.search(query))

    async def _technical_stream(
        self, ticker: str, query: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        from app.services.alpha_vantage import AlphaVantageService
        from app.services.technical_analysis import TechnicalAnalysisService

        yield {"type": "intent", "content": "TECHNICAL_ANALYSIS"}
        yield {"type": "ticker", "content": ticker}

        # Fetch OHLCV (TCBS → Yahoo fallback đã có trong AlphaVantageService)
        try:
            data = await AlphaVantageService.fetch_stock_data(ticker)
        except Exception as e:
            yield {"type": "token", "content": f"Không thể lấy dữ liệu giá cho **{ticker}**: {e}"}
            return

        prices = data.get("prices", [])
        if not prices or len(prices) < 30:
            yield {"type": "token", "content": f"Không đủ dữ liệu lịch sử để phân tích kỹ thuật **{ticker}** (cần ≥ 30 phiên)."}
            return

        # Tính chỉ báo kỹ thuật
        ta = TechnicalAnalysisService.compute(prices)
        if ta is None:
            yield {"type": "token", "content": f"Không thể tính chỉ báo kỹ thuật cho **{ticker}**. Vui lòng thử lại."}
            return

        # Technical Anchor — deterministic BUY/SELL/HOLD, Single Source of Truth
        anchor      = InvestmentRuleEngine.compute_anchor(ticker, ta)
        anchor_text = InvestmentRuleEngine.format_for_llm(anchor)
        yield {"type": "anchor", "content": InvestmentRuleEngine.format_for_ui(anchor)}

        # Context = TA indicators + Anchor (LLM bắt buộc dùng anchor)
        context  = TechnicalAnalysisService.format_for_llm(ticker, ta) + "\n\n" + anchor_text
        messages = self._build_messages(
            _TECHNICAL_SYSTEM, context,
            f"Phân tích kỹ thuật cổ phiếu {ticker} và đưa ra nhận định",
        )

        source = data.get("data_source", "TCBS/Yahoo Finance")
        yield {"type": "sources", "content": [{"source": source, "ticker": ticker, "doc_type": "Dữ liệu kỹ thuật"}]}

        # LLMProvider: Gemini → Groq → Anchor text fallback
        if self._llm_provider:
            async for token in self._llm_provider.stream(messages, anchor_text=anchor_text, timeout=60.0):
                yield {"type": "token", "content": token}
        else:
            async for token in self._stream_with_timeout(messages, timeout=60.0):
                yield {"type": "token", "content": token}

        yield {"type": "disclaimer", "content": ADVISORY_DISCLAIMER}

    # ─── Market Overview Pipeline ────────────────────────────────────────────

    @staticmethod
    def _is_market_overview_query(query: str) -> bool:
        return bool(_MARKET_OVERVIEW_RE.search(query))

    async def _market_overview_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        from app.services.market_overview import MarketOverviewService

        yield {"type": "intent", "content": "MARKET_OVERVIEW"}

        try:
            overview = await asyncio.wait_for(
                MarketOverviewService.get_overview(), timeout=12.0
            )
        except asyncio.TimeoutError:
            yield {"type": "token", "content": "Không thể lấy dữ liệu thị trường lúc này. Vui lòng thử lại sau."}
            return
        except Exception as e:
            logger.error(f"Market overview error: {e}")
            yield {"type": "token", "content": "Lỗi khi lấy dữ liệu thị trường. Vui lòng thử lại."}
            return

        context = await MarketOverviewService.format_for_llm(overview)
        messages = self._build_messages(
            _MARKET_SYSTEM, context,
            "Tóm tắt và nhận định tổng quan thị trường chứng khoán Việt Nam hôm nay",
        )

        yield {"type": "sources", "content": [{"source": "TCBS Public API", "doc_type": "Dữ liệu thị trường"}]}

        async for token in self._stream_with_timeout(messages, timeout=45.0):
            yield {"type": "token", "content": token}

    # ─── VN News Pipeline ────────────────────────────────────────────────────

    @staticmethod
    def _is_news_query(query: str) -> bool:
        return bool(_NEWS_QUERY_RE.search(query))

    async def _news_stream(
        self, ticker: Optional[str], query: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        from app.services.vn_news import VnNewsService

        yield {"type": "intent", "content": "NEWS"}
        if ticker:
            yield {"type": "ticker", "content": ticker}

        try:
            news = await asyncio.wait_for(
                VnNewsService.fetch_news(ticker, max_items=8), timeout=10.0
            )
        except asyncio.TimeoutError:
            yield {"type": "token", "content": "Không thể lấy tin tức lúc này. Vui lòng thử lại sau."}
            return
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            yield {"type": "token", "content": "Lỗi khi lấy tin tức. Vui lòng thử lại."}
            return

        if not news:
            subject = f"mã **{ticker}**" if ticker else "thị trường"
            yield {"type": "token", "content": f"Hiện không tìm thấy tin tức mới về {subject}."}
            return

        context = VnNewsService.format_for_llm(ticker, news)
        user_q  = query if query else (f"Tóm tắt tin tức về {ticker}" if ticker else "Tóm tắt tin tức thị trường chứng khoán")
        messages = self._build_messages(_NEWS_SYSTEM, context, user_q)

        sources = [{"source": item.get("source",""), "doc_type": "Tin tức"} for item in news[:3]]
        yield {"type": "sources", "content": sources}

        async for token in self._stream_with_timeout(messages, timeout=45.0):
            yield {"type": "token", "content": token}

    # ─── Advisory Pipeline ───────────────────────────────────────────────────

    @staticmethod
    def _trim_history(history: Optional[List[Dict[str, str]]]) -> Optional[List[Dict[str, str]]]:
        """Giới hạn độ dài bot messages trong history để tránh lãng phí context window."""
        if not history:
            return history
        trimmed = []
        for msg in history[-10:]:
            if msg["role"] == "assistant" and len(msg["content"]) > 400:
                trimmed.append({"role": "assistant", "content": msg["content"][:400] + "…"})
            else:
                trimmed.append(msg)
        return trimmed

    async def _advisory_answer(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        ticker = await self._extract_ticker(query)
        filter_meta = {"ticker": ticker} if ticker else {}

        # Rewrite vague queries trước khi retrieve
        retrieval_query = await self._rewrite_query(query)

        t0 = time.time()

        # asyncio.to_thread: tránh block event loop (Pinecone SDK là synchronous)
        docs = await asyncio.to_thread(
            self.vector_store.search_advisory, retrieval_query, 10, filter_meta
        )
        trace = {"retrieval_ms": int((time.time() - t0) * 1000)}

        # Retrieval Guard
        rg = self._retrieval_guard.check_advisory(docs)
        if not rg.passed:
            return {
                "answer": INSUFFICIENT_DOCS_RESPONSE,
                "intent": Intent.ADVISORY,
                "ticker_identified": ticker,
                "sources": [],
                "confidence": 0.0,
                "crag_status": "INCORRECT",
            }

        # CRAG evaluation
        t1 = time.time()
        crag_status = await self._crag.evaluate(query, rg.filtered_docs)
        trace["crag_ms"] = int((time.time() - t1) * 1000)

        if crag_status == CRAGEvaluator.INCORRECT:
            return {
                "answer": INSUFFICIENT_DOCS_RESPONSE,
                "intent": Intent.ADVISORY,
                "ticker_identified": ticker,
                "sources": [],
                "confidence": 0.0,
                "crag_status": crag_status,
            }

        # AMBIGUOUS: giảm quality score để output guard chặt hơn
        effective_quality = (
            rg.quality_score * 0.70
            if crag_status == CRAGEvaluator.AMBIGUOUS
            else rg.quality_score
        )
        if crag_status == CRAGEvaluator.AMBIGUOUS:
            logger.info(f"CRAG AMBIGUOUS — quality penalized: {rg.quality_score:.2f} → {effective_quality:.2f}")

        context = self._format_context(rg.filtered_docs)
        messages = self._build_messages(
            _ADVISORY_SYSTEM, context, query, self._trim_history(history)
        )

        t2 = time.time()
        raw_answer = await self._invoke(messages)
        trace["llm_ms"] = int((time.time() - t2) * 1000)
        trace["total_ms"] = int((time.time() - t0) * 1000)
        logger.info(f"Advisory trace: {trace}")

        # Output Guard
        og = self._output_guard.check_advisory(
            raw_answer, effective_quality, len(rg.filtered_docs)
        )

        # Log eval metrics (không dùng LLM — chỉ ghi số liệu vào MongoDB)
        asyncio.create_task(self._log_retrieval_metric(
            intent="ADVISORY",
            docs=rg.filtered_docs,
            crag_status=crag_status,
            latency_ms=trace["total_ms"],
        ))
        # _log_groundedness đã bị tắt (no-op) để tiết kiệm Gemini quota

        return {
            "answer":              og.final_answer,
            "intent":              Intent.ADVISORY,
            "ticker_identified":   ticker,
            "sources":             self._extract_sources(rg.filtered_docs),
            "confidence":          og.confidence,
            "crag_status":         crag_status,
            "disclaimer_injected": og.disclaimer_injected,
            "trace":               trace,
        }

    async def _advisory_stream(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        ticker = await self._extract_ticker(query)
        yield {"type": "ticker", "content": ticker}

        # Rewrite vague queries trước khi retrieve
        retrieval_query = await self._rewrite_query(query)

        filter_meta = {"ticker": ticker} if ticker else {}
        docs = await asyncio.to_thread(
            self.vector_store.search_advisory, retrieval_query, 10, filter_meta
        )

        rg = self._retrieval_guard.check_advisory(docs)
        if not rg.passed:
            yield {"type": "error", "content": INSUFFICIENT_DOCS_RESPONSE}
            return

        crag_status = await self._crag.evaluate(query, rg.filtered_docs)
        yield {"type": "crag_status", "content": crag_status}

        if crag_status == CRAGEvaluator.INCORRECT:
            yield {"type": "error", "content": INSUFFICIENT_DOCS_RESPONSE}
            return

        effective_quality = (
            rg.quality_score * 0.70
            if crag_status == CRAGEvaluator.AMBIGUOUS
            else rg.quality_score
        )

        # Pre-stream gate: dùng ngưỡng thấp hơn 1 bậc (để OutputGuard quyết định cuối cùng)
        if effective_quality < (self._output_guard.CONFIDENCE_GATE_ADVISORY - 0.10):
            logger.warning(
                f"Pre-stream gate blocked: quality={effective_quality:.2f} "
                f"< {self._output_guard.CONFIDENCE_GATE_ADVISORY}"
            )
            yield {"type": "error", "content": INSUFFICIENT_DOCS_RESPONSE}
            return

        yield {"type": "sources", "content": self._extract_sources(rg.filtered_docs)}

        context = self._format_context(rg.filtered_docs)
        messages = self._build_messages(
            _ADVISORY_SYSTEM, context, query, self._trim_history(history)
        )

        full_answer = ""
        async for token in self._stream_with_timeout(messages):
            full_answer += token
            yield {"type": "token", "content": token}

        og = self._output_guard.check_advisory(
            full_answer, effective_quality, len(rg.filtered_docs)
        )
        yield {"type": "disclaimer", "content": ADVISORY_DISCLAIMER}
        yield {"type": "confidence", "content": og.confidence}

    # ─── Knowledge Pipeline ──────────────────────────────────────────────────

    async def _knowledge_answer(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        ticker = await self._extract_ticker(query)
        filter_meta = {"ticker": ticker} if ticker else {}

        docs = await asyncio.to_thread(
            self.vector_store.search_knowledge, query, 10, filter_meta
        )

        rg = self._retrieval_guard.check_knowledge(docs)
        context = self._format_context(rg.filtered_docs) if rg.filtered_docs else ""
        system  = _KNOWLEDGE_SYSTEM if rg.filtered_docs else _FALLBACK_SYSTEM

        messages = self._build_messages(system, context, query, self._trim_history(history))
        t0 = time.time()
        raw_answer = await self._invoke(messages)

        og = self._output_guard.check_knowledge(
            raw_answer, rg.quality_score, len(rg.filtered_docs)
        )

        asyncio.create_task(self._log_retrieval_metric(
            intent="KNOWLEDGE",
            docs=rg.filtered_docs,
            crag_status="N/A",
            latency_ms=int((time.time() - t0) * 1000),
        ))

        return {
            "answer":           og.final_answer,
            "intent":           Intent.KNOWLEDGE,
            "ticker_identified": ticker,
            "sources":          self._extract_sources(rg.filtered_docs),
            "confidence":       og.confidence,
        }

    async def _knowledge_stream(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        ticker = await self._extract_ticker(query)
        yield {"type": "ticker", "content": ticker}

        filter_meta = {"ticker": ticker} if ticker else {}
        docs = await asyncio.to_thread(
            self.vector_store.search_knowledge, query, 10, filter_meta
        )

        rg = self._retrieval_guard.check_knowledge(docs)
        if rg.filtered_docs:
            yield {"type": "sources", "content": self._extract_sources(rg.filtered_docs)}

        context = self._format_context(rg.filtered_docs) if rg.filtered_docs else ""
        system  = _KNOWLEDGE_SYSTEM if rg.filtered_docs else _FALLBACK_SYSTEM
        messages = self._build_messages(system, context, query, self._trim_history(history))

        async for token in self._stream_with_timeout(messages, timeout=60.0):
            yield {"type": "token", "content": token}

    # ─── Complaint Pipeline ──────────────────────────────────────────────────

    async def _complaint_answer(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        docs = await asyncio.to_thread(self.vector_store.search_faq, query, 3)
        rg = self._retrieval_guard.check_complaint(docs)

        context = self._format_context(rg.filtered_docs) if rg.filtered_docs else ""
        messages = self._build_messages(
            _COMPLAINT_SYSTEM, context, query, self._trim_history(history)
        )
        raw_answer = await self._invoke(messages)

        og = self._output_guard.check_complaint(raw_answer)
        return {
            "answer":    og.final_answer,
            "intent":    Intent.COMPLAINT,
            "sources":   self._extract_sources(rg.filtered_docs),
            "confidence": og.confidence,
        }

    async def _complaint_stream(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        docs = await asyncio.to_thread(self.vector_store.search_faq, query, 3)
        rg = self._retrieval_guard.check_complaint(docs)

        if rg.filtered_docs:
            yield {"type": "sources", "content": self._extract_sources(rg.filtered_docs)}

        context = self._format_context(rg.filtered_docs) if rg.filtered_docs else ""
        messages = self._build_messages(
            _COMPLAINT_SYSTEM, context, query, self._trim_history(history)
        )

        async for token in self._stream_with_timeout(messages, timeout=30.0):
            yield {"type": "token", "content": token}

    # ─── Native Tool Calling Pipeline ───────────────────────────────────────

    async def _native_tool_stream(
        self,
        query: str,
        ticker: Optional[str],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        LLM tự suy luận chọn 1 hoặc nhiều tools đồng thời, sau đó synthesis.

        Flow:
          Round 1 (non-stream) — LLM chọn tools (~1-2s)
          Parallel              — execute tất cả tools song song
          Hard-Abort            — nếu advisory tools fail hoàn toàn
          Round 2 (stream)      — LLM synthesis với Technical Anchor bắt buộc
          Fallback              — nếu tool calling fail → intent router cũ
        """
        if self._tool_executor is None or self._llm_provider is None or self._llm_provider.primary is None:
            async for chunk in self._fallback_intent_stream(query, history):
                yield chunk
            return

        # ── Round 1: tool selection ──────────────────────────────────────────
        tool_hint = f" Mã cổ phiếu đang bàn: **{ticker}**." if ticker else ""
        selection_system = (
            "Bạn là trợ lý tài chính AI. Phân tích câu hỏi và chọn ĐÚNG tool cần thiết.\n"
            "Với câu hỏi tư vấn đầu tư (nên mua/bán/giữ), hãy gọi ĐỒNG THỜI "
            "get_technical_analysis VÀ get_rag_advisory để có đánh giá toàn diện.\n"
            "Với câu hỏi về báo cáo tài chính, báo cáo thường niên, kết quả kinh doanh, "
            "doanh thu, lợi nhuận, chiến lược công ty từ tài liệu: dùng get_rag_advisory "
            "(KHÔNG dùng get_stock_news cho loại câu hỏi này).\n"
            "Với câu hỏi về kiến thức chứng khoán, thuật ngữ, quy định: get_rag_knowledge.\n"
            "Với khiếu nại/hỗ trợ tài khoản: get_faq.\n"
            "Với tin tức thị trường mới nhất: get_stock_news.\n"
            "Với tổng quan thị trường hôm nay: get_market_overview." + tool_hint
        )
        try:
            llm_with_tools = self._llm_provider.primary.bind_tools(TOOL_DEFINITIONS)
            response = await asyncio.wait_for(
                llm_with_tools.ainvoke([
                    SystemMessage(content=selection_system),
                    HumanMessage(content=query),
                ]),
                timeout=15.0,
            )
        except Exception as e:
            logger.warning(f"Tool selection failed: {e} — falling back to intent router")
            async for chunk in self._fallback_intent_stream(query, history):
                yield chunk
            return

        # LLM không gọi tool → ngoài phạm vi hoặc câu trả lời trực tiếp
        if not getattr(response, "tool_calls", None):
            content = getattr(response, "content", "") or ""
            if content.strip():
                yield {"type": "intent",  "content": "DIRECT"}
                yield {"type": "token",   "content": content}
            else:
                yield {"type": "token", "content": OUT_OF_SCOPE_RESPONSE}
            return

        # ── Execute tools in parallel ────────────────────────────────────────
        tool_names = [tc.get("name") for tc in response.tool_calls]
        yield {"type": "intent",       "content": "TOOL_CALLING"}
        yield {"type": "tools_called", "content": tool_names}
        logger.info(f"NativeToolCall: {tool_names} | '{query[:60]}'")

        results = await self._tool_executor.execute_all(response.tool_calls)

        # ── Hard-Abort: tất cả tools đều fail hoàn toàn ────────────────────
        # Dùng pattern matching thay vì độ dài để tránh false positive
        _NO_DATA_PATTERNS = ("Lỗi ", "Không tìm thấy", "Không đủ dữ liệu",
                             "Không thể", "Không kết nối", "[Lỗi")
        advisory_tools  = {"get_technical_analysis", "get_rag_advisory"}
        called_advisory = any(n in advisory_tools for n in tool_names)
        if called_advisory:
            has_data = any(
                text and not any(text.startswith(p) for p in _NO_DATA_PATTERNS)
                for _, _, text, _, _ in results
            )
            if not has_data:
                yield {"type": "error", "content": INSUFFICIENT_DOCS_RESPONSE}
                return

        # ── Collect anchors (technical) + actual PDF sources (RAG) ──────────
        anchors     = extract_anchors(results)
        anchor_text = "\n\n".join(
            InvestmentRuleEngine.format_for_llm(a) for a in anchors
        ) if anchors else ""

        if anchors:
            yield {"type": "anchor", "content": InvestmentRuleEngine.format_for_ui(anchors[0])}

        # Ưu tiên lấy sources thật từ PDF (tên file + số trang)
        pdf_sources = extract_rag_sources(results)
        if pdf_sources:
            # Có sources thật từ PDF → dùng luôn
            yield {"type": "sources", "content": pdf_sources}
        else:
            # Không có RAG docs → dùng sources từ tools khác
            fallback_sources = []
            for _, tc_name, text, _, _ in results:
                if any(text.startswith(p) for p in _NO_DATA_PATTERNS):
                    continue  # tool này không có data
                if tc_name in ("get_technical_analysis", "get_price_info"):
                    fallback_sources.append({"source": "TCBS/Yahoo Finance", "ticker": ticker or "", "doc_type": "Dữ liệu thị trường"})
                elif tc_name == "get_stock_news":
                    fallback_sources.append({"source": "VnNews / CafeF", "doc_type": "Tin tức"})
                elif tc_name == "get_market_overview":
                    fallback_sources.append({"source": "TCBS Public API", "doc_type": "Dữ liệu thị trường"})
            if fallback_sources:
                yield {"type": "sources", "content": fallback_sources}

        # ── Round 2: Synthesis ───────────────────────────────────────────────
        system = _SYNTHESIS_WITH_ANCHOR if anchors else _SYNTHESIS_NO_ANCHOR
        if anchor_text:
            system += f"\n\nTECHNICAL ANCHOR HIỆN TẠI:\n{anchor_text}"
        if pdf_sources:
            src_names = ", ".join(
                f"{s.get('source','?')} tr.{s.get('page','?')}"
                for s in pdf_sources[:4]
            )
            system += f"\n\nTRÍCH DẪN NGUỒN BẮT BUỘC: {src_names} — hãy đề cập trong câu trả lời."

        tool_messages = build_tool_messages(results)
        history_msgs  = []
        if history:
            for msg in (history[-6:]):
                role_cls = HumanMessage if msg["role"] == "user" else AIMessage
                content  = msg["content"]
                if msg["role"] == "assistant" and len(content) > 300:
                    content = content[:300] + "…"
                history_msgs.append(role_cls(content=content))

        synthesis_msgs = (
            [SystemMessage(content=system)]
            + tool_messages
            + history_msgs
            + [HumanMessage(content=query)]
        )

        async for token in self._llm_provider.stream(
            synthesis_msgs, anchor_text=anchor_text, timeout=90.0
        ):
            yield {"type": "token", "content": token}

        if anchors:
            yield {"type": "disclaimer", "content": ADVISORY_DISCLAIMER}

    async def _fallback_intent_stream(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Fallback về intent router cũ khi native tool calling không hoạt động."""
        intent_result = await self._intent_router.classify_with_llm(query)
        yield {"type": "intent", "content": intent_result.intent}
        logger.info(f"FallbackIntent: {intent_result.intent} | '{query[:60]}'")

        if intent_result.needs_clarification:
            yield {"type": "token", "content": IntentRouter.get_clarification_message(query)}
            return
        if intent_result.intent == Intent.OUT_OF_SCOPE:
            yield {"type": "token", "content": OUT_OF_SCOPE_RESPONSE}
            return
        if intent_result.intent == Intent.ADVISORY:
            async for chunk in self._advisory_stream(query, history):
                yield chunk
            return
        if intent_result.intent == Intent.COMPLAINT:
            async for chunk in self._complaint_stream(query, history):
                yield chunk
            return
        async for chunk in self._knowledge_stream(query, history):
            yield chunk

    # ─── Public API ──────────────────────────────────────────────────────────

    async def answer_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if not self._is_ready():
            return {
                "answer": "Hệ thống RAG chưa sẵn sàng. Kiểm tra kết nối Ollama.",
                "sources": [],
            }

        ig = self._input_guard.check(query)
        if not ig.passed:
            return {"answer": ig.rejection_reason, "sources": [], "confidence": 0.0}

        q_lower = ig.sanitized_query.lower()
        
        # Shortcut for Clarification Options
        if q_lower in ("tư vấn đầu tư", "tu van dau tu"):
            return {"answer": "Dạ vâng, để em có thể tư vấn đầu tư chính xác nhất, Anh/Chị vui lòng cung cấp mã cổ phiếu cụ thể (Ví dụ: FPT, HPG) hoặc cho em biết Anh/Chị đang quan tâm đến ngành nghề/chiến lược nào ạ?", "intent": "ADVISORY", "sources": [], "confidence": 1.0}
        if q_lower in ("giải đáp kiến thức", "giai dap kien thuc"):
            return {"answer": "Dạ, Anh/Chị cần giải đáp về thuật ngữ chứng khoán (VD: RSI, MACD, P/E), quy định pháp luật, hay cách đọc báo cáo tài chính ạ? Anh/Chị cứ hỏi tự nhiên nhé!", "intent": "KNOWLEDGE", "sources": [], "confidence": 1.0}
        if q_lower in ("hỗ trợ tài khoản", "ho tro tai khoan"):
            return {"answer": "Dạ, Anh/Chị đang gặp khó khăn khi đăng nhập, nạp/rút tiền, lỗi đặt lệnh hay cần hướng dẫn sử dụng ứng dụng ạ? Anh/Chị mô tả chi tiết vấn đề để em hỗ trợ nhé.", "intent": "COMPLAINT", "sources": [], "confidence": 1.0}

        # Price compare shortcut — 2+ mã, không tốn Gemini call
        if self._is_price_compare_query(ig.sanitized_query):
            tickers = self._extract_compare_tickers(ig.sanitized_query)
            if len(tickers) >= 2:
                logger.info(f"Price compare detected: {tickers}")
                return await self._fetch_price_compare_response(tickers)

        # Price query shortcut — 1 mã, không tốn Gemini call
        ticker = await self._extract_ticker(ig.sanitized_query)
        if ticker and self._is_price_query(ig.sanitized_query):
            logger.info(f"Price query detected: ticker={ticker}")
            return await self._fetch_price_response(ticker)

        # Intent classification
        intent_result: IntentResult = await self._intent_router.classify_with_llm(ig.sanitized_query)
        logger.info(
            f"Intent: {intent_result.intent} (conf={intent_result.confidence:.2f}) "
            f"| Query: '{ig.sanitized_query[:60]}'"
        )

        if intent_result.needs_clarification:
            return {
                "answer": IntentRouter.get_clarification_message(ig.sanitized_query),
                "intent": "UNCLEAR",
                "sources": [],
                "confidence": intent_result.confidence,
            }

        query_clean = ig.sanitized_query

        if intent_result.intent == Intent.OUT_OF_SCOPE:
            return {"answer": OUT_OF_SCOPE_RESPONSE, "intent": Intent.OUT_OF_SCOPE, "sources": []}

        if intent_result.intent == Intent.ADVISORY:
            return await self._advisory_answer(query_clean, conversation_history)

        if intent_result.intent == Intent.COMPLAINT:
            return await self._complaint_answer(query_clean, conversation_history)

        # Default: KNOWLEDGE
        return await self._knowledge_answer(query_clean, conversation_history)

    async def answer_query_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not self._is_ready():
            yield {"type": "error", "content": "Hệ thống RAG chưa sẵn sàng."}
            return

        # Input Guard
        ig = self._input_guard.check(query)
        if not ig.passed:
            yield {"type": "error", "content": ig.rejection_reason}
            return

        q_lower = ig.sanitized_query.lower()
        
        # Shortcut for Clarification Options (Make bot smarter)
        if q_lower in ("tư vấn đầu tư", "tu van dau tu"):
            yield {"type": "intent", "content": "ADVISORY"}
            yield {"type": "token", "content": "Dạ vâng, để em có thể tư vấn đầu tư chính xác nhất, Anh/Chị vui lòng cung cấp mã cổ phiếu cụ thể (Ví dụ: FPT, HPG) hoặc cho em biết Anh/Chị đang quan tâm đến ngành nghề/chiến lược nào ạ?"}
            return
            
        if q_lower in ("giải đáp kiến thức", "giai dap kien thuc"):
            yield {"type": "intent", "content": "KNOWLEDGE"}
            yield {"type": "token", "content": "Dạ, Anh/Chị cần giải đáp về thuật ngữ chứng khoán (VD: RSI, MACD, P/E), quy định pháp luật, hay cách đọc báo cáo tài chính ạ? Anh/Chị cứ hỏi tự nhiên nhé!"}
            return
            
        if q_lower in ("hỗ trợ tài khoản", "ho tro tai khoan"):
            yield {"type": "intent", "content": "COMPLAINT"}
            yield {"type": "token", "content": "Dạ, Anh/Chị đang gặp khó khăn khi đăng nhập, nạp/rút tiền, lỗi đặt lệnh hay cần hướng dẫn sử dụng ứng dụng ạ? Anh/Chị mô tả chi tiết vấn đề để em hỗ trợ nhé."}
            return

        # Top BUY shortcut — "top mã BUY hôm nay", "danh sách mã mua"
        if self._is_top_buy_query(ig.sanitized_query):
            logger.info("Stream top-buy query detected")
            async for chunk in self._top_buy_stream():
                yield chunk
            return

        # Price compare shortcut — 2+ mã, không tốn Gemini call
        if self._is_price_compare_query(ig.sanitized_query):
            tickers = self._extract_compare_tickers(ig.sanitized_query)
            if len(tickers) >= 2:
                logger.info(f"Stream price compare detected: {tickers}")
                yield {"type": "intent", "content": "PRICE_COMPARE"}
                async for chunk in self._fetch_price_compare_stream(tickers):
                    yield chunk
                return

        # ── Ticker extraction + Redis context cache ──────────────────────────
        extracted_ticker = await self._extract_ticker(ig.sanitized_query)
        if self._ticker_cache and session_id:
            ticker = await self._ticker_cache.resolve_ticker(session_id, extracted_ticker)
        else:
            ticker = extracted_ticker
        if ticker:
            yield {"type": "ticker", "content": ticker}

        # Price query shortcut — 1 mã, không tốn Gemini call
        if ticker and self._is_price_query(ig.sanitized_query):
            logger.info(f"Stream price query: ticker={ticker}")
            yield {"type": "intent", "content": "PRICE_QUERY"}
            async for chunk in self._fetch_price_stream(ticker):
                yield chunk
            return

        # Market overview shortcut — "thị trường hôm nay", "VN-Index"
        if self._is_market_overview_query(ig.sanitized_query):
            logger.info("Stream market overview detected")
            async for chunk in self._market_overview_stream():
                yield chunk
            return

        # Technical analysis shortcut — "phân tích kỹ thuật FPT", "RSI VNM"
        # (giữ shortcut vì nhanh, đã tích hợp TechnicalAnchor ở _technical_stream)
        if self._is_technical_query(ig.sanitized_query) and ticker:
            logger.info(f"Stream technical analysis: ticker={ticker}")
            async for chunk in self._technical_stream(ticker, ig.sanitized_query):
                yield chunk
            return

        # News shortcut — "tin tức FPT", "FPT có tin gì"
        if self._is_news_query(ig.sanitized_query):
            logger.info(f"Stream news: ticker={ticker}")
            async for chunk in self._news_stream(ticker, ig.sanitized_query):
                yield chunk
            return

        # ── Conversation memory: load server-side history from Redis ────────
        from app.db.cache_service import CacheService, ConversationMemory
        effective_history = conversation_history  # fallback: client-sent
        if session_id:
            redis_history = await ConversationMemory.load(session_id, max_turns=8)
            if redis_history:
                effective_history = redis_history

        # ── Redis response cache (skip when session has prior history) ───────
        # Queries with context are dependent on history → cannot reuse cached answer.
        _cache_key: Optional[str] = None
        if not effective_history:
            _cache_key = hashlib.md5(
                f"{ig.sanitized_query.lower().strip()}:{ticker or ''}".encode()
            ).hexdigest()
            cached = await CacheService.get("rag_response", _cache_key)
            if cached:
                logger.info(f"RAG cache HIT: {ig.sanitized_query[:60]}")
                yield {"type": "intent", "content": cached.get("intent", "CACHE")}
                if cached.get("ticker"):
                    yield {"type": "ticker", "content": cached["ticker"]}
                chunk_size = 60
                text = cached.get("text", "")
                for i in range(0, len(text), chunk_size):
                    yield {"type": "token", "content": text[i:i + chunk_size]}
                    await asyncio.sleep(0)
                yield {"type": "done", "content": ""}
                return

        # ── Native tool calling — LLM tự chọn và kết hợp tools ─────────────
        _accumulated_text = []
        _intent_seen = "GENERAL"
        async for chunk in self._native_tool_stream(
            ig.sanitized_query, ticker, effective_history
        ):
            if chunk.get("type") == "intent":
                _intent_seen = chunk.get("content", "GENERAL")
            if chunk.get("type") == "token":
                _accumulated_text.append(chunk.get("content", ""))
            yield chunk

        # ── Post-response: lưu cache và conversation memory ──────────────────
        full_text = "".join(_accumulated_text)
        if full_text:
            if _cache_key:
                await CacheService.set("rag_response", _cache_key, {
                    "text":   full_text,
                    "intent": _intent_seen,
                    "ticker": ticker,
                })
            if session_id:
                await ConversationMemory.save_turn(
                    session_id, ig.sanitized_query, full_text
                )

    # ─── Multi-ticker comparison ─────────────────────────────────────────────

    async def compare_tickers_stream(
        self,
        query: str,
        tickers: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not self._is_ready():
            yield {"type": "error", "content": "Hệ thống RAG chưa sẵn sàng."}
            return

        ig = self._input_guard.check(query)
        if not ig.passed:
            yield {"type": "error", "content": ig.rejection_reason}
            return

        tickers = tickers[:3]
        if len(tickers) < 2:
            yield {"type": "error", "content": "Cần ít nhất 2 mã cổ phiếu để so sánh."}
            return

        yield {"type": "tickers", "content": tickers}

        # Retrieve docs cho mỗi ticker song song
        async def fetch(ticker: str):
            try:
                docs = await asyncio.to_thread(
                    self.vector_store.search_advisory,
                    ig.sanitized_query, 4, {"ticker": ticker},
                )
                return ticker, docs
            except Exception as e:
                logger.error(f"Compare fetch {ticker}: {e}")
                return ticker, []

        results = await asyncio.gather(*[fetch(t) for t in tickers])
        docs_by_ticker = dict(results)

        has_any_docs = any(len(docs) > 0 for docs in docs_by_ticker.values())
        if not has_any_docs:
            logger.info(f"Compare tickers: no docs found for {tickers}, falling back to price compare.")
            yield {"type": "intent", "content": "PRICE_COMPARE"}
            async for chunk in self._fetch_price_compare_stream(tickers):
                yield chunk
            return

        sources_by_ticker = {
            t: self._extract_sources(docs_by_ticker.get(t, []))
            for t in tickers
        }
        yield {"type": "sources", "content": sources_by_ticker}

        # Build context per ticker
        parts = []
        for t in tickers:
            docs = docs_by_ticker.get(t, [])
            if not docs:
                parts.append(f"=== {t} ===\nKhông có dữ liệu cho {t} trong hệ thống.")
            else:
                rg = self._retrieval_guard.check_advisory(docs)
                parts.append(f"=== {t} ===\n{self._format_context(rg.filtered_docs)}")

        context = "\n\n---\n\n".join(parts)
        system = (
            "Bạn là chuyên gia phân tích tài chính. So sánh các mã cổ phiếu DỰA HOÀN TOÀN "
            "trên tài liệu. Trình bày bảng markdown với các tiêu chí tài chính quan trọng. "
            "Sau bảng viết 2-3 câu nhận xét. CHỈ dùng thông tin trong ngữ cảnh. "
            "Thiếu dữ liệu ghi N/A. Trả lời bằng tiếng Việt."
        )
        messages = self._build_messages(
            system, context, ig.sanitized_query, self._trim_history(conversation_history)
        )

        full_answer = ""
        async for token in self._stream_with_timeout(messages, timeout=60.0):
            full_answer += token
            yield {"type": "token", "content": token}

        yield {"type": "disclaimer", "content": ADVISORY_DISCLAIMER}

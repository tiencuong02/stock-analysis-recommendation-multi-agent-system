from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel, Field
from typing import Optional, List
from app.models.user import User
from app.api.endpoints.auth import get_current_user, check_admin_role
from app.services.rag.vector_store import VectorStoreService, NAMESPACE_ADVISORY, NAMESPACE_KNOWLEDGE, NAMESPACE_FAQ
from app.services.rag.rag_pipeline import RAGPipelineService
from app.services.rag.pdf_processor import PDFProcessorService
from app.db.mongodb import get_db
from bson import ObjectId
import tempfile, os, datetime, logging, uuid, json

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Rate Limiter helper (dùng app.state.rate_limiter — Redis hoặc InMemory) ──

async def _check_rate_limit(request: Request, user_id: str, is_stream: bool = False):
    """
    Lấy rate limiter từ app.state (Redis-backed nếu có, InMemory fallback).
    stream: 30 req/min | query: 60 req/min
    """
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return  # limiter chưa init → bỏ qua

    max_calls     = 30 if is_stream else 60
    window        = 60
    limiter_key   = f"{user_id}:{'stream' if is_stream else 'query'}"

    allowed = await limiter.is_allowed(limiter_key, max_calls, window)
    if not allowed:
        retry = await limiter.retry_after(limiter_key, window)
        raise HTTPException(
            status_code=429,
            detail=f"Quá nhiều yêu cầu. Vui lòng thử lại sau {retry} giây.",
            headers={"Retry-After": str(retry)},
        )

# Mapping doc_type → namespace
_NAMESPACE_MAP = {
    "advisory":  NAMESPACE_ADVISORY,
    "knowledge": NAMESPACE_KNOWLEDGE,
    "faq":       NAMESPACE_FAQ,
}
_DEFAULT_UPLOAD_NAMESPACE = NAMESPACE_ADVISORY  # tài liệu tư vấn là mặc định

# Singleton dependencies from app.state (initialized once at startup)
def get_rag_service(request: Request) -> RAGPipelineService:
    service = getattr(request.app.state, "rag_pipeline", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service chưa sẵn sàng. Vui lòng thử lại sau.")
    return service

def get_vector_store(request: Request) -> VectorStoreService:
    store = getattr(request.app.state, "vector_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Vector store chưa sẵn sàng. Vui lòng thử lại sau.")
    return store

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=2000)

class RAGQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Câu hỏi (tối đa 500 ký tự)")
    conversation_history: Optional[List[ChatMessage]] = Field(default=None, max_length=10, description="Lịch sử hội thoại (tối đa 10 messages)")
    session_id: Optional[str] = Field(default=None, max_length=128, description="Session ID để duy trì ngữ cảnh mã cổ phiếu (từ frontend)")

class CompareQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Câu hỏi so sánh (tối đa 500 ký tự)")
    tickers: Optional[List[str]] = Field(default=None, max_length=3, description="Danh sách mã cổ phiếu để so sánh (tối đa 3)")
    conversation_history: Optional[List[ChatMessage]] = Field(default=None, max_length=10, description="Lịch sử hội thoại (tối đa 10 messages)")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_history(msgs):
    return [{"role": m.role, "content": m.content} for m in msgs] if msgs else None

async def _audit_log(db, user_id: str, event: str, detail: dict):
    """Ghi audit log vào MongoDB — mọi query/response đều được lưu để compliance."""
    if db is None:
        return
    try:
        await db["rag_audit_logs"].insert_one({
            "user_id":    user_id,
            "event":      event,
            "detail":     detail,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
    except Exception as e:
        logger.warning(f"Audit log failed: {e}")


# ─── Query Endpoint ───────────────────────────────────────────────────────────

@router.post("/query/")
async def process_rag_query(
    req: Request,
    request: RAGQuery,
    service: RAGPipelineService = Depends(get_rag_service),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    await _check_rate_limit(req, current_user.username, is_stream=False)
    try:
        history = _to_history(request.conversation_history)
        response = await service.answer_query(request.query, conversation_history=history)
        await _audit_log(db, current_user.username, "query", {
            "query":   request.query[:200],
            "intent":  response.get("intent"),
            "confidence": response.get("confidence"),
            "sources_count": len(response.get("sources", [])),
        })
        return response
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail="Lỗi xử lý câu hỏi. Vui lòng thử lại.")


# ─── Streaming Query Endpoint (SSE) ──────────────────────────────────────────

@router.post("/query/stream")
async def process_rag_query_stream(
    req: Request,
    request: RAGQuery,
    service: RAGPipelineService = Depends(get_rag_service),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    await _check_rate_limit(req, current_user.username, is_stream=True)
    history = _to_history(request.conversation_history)
    user_id = current_user.username

    async def event_generator():
        intent_seen = None
        try:
            async for chunk in service.answer_query_stream(request.query, conversation_history=history, session_id=request.session_id):
                if chunk.get("type") == "intent":
                    intent_seen = chunk.get("content")
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"RAG stream failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            await _audit_log(db, user_id, "stream_query", {
                "query":  request.query[:200],
                "intent": str(intent_seen),
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ─── Comparison Query Endpoint (SSE) ─────────────────────────────────────────

@router.post("/query/compare/stream")
async def compare_tickers_stream(
    req: Request,
    request: CompareQuery,
    service: RAGPipelineService = Depends(get_rag_service),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    await _check_rate_limit(req, current_user.username, is_stream=True)
    history = _to_history(request.conversation_history)
    tickers = request.tickers or service._extract_tickers_multi(request.query)
    user_id = current_user.username

    if len(tickers) < 2:
        async def err_gen():
            yield f"data: {json.dumps({'type': 'error', 'content': 'Can ít nhất 2 mã cổ phiếu để so sánh.'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def event_generator():
        try:
            async for chunk in service.compare_tickers_stream(request.query, tickers, conversation_history=history):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"RAG compare stream failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            await _audit_log(db, user_id, "compare_stream", {
                "query": request.query[:200], "tickers": tickers,
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ─── Background task: embed + upsert (chạy sau khi trả response) ─────────────

_EMBED_BATCH_SIZE = 100  # MiniLM-L12 nhẹ, nhưng giữ 100 để tránh OOM trên file lớn

async def _bg_embed_upsert(
    chunks: list,
    namespace: str,
    vector_store: VectorStoreService,
    db,
    document_id: str,
    username: str,
    filename: str,
    ticker: str,
):
    """Embed + upsert theo batch để tránh OOM và cập nhật tiến trình liên tục."""
    total = len(chunks)
    success = True
    failed_batches = 0
    import time as _time
    t0 = _time.time()

    logger.info(f"[BG] Starting embedding {total} chunks for {document_id} (batch={_EMBED_BATCH_SIZE})...")

    for i in range(0, total, _EMBED_BATCH_SIZE):
        batch = chunks[i : i + _EMBED_BATCH_SIZE]
        batch_num = i // _EMBED_BATCH_SIZE + 1
        try:
            ok = await asyncio.to_thread(vector_store.upsert_chunks, batch, namespace)
            if not ok:
                success = False
                failed_batches += 1
                logger.error(f"[BG] {document_id} batch {batch_num} returned False")
        except Exception as e:
            success = False
            failed_batches += 1
            logger.error(f"[BG] {document_id} batch {batch_num} exception: {type(e).__name__}: {e}")
            # Nếu lỗi dimension mismatch hoặc auth → dừng ngay, không retry
            err_str = str(e).lower()
            if any(x in err_str for x in ("dimension", "unauthorized", "invalid api", "unauthenticated")):
                logger.error(f"[BG] {document_id}: Fatal error, aborting remaining batches.")
                break

        processed = min(i + _EMBED_BATCH_SIZE, total)
        elapsed = _time.time() - t0
        logger.info(f"[BG] {document_id}: {processed}/{total} chunks | {elapsed:.1f}s elapsed")
        if db is not None:
            try:
                await db["knowledge_base"].update_one(
                    {"document_id": document_id},
                    {"$set": {"chunks_count": processed}},
                )
            except Exception:
                pass

    elapsed_total = _time.time() - t0
    status = "indexed" if success else "failed"
    logger.info(f"[BG] {document_id}: {status} ({total} chunks, {failed_batches} failed batches, {elapsed_total:.1f}s)")

    if db is not None:
        try:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            await db["knowledge_base"].update_one(
                {"document_id": document_id},
                {"$set": {"status": status, "indexed_at": now, "chunks_count": total}},
            )
            await _audit_log(db, username, "pdf_indexed", {
                "document_id": document_id,
                "filename": filename,
                "ticker": ticker,
                "status": status,
                "chunks": total,
                "duration_sec": round(elapsed_total, 1),
            })
        except Exception as e:
            logger.error(f"[BG] MongoDB status update for {document_id} failed: {e}")


# ─── Upload PDF Endpoint (Admin Only) ─────────────────────────────────────────

@router.post("/upload/")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ticker: str = Form(default=""),           # Bắt buộc với advisory, optional với knowledge/faq
    doc_type: str = Form("Báo cáo tài chính"),
    namespace_type: str = Form("advisory"),   # "advisory" | "knowledge" | "faq"
    period: str = Form(""),
    year: str = Form("2024"),
    current_user: User = Depends(check_admin_role),
    vector_store: VectorStoreService = Depends(get_vector_store),
    db=Depends(get_db),
):
    """Admin: upload PDF → hierarchical chunking → embed → Qdrant namespace chỉ định."""

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF.")

    MAX_FILE_SIZE = 50 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File quá lớn. Tối đa {MAX_FILE_SIZE//(1024*1024)}MB.")

    # Ticker bắt buộc với advisory (tài liệu gắn với mã cụ thể).
    # Knowledge/FAQ là tài liệu chung → dùng "GENERAL" nếu không cung cấp.
    ns_lower = namespace_type.lower()
    if ns_lower == "advisory" and not ticker.strip():
        raise HTTPException(status_code=400, detail="Mã cổ phiếu là bắt buộc với tài liệu Tư vấn đầu tư.")
    effective_ticker = ticker.strip().upper() if ticker.strip() else "GENERAL"

    target_namespace = _NAMESPACE_MAP.get(ns_lower, _DEFAULT_UPLOAD_NAMESPACE)

    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, "wb") as f:
            f.write(content)

        processor = PDFProcessorService()
        short_id   = uuid.uuid4().hex[:6]
        document_id = f"doc_{effective_ticker.lower()}_{year}_{short_id}"

        metadata = {
            "ticker":    effective_ticker,
            "doc_type":  doc_type,
            "period":    period,
            "year":      int(year) if year.isdigit() else year,
            "source":    file.filename,
            "namespace": target_namespace,
        }

        chunks = await asyncio.to_thread(
            processor.process_and_chunk_pdf, tmp_path, metadata, document_id
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Không trích xuất được nội dung từ file PDF. "
                    "File có thể là PDF scan (chỉ chứa ảnh, không có text layer). "
                    "Vui lòng dùng PDF gốc có text layer hoặc chạy OCR trước khi upload."
                ),
            )

        valid_chunks = [c for c in chunks if c.get("text") and len(c.get("text", "").strip()) > 10]
        if not valid_chunks:
            raise HTTPException(
                status_code=400,
                detail=(
                    "File PDF không chứa nội dung văn bản hợp lệ. "
                    "File có thể là PDF scan hoặc chỉ chứa hình ảnh."
                ),
            )

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        doc_year = int(year) if year.isdigit() else year

        doc_record = {
            "document_id":       document_id,
            "filename":          file.filename,
            "ticker":            effective_ticker,
            "doc_type":          doc_type,
            "namespace_type":    namespace_type.lower(),
            "qdrant_namespace":  target_namespace,
            "period":            period,
            "year":              doc_year,
            "embedding_model":   "paraphrase-multilingual-MiniLM-L12-v2",
            "chunking_strategy": "hierarchical",
            "index_version":     2,
            "status":            "processing",   # background task sẽ update → "indexed"/"failed"
            "uploaded_by":       current_user.username,
            "uploaded_at":       now,
            "indexed_at":        None,
        }

        if db is not None:
            result = await db["knowledge_base"].insert_one(doc_record)
            doc_record["_id"] = str(result.inserted_id)
            await _audit_log(db, current_user.username, "pdf_upload", {
                "filename":    file.filename,
                "ticker":      effective_ticker,
                "namespace":   target_namespace,
                "chunks":      len(valid_chunks),
                "document_id": document_id,
            })

        # Chạy embedding và upsert ở background
        background_tasks.add_task(
            _bg_embed_upsert,
            valid_chunks, target_namespace, vector_store, db,
            document_id, current_user.username, file.filename, effective_ticker
        )

        logger.info(
            f"PDF queued for embedding: {file.filename} | Ticker: {effective_ticker} | "
            f"Namespace: {target_namespace} | Chunks: {len(chunks)}"
        )

        return {
            "status":           "processing",
            "message":          f"Đã nhận {file.filename} — đang embedding {len(chunks)} chunks...",
            "chunks_processed": len(chunks),
            "document": {
                "id":                doc_record.get("_id", ""),
                "document_id":       document_id,
                "filename":          file.filename,
                "ticker":            effective_ticker,
                "doc_type":          doc_type,
                "namespace_type":    namespace_type,
                "qdrant_namespace":  target_namespace,
                "period":            period,
                "year":              doc_year,
                "chunks_count":      len(chunks),
                "embedding_model":   "paraphrase-multilingual-MiniLM-L12-v2",
                "chunking_strategy": "hierarchical",
                "index_version":     2,
                "status":            "processing",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Lỗi xử lý file PDF. Vui lòng kiểm tra lại file và thử lại.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ─── Document Suggestions (Public) ───────────────────────────────────────────
@router.get("/suggestions/")
async def get_document_suggestions(db=Depends(get_db)):
    """Trả về danh sách tài liệu đã index để UI sinh quick chips.
    Public — chỉ lộ ticker, doc_type, year (không có nội dung nhạy cảm).
    """
    if db is None:
        return []
    try:
        cursor = (
            db["knowledge_base"]
            .find({"status": "indexed"}, {"ticker": 1, "doc_type": 1, "year": 1, "_id": 0})
            .sort("uploaded_at", -1)
            .limit(6)
        )
        suggestions = []
        async for doc in cursor:
            suggestions.append(doc)
        return suggestions
    except Exception as e:
        logger.error(f"Failed to load suggestions: {e}")
        return []


# ─── List Documents Endpoint (Admin Only) ─────────────────────────────────────
@router.get("/documents/")
async def list_documents(
    _: User = Depends(check_admin_role),
    db=Depends(get_db)
):
    """Get all ingested documents from knowledge base."""
    if db is None:
        return []
    
    try:
        cursor = db["knowledge_base"].find().sort("uploaded_at", -1)
        documents = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            documents.append(doc)
        return documents
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Loi tai danh sach tai lieu.")

# ─── Delete Document Endpoint (Admin Only) ────────────────────────────────────
@router.delete("/documents/{doc_id}/")
async def delete_document(
    doc_id: str,
    current_user: User = Depends(check_admin_role),
    vector_store: VectorStoreService = Depends(get_vector_store),
    db=Depends(get_db)
):
    """Delete a document record and its vectors from Qdrant."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    # Validate ObjectId format
    try:
        obj_id = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="ID tài liệu không hợp lệ")

    try:
        # Lấy thông tin document trước khi xóa (để biết source filename)
        doc = await db["knowledge_base"].find_one({"_id": obj_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Xóa vectors trong Pinecone: ưu tiên theo document_id, fallback về source filename
        doc_id_field = doc.get("document_id")
        source_filename = doc.get("filename")
        if doc_id_field:
            vector_store.delete_by_metadata({"document_id": doc_id_field})
        elif source_filename:
            vector_store.delete_by_metadata({"source": source_filename})

        # Xóa record trong MongoDB
        await db["knowledge_base"].delete_one({"_id": obj_id})

        logger.info(f"Document {doc_id} + vectors deleted by {current_user.username}")
        return {"status": "success", "message": "Đã xóa tài liệu và vectors"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document.")

# ─── Re-index Document Endpoint (Admin Only) ──────────────────────────────────
@router.post("/documents/{doc_id}/reindex")
async def reindex_document(
    doc_id: str,
    target_namespace_type: str = Form(...), # "advisory", "knowledge", "faq"
    current_user: User = Depends(check_admin_role),
    vector_store: VectorStoreService = Depends(get_vector_store),
    db=Depends(get_db)
):
    """
    Di chuyển vectors của một tài liệu sang Namespace khác (Re-index từng phần).
    Với Qdrant, chỉ cần cập nhật payload field 'namespace' — không cần copy/delete vectors.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    try:
        obj_id = ObjectId(doc_id)
        doc = await db["knowledge_base"].find_one({"_id": obj_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Ưu tiên qdrant_namespace → pinecone_namespace (backwards compat) → fallback
        current_ns = (
            doc.get("qdrant_namespace")
            or doc.get("pinecone_namespace")
            or doc.get("namespace")
            or _NAMESPACE_MAP.get(doc.get("namespace_type", "advisory"), NAMESPACE_ADVISORY)
        )
        target_ns = _NAMESPACE_MAP.get(target_namespace_type.lower(), NAMESPACE_ADVISORY)

        if current_ns == target_ns:
            return {"status": "success", "message": "Tài liệu đã ở namespace này rồi."}

        doc_id_field = doc.get("document_id")
        if not doc_id_field:
            raise HTTPException(status_code=400, detail="Tài liệu không có document_id.")

        if vector_store._client is None:
            raise HTTPException(status_code=503, detail="Qdrant client chưa sẵn sàng.")

        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from app.core.config import settings as _cfg

        scroll_filter = Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=doc_id_field)),
            FieldCondition(key="namespace",    match=MatchValue(value=current_ns)),
        ])

        # 1. Scroll qua tất cả points khớp điều kiện
        all_ids = []
        offset = None
        while True:
            result, next_offset = await asyncio.to_thread(
                vector_store._client.scroll,
                collection_name=_cfg.QDRANT_COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            all_ids.extend([p.id for p in result])
            if next_offset is None:
                break
            offset = next_offset

        if not all_ids:
            raise HTTPException(status_code=404, detail="Không tìm thấy vectors trong Qdrant cho tài liệu này.")

        # 2. Cập nhật payload field 'namespace' — không cần copy/delete vector
        await asyncio.to_thread(
            vector_store._client.set_payload,
            collection_name=_cfg.QDRANT_COLLECTION_NAME,
            payload={"namespace": target_ns},
            points=all_ids,
        )

        # 3. Cập nhật MongoDB
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        await db["knowledge_base"].update_one(
            {"_id": obj_id},
            {"$set": {
                "namespace":       target_ns,
                "qdrant_namespace": target_ns,
                "namespace_type":  target_namespace_type.lower(),
                "reindexed_at":    now,
                "chunks_count":    len(all_ids),
            }}
        )

        logger.info(f"Moved {doc_id_field}: {current_ns} → {target_ns} ({len(all_ids)} points)")
        return {"status": "success", "message": f"Đã chuyển {len(all_ids)} vectors sang ngăn {target_namespace_type.upper()}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Re-indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi di chuyển dữ liệu: {str(e)}")

# ─── Backfill Missing Fields (Admin Only) ────────────────────────────────────
@router.patch("/documents/backfill/")
async def backfill_documents(
    _: User = Depends(check_admin_role),
    db=Depends(get_db)
):
    """Backfill missing fields for old document records in knowledge_base."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    try:
        # Find documents missing any of the required fields
        cursor = db["knowledge_base"].find({
            "$or": [
                {"document_id": {"$exists": False}},
                {"vector_store": {"$exists": False}},
                {"namespace": {"$exists": False}},
                {"status": {"$exists": False}},
                {"embedding_model": {"$exists": False}},
                {"index_version": {"$exists": False}},
                {"indexed_at": {"$exists": False}},
            ]
        })

        updated_count = 0
        async for doc in cursor:
            ticker = doc.get("ticker", "unknown").lower()
            year = doc.get("year", "2024")
            short_id = uuid.uuid4().hex[:6]

            update_fields = {}
            if "document_id" not in doc:
                update_fields["document_id"] = f"doc_{ticker}_{year}_{short_id}"
            if "vector_store" not in doc:
                update_fields["vector_store"] = "qdrant"
            if "namespace" not in doc:
                update_fields["namespace"] = NAMESPACE_ADVISORY
            if "embedding_model" not in doc:
                update_fields["embedding_model"] = "paraphrase-multilingual-MiniLM-L12-v2"
            if "index_version" not in doc:
                update_fields["index_version"] = 1
            if "status" not in doc:
                update_fields["status"] = "indexed"
            if "indexed_at" not in doc:
                update_fields["indexed_at"] = doc.get("uploaded_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
            # Convert year string to int if needed
            if isinstance(doc.get("year"), str) and doc["year"].isdigit():
                update_fields["year"] = int(doc["year"])

            if update_fields:
                await db["knowledge_base"].update_one(
                    {"_id": doc["_id"]},
                    {"$set": update_fields}
                )
                updated_count += 1

        return {
            "status": "success",
            "message": f"Đã cập nhật {updated_count} document(s)",
            "updated_count": updated_count,
        }
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        raise HTTPException(status_code=500, detail="Backfill failed.")

# ─── Chat History Endpoints (multi-session) ──────────────────────────────────

class SaveChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, description="UUID của tab/session")
    messages: List[ChatMessage] = Field(..., max_length=200, description="Danh sách messages")

@router.post("/chat/save")
async def save_chat_history(
    request: SaveChatRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """Lưu chat session theo session_id — mỗi tab là 1 session độc lập."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    try:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        session_data = {
            "user_id":       current_user.username,
            "session_id":    request.session_id,
            "messages":      [{"role": m.role, "content": m.content} for m in request.messages],
            "message_count": len(request.messages),
            "updated_at":    now,
        }
        # Upsert theo (user_id, session_id) — mỗi tab có session riêng
        await db["chat_sessions"].update_one(
            {"user_id": current_user.username, "session_id": request.session_id},
            {"$set": session_data, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        return {"status": "success", "message_count": len(request.messages)}
    except Exception as e:
        logger.error(f"Save chat failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to save chat session.")

@router.get("/chat/history")
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """Load chat session theo session_id — chỉ trả về session của user hiện tại."""
    if db is None:
        return {"messages": []}

    try:
        session = await db["chat_sessions"].find_one(
            {"user_id": current_user.username, "session_id": session_id}
        )
        if not session:
            return {"messages": []}
        return {
            "messages":   session.get("messages", []),
            "updated_at": session.get("updated_at"),
        }
    except Exception as e:
        logger.error(f"Load chat failed: {e}")
        return {"messages": []}

@router.get("/chat/sessions")
async def list_chat_sessions(
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """Liệt kê tất cả sessions của user (tối đa 20, mới nhất trước)."""
    if db is None:
        return {"sessions": []}
    try:
        cursor = db["chat_sessions"].find(
            {"user_id": current_user.username},
            {"session_id": 1, "message_count": 1, "updated_at": 1, "created_at": 1},
        ).sort("updated_at", -1).limit(20)
        sessions = []
        async for s in cursor:
            s.pop("_id", None)
            sessions.append(s)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"List sessions failed: {e}")
        return {"sessions": []}

@router.delete("/chat/history")
async def clear_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """Xóa 1 session cụ thể của user (MongoDB + Redis conversation memory)."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    await db["chat_sessions"].delete_one(
        {"user_id": current_user.username, "session_id": session_id}
    )

    from app.db.cache_service import ConversationMemory
    await ConversationMemory.clear(session_id)

    return {"status": "success", "message": "Đã xóa lịch sử chat"}


# ─── Debug / Diagnostic Endpoint (Admin Only) ────────────────────────────────

@router.get("/debug/search")
async def debug_search(
    query: str,
    ticker: str = "",
    _: User = Depends(check_admin_role),
    vector_store: VectorStoreService = Depends(get_vector_store),
):
    """
    Admin debug: kiểm tra raw similarity scores từ Qdrant — không qua threshold filter.
    Giúp chẩn đoán vì sao chatbot báo 'chưa đủ tài liệu'.
    """
    from app.services.rag.vector_store import (
        NAMESPACE_ADVISORY, NAMESPACE_KNOWLEDGE, NAMESPACE_FAQ,
        SIMILARITY_THRESHOLD_ADVISORY,
        _NS_FIELD,
    )
    from app.core.config import settings as _cfg
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    embedding_model = "unknown"
    if vector_store.embeddings:
        try:
            embedding_model = vector_store.embeddings.model_name
        except Exception:
            embedding_model = str(type(vector_store.embeddings))

    # Qdrant collection stats
    index_stats = {}
    if vector_store._client:
        try:
            col_info = vector_store._client.get_collection(_cfg.QDRANT_COLLECTION_NAME)
            ns_stats = {}
            for ns_name in [NAMESPACE_ADVISORY, NAMESPACE_KNOWLEDGE, NAMESPACE_FAQ]:
                ns_filter = Filter(must=[FieldCondition(key=_NS_FIELD, match=MatchValue(value=ns_name))])
                cnt = vector_store._client.count(
                    collection_name=_cfg.QDRANT_COLLECTION_NAME,
                    count_filter=ns_filter,
                    exact=True,
                )
                ns_stats[ns_name] = {"vector_count": cnt.count}
            dim = getattr(col_info.config.params.vectors, "size", None)
            index_stats = {
                "dimension":     dim,
                "total_vectors": col_info.points_count,
                "namespaces":    ns_stats,
            }
        except Exception as e:
            index_stats = {"error": str(e)}

    # Raw search without threshold filter — dùng _store với Qdrant filter
    raw_results = {}
    if vector_store._store is not None:
        for ns in [NAMESPACE_ADVISORY, NAMESPACE_KNOWLEDGE, NAMESPACE_FAQ]:
            extra = {"ticker": ticker.upper()} if ticker else None
            qdrant_filter = vector_store._build_filter(ns, extra)
            try:
                hits = vector_store._store.similarity_search_with_score(
                    query=query, k=5, filter=qdrant_filter
                )
                raw_results[ns] = [
                    {
                        "score":             round(float(sc), 4),
                        "passes_threshold":  float(sc) >= SIMILARITY_THRESHOLD_ADVISORY,
                        "ticker":            doc.metadata.get("ticker", "?"),
                        "source":            doc.metadata.get("source", "?"),
                        "page":              doc.metadata.get("page", "?"),
                        "preview":           doc.page_content[:120].replace("\n", " "),
                    }
                    for doc, sc in hits
                ]
            except Exception as e:
                raw_results[ns] = {"error": str(e)}

    return {
        "query":              query,
        "ticker_filter":      ticker.upper() if ticker else None,
        "embedding_model":    embedding_model,
        "advisory_threshold": SIMILARITY_THRESHOLD_ADVISORY,
        "qdrant_collection":  index_stats,
        "raw_results":        raw_results,
    }


# ─── RAG Eval Metrics Endpoint (Admin) ───────────────────────────────────────

@router.get("/metrics/rag-summary")
async def rag_metrics_summary(
    days: int = 7,
    _: User = Depends(check_admin_role),
    db=Depends(get_db),
):
    """
    Tổng hợp RAG metrics trong N ngày qua.
    Hiển thị: hit_rate, avg_similarity, CRAG status distribution, avg_latency.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=days)
    ).isoformat()

    try:
        pipeline = [
            {"$match": {"event": "retrieval", "ts": {"$gte": cutoff}}},
            {"$group": {
                "_id":             "$intent",
                "total":           {"$sum": 1},
                "hit_rate":        {"$avg": {"$cond": [{"$gt": ["$docs_count", 0]}, 1, 0]}},
                "avg_similarity":  {"$avg": "$mean_similarity"},
                "avg_latency_ms":  {"$avg": "$latency_ms"},
                "crag_correct":    {"$avg": {"$cond": [{"$eq": ["$crag_status", "CORRECT"]}, 1, 0]}},
                "crag_ambiguous":  {"$avg": {"$cond": [{"$eq": ["$crag_status", "AMBIGUOUS"]}, 1, 0]}},
                "crag_incorrect":  {"$avg": {"$cond": [{"$eq": ["$crag_status", "INCORRECT"]}, 1, 0]}},
            }},
            {"$sort": {"total": -1}},
        ]
        rows = await db["rag_metrics"].aggregate(pipeline).to_list(10)

        # Groundedness sampling stats
        ground_pipeline = [
            {"$match": {"event": "groundedness", "ts": {"$gte": cutoff}}},
            {"$group": {
                "_id":             "$label",
                "count":           {"$sum": 1},
            }},
        ]
        ground_rows = await db["rag_metrics"].aggregate(ground_pipeline).to_list(5)

        return {
            "period_days":   days,
            "retrieval":     rows,
            "groundedness":  ground_rows,
        }
    except Exception as e:
        logger.error(f"RAG metrics query failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics query failed.")

"""
Optimized RAG System v9.0 - Light Weight & Robust
============================================================================

🚀 LIGHTWEIGHT & ROBUST EDITION:
• ⚡ INSTANT STARTUP: Heavy libraries (Torch, Transformers, FAISS) load ONLY when needed.
• 🛡️ CRASH PROTECTION: System waits for user input before closing.
• 💾 OPTIMIZED MEMORY: Lazy loading prevents unused memory consumption.
• 🔄 ROBUST ERROR HANDLING: Catches and displays errors without closing window.

Author: Enhanced RAG Development Team
Version: 9.0 (Lightweight Edition)
License: MIT
"""

import os
import sys
import re
import json
import time
import hashlib
import asyncio
import logging
import gc
import aiohttp
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import pyperclip
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# Fix for Windows Unicode handling in terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Load environment variables (force override to ensure correct keys in this session)
load_dotenv(override=True)

# Progress bar (lazy load if needed, but tqdm is light)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_v9_light.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RAG_v9")

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                                OPTIMIZED CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizedConfig:
    """Optimized configuration with performance-first defaults"""
    
    # API Keys
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    HF_TOKEN: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    
    # API Endpoints
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    
    # Optimized Model Configuration
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OLLAMA_MODEL: str = "llama3.2:3b"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Performance Settings
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    MAX_CHUNKS_FOR_CONTEXT: int = 5
    BATCH_SIZE: int = 64
    MAX_WORKERS: int = 4
    
    # FAISS Settings
    FAISS_HNSW_M: int = 16
    FAISS_EF_CONSTRUCTION: int = 100
    FAISS_EF_SEARCH: int = 50
    
    # Cache Settings
    CACHE_DURATION_HOURS: int = 24
    MAX_CACHE_ENTRIES: int = 100
    
    # Storage
    PROJECT_DATA_DIR: str = ".rag_projects"
    CACHE_DIR: str = ".rag_cache"
    USAGE_TRACKING_FILE: str = ".api_usage.json"
    
    # API Limits
    GROQ_REQUESTS_PER_MINUTE: int = 30
    OLLAMA_REQUESTS_PER_MINUTE: int = 60
    DAILY_TOKEN_LIMIT: int = 50000
    MAX_TOKENS_PER_REQUEST: int = 800
    
    # Feature Toggles
    ENABLE_REMOTE_APIS: bool = field(default_factory=lambda: os.getenv("ENABLE_REMOTE_APIS", "1") != "0")
    ENABLE_RERANKING: bool = True
    USE_FLOAT16: bool = True

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                                    DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════════════════

class Chunk(BaseModel):
    """Lightweight chunk model"""
    content: str
    source: str
    page_number: int
    chunk_id: int = Field(default_factory=lambda: int(time.time() * 1000000) % 1000000)

class Answer(BaseModel):
    """Answer model with metrics"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    api_used: str = "unknown"
    tokens_used: int = 0
    processing_time: float = 0.0
    cache_used: bool = False

class Project(BaseModel):
    """Project model"""
    name: str
    domain: Optional[str] = None
    pdf_paths: List[str] = Field(default_factory=list)
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                                    UTILITY CLASSES
# ═══════════════════════════════════════════════════════════════════════════════════════════

class SimpleCache:
    """Lightweight caching system"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.stats = {"hits": 0, "misses": 0}
    
    def _get_key(self, query: str, context_hash: str) -> str:
        combined = f"{query}_{context_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            age = datetime.now() - timestamp
            if age < timedelta(hours=self.config.CACHE_DURATION_HOURS):
                self.stats["hits"] += 1
                return data
            else:
                del self.cache[key]
        
        self.stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any):
        if len(self.cache) >= self.config.MAX_CACHE_ENTRIES:
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.now())
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        return {
            "hit_rate": round(hit_rate, 2),
            "cache_size": len(self.cache),
            **self.stats
        }

class UsageTracker:
    """Simple usage tracking"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.usage_file = config.USAGE_TRACKING_FILE
        self.data = self._load()
        self.request_times: Dict[str, List[float]] = {"ollama": [], "groq": []}
    
    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "daily_tokens": 0,
            "last_reset": str(datetime.now().date()),
            "api_calls": {"ollama": 0, "groq": 0}
        }
    
    def _save(self):
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save usage data: {e}")
    
    def can_make_request(self, api_type: str, tokens: int = 100) -> bool:
        today = str(datetime.now().date())
        if self.data["last_reset"] != today:
            self.data["daily_tokens"] = 0
            self.data["last_reset"] = today
        
        if self.data["daily_tokens"] + tokens > self.config.DAILY_TOKEN_LIMIT:
            return False
        
        now = time.time()
        minute_ago = now - 60
        self.request_times[api_type] = [t for t in self.request_times[api_type] if t > minute_ago]
        
        limits = {
            "ollama": self.config.OLLAMA_REQUESTS_PER_MINUTE,
            "groq": self.config.GROQ_REQUESTS_PER_MINUTE
        }
        
        return len(self.request_times[api_type]) < limits.get(api_type, 10)
    
    def record_request(self, api_type: str, tokens: int):
        self.request_times[api_type].append(time.time())
        self.data["daily_tokens"] += tokens
        self.data["api_calls"][api_type] = self.data["api_calls"].get(api_type, 0) + 1
        self._save()

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                                OPTIMIZED RAG SERVICE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class OptimizedRAGService:
    """High-performance RAG service with lazy loading and batch processing"""
    
    def __init__(self, config: OptimizedConfig = None):
        self.config = config or OptimizedConfig()
        
        # Lazy-loaded models (initially None)
        self.embedding_model = None
        self.reranker_model = None
        
        # Initialize lightweight components only
        self.cache = SimpleCache(self.config)
        self.usage_tracker = UsageTracker(self.config)
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        
        # State
        self.current_project: Optional[Project] = None
        self.active_chunks: List[Chunk] = []
        self.faiss_index = None  # Lazy type hint
        self.chunk_embeddings = None # Lazy type hint
        
        # Storage
        os.makedirs(self.config.PROJECT_DATA_DIR, exist_ok=True)
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)
        self.projects: Dict[str, Project] = self._load_projects()
        
        logger.info("⚡ Lightweight RAG v9.0 initialized (Lazy Mode)")
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                            LAZY MODEL LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _ensure_embedding_model(self):
        """Lazy load embedding model on first use"""
        if self.embedding_model is not None:
            return
        
        logger.info("🔧 Loading AI engine (this happens once)...")
        print("⏳ Loading AI models (first run only, please wait)...")
        start = time.time()
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            # Defensively force CPU to avoid meta-tensor issues on Windows
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL, device='cpu')
            elapsed = time.time() - start
            logger.info(f"✅ Embedding model loaded in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _ensure_reranker_model(self):
        """Lazy load reranker model on first use"""
        if not self.config.ENABLE_RERANKING:
            return
        
        if self.reranker_model is not None:
            return
        
        try:
            logger.info("🔧 Loading reranker model...")
            from sentence_transformers import CrossEncoder
            import torch
            self.reranker_model = CrossEncoder(self.config.RERANKER_MODEL, device='cpu')
            logger.info("✅ Reranker model loaded.")
        except Exception as e:
            logger.warning(f"Reranker not available: {e}")
            self.reranker_model = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                            PROJECT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _load_projects(self) -> Dict[str, Project]:
        """Load existing projects"""
        projects = {}
        project_file = os.path.join(self.config.PROJECT_DATA_DIR, "projects.json")
        
        if os.path.exists(project_file):
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for name, proj_data in data.items():
                    if isinstance(proj_data.get('created_at'), str):
                        proj_data['created_at'] = datetime.fromisoformat(proj_data['created_at'])
                    if isinstance(proj_data.get('last_updated'), str):
                        proj_data['last_updated'] = datetime.fromisoformat(proj_data['last_updated'])
                    projects[name] = Project(**proj_data)
                
                logger.info(f"📁 Loaded {len(projects)} projects")
            except Exception as e:
                logger.error(f"Failed to load projects: {e}")
        
        return projects
    
    def _save_projects(self):
        """Save all projects"""
        try:
            project_file = os.path.join(self.config.PROJECT_DATA_DIR, "projects.json")
            data = {}
            
            for name, project in self.projects.items():
                proj_dict = project.model_dump()
                proj_dict['created_at'] = project.created_at.isoformat()
                proj_dict['last_updated'] = project.last_updated.isoformat()
                data[name] = proj_dict
            
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save projects: {e}")
    
    def create_project(self, name: str, domain: Optional[str] = None) -> bool:
        """Create new project"""
        if name in self.projects:
            logger.warning(f"Project '{name}' already exists")
            return False
        
        self.projects[name] = Project(name=name, domain=domain)
        self._save_projects()
        logger.info(f"✅ Created project: {name}")
        return True
    
    def load_project(self, name: str) -> bool:
        """Load project and rebuild index"""
        if name not in self.projects:
            logger.error(f"Project '{name}' not found")
            return False
        
        self.current_project = self.projects[name]
        logger.info(f"📂 Loading project: {name}")
        
        # Rebuild chunks
        self.active_chunks = []
        for chunk_data in self.current_project.chunks:
            try:
                self.active_chunks.append(Chunk(**chunk_data))
            except Exception as e:
                logger.warning(f"Failed to load chunk: {e}")
        
        # Rebuild index if we have chunks
        if self.active_chunks:
            self._rebuild_index()
            logger.info(f"📊 Loaded {len(self.active_chunks)} chunks")
            self.faiss_index = None
            logger.info("📭 No chunks in project")
        
        return True

    def delete_project(self, name: str) -> bool:
        """Permanently delete a project and its metadata"""
        if name not in self.projects:
            logger.error(f"Project '{name}' not found for deletion")
            return False
        
        try:
            # If it's the current project, clear state
            if self.current_project and self.current_project.name == name:
                self.current_project = None
                self.active_chunks = []
                self.faiss_index = None
                self.chunk_embeddings = None
            
            # Remove from project map
            del self.projects[name]
            
            # Save updated projects list
            self._save_projects()
            
            logger.info(f"🗑️ Deleted project: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete project {name}: {e}")
            return False
    
    def list_projects(self) -> List[str]:
        return list(self.projects.keys())
    
    def get_project_info(self, name: str) -> Optional[Dict[str, Any]]:
        if name not in self.projects:
            return None
        
        proj = self.projects[name]
        return {
            "name": proj.name,
            "domain": proj.domain,
            "total_pdfs": len(proj.pdf_paths),
            "total_chunks": len(proj.chunks),
            "created_at": proj.created_at.isoformat(),
            "last_updated": proj.last_updated.isoformat()
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                        OPTIMIZED PDF PROCESSING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_pdf_to_project(self, pdf_path: str) -> bool:
        """Add PDF with optimized batch processing"""
        if not self.current_project:
            logger.error("No active project")
            return False
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return False
        
        logger.info(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
        
        try:
            # Ensure embedding model is loaded
            self._ensure_embedding_model()
            
            # Extract chunks with parallel processing
            new_chunks = self._extract_chunks_parallel(pdf_path)
            
            if not new_chunks:
                logger.error("No content extracted")
                return False
            
            # Compute embeddings in batch
            logger.info(f"🔢 Computing embeddings for {len(new_chunks)} chunks...")
            texts = [chunk.content for chunk in new_chunks]
            
            # Batch compute embeddings
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=self.config.BATCH_SIZE,
                show_progress_bar=TQDM_AVAILABLE,
                convert_to_numpy=True
            )
            
            # Convert to float16 for memory savings
            if self.config.USE_FLOAT16:
                embeddings = embeddings.astype('float16')
            
            # Update project
            self.current_project.pdf_paths.append(pdf_path)
            chunk_dicts = [chunk.model_dump() for chunk in new_chunks]
            self.current_project.chunks.extend(chunk_dicts)
            self.current_project.last_updated = datetime.now()
            
            # Add to active chunks
            self.active_chunks.extend(new_chunks)
            
            # Incremental index update
            self._add_to_index(embeddings)
            
            self._save_projects()
            
            # Cleanup
            gc.collect()
            
            logger.info(f"✅ Added {len(new_chunks)} chunks successfully")
            print(f"✅ Parsed PDF into {len(new_chunks)} knowledge chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            return False
    
    def _extract_chunks_parallel(self, pdf_path: str) -> List[Chunk]:
        """Extract chunks with parallel page processing"""
        import fitz  # Lazy import PyMuPDF
        
        chunks = []
        
        try:
            doc = fitz.open(pdf_path)
            source_filename = os.path.basename(pdf_path)
            total_pages = len(doc)
            
            logger.info(f"📑 Processing {total_pages} pages in parallel... (Source: {source_filename})")
            
            # Process pages in parallel
            def process_page(page_num):
                page = doc[page_num]
                text = page.get_text()
                
                if not text.strip():
                    logger.debug(f"⚠️ No text found on page {page_num + 1} of {source_filename}")
                    return []
                
                # Clean and chunk
                text = self._clean_text(text)
                text_chunks = self._fast_chunking(text)
                
                page_chunks = []
                for chunk_text in text_chunks:
                    if len(chunk_text.strip()) < 50:
                        continue
                    
                    chunk = Chunk(
                        content=chunk_text,
                        source=source_filename,
                        page_number=page_num + 1
                    )
                    page_chunks.append(chunk)
                
                return page_chunks
            
            # Use ThreadPoolExecutor for parallel processing
            page_iterator = range(total_pages)
            if TQDM_AVAILABLE:
                page_iterator = tqdm(range(total_pages), desc="Pages", unit="page")
            
            futures = []
            for page_num in page_iterator:
                future = self.executor.submit(process_page, page_num)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                page_chunks = future.result()
                chunks.extend(page_chunks)
            
            doc.close()
            
            if not chunks:
                logger.error(f"❌ Extraction failed: No text content found in {source_filename}. This PDF might be scanned or image-based.")
            else:
                logger.info(f"✅ Successfully extracted {len(chunks)} chunks from {source_filename}")
            
        except Exception as e:
            logger.error(f"Error extracting chunks: {e}")
            raise
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        return text.strip()
    
    def _fast_chunking(self, text: str) -> List[str]:
        if len(text) <= self.config.CHUNK_SIZE:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.CHUNK_SIZE
            
            if end < len(text):
                search_end = min(end + 100, len(text))
                chunk_text = text[start:search_end]
                
                last_period = max(
                    chunk_text.rfind('. '),
                    chunk_text.rfind('! '),
                    chunk_text.rfind('? ')
                )
                
                if last_period > self.config.CHUNK_SIZE * 0.7:
                    end = start + last_period + 2
            
            chunks.append(text[start:end].strip())
            start = end - self.config.CHUNK_OVERLAP
        
        return chunks
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                        OPTIMIZED INDEX MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _rebuild_index(self):
        """Rebuild FAISS index from scratch"""
        if not self.active_chunks:
            return
        
        import faiss # Lazy import
        import numpy as np # Lazy import
        
        logger.info("🔍 Building FAISS index...")
        self._ensure_embedding_model()
        
        texts = [chunk.content for chunk in self.active_chunks]
        
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        if self.config.USE_FLOAT16:
            embeddings = embeddings.astype('float16')
        
        # Build HNSW index
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, self.config.FAISS_HNSW_M)
        index.hnsw.efConstruction = self.config.FAISS_EF_CONSTRUCTION
        index.hnsw.efSearch = self.config.FAISS_EF_SEARCH
        
        # Add embeddings
        index.add(embeddings.astype('float32'))
        
        self.faiss_index = index
        self.chunk_embeddings = embeddings
        
        logger.info(f"✅ Index built with {len(self.active_chunks)} chunks")
    
    def _add_to_index(self, new_embeddings):
        """Incrementally add embeddings to existing index"""
        import faiss # Lazy import
        import numpy as np # Lazy import
        
        if self.faiss_index is None:
            dimension = new_embeddings.shape[1]
            self.faiss_index = faiss.IndexHNSWFlat(dimension, self.config.FAISS_HNSW_M)
            self.faiss_index.hnsw.efConstruction = self.config.FAISS_EF_CONSTRUCTION
            self.faiss_index.hnsw.efSearch = self.config.FAISS_EF_SEARCH
            self.chunk_embeddings = new_embeddings
        else:
            if self.chunk_embeddings is not None:
                self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_embeddings])
            else:
                self.chunk_embeddings = new_embeddings
        
        self.faiss_index.add(new_embeddings.astype('float32'))
        
        logger.info(f"✅ Added {len(new_embeddings)} vectors to index")
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                        OPTIMIZED RETRIEVAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _retrieve_chunks(self, query: str, top_k: int = None) -> List[Chunk]:
        """Retrieve relevant chunks with optional reranking"""
        if not self.faiss_index or not self.active_chunks:
            return []
        
        if top_k is None:
            top_k = self.config.MAX_CHUNKS_FOR_CONTEXT
        
        self._ensure_embedding_model()
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        search_k = min(top_k * 2, len(self.active_chunks))
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)
        
        # Get candidate chunks
        candidates = []
        for idx in indices[0]:
            if 0 <= idx < len(self.active_chunks):
                candidates.append(self.active_chunks[idx])
        
        # Rerank if enabled and beneficial
        if self.config.ENABLE_RERANKING and len(candidates) > top_k:
            self._ensure_reranker_model()
            
            if self.reranker_model:
                try:
                    pairs = [(query, chunk.content) for chunk in candidates]
                    scores = self.reranker_model.predict(pairs)
                    scored = list(zip(scores, candidates))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return [chunk for _, chunk in scored[:top_k]]
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")
        
        return candidates[:top_k]
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                        OPTIMIZED API QUERIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _query_with_fallback(self, messages: List[Dict[str, str]], max_tokens: int = 800) -> Tuple[str, str, int]:
        """Simplified API fallback: Ollama → Groq → Local"""
        
        # Try Ollama first
        if self.usage_tracker.can_make_request("ollama", 0):
            try:
                response = await self._query_ollama(messages, max_tokens)
                if response and response.strip():
                    self.usage_tracker.record_request("ollama", 0)
                    return response, "ollama", 0
            except Exception as e:
                logger.debug(f"Ollama failed: {e}")
        
        # Try Groq
        if self.config.ENABLE_REMOTE_APIS and self.config.GROQ_API_KEY:
            if self.usage_tracker.can_make_request("groq", max_tokens):
                try:
                    response = await self._query_groq(messages, max_tokens)
                    if response and response.strip():
                        self.usage_tracker.record_request("groq", max_tokens)
                        return response, "groq", max_tokens
                except Exception as e:
                    logger.debug(f"Groq failed: {e}")
        
        # Local fallback
        logger.warning(f"⚠️ All APIs unavailable. Ollama Status: {self.config.OLLAMA_BASE_URL}, Groq Key: {'Yes' if self.config.GROQ_API_KEY else 'No'}")
        return self._local_fallback(messages), "local", 0
    
    async def _query_ollama(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=60
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('response', '').strip()
                else:
                    raise Exception(f"Ollama error: {response.status}")
    
    async def _query_groq(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.GROQ_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": min(max_tokens, 1000)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.GROQ_API_URL,
                json=payload,
                headers=headers,
                timeout=20
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    raise Exception(f"Groq error: {response.status}")
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                parts.append(f"System: {content}")
            elif role == 'user':
                parts.append(f"Human: {content}")
        
        return "\n\n".join(parts) + "\n\nAssistant: "

    async def _web_search(self, query: str) -> List[Dict[str, str]]:
        """Search the web using DuckDuckGo and return structured results"""
        if not DDGS_AVAILABLE:
            logger.warning("duckduckgo_search not installed, skipping web search")
            return []
        
        try:
            logger.info(f"🌐 Searching the web for: {query}")
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append({
                        "source": r['href'],
                        "content": r['body']
                    })
            
            return results
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def _local_fallback(self, messages: List[Dict[str, str]]) -> str:
        return "I found relevant information in your documents, but AI generation is currently unavailable. Please check the sources below for detailed information."
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                        QUESTION ANSWERING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def ask_question(self, query: str, mode: str = "project", external_context: str = None) -> Answer:
        """Ask question with optimized pipeline. Supports 'project', 'web', and 'clipboard' (via external_context)."""
        start_time = time.time()
        
        if not self.current_project or not self.faiss_index:
            return Answer(
                answer="❌ No active project or knowledge base.",
                sources=[],
                confidence=0.0,
                api_used="none",
                processing_time=0.0
            )
        
        # Check cache
        context_hash = str(hash(str([c.chunk_id for c in self.active_chunks[:10]])))
        cache_key = self.cache._get_key(query, context_hash)
        
        cached = self.cache.get(cache_key)
        if cached:
            cached["cache_used"] = True
            cached["processing_time"] = time.time() - start_time
            return Answer(**cached)
        
        # Build context based on mode
        context = ""
        sources = []
        
        if external_context:
            context = f"CLIPBOARD CONTENT:\n{external_context}"
            sources = [{"source": "Clipboard Content", "page": 0, "chunk_id": 0}]
            logger.info(f"📋 Using clipboard content ({len(external_context)} chars) for context")
            logger.debug(f"Clipboard snippet: {external_context[:100]}...")
        elif mode == "web":
            web_results = await self._web_search(query)
            if web_results:
                context_parts = []
                for res in web_results:
                    context_parts.append(f"Source: {res['source']}\n{res['content']}")
                    sources.append({"source": res['source'], "page": 0, "chunk_id": 0})
                context = "\n\n".join(context_parts)
            else:
                context = "No relevant web results found."
            logger.info(f"🌐 Using {len(web_results)} web results for context")
        else:
            # Standard Project RAG mode
            chunks = self._retrieve_chunks(query)
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                snippet = chunk.content
                context_parts.append(f"SOURCE: {chunk.source} | PAGE: {chunk.page_number}\nCONTENT: {snippet}")
                sources.append({
                    "source": chunk.source,
                    "page": chunk.page_number,
                    "chunk_id": chunk.chunk_id
                })
            context = "\n\n".join(context_parts)
            logger.info(f"📂 Using {len(chunks)} project chunks for context")
        
        # Create messages
        domain = self.current_project.domain or "general"
        
        system_msg = f"""You are a Distinguished Research Scientist with expertise in {domain} and scholarly analysis.
Your primary objective is to provide high-fidelity, evidence-based responses derived from the provided context.

Guidelines for Analysis:
1. **Academic Rigor**: Maintain a professional, objective, and analytical tone.
2. **Structural Clarity**: Use structured markdown (bold headers, bulleted lists, and tables where appropriate) to organize complex information.
3. **Source Integrity**: Explicitly attribute information to the provided sources using [Source Name, Page X] notation.
4. **Synthesis**: Connect related concepts across different document chunks to provide a holistic view.
5. **Precision**: Use domain-specific terminology accurately. If the context is insufficient, specify exactly what is missing rather than speculating.

Your response should be formatted for visual excellence in a research environment."""
        
        user_msg = f"""Question: {query}

Context:
{context}

Provide a comprehensive answer:"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        # Generate answer
        answer_text, api_used, tokens = await self._query_with_fallback(messages, max_tokens=600)
        
        # Calculate confidence
        # For non-project modes, we use a simpler confidence calculation or slightly adjust it
        confidence = self._calculate_confidence(query, [], answer_text) if mode != "project" else self._calculate_confidence(query, chunks if 'chunks' in locals() else [], answer_text)
        
        processing_time = time.time() - start_time
        
        result = Answer(
            answer=answer_text.strip(),
            sources=sources,
            confidence=confidence,
            api_used=api_used,
            tokens_used=tokens,
            processing_time=processing_time,
            cache_used=False
        )
        
        # Cache result
        self.cache.set(cache_key, result.model_dump())
        
        return result
    
    def _calculate_confidence(self, query: str, chunks: List[Chunk], answer: str) -> float:
        base = 0.5
        source_factor = min(1.0, len(chunks) * 0.15)
        length_factor = min(1.0, len(answer) / 300)
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words.intersection(answer_words))
        overlap_factor = min(1.0, overlap / max(1, len(query_words)))
        
        confidence = base + (source_factor * 0.3) + (length_factor * 0.1) + (overlap_factor * 0.1)
        return min(1.0, max(0.0, confidence))
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                        STATUS & UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def print_status(self):
        """Print system status"""
        print("\n" + "="*70)
        print("🚀 LIGHTWEIGHT RAG v9.0 - STATUS")
        print("="*70)
        
        try:
            response = requests.get(f"{self.config.OLLAMA_BASE_URL}/api/tags", timeout=3)
            if response.status_code == 200:
                print("✅ Ollama: AVAILABLE")
            else:
                print("❌ Ollama: NOT RUNNING")
        except:
            print("❌ Ollama: NOT AVAILABLE")
        
        if self.config.GROQ_API_KEY:
            print("✅ Groq API: KEY PROVIDED")
        else:
            print("⚠️  Groq API: NO KEY")
        
        cache_stats = self.cache.get_stats()
        print(f"\n💾 Cache: {cache_stats['hit_rate']:.1f}% hit rate, {cache_stats['cache_size']} entries")
        
        print(f"\n🤖 Models:")
        print(f"   Embedding: {'Loaded' if self.embedding_model else 'Not loaded (lazy)'}")
        print(f"   Reranker: {'Loaded' if self.reranker_model else 'Not loaded (lazy)'}")
        print("="*70)

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                                   CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════════════════

async def run_cli():
    """Optimized CLI interface"""
    
    print("\n" + "="*70)
    print("🚀 LIGHTWEIGHT RAG v9.0")
    print("Zero-lag Startup | Crash Protection | Lazy Loading")
    print("="*70)
    
    try:
        config = OptimizedConfig()
        rag = OptimizedRAGService(config)
        print("\n✅ System ready! (Heavy imports will load on first use)")
    except Exception as e:
        print(f"\n❌ Initialization error: {e}")
        return
    
    while True:
        try:
            # Project selection
            print("\n" + "="*70)
            print("📁 PROJECT MANAGEMENT")
            print("="*70)
            
            projects = rag.list_projects()
            
            if projects:
                print(f"\n📂 Available Projects:")
                for i, name in enumerate(projects, 1):
                    info = rag.get_project_info(name)
                    domain = f" ({info['domain']})" if info and info['domain'] else ""
                    pdfs = info['total_pdfs'] if info else 0
                    chunks = info['total_chunks'] if info else 0
                    print(f"   {i}. {name}{domain} - {pdfs} PDFs, {chunks} chunks")
                
                print(f"   {len(projects) + 1}. ➕ Create new project")
                print(f"   {len(projects) + 2}. 🔍 System status")
                print(f"   {len(projects) + 3}. ❌ Exit")
                
                choice = input(f"\n📋 Select (1-{len(projects) + 3}): ").strip()
                
                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(projects):
                        if rag.load_project(projects[choice_num - 1]):
                            break
                    elif choice_num == len(projects) + 1:
                        name = input("\n📝 Project name: ").strip()
                        if name:
                            domain = input("🏷️  Domain (optional): ").strip() or None
                            if rag.create_project(name, domain):
                                if rag.load_project(name):
                                    break
                    elif choice_num == len(projects) + 2:
                        rag.print_status()
                        input("\nPress Enter to continue...")
                        continue
                    elif choice_num == len(projects) + 3:
                        print("\n👋 Goodbye!")
                        return
                except ValueError:
                    print("❌ Invalid input")
                    continue
            else:
                # No projects
                name = input("\n📝 Create new project (name): ").strip()
                if name:
                    domain = input("🏷️  Domain (optional): ").strip() or None
                    if rag.create_project(name, domain):
                        if rag.load_project(name):
                            break
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted")
            return
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue
    
    # PDF Management
    print("\n" + "="*70)
    print("📚 DOCUMENT MANAGEMENT")
    print("="*70)
    
    current_pdfs = len(rag.current_project.pdf_paths)
    print(f"📄 Current PDFs: {current_pdfs}")
    
    while True:
        pdf_path = input("\n📄 Enter PDF path (or 'done' to finish): ").strip()
        
        if pdf_path.lower() == 'done':
            if current_pdfs == 0:
                print("❌ At least one PDF required")
                continue
            break
        
        pdf_path = pdf_path.strip('"\'')
        
        if rag.add_pdf_to_project(pdf_path):
            current_pdfs += 1
            print(f"✅ Added! Total: {current_pdfs} PDFs")
        else:
            print("❌ Failed to add PDF")
    
    if not rag.active_chunks:
        print("\n❌ No content extracted")
        return
    
    # Q&A Loop
    print("\n" + "="*70)
    print("🎯 QUESTION & ANSWER")
    print("="*70)
    print(f"✅ Ready with {len(rag.active_chunks)} chunks from {current_pdfs} PDFs\n")
    print("Commands: 'status', 'cache', 'project', 'switch', 'exit'\n")
    
    while True:
        question = input("❓ Your question: ").strip()
        
        if not question:
            continue
        
        if question.lower() == 'exit':
            print("\n👋 Goodbye!")
            return
        
        if question.lower() == 'switch':
            break
        
        if question.lower() == 'status':
            rag.print_status()
            continue

        if question.lower() == 'project':
            info = rag.get_project_info(rag.current_project.name)
            if info:
                 print(f"\n📂 {info['name']} ({info['domain'] or 'General'})")
                 print(f"   PDFs: {info['total_pdfs']}, Chunks: {info['total_chunks']}")
            continue
        
        if question.lower() == 'cache':
            stats = rag.cache.get_stats()
            print(f"\n📊 Cache: {stats['hit_rate']:.1f}% hit rate, {stats['cache_size']} entries")
            continue
        
        # Process question
        try:
            print("\n🔍 Processing...")
            result = await rag.ask_question(question)
            
            print("\n" + "="*70)
            print("📋 ANSWER")
            print("="*70)
            print(result.answer)
            
            print("\n" + "-"*50)
            print("📊 METRICS")
            print("-"*50)
            print(f"🎯 Confidence: {result.confidence:.3f}")
            print(f"⚡ API: {result.api_used}")
            print(f"⏱️  Time: {result.processing_time:.2f}s")
            print(f"📚 Sources: {len(result.sources)}")
            print(f"💾 Cache: {'✓' if result.cache_used else '✗'}")
            
            if result.sources:
                print("\n" + "-"*50)
                print("📚 SOURCES")
                print("-"*50)
                for i, src in enumerate(result.sources[:5], 1):
                    print(f"{i}. 📄 {src['source']} (page {src['page']})")
            
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Catchable Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                                    MAIN ENTRY
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        asyncio.run(run_cli())
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        print(f"\n❌ Critical error: {e}")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        # 🛡️ SHUTDOWN PROTECTION
        print("\n" + "="*70)
        print("🛑 SYSTEM HALTED")
        print("="*70)
        input("Press Enter to close this window...")

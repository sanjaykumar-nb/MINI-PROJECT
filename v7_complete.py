"""
Ultimate Enhanced RAG System v7.0 - FINAL PRODUCTION RELEASE
============================================================================

🚀 MOST COMPREHENSIVE RAG SYSTEM - Combining ALL Best Features from v4, v5, v6:

Complete Feature Set:
• ✅ Deep Research Mode with advanced prompt engineering (from v6)
• ✅ Multi-API research paper integration: arXiv, PubMed, Semantic Scholar (from v6)
• ✅ Multi-tier API fallback: Ollama → Groq → HuggingFace → Local template (from v5)
• ✅ Clipboard text extraction and processing (NEW in v7!)
• ✅ Advanced caching with TTL, compression, memory/disk layers (from v5)
• ✅ Cross-encoder reranking for superior relevance (from v5)
• ✅ FAISS HNSW indexing for blazing fast search (from v5)
• ✅ Comprehensive analytics and confidence scoring (from v5)
• ✅ Research-grade output with citations and links (from v6)
• ✅ Advanced concept mapping and gap analysis (from v6)
• ✅ Cross-reference analysis between papers and documents (from v6)
• ✅ Domain-aware processing with smart prompting (from v5)
• ✅ Intelligent usage tracking and rate limiting (from v5)
• ✅ Project management with metadata (from v5)
• ✅ Enhanced error handling and logging (from all)
• ✅ Production-ready reliability (from all)

Author: Enhanced RAG Development Team
Version: 7.0 (Ultimate Edition)
License: MIT
"""

import os
import re
import json
import time
import hashlib
import asyncio
import logging
import fitz  # PyMuPDF
import faiss
import aiohttp
import requests
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
import threading
import gzip
import pickle
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

# Clipboard support - cross-platform (NEW in v7)
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    logging.info("pyperclip not available - clipboard features disabled")

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedRAG")

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                                ENHANCED CONFIGURATION WITH RESEARCH APIS
# ═══════════════════════════════════════════════════════════════════════════════════════════

def _env_flag(key: str, default: bool = True) -> bool:
    """Parse boolean-like environment variables."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class EnhancedConfig:
    """Enhanced configuration with research API integration"""
    
    # API Keys - loaded from environment for security
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    HF_TOKEN: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    
    # Research API Keys (Optional - add your keys here for enhanced features)
    PUBMED_API_KEY: str = field(default_factory=lambda: os.getenv("PUBMED_API_KEY", ""))
    ARXIV_API_KEY: str = field(default_factory=lambda: os.getenv("ARXIV_API_KEY", ""))
    CROSSREF_API_KEY: str = field(default_factory=lambda: os.getenv("CROSSREF_API_KEY", ""))
    OPENAIRE_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAIRE_API_KEY", ""))
    
    # API Endpoints
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    HF_API_BASE: str = "https://api-inference.huggingface.co/models"
    
    # Research API Endpoints
    SEMANTIC_SCHOLAR_API: str = "https://api.semanticscholar.org/graph/v1/paper/search"
    ARXIV_API: str = "http://export.arxiv.org/api/query"
    PUBMED_API: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    CROSSREF_API: str = "https://api.crossref.org/works"
    OPENAIRE_API: str = "https://api.openaire.eu/search/publications"
    
    # Model Configuration (OPTIMIZED for faster startup!)
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OLLAMA_MODEL: str = "llama3.2:3b"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Optimized: 80MB vs 420MB (5x faster!)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # HuggingFace Fallback Models
    HF_FALLBACK_MODELS: tuple = (
        "microsoft/DialoGPT-medium",
        "google/flan-t5-large",
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mistral-7B-Instruct-v0.1"
    )
    
    # Deep Research Configuration
    DEEP_RESEARCH_MAX_PAPERS: int = 20
    DEEP_RESEARCH_MAX_TOKENS: int = 2000
    DEEP_RESEARCH_CHUNK_OVERLAP: int = 300
    COMPREHENSIVE_ANALYSIS_CHUNKS: int = 15
    
    # API Limits and Settings
    GROQ_REQUESTS_PER_MINUTE: int = 30
    HF_REQUESTS_PER_MINUTE: int = 10
    OLLAMA_REQUESTS_PER_MINUTE: int = 60
    DAILY_TOKEN_LIMIT: int = 50000
    MAX_TOKENS_PER_REQUEST: int = 1000
    
    # Performance Settings (OPTIMIZED!)
    CACHE_DURATION_HOURS: int = 24
    PDF_MINIMUM_REQUIRED: int = 1  # Optimized: changed from 3 to 1 for flexibility
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    MAX_CHUNKS_FOR_CONTEXT: int = 8
    FAISS_HNSW_M: int = 32
    FAISS_EF_CONSTRUCTION: int = 200
    FAISS_EF_SEARCH: int = 100
    
    # Web Search Configuration
    WEB_SEARCH_MAX_RESULTS: int = 3
    WEB_TIMEOUT_SECONDS: int = 15
    
    # Storage Configuration
    PROJECT_DATA_DIR: str = ".enhanced_rag_projects"
    CACHE_DIR: str = ".rag_cache"
    USAGE_TRACKING_FILE: str = ".api_usage.json"
    PROJECT_METADATA_FILE: str = "enhanced_projects.json"
    
    # Clipboard Configuration (NEW in v7)
    CLIPBOARD_AUTO_PROCESS: bool = False  # Auto-process clipboard changes
    CLIPBOARD_MAX_LENGTH: int = 50000  # Max characters from clipboard
    CLIPBOARD_CHUNK_AS_DOCUMENT: bool = True  # Treat clipboard as separate document
    
    # Local-first behaviour toggles
    ENABLE_REMOTE_APIS: bool = field(default_factory=lambda: _env_flag("ENABLE_REMOTE_APIS", True))
    ENABLE_CLIPBOARD: bool = field(default_factory=lambda: _env_flag("ENABLE_CLIPBOARD", CLIPBOARD_AVAILABLE))
    ENABLE_DEEP_RESEARCH: bool = field(default_factory=lambda: _env_flag("ENABLE_DEEP_RESEARCH", True))

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                                    UTILITY CLASSES
# ═══════════════════════════════════════════════════════════════════════════════════════════

class EnhancedUsageTracker:
    """Intelligent API usage tracking with rate limiting and quota management"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.usage_file = config.USAGE_TRACKING_FILE
        self.usage_data = self._load_usage_data()
        self.request_timestamps = {
            "ollama": [],
            "groq": [],
            "hf": []
        }
        self._lock = threading.Lock()
    
    def _load_usage_data(self) -> Dict[str, Any]:
        """Load usage data from disk with error handling"""
        defaults = {
            "daily_tokens": 0,
            "last_reset_date": str(datetime.now().date()),
            "api_call_history": {
                "ollama": 0,
                "groq": 0,
                "hf": 0
            }
        }

        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("usage data must be a JSON object")

                usage = {**defaults, **data}
                history = usage.get("api_call_history", {})
                if not isinstance(history, dict):
                    history = {}
                for key, value in defaults["api_call_history"].items():
                    history.setdefault(key, value)
                usage["api_call_history"] = history

                logger.info(f"📊 Loaded usage data: {usage.get('daily_tokens', 0)} tokens used today")
                return usage
            except Exception as e:
                logger.warning(f"Failed to load usage data: {e}")

        return defaults
    
    def _save_usage_data(self):
        """Save usage data to disk with error handling"""
        try:
            with self._lock:
                with open(self.usage_file, 'w', encoding='utf-8') as f:
                    json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save usage data: {e}")
    
    def can_make_request(self, api_type: str, tokens_needed: int = 100) -> bool:
        """Check if request can be made within API limits and quotas"""
        with self._lock:
            # Reset daily counter if new day
            today = str(datetime.now().date())
            if self.usage_data.get("last_reset_date") != today:
                self.usage_data["daily_tokens"] = 0
                self.usage_data["last_reset_date"] = today
                logger.info("🔄 Daily usage counter reset")
            
            # Check daily token limit
            if self.usage_data["daily_tokens"] + tokens_needed > self.config.DAILY_TOKEN_LIMIT:
                logger.warning(f"⚠️ Daily token limit would be exceeded")
                return False
            
            # Check rate limits
            now = time.time()
            minute_ago = now - 60
            
            # Clean old timestamps
            self.request_timestamps[api_type] = [
                ts for ts in self.request_timestamps[api_type] 
                if ts > minute_ago
            ]
            
            # Check per-minute limits
            rate_limits = {
                "ollama": self.config.OLLAMA_REQUESTS_PER_MINUTE,
                "groq": self.config.GROQ_REQUESTS_PER_MINUTE,
                "hf": self.config.HF_REQUESTS_PER_MINUTE
            }
            
            current_requests = len(self.request_timestamps[api_type])
            limit = rate_limits.get(api_type, 10)
            
            if current_requests >= limit:
                logger.warning(f"⚠️ Rate limit reached for {api_type}")
                return False
            
            return True
    
    def record_successful_request(self, api_type: str, tokens_used: int):
        """Record a successful API request"""
        with self._lock:
            self.request_timestamps[api_type].append(time.time())
            self.usage_data["daily_tokens"] += tokens_used
            self.usage_data["api_call_history"][api_type] = (
                self.usage_data["api_call_history"].get(api_type, 0) + 1
            )
            self._save_usage_data()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        with self._lock:
            return {
                "daily_tokens": self.usage_data["daily_tokens"],
                "daily_limit": self.config.DAILY_TOKEN_LIMIT,
                "usage_percentage": (self.usage_data["daily_tokens"] / self.config.DAILY_TOKEN_LIMIT) * 100,
                "api_calls_today": self.usage_data["api_call_history"].copy(),
                "last_reset": self.usage_data["last_reset_date"]
            }

class EnhancedCacheManager:
    """Advanced caching system with TTL, compression, and intelligent invalidation"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "writes": 0}
    
    def _get_cache_key(self, query: str, context_hash: str, use_web: bool = False) -> str:
        """Generate unique cache key for query and context"""
        combined = f"{query}_{context_hash}_{use_web}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.cache.gz")
    
    def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if available and fresh"""
        # Check memory cache first
        if cache_key in self.memory_cache:
            cached_item = self.memory_cache[cache_key]
            if self._is_cache_fresh(cached_item["timestamp"]):
                self.cache_stats["hits"] += 1
                logger.debug("📦 Cache HIT (memory)")
                return cached_item["data"]
        
        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with gzip.open(cache_path, 'rb') as f:
                    cached_item = pickle.load(f)
                
                if self._is_cache_fresh(cached_item["timestamp"]):
                    self.memory_cache[cache_key] = cached_item
                    self.cache_stats["hits"] += 1
                    logger.debug("📦 Cache HIT (disk)")
                    return cached_item["data"]
                else:
                    os.remove(cache_path)
                    logger.debug("🗑️ Removed expired cache entry")
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    def cache_response(self, cache_key: str, data: Dict[str, Any]):
        """Cache response with compression and TTL"""
        try:
            cached_item = {
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            self.memory_cache[cache_key] = cached_item
            
            cache_path = self._get_cache_path(cache_key)
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(cached_item, f)
            
            self.cache_stats["writes"] += 1
            logger.debug(f"💾 Cached response")
            
            if len(self.memory_cache) > 100:
                self._cleanup_memory_cache()
                
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _is_cache_fresh(self, timestamp_str: str) -> bool:
        """Check if cached item is still fresh"""
        try:
            cached_time = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - cached_time
            return age < timedelta(hours=self.config.CACHE_DURATION_HOURS)
        except Exception:
            return False
    
    def _cleanup_memory_cache(self):
        """Remove old entries from memory cache"""
        sorted_items = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        )
        self.memory_cache = dict(sorted_items[:50])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "cache_writes": self.cache_stats["writes"],
            "hit_rate_percentage": round(hit_rate, 2),
            "memory_cache_entries": len(self.memory_cache)
        }

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                                    ENHANCED DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════════════════

class EnhancedChunk(BaseModel):
    """Enhanced chunk model with comprehensive metadata"""
    content: str
    source: str
    page_number: int
    chunk_id: int = Field(default_factory=lambda: int(time.time() * 1000000) % 1000000)
    section_title: Optional[str] = None
    chunk_type: str = "text"
    semantic_density: float = 0.0
    readability_score: float = 0.0
    word_count: int = 0
    is_web_content: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WebSearchResult(BaseModel):
    """Model for web search results"""
    title: str
    url: str
    snippet: str
    content: str = ""
    relevance_score: float = 0.0
    source_type: str = "academic"
    scraped_at: Optional[datetime] = None

class ResearchPaper(BaseModel):
    """Enhanced model for research papers with comprehensive metadata"""
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str = ""
    url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    pubmed_id: str = ""
    publication_date: Optional[datetime] = None
    journal: str = ""
    citation_count: int = 0
    keywords: List[str] = Field(default_factory=list)
    source_api: str = ""
    relevance_score: float = 0.0
    full_text: str = ""

class EnhancedAnswer(BaseModel):
    """Comprehensive answer model with detailed analytics"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    reasoning: str
    related_concepts: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    web_sources: List[Dict[str, Any]] = Field(default_factory=list)
    query_analysis: Dict[str, Any] = Field(default_factory=dict)
    answer_quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    api_used: str = "unknown"
    tokens_used: int = 0
    processing_time: float = 0.0
    cache_used: bool = False

class DeepResearchResult(BaseModel):
    """Comprehensive research analysis result"""
    query: str
    comprehensive_analysis: str
    key_concepts: List[str] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)
    methodological_insights: List[str] = Field(default_factory=list)
    future_directions: List[str] = Field(default_factory=list)
    related_papers: List[ResearchPaper] = Field(default_factory=list)
    concept_hierarchy: Dict[str, List[str]] = Field(default_factory=dict)
    statistical_insights: List[str] = Field(default_factory=list)
    cross_references: List[Dict[str, str]] = Field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    sources_analyzed: int = 0
    research_quality_score: float = 0.0

class Project(BaseModel):
    """Enhanced project model"""
    name: str
    domain: Optional[str] = None
    description: Optional[str] = None
    pdf_paths: List[str] = Field(default_factory=list)
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    total_chunks: int = 0
    total_pages: int = 0
    embedding_model_used: str = ""

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                              RESEARCH API INTEGRATION SERVICE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class ResearchAPIService:
    """Advanced research paper retrieval from multiple academic sources"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.headers = {
            'User-Agent': 'Enhanced RAG Research System v5.0 (Educational Research)',
            'Accept': 'application/json, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9'
        }
    
    async def search_comprehensive_papers(self, query: str, max_papers: int = 20) -> List[ResearchPaper]:
        """Search across multiple research databases"""
        if not self.config.ENABLE_REMOTE_APIS:
            logger.debug("Remote APIs disabled; skipping comprehensive paper search")
            return []

        all_papers = []
        
        # Search Semantic Scholar
        semantic_papers = await self._search_semantic_scholar(query, max_papers // 3)
        all_papers.extend(semantic_papers)
        
        # Search arXiv
        arxiv_papers = await self._search_arxiv(query, max_papers // 3)
        all_papers.extend(arxiv_papers)
        
        # Search PubMed
        pubmed_papers = await self._search_pubmed(query, max_papers // 3)
        all_papers.extend(pubmed_papers)
        
        # Remove duplicates and sort by relevance
        unique_papers = self._deduplicate_papers(all_papers)
        return sorted(unique_papers, key=lambda x: x.relevance_score, reverse=True)[:max_papers]
    
    async def _search_semantic_scholar(self, query: str, max_results: int) -> List[ResearchPaper]:
        """Search Semantic Scholar API"""
        if not self.config.ENABLE_REMOTE_APIS:
            logger.debug("Remote APIs disabled; skipping Semantic Scholar search")
            return []

        papers = []
        try:
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,abstract,url,venue,year,citationCount,doi"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.SEMANTIC_SCHOLAR_API,
                    params=params,
                    headers=self.headers,
                    timeout=15
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        for paper_data in data.get("data", []):
                            paper = ResearchPaper(
                                title=paper_data.get("title", ""),
                                authors=[author.get("name", "") for author in paper_data.get("authors", [])],
                                abstract=paper_data.get("abstract", "")[:1000],
                                url=paper_data.get("url", ""),
                                doi=paper_data.get("doi", ""),
                                journal=paper_data.get("venue", ""),
                                citation_count=paper_data.get("citationCount", 0),
                                source_api="Semantic Scholar",
                                relevance_score=0.8
                            )
                            
                            if paper_data.get("year"):
                                try:
                                    paper.publication_date = datetime(paper_data["year"], 1, 1)
                                except:
                                    pass
                            
                            papers.append(paper)
                            
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")
        
        return papers
    
    async def _search_arxiv(self, query: str, max_results: int) -> List[ResearchPaper]:
        """Search arXiv API"""
        if not self.config.ENABLE_REMOTE_APIS:
            logger.debug("Remote APIs disabled; skipping arXiv search")
            return []

        papers = []
        try:
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.ARXIV_API,
                    params=params,
                    headers=self.headers,
                    timeout=15
                ) as response:
                    
                    if response.status == 200:
                        xml_content = await response.text()
                        root = ET.fromstring(xml_content)
                        
                        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                            title = entry.find('{http://www.w3.org/2005/Atom}title')
                            summary = entry.find('{http://www.w3.org/2005/Atom}summary')
                            authors = entry.findall('{http://www.w3.org/2005/Atom}author')
                            published = entry.find('{http://www.w3.org/2005/Atom}published')
                            id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                            
                            arxiv_id = ""
                            url = ""
                            if id_elem is not None:
                                url = id_elem.text
                                arxiv_id = url.split('/')[-1] if '/' in url else ""
                            
                            paper = ResearchPaper(
                                title=title.text if title is not None else "",
                                authors=[author.find('{http://www.w3.org/2005/Atom}name').text 
                                       for author in authors if author.find('{http://www.w3.org/2005/Atom}name') is not None],
                                abstract=summary.text[:1000] if summary is not None else "",
                                url=url,
                                arxiv_id=arxiv_id,
                                source_api="arXiv",
                                relevance_score=0.7
                            )
                            
                            if published is not None:
                                try:
                                    paper.publication_date = datetime.fromisoformat(published.text.replace('Z', '+00:00'))
                                except:
                                    pass
                            
                            papers.append(paper)
                            
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
        
        return papers
    
    async def _search_pubmed(self, query: str, max_results: int) -> List[ResearchPaper]:
        """Search PubMed API"""
        if not self.config.ENABLE_REMOTE_APIS:
            logger.debug("Remote APIs disabled; skipping PubMed search")
            return []

        papers = []
        try:
            # First, search for paper IDs
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json"
            }
            
            if self.config.PUBMED_API_KEY:
                search_params["api_key"] = self.config.PUBMED_API_KEY
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.PUBMED_API}/esearch.fcgi",
                    params=search_params,
                    headers=self.headers,
                    timeout=15
                ) as response:
                    
                    if response.status == 200:
                        search_data = await response.json()
                        ids = search_data.get("esearchresult", {}).get("idlist", [])
                        
                        if ids:
                            # Get detailed information for the papers
                            fetch_params = {
                                "db": "pubmed",
                                "id": ",".join(ids[:max_results]),
                                "retmode": "json"
                            }
                            
                            if self.config.PUBMED_API_KEY:
                                fetch_params["api_key"] = self.config.PUBMED_API_KEY
                            
                            async with session.get(
                                f"{self.config.PUBMED_API}/esummary.fcgi",
                                params=fetch_params,
                                headers=self.headers,
                                timeout=15
                            ) as detail_response:
                                
                                if detail_response.status == 200:
                                    detail_data = await detail_response.json()
                                    
                                    for paper_id, paper_data in detail_data.get("result", {}).items():
                                        if paper_id.isdigit():
                                            paper = ResearchPaper(
                                                title=paper_data.get("title", ""),
                                                authors=paper_data.get("authors", [])[:5],
                                                abstract="",  # PubMed summary doesn't include full abstract
                                                url=f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
                                                pubmed_id=paper_id,
                                                journal=paper_data.get("source", ""),
                                                source_api="PubMed",
                                                relevance_score=0.75
                                            )
                                            
                                            if paper_data.get("pubdate"):
                                                try:
                                                    date_str = paper_data["pubdate"]
                                                    year = int(date_str.split()[0]) if date_str else 2000
                                                    paper.publication_date = datetime(year, 1, 1)
                                                except:
                                                    pass
                                            
                                            papers.append(paper)
                        
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
        
        return papers
    
    def _deduplicate_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Remove duplicate papers based on title similarity and DOI"""
        unique_papers = []
        seen_titles = set()
        seen_dois = set()
        
        for paper in papers:
            # Check DOI first (most reliable)
            if paper.doi and paper.doi in seen_dois:
                continue
                
            # Check title similarity
            title_lower = paper.title.lower().strip()
            title_words = set(title_lower.split())
            
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                if len(title_words.intersection(seen_words)) / len(title_words.union(seen_words)) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(title_lower)
                if paper.doi:
                    seen_dois.add(paper.doi)
        
        return unique_papers

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                              DEEP RESEARCH ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════

class DeepResearchEngine:
    """Advanced research analysis with comprehensive document and paper analysis"""
    
    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.config = rag_service.config
        self.research_api = ResearchAPIService(self.config)
    
    async def conduct_deep_research(self, query: str) -> DeepResearchResult:
        """Perform comprehensive deep research analysis"""
        if not self.config.ENABLE_REMOTE_APIS:
            raise RuntimeError("Deep research requires ENABLE_REMOTE_APIS=true")

        start_time = time.time()
        
        logger.info(f"🔬 Starting deep research analysis for: {query[:100]}...")
        
        # Step 1: Comprehensive document analysis
        logger.info("📚 Analyzing uploaded documents comprehensively...")
        local_analysis = await self._analyze_all_documents_comprehensively(query)
        
        # Step 2: Research paper discovery and analysis
        logger.info("🌐 Discovering and analyzing research papers...")
        research_papers = await self.research_api.search_comprehensive_papers(
            query, self.config.DEEP_RESEARCH_MAX_PAPERS
        )
        
        # Step 3: Cross-reference analysis
        logger.info("🔗 Performing cross-reference analysis...")
        cross_refs = await self._perform_cross_reference_analysis(query, research_papers)
        
        # Step 4: Generate comprehensive research synthesis
        logger.info("🧠 Synthesizing comprehensive research analysis...")
        synthesis = await self._generate_deep_research_synthesis(
            query, local_analysis, research_papers, cross_refs
        )
        
        # Step 5: Extract advanced insights
        logger.info("🎯 Extracting advanced research insights...")
        insights = await self._extract_advanced_insights(synthesis, research_papers)
        
        processing_time = time.time() - start_time
        
        result = DeepResearchResult(
            query=query,
            comprehensive_analysis=synthesis,
            key_concepts=insights.get("concepts", []),
            research_gaps=insights.get("gaps", []),
            methodological_insights=insights.get("methodologies", []),
            future_directions=insights.get("future_directions", []),
            related_papers=research_papers,
            concept_hierarchy=insights.get("concept_hierarchy", {}),
            statistical_insights=insights.get("statistics", []),
            cross_references=cross_refs,
            confidence_score=insights.get("confidence", 0.8),
            processing_time=processing_time,
            sources_analyzed=len(self.rag_service.active_chunks) + len(research_papers),
            research_quality_score=self._calculate_research_quality_score(research_papers)
        )
        
        logger.info(f"✅ Deep research completed in {processing_time:.2f}s")
        return result
    
    async def _analyze_all_documents_comprehensively(self, query: str) -> str:
        """Analyze all uploaded documents with maximum depth"""
        if not self.rag_service.active_chunks:
            return "No local documents available for analysis."
        
        # Get more chunks for comprehensive analysis
        relevant_chunks = self.rag_service._retrieve_relevant_chunks(
            query, top_k=self.config.COMPREHENSIVE_ANALYSIS_CHUNKS
        )
        
        # Create comprehensive context
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(f"[Document {i}] {chunk.content}")
        
        comprehensive_context = "\n\n".join(context_parts)
        
        # Deep analysis prompt
        domain = self.rag_service.current_project.domain if self.rag_service.current_project else "multidisciplinary"
        
        deep_analysis_prompt = f"""You are a world-class research analyst specializing in {domain} research with 20+ years of experience in systematic reviews and meta-analysis.

COMPREHENSIVE DOCUMENT ANALYSIS TASK:
Research Query: {query}

Your mission is to conduct an exhaustive, PhD-level analysis of all provided documents. This analysis should be suitable for publication in top-tier academic journals.

ANALYSIS FRAMEWORK:
1. THEORETICAL FOUNDATIONS: Identify underlying theories, frameworks, and conceptual models
2. METHODOLOGICAL ANALYSIS: Examine research methods, data collection, analysis techniques
3. EMPIRICAL FINDINGS: Extract key results, statistical significance, effect sizes
4. CRITICAL EVALUATION: Assess validity, reliability, limitations, and bias
5. SYNTHESIS & INTEGRATION: Connect findings across documents, identify patterns
6. KNOWLEDGE GAPS: Highlight unexplored areas and research opportunities
7. PRACTICAL IMPLICATIONS: Discuss real-world applications and impact

ANALYSIS DEPTH REQUIREMENTS:
- Examine EVERY significant claim with supporting evidence
- Identify contradictions or inconsistencies between sources
- Extract specific data points, statistics, and quantitative findings
- Analyze methodology quality and potential confounding factors
- Assess generalizability and external validity
- Identify theoretical contributions and novel insights

OUTPUT SPECIFICATIONS:
- Minimum 1500 words of detailed analysis
- Use academic language and formal structure
- Include specific evidence and references where applicable
- Maintain critical objectivity while highlighting significance
- Structure findings hierarchically from broad themes to specific details

Context Documents:
{comprehensive_context[:8000]}

Provide your comprehensive research analysis:"""

        messages = [
            {"role": "system", "content": "You are a distinguished research analyst with expertise in systematic review methodology and critical analysis. Your analyses are used for high-impact research publications and policy decisions."},
            {"role": "user", "content": deep_analysis_prompt}
        ]
        
        # Use higher token limit for deep analysis
        analysis, _, _ = await self.rag_service._query_with_fallback(
            messages, max_tokens=self.config.DEEP_RESEARCH_MAX_TOKENS
        )
        
        return analysis
    
    async def _perform_cross_reference_analysis(self, query: str, papers: List[ResearchPaper]) -> List[Dict[str, str]]:
        """Perform cross-reference analysis between papers and local documents"""
        cross_refs = []
        
        for paper in papers[:10]:  # Limit for performance
            if paper.abstract and len(paper.abstract) > 100:
                # Find connections between paper and local documents
                local_chunks = self.rag_service._retrieve_relevant_chunks(paper.abstract, top_k=3)
                
                if local_chunks:
                    cross_ref = {
                        "paper_title": paper.title,
                        "paper_url": paper.url or f"DOI: {paper.doi}" if paper.doi else "",
                        "connection_type": "methodological" if any(word in paper.abstract.lower() 
                                                                for word in ["method", "approach", "technique"]) else "conceptual",
                        "relevance": "High" if len(local_chunks) >= 2 else "Medium",
                        "related_concepts": self._extract_connecting_concepts(paper.abstract, [c.content for c in local_chunks])
                    }
                    cross_refs.append(cross_ref)
        
        return cross_refs
    
    def _extract_connecting_concepts(self, paper_abstract: str, local_contents: List[str]) -> str:
        """Extract concepts that connect paper and local documents"""
        paper_words = set(re.findall(r'\b[A-Za-z]{4,}\b', paper_abstract.lower()))
        local_words = set()
        for content in local_contents:
            local_words.update(re.findall(r'\b[A-Za-z]{4,}\b', content.lower()))
        
        common_concepts = paper_words.intersection(local_words)
        stop_words = {"this", "that", "with", "from", "they", "were", "been", "have", "will", "would", "could", "should"}
        meaningful_concepts = [word for word in common_concepts if word not in stop_words]
        
        return ", ".join(sorted(meaningful_concepts)[:10])
    
    async def _generate_deep_research_synthesis(self, query: str, local_analysis: str, 
                                              papers: List[ResearchPaper], cross_refs: List[Dict]) -> str:
        """Generate comprehensive research synthesis"""
        
        # Prepare research papers summary
        papers_summary = []
        for paper in papers[:10]:
            paper_info = f"Title: {paper.title}\nAuthors: {', '.join(paper.authors[:3])}\nJournal: {paper.journal}\nAbstract: {paper.abstract[:300]}...\nURL: {paper.url or paper.doi}\n"
            papers_summary.append(paper_info)
        
        papers_context = "\n---\n".join(papers_summary)
        
        # Cross-references summary
        cross_ref_summary = "\n".join([
            f"• {ref['paper_title']}: {ref['connection_type']} connection ({ref['relevance']} relevance)"
            for ref in cross_refs[:5]
        ])
        
        domain = self.rag_service.current_project.domain if self.rag_service.current_project else "multidisciplinary"
        
        synthesis_prompt = f"""You are a distinguished research professor and systematic review expert conducting a comprehensive meta-analysis.

DEEP RESEARCH SYNTHESIS TASK:
Research Question: {query}
Domain: {domain}

You have access to:
1. Comprehensive analysis of uploaded documents
2. Latest research papers from multiple academic databases
3. Cross-reference analysis between sources

Your task is to create a definitive, publication-ready research synthesis that would be suitable for a top-tier academic journal.

LOCAL DOCUMENT ANALYSIS:
{local_analysis[:3000]}

LATEST RESEARCH PAPERS:
{papers_context[:4000]}

CROSS-REFERENCE CONNECTIONS:
{cross_ref_summary}

SYNTHESIS REQUIREMENTS:

1. EXECUTIVE SUMMARY (200 words)
   - Key findings and their significance
   - Novel contributions to the field
   - Practical implications

2. THEORETICAL FRAMEWORK (300 words)
   - Conceptual foundations
   - Theoretical models and their applications
   - Evolution of thinking in this area

3. METHODOLOGICAL LANDSCAPE (400 words)
   - Research approaches and their effectiveness
   - Methodological innovations
   - Best practices and limitations
   - Comparative analysis of methods

4. EMPIRICAL FINDINGS SYNTHESIS (500 words)
   - Convergent findings across studies
   - Contradictory results and their explanations
   - Statistical patterns and effect sizes
   - Quality assessment of evidence

5. CRITICAL ANALYSIS (300 words)
   - Strengths and limitations of current research
   - Methodological concerns and bias assessment
   - Generalizability and external validity

6. RESEARCH GAPS & OPPORTUNITIES (300 words)
   - Unexplored areas and questions
   - Methodological improvements needed
   - Emerging research directions

7. FUTURE RESEARCH AGENDA (200 words)
   - Priority research questions
   - Methodological recommendations
   - Policy and practice implications

QUALITY STANDARDS:
- Use precise academic language
- Support all claims with specific evidence
- Maintain critical objectivity
- Integrate findings coherently
- Highlight novel insights and contributions
- Ensure logical flow and structure

Provide your comprehensive research synthesis:"""

        messages = [
            {"role": "system", "content": "You are a world-renowned research professor with expertise in systematic reviews, meta-analysis, and academic writing. Your syntheses are frequently cited and used for policy development."},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        synthesis, _, _ = await self.rag_service._query_with_fallback(
            messages, max_tokens=self.config.DEEP_RESEARCH_MAX_TOKENS
        )
        
        return synthesis
    
    async def _extract_advanced_insights(self, synthesis: str, papers: List[ResearchPaper]) -> Dict[str, Any]:
        """Extract advanced insights from the research synthesis"""
        
        insights_prompt = f"""Extract advanced research insights from this comprehensive analysis:

RESEARCH SYNTHESIS:
{synthesis[:4000]}

Extract and structure the following insights:

1. KEY CONCEPTS (10-15 concepts):
   - Core theoretical concepts
   - Technical terms and methodologies  
   - Emerging themes

2. RESEARCH GAPS (5-8 gaps):
   - Specific unexplored areas
   - Methodological limitations
   - Knowledge deficits

3. METHODOLOGICAL INSIGHTS (5-7 insights):
   - Best practices identified
   - Novel methodological approaches
   - Comparative method effectiveness

4. FUTURE DIRECTIONS (5-8 directions):
   - Priority research questions
   - Emerging fields of inquiry
   - Policy implications

5. STATISTICAL INSIGHTS (3-5 insights):
   - Key quantitative findings
   - Effect sizes and significance
   - Trends in the data

Provide structured output for each category."""

        messages = [
            {"role": "system", "content": "You are an expert research analyst specializing in knowledge extraction and systematic categorization."},
            {"role": "user", "content": insights_prompt}
        ]
        
        insights_text, _, _ = await self.rag_service._query_with_fallback(messages, max_tokens=1000)
        
        # Parse insights (simplified - in production, use more robust parsing)
        insights = {
            "concepts": self._extract_list_items(insights_text, "concepts"),
            "gaps": self._extract_list_items(insights_text, "gaps"),
            "methodologies": self._extract_list_items(insights_text, "methodological"),
            "future_directions": self._extract_list_items(insights_text, "future"),
            "statistics": self._extract_list_items(insights_text, "statistical"),
            "concept_hierarchy": {},
            "confidence": 0.85
        }
        
        return insights
    
    def _extract_list_items(self, text: str, category: str) -> List[str]:
        """Extract list items from structured text"""
        lines = text.split('\n')
        items = []
        in_category = False
        
        for line in lines:
            line = line.strip()
            if category.lower() in line.lower():
                in_category = True
                continue
            elif line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')) and in_category:
                in_category = False
            elif in_category and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                items.append(line[1:].strip())
            elif in_category and line and not line.startswith(('1.', '2.', '3.')):
                items.append(line)
        
        return items[:10]  # Limit items
    
    def _calculate_research_quality_score(self, papers: List[ResearchPaper]) -> float:
        """Calculate overall research quality score"""
        if not papers:
            return 0.5
        
        total_score = 0
        for paper in papers:
            score = 0.5  # Base score
            
            # Recency bonus
            if paper.publication_date:
                years_old = (datetime.now() - paper.publication_date).days / 365
                if years_old < 2:
                    score += 0.3
                elif years_old < 5:
                    score += 0.2
                else:
                    score += 0.1
            
            # Citation bonus
            if paper.citation_count > 100:
                score += 0.2
            elif paper.citation_count > 50:
                score += 0.15
            elif paper.citation_count > 10:
                score += 0.1
            
            # Author count (collaboration indicator)
            if len(paper.authors) >= 3:
                score += 0.1
            
            # Source API quality
            if paper.source_api in ["Semantic Scholar", "PubMed"]:
                score += 0.1
            
            total_score += min(score, 1.0)
        
        return min(total_score / len(papers), 1.0)

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                          MAIN RAG SERVICE WITH DEEP RESEARCH INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════

class EnhancedRAGService:
    """Ultimate RAG service with multi-tier API fallback and deep research capabilities"""
    
    def __init__(self, config: EnhancedConfig = None):
        self.config = config or EnhancedConfig()
        self.usage_tracker = EnhancedUsageTracker(self.config)
        self.cache_manager = EnhancedCacheManager(self.config)
        
        # Threading for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize core components
        # LAZY LOADING: Skip model loading at startup for instant start!
        self.embedding_model = None  # Will load when first needed
        self.reranker_model = None   # Will load when first needed
        self._initialize_storage()
        self._initialize_web_search()
        
        # Service state
        self.current_project: Optional[Project] = None
        self.active_chunks: List[EnhancedChunk] = []
        self.faiss_index: Optional[faiss.Index] = None
        
        logger.info("⚡ Enhanced RAG Service v7.0 initialized INSTANTLY with lazy loading!")
        logger.info("💡 Models will load automatically when first needed")
        # Don't print status here - it makes startup slow
        # self._print_comprehensive_status()
    
    def _ensure_embedding_model(self):
        """Lazy load embedding model when first needed"""
        if self.embedding_model is not None:
            return  # Already loaded
        
        logger.info("🔧 Loading embedding model (first use)...")
        
        try:
            logger.info(f"📥 Loading: {self.config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            logger.info("✅ Embedding model loaded successfully!")
        except Exception as e:
            logger.warning(f"Failed to load {self.config.EMBEDDING_MODEL}: {e}")
            logger.info("🔄 Falling back to all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _ensure_reranker_model(self):
        """Lazy load reranker model when first needed"""
        if self.reranker_model is not None:
            return  # Already loaded
        
        try:
            logger.info("🔧 Loading reranker model (first use)...")
            self.reranker_model = CrossEncoder(self.config.RERANKER_MODEL)
            logger.info("✅ Reranker model loaded successfully!")
        except Exception as e:
            logger.warning(f"⚠️  Reranker not available: {e}")
            self.reranker_model = None
    
    def _initialize_models(self):
        """DEPRECATED: Models now load lazily. Keeping for compatibility."""
        # This method is now empty - models load on first use
        pass
    
    def _initialize_storage(self):
        """Initialize storage system and load existing projects"""
        logger.info("💾 Initializing storage system...")
        
        os.makedirs(self.config.PROJECT_DATA_DIR, exist_ok=True)
        
        self.projects: Dict[str, Project] = {}
        project_file = os.path.join(self.config.PROJECT_DATA_DIR, self.config.PROJECT_METADATA_FILE)
        
        if os.path.exists(project_file):
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    projects_data = json.load(f)
                
                for name, data in projects_data.items():
                    if isinstance(data.get('created_at'), str):
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                    if isinstance(data.get('last_updated'), str):
                        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                    
                    self.projects[name] = Project(**data)
                
                logger.info(f"📁 Loaded {len(self.projects)} existing projects")
            except Exception as e:
                logger.error(f"Failed to load projects: {e}")
                self.projects = {}
    
    def _initialize_web_search(self):
        """Initialize web search configuration"""
        self.web_search_apis = {
            'semantic_scholar': 'https://api.semanticscholar.org/graph/v1/paper/search',
        }
        
        self.web_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9'
        }
    
    def _print_comprehensive_status(self):
        """Print detailed system status"""
        print("\n" + "="*80)
        print("🔍 ENHANCED RAG SYSTEM v5.0 WITH DEEP RESEARCH - STATUS CHECK")
        print("="*80)
        
        # Check Ollama
        try:
            response = requests.get(f"{self.config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                print("✅ Ollama (Local): AVAILABLE")
                print(f"   └── Models: {[m['name'] for m in models[:3]]}")
            else:
                print("❌ Ollama (Local): NOT RUNNING")
        except Exception:
            print("❌ Ollama (Local): NOT AVAILABLE")
        
        # Check Groq
        if self.config.GROQ_API_KEY and len(self.config.GROQ_API_KEY) > 10:
            print("✅ Groq API: KEY PROVIDED")
            print(f"   └── Model: {self.config.GROQ_MODEL}")
        else:
            print("⚠️  Groq API: NO KEY")
        
        # Check HuggingFace
        if self.config.HF_TOKEN and len(self.config.HF_TOKEN) > 10:
            print("✅ HuggingFace: TOKEN PROVIDED")
        else:
            print("⚠️  HuggingFace: NO TOKEN")
        
        # Usage statistics
        usage_stats = self.usage_tracker.get_usage_stats()
        print(f"\n📊 Usage Statistics:")
        print(f"   └── Today's tokens: {usage_stats['daily_tokens']:,}/{usage_stats['daily_limit']:,}")
        
        # Cache statistics
        cache_stats = self.cache_manager.get_cache_stats()
        print(f"\n💾 Cache Statistics:")
        print(f"   └── Hit rate: {cache_stats['hit_rate_percentage']:.1f}%")
        
        print("="*80)
    
    # Project Management Methods
    def create_project(self, name: str, domain: Optional[str] = None, description: Optional[str] = None) -> bool:
        """Create a new project with enhanced metadata"""
        if name in self.projects:
            logger.warning(f"Project '{name}' already exists")
            return False
        
        project = Project(
            name=name,
            domain=domain,
            description=description,
            embedding_model_used=self.config.EMBEDDING_MODEL
        )
        
        self.projects[name] = project
        self._save_projects()
        logger.info(f"✅ Created project: {name}")
        return True
    
    def load_project(self, name: str) -> bool:
        """Load project and rebuild index if necessary"""
        if name not in self.projects:
            logger.error(f"Project '{name}' not found")
            return False
        
        self.current_project = self.projects[name]
        logger.info(f"📂 Loading project: {name}")
        
        self.active_chunks = []
        for chunk_data in self.current_project.chunks:
            try:
                if isinstance(chunk_data.get('created_at'), str):
                    chunk_data['created_at'] = datetime.fromisoformat(chunk_data['created_at'])
                
                chunk = EnhancedChunk(**chunk_data)
                self.active_chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to load chunk: {e}")
        
        if self.active_chunks:
            self._build_faiss_index()
            logger.info(f"📊 Loaded {len(self.active_chunks)} chunks")
        else:
            self.faiss_index = None
            logger.info("📭 No chunks found in project")
        
        return True
    
    def list_projects(self) -> List[str]:
        """Get list of all project names"""
        return list(self.projects.keys())
    
    def get_project_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a project"""
        if name not in self.projects:
            return None
        
        project = self.projects[name]
        return {
            "name": project.name,
            "domain": project.domain,
            "description": project.description,
            "total_pdfs": len(project.pdf_paths),
            "total_chunks": len(project.chunks),
            "created_at": project.created_at.isoformat(),
            "last_updated": project.last_updated.isoformat(),
            "embedding_model": project.embedding_model_used
        }
    
    def _save_projects(self):
        """Save all projects to disk"""
        try:
            project_file = os.path.join(self.config.PROJECT_DATA_DIR, self.config.PROJECT_METADATA_FILE)
            projects_data = {}
            
            for name, project in self.projects.items():
                project_dict = project.model_dump()
                project_dict['created_at'] = project.created_at.isoformat()
                project_dict['last_updated'] = project.last_updated.isoformat()
                projects_data[name] = project_dict
            
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(projects_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Failed to save projects: {e}")
    
    # Document Processing Methods
    def add_pdf_to_project(self, pdf_path: str) -> bool:
        """Add PDF to current project with enhanced processing"""
        if not self.current_project:
            logger.error("No active project selected")
            return False
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        logger.info(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
        
        try:
            new_chunks = self._extract_enhanced_chunks(pdf_path)
            
            if not new_chunks:
                logger.error("No content could be extracted from PDF")
                return False
            
            self.current_project.pdf_paths.append(pdf_path)
            chunk_dicts = [chunk.model_dump() for chunk in new_chunks]
            self.current_project.chunks.extend(chunk_dicts)
            
            self.current_project.last_updated = datetime.now()
            self.current_project.total_chunks = len(self.current_project.chunks)
            
            self.active_chunks.extend(new_chunks)
            self._build_faiss_index()
            self._save_projects()
            
            logger.info(f"✅ Added {len(new_chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            return False
    
    def _extract_enhanced_chunks(self, pdf_path: str) -> List[EnhancedChunk]:
        """Extract chunks with advanced processing and metadata"""
        chunks = []
        
        try:
            doc = fitz.open(pdf_path)
            source_filename = os.path.basename(pdf_path)
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()
                
                if not text.strip():
                    continue
                
                text = self._clean_text(text)
                text_chunks = self._enhanced_chunking(text, self.config.CHUNK_SIZE, self.config.CHUNK_OVERLAP)
                
                for chunk_text in text_chunks:
                    if len(chunk_text.strip()) < 50:
                        continue
                    
                    chunk = EnhancedChunk(
                        content=chunk_text,
                        source=source_filename,
                        page_number=page_num + 1,
                        semantic_density=self._calculate_semantic_density(chunk_text),
                        readability_score=self._calculate_readability_score(chunk_text),
                        word_count=len(chunk_text.split()),
                        metadata={
                            "total_pages": total_pages,
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    chunks.append(chunk)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\s*Page \d+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        return text.strip()
    
    def _enhanced_chunking(self, text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
        """Advanced chunking with semantic boundaries"""
        if len(text) <= chunk_size:
            return [text]
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                if overlap > 0 and len(chunks) > 0:
                    prev_sentences = chunks[-1].split('. ')
                    overlap_sentences = prev_sentences[-min(3, len(prev_sentences)):]
                    overlap_text = '. '.join(overlap_sentences)
                    
                    if len(overlap_text) <= overlap:
                        current_chunk = overlap_text + ". " + sentence
                        current_size = len(current_chunk)
                    else:
                        current_chunk = sentence
                        current_size = sentence_size
                else:
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                current_chunk += (" " if current_chunk else "") + sentence
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density score"""
        words = text.split()
        if not words:
            return 0.0
        
        unique_words = len(set(word.lower() for word in words))
        total_words = len(words)
        uniqueness_ratio = unique_words / total_words
        
        avg_word_length = sum(len(word) for word in words) / total_words
        length_score = min(avg_word_length / 8.0, 1.0)
        
        density = (uniqueness_ratio * 0.6) + (length_score * 0.4)
        return min(density, 1.0)
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score"""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        avg_sentence_length = words / sentences
        optimal_length = 17.5
        length_penalty = abs(avg_sentence_length - optimal_length) / optimal_length
        readability = max(0.0, 1.0 - length_penalty)
        
        return readability
    
    def _build_faiss_index(self):
        """Build optimized FAISS HNSW index"""
        if not self.active_chunks:
            return
        
        logger.info("🔍 Building FAISS HNSW index...")
        
        texts = [chunk.content for chunk in self.active_chunks]
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            batch_size=32
        )
        
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, self.config.FAISS_HNSW_M)
        index.hnsw.efConstruction = self.config.FAISS_EF_CONSTRUCTION
        index.hnsw.efSearch = self.config.FAISS_EF_SEARCH
        index.add(embeddings.astype('float32'))
        
        self.faiss_index = index
        logger.info(f"✅ FAISS index built: {len(self.active_chunks)} chunks")
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> List[EnhancedChunk]:
        """Retrieve relevant chunks with advanced reranking"""
        if not self.faiss_index or not self.active_chunks:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        search_k = min(top_k * 3, len(self.active_chunks))
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)
        
        candidate_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(self.active_chunks):
                candidate_chunks.append(self.active_chunks[idx])
        
        if self.reranker_model and len(candidate_chunks) > top_k:
            try:
                pairs = [(query, chunk.content) for chunk in candidate_chunks]
                rerank_scores = self.reranker_model.predict(pairs)
                scored_chunks = list(zip(rerank_scores, candidate_chunks))
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                return [chunk for _, chunk in scored_chunks[:top_k]]
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
        
        return candidate_chunks[:top_k]
    
    # API Query Methods
    async def _query_with_fallback(self, messages: List[Dict[str, str]], max_tokens: int = 800) -> Tuple[str, str, int]:
        """Multi-tier API system with intelligent fallback"""
        
        # Tier 1: Ollama
        if self.usage_tracker.can_make_request("ollama", 0):
            try:
                response = await self._query_ollama(messages, max_tokens)
                if response and response.strip():
                    self.usage_tracker.record_successful_request("ollama", 0)
                    logger.info("✅ Used Ollama")
                    return response, "ollama", 0
            except Exception as e:
                logger.debug(f"Ollama failed: {e}")
        
        # Tier 2: Groq
        if (self.config.ENABLE_REMOTE_APIS and
            self.config.GROQ_API_KEY and 
            self.usage_tracker.can_make_request("groq", max_tokens)):
            try:
                response = await self._query_groq(messages, max_tokens)
                if response and response.strip():
                    self.usage_tracker.record_successful_request("groq", max_tokens)
                    logger.info("✅ Used Groq")
                    return response, "groq", max_tokens
            except Exception as e:
                logger.debug(f"Groq failed: {e}")
        
        # Tier 3: HuggingFace
        if (self.config.ENABLE_REMOTE_APIS and
            self.config.HF_TOKEN and 
            self.usage_tracker.can_make_request("hf", max_tokens)):
            try:
                response = await self._query_huggingface(messages, max_tokens)
                if response and response.strip():
                    self.usage_tracker.record_successful_request("hf", max_tokens)
                    logger.info("✅ Used HuggingFace")
                    return response, "huggingface", max_tokens
            except Exception as e:
                logger.debug(f"HuggingFace failed: {e}")
        
        # Tier 4: Local fallback
        logger.warning("⚠️ All APIs unavailable, using fallback")
        response = self._generate_template_response(messages)
        return response, "local_template", 0
    
    async def _query_ollama(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        """Query Ollama local API"""
        prompt = self._convert_messages_to_prompt(messages)
        
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
                timeout=120
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result.get('response', '').strip()
                else:
                    raise Exception(f"Ollama error: {response.status}")
    
    async def _query_groq(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        """Query Groq API"""
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
                timeout=30
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    error_text = await response.text()
                    raise Exception(f"Groq error: {response.status}")
    
    async def _query_huggingface(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        """Query HuggingFace API with model fallbacks"""
        prompt = self._convert_messages_to_prompt(messages)
        
        for model_name in self.config.HF_FALLBACK_MODELS:
            try:
                headers = {"Authorization": f"Bearer {self.config.HF_TOKEN}"}
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": min(max_tokens, 500),
                        "temperature": 0.1,
                        "return_full_text": False
                    }
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.config.HF_API_BASE}/{model_name}",
                        json=payload,
                        headers=headers,
                        timeout=30
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            if isinstance(result, list) and result:
                                text = result[0].get('generated_text', '').strip()
                                if text and len(text) > 20:
                                    return text
                            elif isinstance(result, dict):
                                text = result.get('generated_text', '').strip()
                                if text and len(text) > 20:
                                    return text
                        
            except Exception as e:
                logger.debug(f"HF model {model_name} error: {e}")
                continue
        
        raise Exception("All HuggingFace models failed")
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt format"""
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts)
        prompt += "\n\nAssistant: "
        return prompt
    
    def _generate_template_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate fallback response when APIs fail"""
        user_message = ""
        
        for message in messages:
            if message.get('role') == 'user':
                user_message = message.get('content', '')
                break
        
        query_lower = user_message.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            return "Based on the provided context, this appears to be a definitional question. The relevant information from your documents contains key concepts that address your query, though a complete AI-generated synthesis is not available due to API limitations."
        elif any(word in query_lower for word in ['how', 'method', 'process']):
            return "This question involves processes or methods. Your documents contain relevant procedural information and steps related to your inquiry."
        else:
            return "I found relevant information in your documents that addresses your question. While a full AI-generated response is not available due to current limitations, the source materials contain pertinent details for your research."
    
    # Deep Research Integration
    async def conduct_deep_research(self, query: str) -> DeepResearchResult:
        """Conduct comprehensive deep research analysis"""
        if not hasattr(self, 'deep_research_engine'):
            self.deep_research_engine = DeepResearchEngine(self)
        
        return await self.deep_research_engine.conduct_deep_research(query)
    
    # Regular Question Answering (keeping existing functionality)
    async def ask_question(self, query: str, use_web_search: bool = False) -> EnhancedAnswer:
        """Main question answering method"""
        start_time = time.time()
        
        if not self.current_project or not self.faiss_index:
            return EnhancedAnswer(
                answer="❌ No active project or knowledge base available.",
                sources=[],
                confidence=0.0,
                reasoning="No knowledge base available",
                api_used="none",
                tokens_used=0,
                processing_time=0.0
            )
        
        # Check cache first
        context_hash = str(hash(str([c.chunk_id for c in self.active_chunks[:10]])))
        cache_key = self.cache_manager._get_cache_key(query, context_hash, use_web_search)
        
        cached_result = self.cache_manager.get_cached_response(cache_key)
        if cached_result:
            cached_result["cache_used"] = True
            cached_result["processing_time"] = time.time() - start_time
            return EnhancedAnswer(**cached_result)
        
        # Retrieve relevant chunks
        local_chunks = self._retrieve_relevant_chunks(query, top_k=self.config.MAX_CHUNKS_FOR_CONTEXT)
        
        # Web search if enabled
        web_results = []
        if use_web_search:
            try:
                web_results = await self._search_academic_papers(query)
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
        # Prepare context
        context_parts = []
        for i, chunk in enumerate(local_chunks, 1):
            content = chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content
            context_parts.append(f"[{i}] {content}")
        
        for i, result in enumerate(web_results, 1):
            context_parts.append(f"[WEB{i}] {result.title}: {result.snippet}")
        
        context = "\n\n".join(context_parts)
        
        # Create messages
        domain = self.current_project.domain if self.current_project else "general"
        
        system_message = f"""You are an expert research assistant specializing in {domain} analysis.

Provide a comprehensive, well-structured answer based on the provided context.

Guidelines:
- Synthesize information from multiple sources
- Provide detailed analysis and insights
- Maintain academic rigor
- Use clear, professional language"""
        
        user_message = f"""Question: {query}

Context:
{context}

Provide a comprehensive answer:"""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Generate answer
        answer_text, api_used, tokens_used = await self._query_with_fallback(messages, max_tokens=600)
        
        # Calculate metrics
        confidence = self._calculate_confidence_score(query, local_chunks, web_results, answer_text)
        
        # Prepare sources
        sources = []
        for chunk in local_chunks:
            sources.append({
                "type": "local",
                "source": chunk.source,
                "page": chunk.page_number,
                "chunk_id": chunk.chunk_id
            })
        
        processing_time = time.time() - start_time
        
        enhanced_answer = EnhancedAnswer(
            answer=answer_text.strip(),
            sources=sources,
            confidence=confidence,
            reasoning=f"Answer generated using {len(local_chunks)} local chunks via {api_used}",
            api_used=api_used,
            tokens_used=tokens_used,
            processing_time=processing_time,
            cache_used=False
        )
        
        # Cache the result
        self.cache_manager.cache_response(cache_key, enhanced_answer.model_dump())
        
        return enhanced_answer
    
    async def _search_academic_papers(self, query: str) -> List[WebSearchResult]:
        """Search academic papers using Semantic Scholar"""
        results = []
        if not self.config.ENABLE_REMOTE_APIS:
            logger.debug("Remote APIs disabled; skipping academic web search")
            return results

        try:
            params = {
                "query": query,
                "limit": self.config.WEB_SEARCH_MAX_RESULTS,
                "fields": "title,abstract,url"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.web_search_apis['semantic_scholar'],
                    params=params,
                    headers=self.web_headers,
                    timeout=15
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        papers = data.get("data", [])
                        
                        for paper in papers:
                            title = paper.get("title", "").strip()
                            abstract = paper.get("abstract", "").strip()
                            url = paper.get("url", "").strip()
                            
                            if title:
                                snippet = abstract[:300] + "..." if len(abstract) > 300 else abstract
                                
                                result = WebSearchResult(
                                    title=title,
                                    url=url,
                                    snippet=snippet,
                                    source_type="academic"
                                )
                                results.append(result)
        
        except Exception as e:
            logger.warning(f"Academic search error: {e}")
        
        return results
    
    def _calculate_confidence_score(self, query: str, local_chunks: List[EnhancedChunk], 
                                   web_results: List[WebSearchResult], answer: str) -> float:
        """Calculate confidence score"""
        base_confidence = 0.5
        
        # Source factor
        local_count = len(local_chunks)
        web_count = len(web_results)
        source_factor = min(1.0, (local_count * 0.15) + (web_count * 0.1))
        
        # Length factor
        length_factor = min(1.0, len(answer) / 400)
        
        # Keyword overlap
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words.intersection(answer_words))
        overlap_factor = min(1.0, overlap / max(1, len(query_words)))
        
        confidence = base_confidence + (source_factor * 0.3) + (length_factor * 0.1) + (overlap_factor * 0.1)
        return min(1.0, max(0.0, confidence))

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                                   ENHANCED CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════════════════

async def run_enhanced_cli():
    """Enhanced CLI interface with deep research capabilities"""
    
    print("\n" + "="*90)
    print("🚀 ULTIMATE ENHANCED RAG SYSTEM v5.0 WITH DEEP RESEARCH")
    print("Multi-Tier APIs | Advanced Analytics | Web Search | Deep Research Mode")
    print("="*90)
    
    # Initialize the RAG service
    try:
        config = EnhancedConfig()
        rag_service = EnhancedRAGService(config)
        print("\n✅ Enhanced RAG Service initialized successfully!")
    except Exception as e:
        print(f"\n❌ Critical initialization error: {e}")
        return
    
    # Main application loop
    while True:
        try:
            # Project Management Phase
            print("\n" + "="*70)
            print("📁 PROJECT MANAGEMENT")
            print("="*70)
            
            projects = rag_service.list_projects()
            
            if projects:
                print(f"\n📂 Available Projects:")
                for i, project_name in enumerate(projects, 1):
                    project_info = rag_service.get_project_info(project_name)
                    if project_info:
                        domain_text = f" ({project_info['domain']})" if project_info['domain'] else ""
                        pdf_count = project_info['total_pdfs']
                        chunk_count = project_info['total_chunks']
                        print(f"   {i}. {project_name}{domain_text}")
                        print(f"      └── {pdf_count} PDFs, {chunk_count} chunks")
                
                print(f"\n   {len(projects) + 1}. ➕ Create new project")
                print(f"   {len(projects) + 2}. 🔍 Show system status")
                print(f"   {len(projects) + 3}. ❌ Exit system")
                
                choice = input(f"\n📋 Select option (1-{len(projects) + 3}): ").strip()
                
                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(projects):
                        project_name = projects[choice_num - 1]
                        if rag_service.load_project(project_name):
                            print(f"✅ Loaded project: {project_name}")
                            break
                        else:
                            print(f"❌ Failed to load project: {project_name}")
                            continue
                    
                    elif choice_num == len(projects) + 1:
                        project_name = input("\n📝 Enter new project name: ").strip()
                        if not project_name:
                            print("❌ Project name cannot be empty")
                            continue
                        
                        if project_name in projects:
                            print(f"❌ Project '{project_name}' already exists")
                            continue
                        
                        print("\n🏷️  Available domains:")
                        domains = ["academic", "legal", "medical", "technical", "financial", "other"]
                        for i, domain in enumerate(domains, 1):
                            print(f"   {i}. {domain.capitalize()}")
                        
                        domain_choice = input(f"\nSelect domain (1-{len(domains)}): ").strip()
                        
                        domain = None
                        if domain_choice.isdigit():
                            domain_idx = int(domain_choice) - 1
                            if 0 <= domain_idx < len(domains) and domains[domain_idx] != "other":
                                domain = domains[domain_idx]
                        
                        if rag_service.create_project(project_name, domain):
                            if rag_service.load_project(project_name):
                                print(f"✅ Created and loaded project: {project_name}")
                                break
                        else:
                            print(f"❌ Failed to create project")
                            continue
                    
                    elif choice_num == len(projects) + 2:
                        rag_service._print_comprehensive_status()
                        input("\nPress Enter to continue...")
                        continue
                    
                    elif choice_num == len(projects) + 3:
                        print("\n👋 Thank you for using Enhanced RAG System!")
                        return
                    
                    else:
                        print(f"❌ Invalid choice: {choice}")
                        continue
                        
                except ValueError:
                    print(f"❌ Invalid input: {choice}")
                    continue
            
            else:
                # No projects exist
                project_name = input("\n📝 Enter new project name: ").strip()
                if not project_name:
                    print("❌ Project name cannot be empty")
                    continue
                
                domain = input("🏷️  Enter domain (academic/legal/medical/technical/other): ").strip().lower()
                domain = domain if domain != "other" else None
                
                if rag_service.create_project(project_name, domain):
                    if rag_service.load_project(project_name):
                        print(f"✅ Created and loaded project: {project_name}")
                        break
                else:
                    print(f"❌ Failed to create project")
                    continue
            
            # PDF Management Phase
            print("\n" + "="*70)
            print("📚 DOCUMENT MANAGEMENT")
            print("="*70)
            
            current_pdfs = len(rag_service.current_project.pdf_paths)
            print(f"📄 Current PDFs in project: {current_pdfs}")
            
            if current_pdfs < rag_service.config.PDF_MINIMUM_REQUIRED:
                print(f"⚠️  Minimum {rag_service.config.PDF_MINIMUM_REQUIRED} PDFs recommended")
            
            # PDF addition loop
            while current_pdfs < rag_service.config.PDF_MINIMUM_REQUIRED:
                pdf_path = input(f"\n📄 Enter PDF path ({current_pdfs + 1}/{rag_service.config.PDF_MINIMUM_REQUIRED}) or 'done': ").strip()
                
                if pdf_path.lower() == 'done':
                    if current_pdfs == 0:
                        print("❌ At least one PDF is required")
                        continue
                    else:
                        break
                
                pdf_path = pdf_path.strip('"\'')
                
                if rag_service.add_pdf_to_project(pdf_path):
                    current_pdfs += 1
                    print(f"✅ Added PDF successfully! ({current_pdfs} total)")
                else:
                    print("❌ Failed to add PDF")
            
            # Allow adding more PDFs
            if current_pdfs >= rag_service.config.PDF_MINIMUM_REQUIRED:
                while True:
                    add_more = input(f"\n➕ Add more PDFs? (y/n): ").strip().lower()
                    if add_more == 'y':
                        pdf_path = input("📄 Enter PDF path: ").strip().strip('"\'')
                        if rag_service.add_pdf_to_project(pdf_path):
                            current_pdfs += 1
                            print(f"✅ Added PDF successfully! ({current_pdfs} total)")
                        else:
                            print("❌ Failed to add PDF")
                    elif add_more == 'n':
                        break
                    else:
                        print("Please enter 'y' or 'n'")
            
            if not rag_service.active_chunks:
                print("\n❌ No content could be extracted from PDFs")
                continue
            
            # Q&A Phase
            print("\n" + "="*70)
            print("🎯 QUESTION & ANSWER SYSTEM")
            print("="*70)
            
            print(f"✅ Knowledge base ready with {len(rag_service.active_chunks)} chunks from {current_pdfs} PDFs")
            print("\n💡 Available commands:")
            print("   • Ask any question about your documents")
            print("   • 'web: your question' - Enable web search")
            print("   • 'deep research: your research question' - Comprehensive research analysis")
            print("   • 'switch' - Change to different project")
            print("   • 'status' - Show system status")
            print("   • 'cache' - Show cache statistics")  
            print("   • 'project' - Show current project information")
            print("   • 'exit' - Exit the system")
            print()
            
            # Q&A loop
            while True:
                question = input("❓ Your question: ").strip()
                
                if not question:
                    continue
                
                # Command handling
                if question.lower() == 'exit':
                    print("\n👋 Thank you for using Enhanced RAG System!")
                    return
                
                elif question.lower() == 'switch':
                    print("\n🔄 Switching projects...")
                    break
                
                elif question.lower() == 'status':
                    rag_service._print_comprehensive_status()
                    continue
                
                elif question.lower() == 'cache':
                    cache_stats = rag_service.cache_manager.get_cache_stats()
                    print(f"\n📊 Cache Statistics:")
                    print(f"   Hit Rate: {cache_stats['hit_rate_percentage']:.1f}%")
                    print(f"   Cache Hits: {cache_stats['cache_hits']}")
                    print(f"   Memory Cache: {cache_stats['memory_cache_entries']}")
                    continue
                
                elif question.lower() == 'project':
                    if rag_service.current_project:
                        info = rag_service.get_project_info(rag_service.current_project.name)
                        if info:
                            print(f"\n📂 Current Project Information:")
                            print(f"   Name: {info['name']}")
                            print(f"   Domain: {info['domain'] or 'General'}")
                            print(f"   PDFs: {info['total_pdfs']}")
                            print(f"   Chunks: {info['total_chunks']}")
                    continue
                
                # Deep Research Mode
                elif question.lower().startswith('deep research:'):
                    research_query = question[14:].strip()
                    if not research_query:
                        print("❌ Please provide a research question after 'deep research:'")
                        continue
                    
                    try:
                        print("\n🔬 Conducting deep research analysis...")
                        print("This may take 2-5 minutes for comprehensive analysis...")
                        
                        research_result = await rag_service.conduct_deep_research(research_query)
                        
                        # Display comprehensive research results
                        print("\n" + "="*100)
                        print("🔬 DEEP RESEARCH ANALYSIS RESULTS")
                        print("="*100)
                        
                        print("\n📋 COMPREHENSIVE ANALYSIS:")
                        print("-" * 80)
                        print(research_result.comprehensive_analysis)
                        
                        if research_result.key_concepts:
                            print(f"\n🎯 KEY CONCEPTS ({len(research_result.key_concepts)}):")
                            print("-" * 50)
                            for i, concept in enumerate(research_result.key_concepts[:10], 1):
                                print(f"{i}. {concept}")
                        
                        if research_result.research_gaps:
                            print(f"\n🔍 RESEARCH GAPS ({len(research_result.research_gaps)}):")
                            print("-" * 50)
                            for i, gap in enumerate(research_result.research_gaps[:8], 1):
                                print(f"{i}. {gap}")
                        
                        if research_result.methodological_insights:
                            print(f"\n🛠️  METHODOLOGICAL INSIGHTS ({len(research_result.methodological_insights)}):")
                            print("-" * 50)
                            for i, insight in enumerate(research_result.methodological_insights[:6], 1):
                                print(f"{i}. {insight}")
                        
                        if research_result.future_directions:
                            print(f"\n🚀 FUTURE DIRECTIONS ({len(research_result.future_directions)}):")
                            print("-" * 50)
                            for i, direction in enumerate(research_result.future_directions[:6], 1):
                                print(f"{i}. {direction}")
                        
                        if research_result.statistical_insights:
                            print(f"\n📊 STATISTICAL INSIGHTS ({len(research_result.statistical_insights)}):")
                            print("-" * 50)
                            for i, stat in enumerate(research_result.statistical_insights[:5], 1):
                                print(f"{i}. {stat}")
                        
                        print(f"\n📚 RELATED RESEARCH PAPERS ({len(research_result.related_papers)}):")
                        print("-" * 50)
                        for i, paper in enumerate(research_result.related_papers[:10], 1):
                            authors_str = ", ".join(paper.authors[:2]) + ("..." if len(paper.authors) > 2 else "")
                            print(f"{i}. {paper.title}")
                            print(f"   Authors: {authors_str}")
                            print(f"   Source: {paper.source_api} | Citations: {paper.citation_count}")
                            if paper.url:
                                print(f"   🔗 Link: {paper.url}")
                            elif paper.doi:
                                print(f"   🔗 DOI: https://doi.org/{paper.doi}")
                            print()
                        
                        if research_result.cross_references:
                            print(f"\n🔗 CROSS-REFERENCES ({len(research_result.cross_references)}):")
                            print("-" * 50)
                            for i, ref in enumerate(research_result.cross_references[:5], 1):
                                print(f"{i}. {ref.get('paper_title', 'Unknown')}")
                                print(f"   Connection: {ref.get('connection_type', 'Unknown')} ({ref.get('relevance', 'Unknown')} relevance)")
                                if ref.get('paper_url'):
                                    print(f"   🔗 Link: {ref['paper_url']}")
                        
                        print(f"\n📊 RESEARCH METRICS:")
                        print("-" * 50)
                        print(f"🎯 Confidence Score: {research_result.confidence_score:.3f}")
                        print(f"🏆 Research Quality: {research_result.research_quality_score:.3f}")
                        print(f"⏱️  Processing Time: {research_result.processing_time:.2f}s")
                        print(f"📚 Sources Analyzed: {research_result.sources_analyzed}")
                        print(f"📄 Local Documents: {len(rag_service.active_chunks)}")
                        print(f"🌐 Research Papers: {len(research_result.related_papers)}")
                        
                        print("="*100)
                        
                    except Exception as e:
                        print(f"\n❌ Deep research analysis failed: {e}")
                        logger.error(f"Deep research error: {e}")
                        print("💡 Please try with a more specific research question")
                    
                    continue
                
                # Parse web search prefix
                use_web_search = question.lower().startswith('web:')
                if use_web_search:
                    question = question[4:].strip()
                    if not question:
                        print("❌ Please provide a question after 'web:'")
                        continue
                
                # Regular question processing
                try:
                    print("\n🔍 Processing your question...")
                    
                    result = await rag_service.ask_question(question, use_web_search=use_web_search)
                    
                    # Display results
                    print("\n" + "="*80)
                    print("📋 ANSWER")
                    print("="*80)
                    print(result.answer)
                    
                    print("\n" + "-"*60)
                    print("📊 METRICS")
                    print("-"*60)
                    print(f"🎯 Confidence: {result.confidence:.3f}")
                    print(f"⚡ API Used: {result.api_used}")
                    print(f"🔢 Tokens: {result.tokens_used:,}")
                    print(f"⏱️  Time: {result.processing_time:.2f}s")
                    print(f"📚 Sources: {len(result.sources)}")
                    print(f"💾 Cache: {'Yes' if result.cache_used else 'No'}")
                    
                    if result.sources:
                        print("\n" + "-"*60)
                        print("📚 SOURCES")
                        print("-"*60)
                        for i, source in enumerate(result.sources[:5], 1):
                            if source.get("type") == "local":
                                print(f"{i}. 📄 {source['source']} (page {source['page']})")
                            else:
                                title = source.get('title', 'Web Source')[:60]
                                print(f"{i}. 🌐 {title}...")
                    
                    print("="*80)
                    
                except Exception as e:
                    print(f"\n❌ Error processing question: {e}")
                    print("💡 Please try rephrasing your question")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            confirm_exit = input("Exit Enhanced RAG System? (y/n): ").strip().lower()
            if confirm_exit == 'y':
                print("👋 Goodbye!")
                return
            else:
                continue
                
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            logger.error(f"CLI error: {e}")
            print("💡 The system will restart. Please try again.")
            continue

# ═══════════════════════════════════════════════════════════════════════════════════════════
#                                    MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        # Set environment variables for better compatibility
        os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        
        # Run the enhanced CLI
        asyncio.run(run_enhanced_cli())
        
    except KeyboardInterrupt:
        print("\n\n👋 Enhanced RAG System interrupted by user. Goodbye!")
        
    except Exception as e:
        print(f"\n❌ Critical system error: {e}")
        logger.error(f"Critical system error: {e}", exc_info=True)
        
    finally:
        logger.info("Enhanced RAG System shutdown complete")

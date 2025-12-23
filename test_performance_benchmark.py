"""
Performance Benchmark Tests for RAG System Optimization
=======================================================

Compares v7_complete.py vs v8_optimized.py performance metrics:
- Startup time
- PDF processing speed
- Query response time
- Memory usage
"""

import time
import os
import sys
import psutil
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def get_memory_mb():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def test_startup_time():
    """Test system initialization time"""
    print("\n" + "="*70)
    print("TEST 1: STARTUP TIME")
    print("="*70)
    
    # Test v8_optimized
    print("\n⚡ Testing v8_optimized.py...")
    start = time.time()
    from v8_optimized import OptimizedRAGService, OptimizedConfig
    rag_v8 = OptimizedRAGService(OptimizedConfig())
    v8_time = time.time() - start
    v8_memory = get_memory_mb()
    print(f"✅ v8_optimized startup: {v8_time:.3f}s, Memory: {v8_memory:.1f}MB")
    
    # Clean up
    del rag_v8
    import gc
    gc.collect()
    
    # Test v7_complete (if available)
    try:
        print("\n🐌 Testing v7_complete.py...")
        start = time.time()
        from v7_complete import EnhancedRAGService, EnhancedConfig
        rag_v7 = EnhancedRAGService(EnhancedConfig())
        v7_time = time.time() - start
        v7_memory = get_memory_mb()
        print(f"✅ v7_complete startup: {v7_time:.3f}s, Memory: {v7_memory:.1f}MB")
        
        # Comparison
        speedup = v7_time / v8_time if v8_time > 0 else 0
        memory_reduction = ((v7_memory - v8_memory) / v7_memory * 100) if v7_memory > 0 else 0
        
        print(f"\n📊 RESULTS:")
        print(f"   ⚡ Speedup: {speedup:.1f}x faster")
        print(f"   💾 Memory: {memory_reduction:.1f}% reduction")
        
        # Check if targets met
        if v8_time < 2.0:
            print(f"   ✅ Startup target met (< 2s): {v8_time:.3f}s")
        else:
            print(f"   ❌ Startup target missed (< 2s): {v8_time:.3f}s")
        
        del rag_v7
        gc.collect()
        
    except ImportError:
        print("⚠️  v7_complete.py not available for comparison")
        
        # Check v8 target
        if v8_time < 2.0:
            print(f"\n✅ v8 Startup target met (< 2s): {v8_time:.3f}s")
        else:
            print(f"\n❌ v8 Startup target missed (< 2s): {v8_time:.3f}s")

async def test_pdf_processing_speed(pdf_path: str = None):
    """Test PDF processing speed"""
    print("\n" + "="*70)
    print("TEST 2: PDF PROCESSING SPEED")
    print("="*70)
    
    if not pdf_path or not os.path.exists(pdf_path):
        print("❌ No test PDF provided. Skipping test.")
        print("   Usage: Provide a PDF path as argument")
        return
    
    # Get PDF info
    import fitz
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()
    
    print(f"\n📄 Test PDF: {os.path.basename(pdf_path)}")
    print(f"   Pages: {page_count}")
    
    # Test v8
    print("\n⚡ Testing v8_optimized.py...")
    from v8_optimized import OptimizedRAGService, OptimizedConfig
    
    rag_v8 = OptimizedRAGService(OptimizedConfig())
    rag_v8.create_project("benchmark_v8", "test")
    rag_v8.load_project("benchmark_v8")
    
    start = time.time()
    success = rag_v8.add_pdf_to_project(pdf_path)
    v8_time = time.time() - start
    
    if success:
        pages_per_sec = page_count / v8_time if v8_time > 0 else 0
        print(f"✅ v8_optimized: {v8_time:.2f}s ({pages_per_sec:.2f} pages/sec)")
        
        if pages_per_sec > 5:
            print(f"   ✅ Target met (> 5 pages/sec)")
        else:
            print(f"   ❌ Target missed (> 5 pages/sec)")
    else:
        print(f"❌ v8_optimized failed to process PDF")
    
    # Cleanup
    import shutil
    if os.path.exists(".rag_projects"):
        shutil.rmtree(".rag_projects")

async def test_query_speed():
    """Test query response time"""
    print("\n" + "="*70)
    print("TEST 3: QUERY RESPONSE TIME")
    print("="*70)
    
    # This test requires a pre-loaded project with data
    print("⚠️  This test requires an existing project with PDFs")
    print("   Run the system manually and add PDFs first")

def test_memory_usage():
    """Test memory footprint"""
    print("\n" + "="*70)
    print("TEST 4: MEMORY USAGE")
    print("="*70)
    
    print(f"\n📊 Current memory usage: {get_memory_mb():.1f}MB")
    
    from v8_optimized import OptimizedRAGService, OptimizedConfig
    rag = OptimizedRAGService(OptimizedConfig())
    
    after_init = get_memory_mb()
    print(f"📊 After initialization: {after_init:.1f}MB")
    
    if after_init < 500:
        print(f"✅ Memory target met (< 500MB)")
    else:
        print(f"❌ Memory target exceeded (< 500MB): {after_init:.1f}MB")

def print_summary():
    """Print test summary"""
    print("\n" + "="*70)
    print("🎯 BENCHMARK SUMMARY")
    print("="*70)
    print("""
Performance Targets:
✓ Startup time: < 2 seconds
✓ PDF processing: > 5 pages/second
✓ Query response: < 3 seconds
✓ Memory usage: < 500MB

Run specific tests:
  python test_performance_benchmark.py startup
  python test_performance_benchmark.py pdf <path_to_pdf>
  python test_performance_benchmark.py memory
  python test_performance_benchmark.py all <path_to_pdf>
""")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print_summary()
        sys.exit(0)
    
    test_type = sys.argv[1].lower()
    
    if test_type == "startup":
        test_startup_time()
    
    elif test_type == "pdf":
        if len(sys.argv) < 3:
            print("❌ Please provide PDF path: python test_performance_benchmark.py pdf <path>")
        else:
            asyncio.run(test_pdf_processing_speed(sys.argv[2]))
    
    elif test_type == "query":
        asyncio.run(test_query_speed())
    
    elif test_type == "memory":
        test_memory_usage()
    
    elif test_type == "all":
        pdf_path = sys.argv[2] if len(sys.argv) > 2 else None
        test_startup_time()
        if pdf_path:
            asyncio.run(test_pdf_processing_speed(pdf_path))
        test_memory_usage()
    
    else:
        print(f"❌ Unknown test: {test_type}")
        print_summary()

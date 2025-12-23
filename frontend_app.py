import asyncio
import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

from v6 import DeepResearchResult, EnhancedAnswer, EnhancedRAGService, EnhancedConfig


if os.name == "nt":  # Ensure compatibility with Windows event loop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@st.cache_resource(show_spinner=True)
def get_rag_service() -> EnhancedRAGService:
    """Instantiate the Enhanced RAG service once per Streamlit session."""
    config = EnhancedConfig()
    return EnhancedRAGService(config)


def run_async(coro):
    """Helper to execute async coroutines from Streamlit callbacks."""
    return asyncio.run(coro)


def ensure_project_loaded(service: EnhancedRAGService, project_name: str) -> bool:
    if not project_name:
        return False
    if service.current_project and service.current_project.name == project_name:
        return True
    return service.load_project(project_name)


def persist_uploaded_pdf(uploaded_file, project_name: str, base_dir: Path) -> Path:
    project_dir = base_dir / project_name / "pdfs"
    project_dir.mkdir(parents=True, exist_ok=True)

    clean_name = Path(uploaded_file.name).name.replace(" ", "_")
    dest_path = project_dir / clean_name

    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return dest_path


st.set_page_config(page_title="Enhanced Research RAG", layout="wide")
st.title("📚 Enhanced Research Retrieval-Augmented Generation")

service = get_rag_service()
projects: List[str] = service.list_projects()

if "selected_project" not in st.session_state:
    st.session_state.selected_project = projects[0] if projects else ""

with st.sidebar:
    st.header("🔧 Project Manager")
    with st.expander("Create New Project", expanded=not projects):
        new_project = st.text_input("Project Name")
        domain = st.selectbox(
            "Domain",
            options=["general", "academic", "legal", "medical", "technical", "financial"],
            index=0,
            help="Used to tailor the system prompt"
        )
        description = st.text_area("Description", help="Optional project description")
        if st.button("Create Project", use_container_width=True):
            if not new_project.strip():
                st.warning("Please provide a project name before creating one.")
            elif new_project in projects:
                st.warning("A project with that name already exists. Choose another name.")
            else:
                created = service.create_project(new_project.strip(), domain=domain if domain != "general" else None,
                                                 description=description or None)
                if created:
                    st.success(f"Project '{new_project}' created!", icon="✅")
                    projects = service.list_projects()
                    st.session_state.selected_project = new_project
                else:
                    st.error("Unable to create project. Check logs for details.")

    if projects:
        selected = st.selectbox("Select Project", options=projects, index=projects.index(st.session_state.selected_project) if st.session_state.selected_project in projects else 0)
        st.session_state.selected_project = selected
        if st.button("Load Project", use_container_width=True):
            if ensure_project_loaded(service, selected):
                st.success(f"Loaded project '{selected}'", icon="📂")
            else:
                st.error("Failed to load project. See logs for details.")
    else:
        st.info("Create a project to get started.")

selected_project = st.session_state.selected_project
project_loaded = ensure_project_loaded(service, selected_project) if selected_project else False

if project_loaded and service.current_project:
    info = service.get_project_info(service.current_project.name)
    st.subheader(f"📁 Active Project: {info['name']}")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Domain", info.get("domain") or "General")
    col_b.metric("PDFs", info.get("total_pdfs", 0))
    col_c.metric("Chunks", info.get("total_chunks", 0))
    col_d.metric("Embedding Model", info.get("embedding_model", "-"))

    st.markdown("---")

    st.subheader("📄 Document Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Documents are stored locally per project."
    )

    if uploaded_files and st.button("Ingest Documents", use_container_width=True):
        with st.spinner("Processing PDFs..."):
            successes = 0
            failures = 0
            base_dir = Path(service.config.PROJECT_DATA_DIR)
            for uploaded in uploaded_files:
                try:
                    pdf_path = persist_uploaded_pdf(uploaded, service.current_project.name, base_dir)
                    added = service.add_pdf_to_project(str(pdf_path))
                    if added:
                        successes += 1
                    else:
                        failures += 1
                except Exception as exc:
                    failures += 1
                    st.error(f"Failed to ingest {uploaded.name}: {exc}")
            st.success(f"Added {successes} document(s)")
            if failures:
                st.warning(f"{failures} document(s) failed to ingest. Check logs for details.")

    st.markdown("---")

    st.subheader("🔍 Ask Research Questions")
    query = st.text_area("Enter your research question", height=120)
    col1, col2, col3 = st.columns([1, 1, 2])
    use_web = col1.checkbox("Use web search", value=False, help="Leverages Semantic Scholar when remote APIs are enabled")
    with col2:
        run_button = st.button("Generate Answer", type="primary")

    if run_button:
        if not query.strip():
            st.warning("Please provide a research question before running.")
        else:
            with st.spinner("Synthesizing answer..."):
                try:
                    answer: EnhancedAnswer = run_async(service.ask_question(query.strip(), use_web_search=use_web))
                    st.write(answer.answer)

                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    metrics_col1.metric("Confidence", f"{answer.confidence:.2f}")
                    metrics_col2.metric("Tokens", f"{answer.tokens_used}")
                    metrics_col3.metric("API", answer.api_used)
                    metrics_col4.metric("Latency", f"{answer.processing_time:.2f}s")

                    if answer.key_findings:
                        st.markdown("### 🔑 Key Findings")
                        for finding in answer.key_findings:
                            st.markdown(f"- {finding}")

                    if answer.related_concepts:
                        st.markdown("### 🔗 Related Concepts")
                        st.write(", ".join(answer.related_concepts))

                    if answer.sources:
                        st.markdown("### 📚 Sources")
                        for source in answer.sources:
                            if source.get("type") == "local":
                                st.markdown(f"- {source['source']} (page {source.get('page', '-')})")
                            else:
                                st.markdown(f"- [WEB] {source.get('title', 'Unknown Source')} - {source.get('url', '')}")

                    st.info(answer.reasoning)
                except Exception as exc:
                    st.error(f"Unable to generate answer: {exc}")

    st.markdown("---")
    st.subheader("🧠 Deep Research Mode")
    deep_query = st.text_area("Comprehensive research query", height=120, key="deep_query")
    deep_cols = st.columns([1, 1, 2])
    run_deep = deep_cols[0].button("Run Deep Research")

    if run_deep:
        if not service.config.ENABLE_REMOTE_APIS:
            st.warning("Enable remote APIs (Groq/HF/Semantic Scholar) via environment variables to use deep research mode.")
        elif not deep_query.strip():
            st.warning("Provide a research topic for deep analysis.")
        else:
            with st.spinner("Conducting deep research. This may take a few minutes..."):
                try:
                    research_result: DeepResearchResult = run_async(service.conduct_deep_research(deep_query.strip()))
                    st.markdown("### 📋 Comprehensive Analysis")
                    st.write(research_result.comprehensive_analysis)

                    if research_result.key_concepts:
                        st.markdown("### 🎯 Key Concepts")
                        for concept in research_result.key_concepts:
                            st.markdown(f"- {concept}")

                    if research_result.research_gaps:
                        st.markdown("### 🔍 Research Gaps")
                        for gap in research_result.research_gaps:
                            st.markdown(f"- {gap}")

                    if research_result.future_directions:
                        st.markdown("### 🚀 Future Directions")
                        for direction in research_result.future_directions:
                            st.markdown(f"- {direction}")

                    st.markdown("### 📚 Related Papers")
                    for paper in research_result.related_papers[:10]:
                        st.markdown(f"**{paper.title}**  ")
                        st.caption(", ".join(paper.authors[:3]))
                        if paper.url:
                            st.markdown(f"[View Paper]({paper.url})")

                    st.markdown("### 📊 Research Metrics")
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    metrics_col1.metric("Confidence", f"{research_result.confidence_score:.2f}")
                    metrics_col2.metric("Quality", f"{research_result.research_quality_score:.2f}")
                    metrics_col3.metric("Sources", research_result.sources_analyzed)
                    metrics_col4.metric("Latency", f"{research_result.processing_time:.2f}s")
                except Exception as exc:
                    st.error(f"Deep research failed: {exc}")
else:
    st.info("Select or create a project to begin building your research workspace.")


st.sidebar.markdown("---")
st.sidebar.caption("Enhanced RAG UI • Local-first research assistant")

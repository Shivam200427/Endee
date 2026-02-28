import streamlit as st
from utils import extract_text_from_uploaded, chunk_text, save_uploaded_pdf
from embed import store_chunks_in_endee
from search import semantic_search
from rag import generate_answer

st.set_page_config(
    page_title="Semantic Search — Endee",
    page_icon="🔍",
    layout="wide"
)

st.title("Semantic Search Engine")
st.caption("Powered by Endee Vector Database & Sentence Transformers")

# Track which files have been indexed this session to avoid duplicates
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

# ── Sidebar: Document Upload ────────────────────────────────────
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("Process & Store", type="primary"):
        total_chunks = 0
        progress_bar = st.sidebar.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):
            fname = uploaded_file.name

            # Skip files already indexed in this session
            if fname in st.session_state.indexed_files:
                st.sidebar.info(f"{fname} already indexed — skipping")
                progress_bar.progress((i + 1) / len(uploaded_files))
                continue

            try:
                with st.sidebar.status(f"Processing {fname}...", expanded=True) as status:
                    # Save a local copy
                    save_uploaded_pdf(uploaded_file)

                    # Reset file pointer before reading
                    uploaded_file.seek(0)

                    # Extract text
                    st.write("Extracting text...")
                    raw_text = extract_text_from_uploaded(uploaded_file)

                    if not raw_text.strip():
                        st.warning(f"No text found in {fname}")
                        status.update(label=f"{fname} — no text found", state="error")
                        continue

                    # Chunk the text
                    st.write("Splitting into chunks...")
                    chunks = chunk_text(raw_text, chunk_size=400, overlap=0)
                    st.write(f"Created {len(chunks)} chunks")

                    if not chunks:
                        st.warning(f"Could not create chunks from {fname}")
                        status.update(label=f"{fname} — chunking failed", state="error")
                        continue

                    # Embed and store in Endee
                    st.write("Generating embeddings & storing in Endee...")
                    stored = store_chunks_in_endee(chunks, source_filename=fname)
                    total_chunks += stored

                    st.session_state.indexed_files.add(fname)
                    status.update(label=f"{fname} — {stored} chunks stored", state="complete")

            except Exception as exc:
                st.sidebar.error(f"Error processing {fname}: {exc}")

            progress_bar.progress((i + 1) / len(uploaded_files))

        if total_chunks > 0:
            st.sidebar.success(f"Done — {total_chunks} chunks indexed")
        else:
            st.sidebar.warning("No new chunks were indexed")

# ── Main Area: Search ───────────────────────────────────────────
st.markdown("---")
st.subheader("Search Your Documents")

col_query, col_k = st.columns([4, 1])

with col_query:
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., What are the key findings of the study?"
    )

with col_k:
    top_k = st.number_input("Results", min_value=1, max_value=20, value=5)

if query:
    try:
        with st.spinner("Searching..."):
            results = semantic_search(query, top_k=top_k)

        if results:
            # Generate an LLM answer from the retrieved chunks
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(query, results)
                except Exception as llm_err:
                    answer = f"_LLM answer unavailable: {llm_err}_"

            st.markdown("### Answer")
            st.markdown(answer)

            st.markdown("---")
            st.markdown(f"**Source chunks ({len(results)}):**")

            for rank, result in enumerate(results, 1):
                similarity_pct = result["similarity"] * 100
                with st.expander(
                    f"#{rank} — {result['source']} (similarity: {similarity_pct:.1f}%)",
                    expanded=False
                ):
                    st.markdown(result["text"])
                    st.caption(f"Chunk: {result['chunk_id']} | Vector ID: {result['id']}")
        else:
            st.info("No results found. Upload and process some documents first.")
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        st.info("Make sure you have uploaded and processed at least one document.")

# ── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Endee Vector Database · Sentence Transformers · Streamlit")

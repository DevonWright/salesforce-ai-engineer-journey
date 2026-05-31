import os
import time
import pdfplumber
import chromadb
import gradio as gr
from chromadb.utils import embedding_functions
from openai import OpenAI
from anthropic import Anthropic

# ============================================================
# CORPUS METADATA
# ============================================================

CORPUS = {
    "📐 Well Architected Overview": {
        "questions": [
            "What are the core principles of the Salesforce Well Architected Framework?",
            "How does the Well Architected Framework help architects make better decisions?",
            "What are the three pillars of the Salesforce Well Architected Framework?",
        ]
    },
    "🔒 Trusted": {
        "questions": [
            "What does it mean for a Salesforce solution to be trusted?",
            "How should I approach security architecture in a Well Architected Salesforce solution?",
            "What are the key principles for building reliable Salesforce applications?",
            "How does the Well Architected Framework define compliance for Salesforce solutions?",
        ]
    },
    "⚡ Easy": {
        "questions": [
            "What makes a Salesforce solution easy according to the Well Architected Framework?",
            "How should automation be approached in a Well Architected Salesforce solution?",
            "What are the principles for building engaging Salesforce user experiences?",
            "What does intentional design mean in the context of Salesforce architecture?",
        ]
    },
    "🔧 Adaptable": {
        "questions": [
            "What does composable architecture mean in Salesforce and how do I achieve it?",
            "How should I design Salesforce solutions to be resilient to change?",
            "What are the key principles for building adaptable Salesforce architecture?",
        ]
    },
}


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(filepath: str) -> dict:
    filename = os.path.basename(filepath)
    doc_name = filename.replace(".pdf", "").replace("-", " ").strip()
    doc_name = " ".join(doc_name.split())
    all_text = []

    try:
        with pdfplumber.open(filepath) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                lines = text.split("\n")
                clean_lines = [
                    line.strip() for line in lines
                    if len(line.strip()) >= 20 and not line.strip().isdigit()
                ]
                if clean_lines:
                    # No [Page N] marker — clean text only
                    all_text.append("\n".join(clean_lines))

        full_text = "\n\n".join(all_text)
        return {
            "filename":   filename,
            "doc_name":   doc_name,
            "text":       full_text,
            "page_count": page_count,
            "char_count": len(full_text),
        }
    except Exception as e:
        print(f"Failed to extract {filename}: {e}")
        return None


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 100) -> list:
    chunks   = []
    start    = 0
    text_len = len(text)

    while start < text_len:
        end   = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)
        next_start = start + chunk_size - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(docs_path: str):
    print("Building index...")

    pdf_files = sorted([f for f in os.listdir(docs_path) if f.endswith(".pdf")])
    documents = []

    for filename in pdf_files:
        filepath = os.path.join(docs_path, filename)
        doc = extract_pdf(filepath)
        if doc and doc["char_count"] > 200:
            documents.append(doc)
            print(f"  Extracted: {doc['doc_name']}")

    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id":       f"{doc['filename'].replace('.pdf','').replace(' ','_')}_{i}",
                "text":     chunk,
                "doc_name": doc["doc_name"],
                "filename": doc["filename"],
            })

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )

    client = chromadb.Client()
    try:
        client.delete_collection("salesforce_docs")
    except:
        pass

    collection = client.create_collection(
        name="salesforce_docs",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

    BATCH_SIZE = 50
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        collection.add(
            ids       = [c["id"]      for c in batch],
            documents = [c["text"]    for c in batch],
            metadatas = [{"doc_name": c["doc_name"], "filename": c["filename"]} for c in batch],
        )

    total_chunks = collection.count()
    total_docs   = len(documents)
    print(f"Index ready — {total_chunks} chunks from {total_docs} documents")
    return collection, total_chunks, total_docs


# ============================================================
# RETRIEVAL + GENERATION
# ============================================================

SYSTEM_PROMPT = """You are a senior Salesforce Solution Architect with deep expertise
in the Salesforce Well Architected Framework.

Answer questions using ONLY the context chunks provided from official Salesforce
Well Architected documentation.

Rules:
- Use ONLY the provided context — not your general training knowledge
- Cite sources inline using [Source: document name]
- If the context lacks sufficient information, say so explicitly
- Be specific and technical — the user is an experienced Salesforce architect
- No filler sentences — be direct and complete"""


def retrieve(collection, question: str, top_k: int = 5) -> list:
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text":     text,
            "doc_name": meta["doc_name"],
            "distance": round(dist, 4)
        })
    return chunks


def build_context(chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"[Source {i+1}: {chunk['doc_name']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_openai(question: str, chunks: list) -> dict:
    client  = OpenAI()
    context = build_context(chunks)
    start   = time.time()
    resp    = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=600,
        temperature=0.2
    )
    return {
        "answer":        resp.choices[0].message.content,
        "input_tokens":  resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "total_tokens":  resp.usage.total_tokens,
        "response_time": round(time.time() - start, 2),
    }


def generate_anthropic(question: str, chunks: list) -> dict:
    if not chunks:
        return {"answer": "*No context retrieved.*", "input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0, "response_time": 0}

    client  = Anthropic()
    context = build_context(chunks)
    start   = time.time()
    resp    = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    )
    return {
        "answer":        resp.content[0].text,
        "input_tokens":  resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "total_tokens":  resp.usage.input_tokens + resp.usage.output_tokens,
        "response_time": round(time.time() - start, 2),
    }


def format_chunks_md(chunks: list) -> str:
    if not chunks:
        return "*No chunks retrieved.*"

    import re
    lines = []

    for i, chunk in enumerate(chunks):
        dist = chunk["distance"]

        if dist < 0.3:
            icon, label = "🟢", "strong match"
        elif dist < 0.4:
            icon, label = "🔵", "good match"
        elif dist < 0.5:
            icon, label = "🟡", "fair match"
        else:
            icon, label = "🔴", "weak match"

        # Strip any leftover page markers
        clean_text = re.sub(r'\[Page \d+\]\s*', '', chunk['text']).strip()

        # Truncate very long chunks for display — show first 600 chars
        # Full text was still passed to LLM, this is display only
        if len(clean_text) > 600:
            display_text = clean_text[:600].rsplit(' ', 1)[0] + "…"
        else:
            display_text = clean_text

        lines.append(
            f"#### {icon} Chunk {i+1} — {chunk['doc_name']}\n"
            f"`score: {dist} — {label}`\n\n"
            f"> {display_text}"
        )

    return "\n\n---\n\n".join(lines)


def format_stats(o: dict, a: dict) -> str:
    return (
        f"| | 🤖 GPT-4o-mini | 🤖 Claude Haiku |\n"
        f"|---|---|---|\n"
        f"| Response Time | `{o['response_time']}s` | `{a['response_time']}s` |\n"
        f"| Input Tokens  | `{o['input_tokens']}` | `{a['input_tokens']}` |\n"
        f"| Output Tokens | `{o['output_tokens']}` | `{a['output_tokens']}` |\n"
        f"| Total Tokens  | `{o['total_tokens']}` | `{a['total_tokens']}` |"
    )


# ============================================================
# GRADIO UI
# ============================================================

def launch_app(collection, total_chunks, total_docs):

    all_examples = []
    for data in CORPUS.values():
        for q in data["questions"]:
            all_examples.append([q])

    custom_css = """
        .gradio-container {
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
            background: #0f172a !important;
        }
        body, html { background: #0f172a !important; }
        footer { display: none !important; }
        .built-with { display: none !important; }

        .hero {
            background: linear-gradient(160deg, #1e3a5f 0%, #0f172a 60%);
            border-bottom: 1px solid #1e293b;
            padding: 56px 80px 30px;
        }

        .content {
            padding: 40px 80px;
            max-width: 100%;
            background: #0f172a;
        }

        .question-input textarea {
            font-size: 17px !important;
            padding: 28px 20px 18px 20px !important;
            background: #1e293b !important;
            border: 2px solid #334155 !important;
            border-radius: 12px !important;
            color: #f1f5f9 !important;
            min-height: 90px !important;
            resize: vertical !important;
            transition: border-color 0.15s ease !important;
        }
        .question-input textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
            outline: none !important;
        }

        .ask-btn button {
            width: 100% !important;
            font-size: 15px !important;
            font-weight: 700 !important;
            padding: 14px !important;
            border-radius: 10px !important;
            letter-spacing: 0.04em !important;
            margin-top: 10px !important;
            background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
            border: none !important;
            transition: opacity 0.15s ease !important;
        }
        .ask-btn button:hover { opacity: 0.9 !important; }

        /* Answer panels */
        .answer-panel {
            background: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 14px !important;
            padding: 0 !important;
            min-height: 220px !important;
        }

        /* Padding inside answer markdown */
        #openai-out, #anthropic-out {
            padding: 20px 24px !important;
        }
        #openai-out p, #anthropic-out p,
        #openai-out li, #anthropic-out li {
            color: #e2e8f0 !important;
            line-height: 1.7 !important;
        }

        /* Stat table */
        .stat-wrap table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            font-family: 'DM Mono', monospace;
            margin-top: 4px;
        }
        .stat-wrap th {
            background: #1e293b;
            color: #64748b;
            padding: 8px 14px;
            text-align: left;
            border: 1px solid #334155;
            font-weight: 600;
        }
        .stat-wrap td {
            padding: 8px 14px;
            border: 1px solid #1e293b;
            color: #e2e8f0;
            background: #0f172a;
        }

        /* Constrain all content to max width matching titles */
        .gradio-container .prose,
        .gradio-container .markdown,
        .gradio-container .md {
            max-width: 100% !important;
        }

        /* Fix blockquote text overflow */
        blockquote {
            border-left: 3px solid #3b82f6 !important;
            padding: 12px 16px !important;
            margin: 8px 0 0 0 !important;
            background: #0d1120 !important;
            border-radius: 0 8px 8px 0 !important;
            font-size: 13px !important;
            color: #94a3b8 !important;
            line-height: 1.65 !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            overflow: hidden !important;
            max-width: 100% !important;
        }

        blockquote p {
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            margin: 0 !important;
        }

        /* Constrain section content width */
        #openai-out, #anthropic-out {
            padding: 20px 24px !important;
            max-width: 100% !important;
            overflow-wrap: break-word !important;
            word-wrap: break-word !important;
        }

        /* Stats, chunks full width but contained */
        .stat-wrap, .chunks-wrap {
            max-width: 1280px !important;
            margin: 0 auto !important;
        }

        /* Hide Gradio's injected "Examples" label */
        .example-pills .label-wrap { display: none !important; }
        .example-pills > .wrapper > .label-wrap { display: none !important; }
        .example-pills span.text { display: none !important; }
        #example-pills > div > span { display: none !important; }
        #example-pills .label-wrap { display: none !important; }

        /* Style example pills */
        .example-pills table,
        #example-pills table { 
            border: none !important; 
            background: transparent !important; 
        }
        .example-pills td,
        #example-pills td { 
            border: none !important; 
            background: transparent !important; 
            padding: 4px !important; 
        }
        .example-pills button,
        #example-pills button {
            background: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 20px !important;
            color: #7dd3fc !important;
            font-size: 12px !important;
            padding: 6px 14px !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            white-space: nowrap !important;
        }
        .example-pills button:hover,
        #example-pills button:hover {
            background: #1e3a5f !important;
            border-color: #3b82f6 !important;
            color: #bfdbfe !important;
        }
        
        /* Nuclear option on Examples label */
        #example-pills .label-wrap,
        #example-pills label,
        #example-pills > div > span,
        #example-pills .svelte-label,
        div[id="example-pills"] > div > span { 
            display: none !important; 
        }

        /* Remove Gradio default container backgrounds */
        .gradio-container > div,
        .contain,
        .gap,
        .gr-group {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* Ensure rows have no background */
        .gr-row, .row {
            background: transparent !important;
        }

        /* Constrain answer row to match other sections */
        #answer-row {
            max-width: 1280px !important;
            margin: 0 auto !important;
            padding: 0 80px !important;
        }

        /* Constrain question input and button rows */
        #question-row,
        #button-row,
        #examples-row {
            max-width: 1280px !important;
            margin: 0 auto !important;
            padding: 0 80px !important;
        }

        /* Constrain interactive rows to match other sections */
        #question-input,
        #ask-btn,
        #example-pills {
            max-width: 1280px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: 80px !important;
            padding-right: 80px !important;
        }

        #answer-row {
            max-width: 1280px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: 80px !important;
            padding-right: 80px !important;
        }

        /* Target the parent row containers */
        #question-input > .wrap,
        #ask-btn > .wrap {
            max-width: 1280px !important;
            margin: 0 auto !important;
            padding: 0 80px !important;
        }

        /* Constrain question input and button to match other sections */
        #question-input {
            max-width: 1120px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        #ask-btn {
            max-width: 1120px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        /* Remove inner container box from question input */
        #question-input .wrap,
        #question-input .container,
        #question-input > div,
        #question-input > div > div {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }

        /* Remove column wrapper box */
        #question-input.svelte-kqij2n,
        .svelte-kqij2n {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* Gradio 6 specific — remove form element backgrounds */
        #question-input form,
        #question-input .form,
        #question-input [data-testid="textbox"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }

        /* Push text down from top of input box */
        .question-input,
        .question-input > div,
        .question-input > div > div,
        .question-input > div > div > label,
        .question-input textarea {
            padding-top: 16px !important;
        }

        /* Specifically target the Gradio textbox wrapper */
        [data-testid="textbox"] {
            padding-top: 16px !important;
        }

        [data-testid="textbox"] textarea {
            padding-top: 20px !important;
            margin-top: 8px !important;
        }
    """

    with gr.Blocks(title="Salesforce Well-Architected Framework Q&A — Devon Wright") as demo:

        # ── Hero ─────────────────────────────────────────────
        gr.HTML(f"""
        <div class="hero">
          <div style="max-width:1280px; margin:0 auto;">

            <h1 style="
              font-family:-apple-system,BlinkMacSystemFont,sans-serif;
              font-size:clamp(30px,4vw,52px);
              font-weight:800;
              color:#f8fafc;
              margin:0 0 14px;
              line-height:1.08;
              letter-spacing:-0.02em;
            ">Salesforce Well-Architected<br>
            <span style="color:#3b82f6;">Framework Q&amp;A</span></h1>

            <p style="
              font-family:-apple-system,BlinkMacSystemFont,sans-serif;
              color:#94a3b8; font-size:16px;
              margin:0 0 20px; max-width:640px; line-height:1.65;
            ">Ask questions about Salesforce's
            <a href="https://architect.salesforce.com/docs/architect/well-architected/overview"
               target="_blank"
               style="color:#3b82f6;text-decoration:none;border-bottom:1px solid rgba(59,130,246,0.35);">
               Well-Architected Framework</a>
            — an official Salesforce methodology for building solutions that are
            <strong style="color:#f1f5f9;">Easy</strong>,
            <strong style="color:#f1f5f9;">Trusted</strong>, and
            <strong style="color:#f1f5f9;">Adaptable</strong>.
            Two AI models answer simultaneously from
            <strong style="color:#f1f5f9;">{total_chunks} indexed chunks</strong>
            across <strong style="color:#f1f5f9;">{total_docs} official documents</strong>
            — every answer cited back to its source.</p>

            <a href="https://www.linkedin.com/in/devwright/" target="_blank"
               style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;
                      color:#475569;font-size:13px;text-decoration:none;
                      border-bottom:1px solid #1f2d4a;padding-bottom:2px;">
               Built by Devon Wright</a>
          </div>
        </div>
        """)

        # ── Question + Examples + Answers wrapper ─────────────
        gr.HTML("""
        <div style="max-width:1280px;margin:0 auto;padding:10px 80px 0;">
          <div style="font-family:monospace;font-size:11px;letter-spacing:0.18em;
                      text-transform:uppercase;color:#475569;font-weight:700;
                      margin-bottom:6px;">
            Ask a Question
          </div>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    placeholder="e.g. What are the core principles of the Salesforce Well-Architected Framework?",
                    lines=3,
                    show_label=False,
                    container=False,
                    elem_classes=["question-input"],
                    elem_id="question-input",
                )

        with gr.Row():
            with gr.Column():
                submit_btn = gr.Button(
                    "⚡  Ask Both Models",
                    variant="primary",
                    elem_classes=["ask-btn"],
                    elem_id="ask-btn",
                )

        gr.Examples(
            examples=all_examples[:4],
            inputs=question_input,
            label=None,
            elem_id="example-pills",
        )

        gr.HTML("""
        <div style="max-width:1280px;margin:0 auto;padding:24px 80px 16px;">
          <div style="font-family:monospace;font-size:11px;letter-spacing:0.18em;
                      text-transform:uppercase;color:#475569;font-weight:700;
                      padding-top:24px;border-top:1px solid #1e293b;">
            Answers
          </div>
        </div>
        """)

        with gr.Row(elem_id="answer-row"):
            with gr.Column(elem_classes=["answer-panel"]):
                gr.HTML("""
                <div style="padding:18px 20px 10px;border-bottom:1px solid #1e293b;
                            display:flex;align-items:center;gap:10px;">
                  <span style="font-size:20px;">🤖</span>
                  <div>
                    <div style="font-size:14px;font-weight:700;color:#f1f5f9;
                                font-family:-apple-system,BlinkMacSystemFont,sans-serif;">
                      GPT-4o-mini</div>
                    <div style="font-size:11px;color:#64748b;
                                font-family:-apple-system,BlinkMacSystemFont,sans-serif;">
                      OpenAI</div>
                  </div>
                </div>""")
                openai_output = gr.Markdown(
                    value="*Ask a question above to see the answer here.*",
                    elem_id="openai-out",
                )

            with gr.Column(elem_classes=["answer-panel"]):
                gr.HTML("""
                <div style="padding:18px 20px 10px;border-bottom:1px solid #1e293b;
                            display:flex;align-items:center;gap:10px;">
                  <span style="font-size:20px;">🤖</span>
                  <div>
                    <div style="font-size:14px;font-weight:700;color:#f1f5f9;
                                font-family:-apple-system,BlinkMacSystemFont,sans-serif;">
                      Claude Haiku</div>
                    <div style="font-size:11px;color:#64748b;
                                font-family:-apple-system,BlinkMacSystemFont,sans-serif;">
                      Anthropic</div>
                  </div>
                </div>""")
                anthropic_output = gr.Markdown(
                    value="*Ask a question above to see the answer here.*",
                    elem_id="anthropic-out",
                )

        # ── Performance stats ────────────────────────────────
        with gr.Row():
            with gr.Column():
                # Performance stats label
                gr.HTML("""
                <div style="padding:0 80px;">
                  <div style="max-width:1280px;margin:0 auto;
                              font-family:monospace;font-size:11px;letter-spacing:0.18em;
                              text-transform:uppercase;color:#475569;font-weight:700;
                              margin-top:32px;padding-top:24px;border-top:1px solid #1e293b;">
                    ⚡ Performance Comparison
                  </div>
                </div>
                """)
                stats_output = gr.Markdown(
                    value="*Performance metrics will appear here after your first query.*",
                    elem_classes=["stat-wrap"],
                )

        # ── Retrieved chunks ─────────────────────────────────
        with gr.Row():
            with gr.Column():
                # Chunks label
                gr.HTML("""
                <div style="padding:0 80px;">
                  <div style="max-width:1280px;margin:0 auto;
                              font-family:monospace;font-size:11px;letter-spacing:0.18em;
                              text-transform:uppercase;color:#475569;font-weight:700;
                              margin-top:32px;padding-top:24px;border-top:1px solid #1e293b;">
                    🔍 Retrieved Context — What the Models Saw
                  </div>
                </div>
                """)
                chunks_output = gr.Markdown(
                    value="*Retrieved document chunks will appear here after your first query.*",
                    elem_classes=["chunks-wrap"],
                )

        # ── Pipeline diagram ─────────────────────────────────
        gr.HTML("""
        <div style="background:#0f172a;border-top:1px solid #1e293b;
                    padding:24px 80px;margin-top:40px;">
          <div style="max-width:1280px;margin:0 auto;">
            <div style="font-family:monospace;font-size:10px;letter-spacing:0.2em;
                        text-transform:uppercase;color:#334155;
                        margin-bottom:14px;font-weight:700;">
              How It Works — RAG Pipeline
            </div>
            <div style="display:flex;align-items:center;gap:8px;
                        flex-wrap:wrap;font-family:monospace;font-size:12px;">
              <span style="background:#111827;border:1px solid #1f2d4a;padding:6px 14px;border-radius:6px;color:#7dd3fc;">📄 12 PDFs</span>
              <span style="color:#1f2d4a;font-size:18px;">→</span>
              <span style="background:#111827;border:1px solid #1f2d4a;padding:6px 14px;border-radius:6px;color:#7dd3fc;">pdfplumber</span>
              <span style="color:#1f2d4a;font-size:18px;">→</span>
              <span style="background:#111827;border:1px solid #1f2d4a;padding:6px 14px;border-radius:6px;color:#7dd3fc;">1200-char chunks</span>
              <span style="color:#1f2d4a;font-size:18px;">→</span>
              <span style="background:#111827;border:1px solid #1f2d4a;padding:6px 14px;border-radius:6px;color:#7dd3fc;">text-embedding-3-small</span>
              <span style="color:#1f2d4a;font-size:18px;">→</span>
              <span style="background:#111827;border:1px solid #1f2d4a;padding:6px 14px;border-radius:6px;color:#7dd3fc;">ChromaDB cosine search</span>
              <span style="color:#1f2d4a;font-size:18px;">→</span>
              <span style="background:#111827;border:1px solid #1f2d4a;padding:6px 14px;border-radius:6px;color:#7dd3fc;">top-5 retrieval</span>
              <span style="color:#1f2d4a;font-size:18px;">→</span>
              <span style="background:#0d1b35;border:1px solid #2563eb;padding:6px 14px;border-radius:6px;color:#93c5fd;font-weight:700;">GPT-4o-mini + Claude Haiku</span>
            </div>
          </div>
        </div>
        """)

        # ── Auto-scroll on submit ────────────────────────────
        gr.HTML("""
        <script>
            (function() {
                function scrollToAnswers() {
                    try {
                        // We are inside an iframe on HF Spaces
                        // Target the iframe's own scroll container
                        const scrollTargets = [
                            document.querySelector('.answer-panel'),
                            document.querySelector('#openai-out'),
                            document.querySelector('[id="answer-row"]'),
                        ];

                        for (const el of scrollTargets) {
                            if (el) {
                                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                return;
                            }
                        }

                        // Fallback — scroll the document body directly
                        const scrollAmount = window.innerHeight * 0.8;
                        window.scrollTo({
                            top: window.scrollY + scrollAmount,
                            behavior: 'smooth'
                        });

                    } catch(e) {
                        console.log('Scroll error:', e);
                    }
                }

                function findAndAttach() {
                    const allBtns = document.querySelectorAll('button');
                    let found = false;

                    allBtns.forEach(function(btn) {
                        if (btn.textContent && btn.textContent.includes('Ask Both Models')) {
                            btn.addEventListener('click', function() {
                                setTimeout(scrollToAnswers, 400);
                            });
                            found = true;
                            console.log('✅ Scroll listener attached');
                        }
                    });

                    if (!found) {
                        setTimeout(findAndAttach, 500);
                    }
                }

                // Multiple entry points to handle different load states
                setTimeout(findAndAttach, 500);
                setTimeout(findAndAttach, 1500);
                setTimeout(findAndAttach, 3000);

            })();
        </script>
        """)

        # ── Query handler ────────────────────────────────────
        def run_query(question):
            if not question or not question.strip():
                return (
                    "*Please enter a question.*",
                    "*Please enter a question.*",
                    "*Performance metrics will appear after your first query.*",
                    "*Retrieved chunks will appear after your first query.*",
                )

            print(f"\nQuery: {question}")
            chunks = retrieve(COLLECTION, question)
            print(f"Chunks retrieved: {len(chunks)}")

            if not chunks:
                return (
                    "*No relevant content found.*",
                    "*No relevant content found.*",
                    "*No sources retrieved.*",
                    "*No chunks retrieved.*",
                )

            o_result = generate_openai(question, chunks)
            a_result = generate_anthropic(question, chunks)

            return (
                o_result["answer"],
                a_result["answer"],
                format_stats(o_result, a_result),
                format_chunks_md(chunks),
            )

        submit_btn.click(
            fn=run_query,
            inputs=question_input,
            outputs=[openai_output, anthropic_output, stats_output, chunks_output],
            show_progress="full",
        )

    demo.launch(
        css=custom_css,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("DM Sans"),
        ),
        ssr_mode=False,
    )


# ============================================================
# ENTRY POINT
# ============================================================

COLLECTION = None

if __name__ == "__main__":
    DOCS_PATH                    = "./docs"
    COLLECTION, n_chunks, n_docs = build_index(DOCS_PATH)
    launch_app(COLLECTION, n_chunks, n_docs)
# Project 1 — Salesforce Well-Architected RAG Pipeline

A retrieval-augmented generation system built from scratch over official Salesforce 
Well-Architected Framework documentation. No LangChain abstraction on the deployed 
pipeline — every step is explicit, observable, and evaluated.

🔗 **[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/DevThaDev/salesforce-well-architected-rag)**

---

## What It Does

Ask natural language questions about the Salesforce Well-Architected Framework and 
get answers grounded in official documentation, with source citations, retrieval 
confidence scores, and side-by-side comparison between GPT-4o-mini and Claude Haiku.

**Example questions this system answers:**
- "What are the three pillars of the Salesforce Well-Architected Framework?"
- "What does it mean for a Salesforce solution to be trusted?"
- "How does the Well-Architected Framework define composable architecture?"
- "What role does automation play in building an Easy Salesforce solution?"

---

## Architecture

12 PDFs (Well-Architected Framework docs)
→ pdfplumber extraction
→ 1200-char chunks (100-char overlap)
→ OpenAI text-embedding-3-small
→ ChromaDB (cosine similarity, in-memory)
→ top-5 retrieval
→ GPT-4o-mini + Claude Haiku (simultaneous)
→ cited answer with source confidence scores

---

## Technical Stack

| Component | Technology | Why |
|---|---|---|
| PDF Extraction | pdfplumber | Handles multi-column layouts, cleaner than PyPDF |
| Chunking | Custom Python | Raw implementation — no framework abstraction |
| Embeddings | OpenAI text-embedding-3-small | Fast, cheap, strong on technical docs |
| Vector Store | ChromaDB (in-memory) | Zero setup, Colab-friendly |
| Generation | GPT-4o-mini + Claude Haiku | Provider-agnostic architecture |
| Tracing | LangSmith | Full observability on every retrieval + generation step |
| Validation | Pydantic | Typed inputs and outputs throughout |
| Evaluation | DeepEval | Faithfulness, answer relevancy, context recall |
| UI | Gradio | Interactive side-by-side model comparison |
| Deployment | Hugging Face Spaces | Permanent public URL |

---

## Corpus

12 PDFs sourced from the official 
[Salesforce Well-Architected Framework](https://architect.salesforce.com/docs/architect/well-architected/overview) 
covering all three pillars:

**Trusted** — Secure · Compliant · Reliable  
**Easy** — Intentional · Automated · Engaging  
**Adaptable** — Resilient · Composable  

---

## Evaluation Results

Evaluated using DeepEval across a 20-question golden test set covering all 
three framework pillars. Both pipelines tested on identical questions, corpus, 
and embedding model — only the framework layer differs.

| Metric | Raw Pipeline | LangChain Pipeline | Delta |
|---|---|---|---|
| Faithfulness | 0.9637 | 0.9500 | ▼0.0137 |
| Answer Relevancy | 0.9830 | 0.9549 | ▼0.0280 |
| Context Recall | 1.0000 | 1.0000 | =0.0000 |

> The raw pipeline slightly outperforms LangChain on faithfulness and answer 
> relevancy, confirming that understanding the primitives before reaching for 
> a framework produces a more precise system. Context recall is perfect on 
> both — the retriever finds all relevant chunks regardless of framework layer.

---

## Key Design Decisions

**Raw pipeline before LangChain** — built the full retrieval and generation 
pipeline directly against the OpenAI and Chroma APIs before introducing any 
framework. Evaluation confirms this produced a slightly more precise system 
than the LangChain abstraction layer.

**Provider-agnostic generation** — the retrieval layer is completely decoupled 
from the generation layer. Swapping OpenAI for Anthropic requires changing one 
parameter. Demonstrated live in the app with side-by-side GPT-4o-mini vs 
Claude Haiku comparison.

**Observability from day one** — LangSmith tracing added before the first query 
was run. Every retrieval call, generation call, token count, and latency is 
logged. This is a production habit, not an afterthought.

**Evaluation-driven iteration** — DeepEval integrated into the notebook to 
measure faithfulness, answer relevancy, and context recall before and after 
the LangChain refactor. Scores above 0.95 on all metrics confirm production-ready 
retrieval quality.

**Production considerations** — ChromaDB runs in-memory for this prototype. 
A production deployment would persist embeddings to Pinecone or a disk-backed 
Chroma instance to avoid re-embedding on restart. Re-embedding the full corpus 
currently takes ~60 seconds and costs ~$0.02.

---

## How to Run

1. Open `salesforce-well-architected-rag.ipynb` in Google Colab
2. Add the following to Colab Secrets:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `LANGSMITH_API_KEY`
   - `GOOGLE_DRIVE_FOLDER` (folder name containing your PDFs)
3. Upload the 12 Well-Architected PDFs to a Google Drive folder matching that name
4. Run all cells in order
5. Use Cell 17 to ask your own questions
6. DeepEval evaluation runs in Cells 21–34

---

## What I Learned

- How chunking strategy directly impacts retrieval quality — boundary detection 
  vs fixed-size tradeoffs and why overlap matters at chunk boundaries
- Why the same embedding model must be used at both ingest and query time
- How cosine distance scores surface retrieval confidence without a separate 
  evaluation step
- The difference between PDF text extraction and URL scraping for enterprise 
  RAG — and when each is the right choice
- How to measure RAG quality with DeepEval — faithfulness, answer relevancy, 
  and context recall as distinct signals for different failure modes
- That building raw before using a framework produces measurably better results 
  and deeper debugging intuition

---

## Author

Devon Wright — Salesforce Application Architect  
[LinkedIn](https://www.linkedin.com/in/devwright/) · 
[Hugging Face](https://huggingface.co/DevThaDev)

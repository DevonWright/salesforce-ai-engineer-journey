# Project 2 — Salesforce Data Architect Agent

An agentic system that takes a plain-language business requirement and 
produces a structured Salesforce data model — validating every object name 
and field type against real Salesforce rules, self-correcting when 
validation fails.

🔗 **[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/DevonWright/salesforce-data-architect-agent)**

---

## What It Does

Describe a business need in plain language. The agent:

1. Extracts structured fields from the freeform input
2. Evaluates whether the requirement is clear enough to design from
3. Proposes Salesforce objects, fields, and relationships
4. Validates every object name and field type using on-demand tool calls
5. Self-corrects if validation fails, feeding specific errors back into design
6. Produces a structured data architecture document

**Three observable outcomes:**
- Clean design document with validated data model
- Clarifying questions when the requirement is too vague to design from
- Self-correction loop visible when validation fails and design retries

---

## Architecture
plain text input
→ extraction (requirement, process_flow, edge_cases, current_process)
→ intake node (is this clear enough to design from?)
→ design node + tool calls (propose objects/fields, validate inline)
→ validate node (structural limits check, pass/retry decision)
→ output node (assemble final design document)

---

## Technical Stack

| Component | Technology | Why |
|---|---|---|
| Agent loop | Plain Python control flow | LangGraph evaluated and intentionally not used — the retry loop is a single condition that doesn't warrant graph overhead |
| LLM | GPT-4o-mini | Consistent structured JSON output, low latency |
| Tool calls | Inline Python functions | On-demand naming and field-type validation called during design |
| Validation | Custom structural limits checker | Counts real metadata from the design dict — no fabricated estimates |
| UI | Gradio + Hugging Face Spaces | Single-input public demo |

---

## Design Decisions

**No LangGraph** — the agent has four sequential nodes and one binary retry condition. A plain Python `for` loop expresses that more clearly than a graph framework. LangGraph was evaluated and deliberately skipped. The README says so because that decision is worth more than silently using a framework that wasn't needed.

**No RAG** — the Salesforce Well-Architected Framework principles (Composable, Reliable) are small, stable, and foundational. They live directly in the design node's system prompt as static text. Retrieving them from a vector store every call would add latency and unpredictability with no benefit.

**No fabricated governor estimates** — previous versions attempted to have the LLM estimate SOQL query counts and DML rows. Those are runtime execution metrics that only exist while Apex code runs — this agent designs a data model, not code. The structural limits check (max fields per object, max Master-Detail relationships) uses real Salesforce metadata ceiling values applied to numbers counted directly from the design dict. Nothing is invented.

**Scoped to data architecture only** — the agent proposes objects, fields, and relationships. It does not design automation, Flows, triggers, or Apex. That is a deliberate boundary: a solution architect agent that takes this data model as input and designs the behavior layer on top is a natural Project 3 extension.

**Standard objects preferred** — the design prompt explicitly lists Account, Contact, and User with their purposes, instructing the agent to use them before inventing custom objects. This reflects real Salesforce data modeling practice.

**Single-input UI with extraction layer** — the agent internally requires four structured fields. The public UI accepts one freeform paragraph and runs a lightweight extraction call before the agent sees anything. The extraction is invisible to the user — it's plumbing, not the demonstration.

---

## Agent Nodes

| Node | Job | Calls LLM | Calls Tools |
|---|---|---|---|
| intake | Decides if requirement is clear enough | Yes | No |
| design | Proposes objects/fields/relationships | Yes | Yes — validate_object_name, validate_field_type |
| validate | Checks structural limits, pass/retry | No | Yes — check_structural_limits |
| output | Assembles final design document | Yes | No |

---

## Tool Calls

Two tools called on-demand inside the design node, one called inside validate:

**`validate_object_name(name)`** — checks naming conventions (must end `__c`, cannot start with digit, cannot be a reserved Salesforce object name)

**`validate_field_type(field_type)`** — checks against the list of valid Salesforce field types, catches relationship descriptions accidentally placed in the fields array

**`check_structural_limits(obj)`** — counts fields and Master-Detail relationships per object, compares against real Salesforce metadata ceilings (800 fields max, 2 Master-Detail max)

---

## Self-Correction Loop

When `validate` finds errors, it does not output a failed result. It feeds the specific error messages back into `design` as an override note and retries — up to 3 attempts. Each retry sees exactly what was wrong with the previous attempt. This is the ReAct (Reason → Act → Observe → Reason) pattern made concrete: the tool's output is the observation that drives the next design attempt.

---

## What This Agent Deliberately Does Not Do

- **No automation design** — Flows, triggers, Apex, and validation rules are out of scope by design. A data architect designs the schema; a solution architect designs the behavior.
- **No runtime governor limit estimates** — fabricated SOQL/DML counts were removed after evaluation. Only real, countable structural limits are checked.
- **No Well-Architected RAG** — principles are embedded directly in the system prompt, not retrieved.
- **No brownfield impact analysis** — the agent cannot know what existing automation, sharing rules, or integrations exist in a real org. Impact on existing configuration is a human architect's responsibility.

---

## How to Run Locally

```bash
pip install openai gradio
export OPENAI_API_KEY=your_key
python app.py
```

---

## Author

Devon Wright — Salesforce Application Architect  
[LinkedIn](https://www.linkedin.com/in/devwright/) · 
[Hugging Face](https://huggingface.co/DevThaDev)

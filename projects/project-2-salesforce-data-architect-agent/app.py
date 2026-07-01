import os
import re
import json
import gradio as gr
from openai import OpenAI

# ============================================================
# TOOLS
# ============================================================

RESERVED_OBJECT_NAMES = [
    "User", "Profile", "PermissionSet", "Group", "Territory",
    "Forecast", "Contract", "Order", "Asset", "Case"
]

VALID_FIELD_TYPES = [
    "Text", "Number", "Currency", "Date", "DateTime",
    "Picklist", "MultiSelectPicklist", "Checkbox", "Email",
    "Phone", "URL", "TextArea", "LongTextArea", "Percent"
]

MAX_FIELDS_PER_OBJECT        = 800
MAX_MASTER_DETAIL_PER_OBJECT = 2


def validate_object_name(name: str) -> dict:
    errors = []
    if not name.endswith("__c"):
        errors.append("Must end with __c")
    base = name.replace("__c", "")
    if base and base[0].isdigit():
        errors.append("Cannot start with a number")
    if base in RESERVED_OBJECT_NAMES:
        errors.append(f"{base} is a reserved name")
    return {"name": name, "valid": len(errors) == 0, "errors": errors}


def validate_field_type(field_type: str) -> dict:
    relationship_words = ["lookup to", "master-detail to", "master detail to"]
    if any(w in field_type.lower() for w in relationship_words):
        return {"field_type": field_type, "valid": False,
                "error": f"'{field_type}' is a relationship, not a field type"}
    valid = field_type in VALID_FIELD_TYPES
    return {"field_type": field_type, "valid": valid,
            "error": None if valid else f"'{field_type}' is not a valid field type"}


def check_structural_limits(obj: dict) -> dict:
    field_count = len(obj.get("fields", []))
    md_count = sum(
        1 for r in obj.get("relationships", [])
        if "master-detail" in r.lower() or "master detail" in r.lower()
    )
    errors = []
    if field_count > MAX_FIELDS_PER_OBJECT:
        errors.append(f"{field_count} fields exceeds max of {MAX_FIELDS_PER_OBJECT}")
    if md_count > MAX_MASTER_DETAIL_PER_OBJECT:
        errors.append(f"{md_count} Master-Detail exceeds max of {MAX_MASTER_DETAIL_PER_OBJECT}")
    return {"api_name": obj.get("api_name"), "field_count": field_count,
            "master_detail_count": md_count, "passed": len(errors) == 0, "errors": errors}


# ============================================================
# EXTRACTION
# ============================================================

EXTRACTION_PROMPT = """A user wrote a rough description of a business need.
Extract four things. Write "Not specified" if something isn't mentioned.

USER INPUT: {raw_text}

Respond in this exact JSON format only, no other text:
{{
  "requirement": "the core business need, restated clearly",
  "process_flow": "the sequence of steps described, if any",
  "edge_cases": "exceptions or unusual scenarios mentioned, if any",
  "current_process": "how it is done today, if mentioned"
}}"""


def extract_fields(raw_text: str) -> dict:
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(raw_text=raw_text)}],
        max_tokens=400, temperature=0.1
    )
    raw = re.sub(r"^```[a-z]*\n?|\n?```$", "",
                 resp.choices[0].message.content.strip())
    return json.loads(raw)


# ============================================================
# NODES
# ============================================================

CLARITY_PROMPT = """You are a senior Salesforce Data Architect reviewing a requirement.

Requirement: {requirement}
Process Flow: {process_flow}
Edge Cases: {edge_cases}
Current Process: {current_process}

Is there enough information to design a Salesforce data model?
Only flag as too vague if the core business problem is genuinely unclear
or the input is a single sentence with no process detail at all.
If there is a reasonable description of what needs to be built and how,
mark it as clear even if some fields say Not specified. 

Respond in this exact JSON format only:
{{
  "is_clear": true or false,
  "clarifying_questions": ["question 1", "question 2"]
}}
If is_clear is true, clarifying_questions must be empty."""


def intake(requirement, process_flow, edge_cases, current_process) -> dict:
    client = OpenAI()
    prompt = CLARITY_PROMPT.format(
        requirement=requirement, process_flow=process_flow,
        edge_cases=edge_cases, current_process=current_process
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300, temperature=0.1
    )
    raw = re.sub(r"^```[a-z]*\n?|\n?```$", "",
                 resp.choices[0].message.content.strip())
    return json.loads(raw)


DESIGN_PROMPT = """You are a senior Salesforce Data Architect.
Design ONLY the data model — objects, fields, relationships.
No automation, triggers, Flows, or Apex. Do not mention them.

Design philosophy:
- Relationships should be composable: flexible and reusable
- Field and object structure should protect data reliability
- ALWAYS prefer existing standard objects over custom ones:
    Account: any organization or company
    Contact: any individual person
    User: any internal Salesforce user
  Only create custom objects when data genuinely does not fit above.

REQUIREMENT: {requirement}
PROCESS FLOW: {process_flow}
EDGE CASES: {edge_cases}
CURRENT PROCESS: {current_process}
{override_section}

Rules:
- Custom object API names must end with __c
- Fields format: "Field_Name__c (Type)"
- Valid field types ONLY: Text, Number, Currency, Date, DateTime, Picklist,
  MultiSelectPicklist, Checkbox, Email, Phone, URL, TextArea, LongTextArea, Percent
- Relationships go ONLY in relationships array, never in fields
- Prefer Lookup over Master-Detail unless child cannot exist without parent

Respond in this exact JSON format only, no other text:
{{
  "data_model": [
    {{
      "object_name": "Human readable name",
      "api_name": "API_Name__c",
      "fields": ["Field_Name__c (Type)"],
      "relationships": ["Lookup to Account"]
    }}
  ],
  "alternative_considered": "only if a real trade-off was made, else empty string"
}}"""


def design(requirement, process_flow, edge_cases,
           current_process, override_note="") -> dict:
    client = OpenAI()
    override_section = f"OVERRIDE NOTE: {override_note}" if override_note else ""
    prompt = DESIGN_PROMPT.format(
        requirement=requirement, process_flow=process_flow,
        edge_cases=edge_cases, current_process=current_process,
        override_section=override_section
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200, temperature=0.3
    )
    raw = re.sub(r"^```[a-z]*\n?|\n?```$", "",
                 resp.choices[0].message.content.strip())
    parsed = json.loads(raw)

    for obj in parsed.get("data_model", []):
        name_check = validate_object_name(obj["api_name"])
        obj["naming_valid"]  = name_check["valid"]
        obj["naming_errors"] = name_check["errors"]
        field_results = []
        for field_str in obj.get("fields", []):
            if "(" in field_str and ")" in field_str:
                ft = field_str.split("(")[1].split(")")[0].strip()
                field_results.append(validate_field_type(ft))
        obj["field_check_results"] = field_results
    return parsed


def validate(design_result: dict) -> dict:
    all_errors = []
    structural_results = []
    for obj in design_result.get("data_model", []):
        if not obj.get("naming_valid", True):
            for err in obj.get("naming_errors", []):
                all_errors.append(f"{obj['api_name']}: {err}")
        for fr in obj.get("field_check_results", []):
            if not fr.get("valid", True):
                all_errors.append(f"{obj['api_name']}: {fr['error']}")
        structural = check_structural_limits(obj)
        structural_results.append(structural)
        if not structural["passed"]:
            for err in structural["errors"]:
                all_errors.append(f"{obj['api_name']}: {err}")
    return {"passed": len(all_errors) == 0, "errors": all_errors,
            "structural_results": structural_results}


OUTPUT_PROMPT = """You are a senior Salesforce Data Architect writing
narrative sections of a design document. Do not design anything new.

REQUIREMENT: {requirement}
CURRENT PROCESS: {current_process}
DATA MODEL: {data_model_summary}
VALIDATION STATUS: {validation_status}

Respond in this exact JSON format only:
{{
  "problem": "1-2 sentence problem statement",
  "current_state": "1-2 sentences on what happens today",
  "proposed_solution": "1-2 sentences describing the data model decision"
}}"""


def output_node(requirement, current_process,
                design_result, validation_result,
                clarifying_questions=None) -> dict:
    if clarifying_questions:
        return {
            "problem": None, "current_state": None, "proposed_solution": None,
            "data_model": [], "validation_status": "needs_clarification",
            "alternative_considered": "", "confidence_score": 0.0,
            "clarifying_questions": clarifying_questions
        }
    client = OpenAI()
    summary = ", ".join(
        f"{o['object_name']} ({len(o.get('fields',[]))} fields)"
        for o in design_result.get("data_model", [])
    )
    validation_status = "passed" if validation_result["passed"] else "failed"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": OUTPUT_PROMPT.format(
            requirement=requirement, current_process=current_process,
            data_model_summary=summary, validation_status=validation_status
        )}],
        max_tokens=300, temperature=0.2
    )
    raw = re.sub(r"^```[a-z]*\n?|\n?```$", "",
                 resp.choices[0].message.content.strip())
    narrative = json.loads(raw)
    return {
        "problem":               narrative["problem"],
        "current_state":         narrative["current_state"],
        "proposed_solution":     narrative["proposed_solution"],
        "data_model":            design_result.get("data_model", []),
        "validation_status":     validation_status,
        "alternative_considered": design_result.get("alternative_considered", ""),
        "confidence_score":      1.0 if validation_result["passed"] else 0.5,
        "clarifying_questions":  []
    }


# ============================================================
# AGENT RUNNER
# ============================================================

def run_agent(requirement, process_flow, edge_cases,
              current_process, override_note="", max_retries=3) -> dict:
    intake_result = intake(requirement, process_flow, edge_cases, current_process)

    if not intake_result["is_clear"]:
        return output_node(requirement, current_process, {}, {"passed": False, "errors": []},
                           clarifying_questions=intake_result["clarifying_questions"])

    retry_note = override_note
    for attempt in range(max_retries):
        design_result     = design(requirement, process_flow, edge_cases,
                                   current_process, retry_note)
        validation_result = validate(design_result)
        if validation_result["passed"]:
            break
        error_summary = "; ".join(validation_result["errors"])
        retry_note = f"Previous attempt had these errors, fix them: {error_summary}"

    return output_node(requirement, current_process, design_result, validation_result)


def run_from_raw(raw_text: str) -> dict:
    extracted = extract_fields(raw_text)
    return run_agent(
        requirement=extracted["requirement"],
        process_flow=extracted["process_flow"],
        edge_cases=extracted["edge_cases"],
        current_process=extracted["current_process"]
    )


# ============================================================
# FORMAT OUTPUT FOR UI
# ============================================================

def format_output(result: dict) -> str:
    if result["validation_status"] == "needs_clarification":
        lines = ["## Clarification Needed\n",
                 "The requirement needs more detail before design can begin.\n",
                 "**Questions:**"]
        for i, q in enumerate(result["clarifying_questions"], 1):
            lines.append(f"{i}. {q}")
        return "\n".join(lines)

    lines = []

    lines.append(f"## Problem\n{result['problem']}\n")
    lines.append(f"## Current State\n{result['current_state']}\n")
    lines.append(f"## Proposed Solution\n{result['proposed_solution']}\n")

    lines.append("## Data Model\n")
    for obj in result["data_model"]:
        lines.append(f"### {obj['object_name']} `{obj['api_name']}`")
        lines.append("**Fields:**")
        for f in obj.get("fields", []):
            lines.append(f"- {f}")
        if obj.get("relationships"):
            lines.append("\n**Relationships:**")
            for r in obj.get("relationships", []):
                lines.append(f"- {r}")
        lines.append("")

    lines.append(f"## Validation\n`{result['validation_status']}`\n")

    if result.get("alternative_considered"):
        lines.append(f"## Alternative Considered\n{result['alternative_considered']}\n")

    lines.append(f"## Confidence Score\n`{result['confidence_score']}`")

    return "\n".join(lines)


# ============================================================
# GRADIO UI
# ============================================================

EXAMPLES = [
    # Example 1 — clean design output
    ["A hospital network needs to track patient referrals from primary care physicians to specialists. Referrals go from doctor to specialist office, patient gets scheduled, appointment completed, outcome recorded. Sometimes the specialist is unavailable and the referral needs to be redirected. Currently done via fax and phone calls with no visibility."],

    # Example 2 — triggers clarifying questions (intentionally vague)
    ["We need to track stuff for our team in Salesforce."],

    # Example 3 — forces validation retry (asks for many Master-Detail relationships)
    ["A manufacturing company needs to track equipment inspections across 12 facilities. Each piece of equipment must be a child of a facility, a child of a department, and a child of a compliance record simultaneously — all as required Master-Detail relationships so records cannot exist independently. Inspectors log results, schedule follow-ups, and record parts used. Currently on paper forms."],
]

custom_css = """
    .gradio-container { max-width: 100% !important; background: #0f172a !important; }
    body, html { background: #0f172a !important; }
    footer { display: none !important; }
    .question-input textarea {
        font-size: 16px !important;
        padding: 16px !important;
        background: #1e293b !important;
        border: 2px solid #334155 !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        min-height: 120px !important;
    }
    .question-input textarea:focus {
        border-color: #3b82f6 !important;
        outline: none !important;
    }
    #run-btn {
        max-width: 1120px !important;
        margin: 0 auto !important;
    }
    #run-btn button {
        background: #2563eb !important;
        border: none !important;
    }
    #run-btn button:hover {
        background: #1d4ed8 !important;
        opacity: 1 !important;
    }
    #question-input {
        max-width: 1120px !important;
        margin: 0 auto !important;
    }
    #example-pills table { border: none !important; background: transparent !important; }
    #example-pills td { border: none !important; background: transparent !important; padding: 4px !important; }
    #example-pills button {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 20px !important;
        color: #7dd3fc !important;
        font-size: 12px !important;
        padding: 6px 14px !important;
        cursor: pointer !important;
    }
    #example-pills button:hover {
        background: #1e3a5f !important;
        border-color: #3b82f6 !important;
    }
    .blue-btn { background: #2563eb !important; }
    .blue-btn:hover { background: #1d4ed8 !important; }
"""


def query(raw_text: str) -> str:
    if not raw_text.strip():
        return "*Please enter a requirement.*"
    result = run_from_raw(raw_text)
    return format_output(result)


with gr.Blocks(title="Salesforce Data Architect Agent — Devon Wright",
               css=custom_css) as demo:

    gr.HTML("""
    <div style="background:linear-gradient(160deg,#1e3a5f 0%,#0f172a 60%);
                border-bottom:1px solid #1e293b;padding:56px 80px 48px;">
      <div style="max-width:1280px;margin:0 auto;">
        <h1 style="font-family:-apple-system,sans-serif;font-size:clamp(28px,4vw,48px);
                   font-weight:800;color:#f8fafc;margin:0 0 12px;line-height:1.1;">
          Salesforce Data Architect<br>
          <span style="color:#3b82f6;">Agent</span></h1>
        <p style="font-family:-apple-system,sans-serif;color:#94a3b8;font-size:16px;
                  margin:0 0 20px;max-width:640px;line-height:1.65;">
          Describe a business need in plain language. The agent extracts the
          requirement, designs a Salesforce data model, validates every object
          name and field type, self-corrects if needed, and produces a
          structured design document.</p>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:24px;">
          <span style="background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.3);
                       color:#93c5fd;padding:3px 12px;border-radius:20px;
                       font-size:11px;font-family:monospace;">4-node agent</span>
          <span style="background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.3);
                       color:#93c5fd;padding:3px 12px;border-radius:20px;
                       font-size:11px;font-family:monospace;">on-demand tool calls</span>
          <span style="background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.3);
                       color:#93c5fd;padding:3px 12px;border-radius:20px;
                       font-size:11px;font-family:monospace;">self-correcting retry loop</span>
          <span style="background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.3);
                       color:#93c5fd;padding:3px 12px;border-radius:20px;
                       font-size:11px;font-family:monospace;">GPT-4o-mini</span>
        </div>
        <a href="https://www.linkedin.com/in/devwright/" target="_blank"
           style="color:#475569;font-size:13px;text-decoration:none;
                  border-bottom:1px solid #334155;padding-bottom:2px;">
           Built by Devon Wright</a>
      </div>
    </div>
    """)

    gr.HTML("""
    <div style="max-width:1280px;margin:0 auto;padding:40px 80px 0;">
      <div style="font-family:monospace;font-size:11px;letter-spacing:0.18em;
                  text-transform:uppercase;color:#475569;font-weight:700;
                  margin-bottom:12px;">Describe your business need</div>
    </div>""")

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                placeholder="e.g. We need to track patient referrals from primary care to specialists. Currently done over fax with no visibility...",
                lines=4,
                show_label=False,
                container=False,
                elem_classes=["question-input"],
                elem_id="question-input",
            )

    with gr.Row():
        with gr.Column():
            run_btn = gr.Button("⚡ Generate Data Model",
                    variant="primary", elem_id="run-btn",
                    elem_classes=["blue-btn"])

    gr.HTML("""
    <div style="max-width:1280px;margin:0 auto;padding:16px 80px 0;">
      <div style="font-family:monospace;font-size:11px;letter-spacing:0.18em;
                  text-transform:uppercase;color:#475569;font-weight:700;
                  margin-bottom:8px;">Example inputs</div>
    </div>""")

    gr.Examples(
        examples=EXAMPLES,
        inputs=question_input,
        label=None,
        elem_id="example-pills",
    )

    gr.HTML("""
    <div style="max-width:1280px;margin:0 auto;padding:32px 80px 0;">
      <div style="font-family:monospace;font-size:11px;letter-spacing:0.18em;
                  text-transform:uppercase;color:#475569;font-weight:700;
                  padding-top:24px;border-top:1px solid #1e293b;
                  margin-bottom:16px;">Design Output</div>
    </div>""")

    output_display = gr.Markdown(
        value="*Output will appear here after you submit a requirement.*"
    )

    gr.HTML("""
    <div style="background:#0f172a;border-top:1px solid #1e293b;
                padding:24px 80px;margin-top:40px;">
      <div style="max-width:1280px;margin:0 auto;">
        <div style="font-family:monospace;font-size:10px;letter-spacing:0.2em;
                    text-transform:uppercase;color:#475569;margin-bottom:14px;
                    font-weight:700;">How it works</div>
        <div style="display:flex;align-items:center;gap:8px;
                    flex-wrap:wrap;font-family:monospace;font-size:12px;">
          <span style="background:#1e293b;border:1px solid #334155;
                       padding:6px 14px;border-radius:6px;color:#7dd3fc;">
                       📝 plain text input</span>
          <span style="color:#334155;font-size:18px;">→</span>
          <span style="background:#1e293b;border:1px solid #334155;
                       padding:6px 14px;border-radius:6px;color:#7dd3fc;">
                       extract fields</span>
          <span style="color:#334155;font-size:18px;">→</span>
          <span style="background:#1e293b;border:1px solid #334155;
                       padding:6px 14px;border-radius:6px;color:#7dd3fc;">
                       intake</span>
          <span style="color:#334155;font-size:18px;">→</span>
          <span style="background:#1e293b;border:1px solid #334155;
                       padding:6px 14px;border-radius:6px;color:#7dd3fc;">
                       design + tool calls</span>
          <span style="color:#334155;font-size:18px;">→</span>
          <span style="background:#1e293b;border:1px solid #334155;
                       padding:6px 14px;border-radius:6px;color:#7dd3fc;">
                       validate</span>
          <span style="color:#334155;font-size:18px;">→</span>
          <span style="background:#0d1b35;border:1px solid #2563eb;
                       padding:6px 14px;border-radius:6px;color:#93c5fd;
                       font-weight:700;">design document</span>
        </div>
      </div>
    </div>""")

    run_btn.click(
        fn=query,
        inputs=question_input,
        outputs=output_display,
        show_progress="full",
    )

demo.launch()
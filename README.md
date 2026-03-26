# 🧪 QAForge — AI Test Case Generator

> **QAForge** is a Streamlit web application that automatically generates professional, structured QA test cases from a user story or feature description, using AI (Gemini or OpenAI).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A Google AI Studio API key (Gemini) **or** an OpenAI API key

### Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Configuration
On first launch, enter your API key in the sidebar. Choose between **Gemini** (recommended, free tier available) or **OpenAI**.

---

## 🗺️ How It Works — The 3-Phase Pipeline

QAForge generates test cases through a structured 3-phase process:

```
Phase 1 — Analysis        Phase 2 — Planning         Phase 3 — Generation
─────────────────────     ──────────────────────     ──────────────────────
User story input      →   Acceptance Criteria    →   Full test cases
+ Attachments             + Test Plan outline         (Markdown + JSON/CSV)
```

### Phase 1 — Story Analysis
The user pastes their **user story** (and optionally attaches files/screenshots). The AI analyzes the story and extracts:
- Functional requirements
- Edge cases and constraints
- Suggested acceptance criteria

### Phase 2 — Test Plan
The AI generates a **structured test plan** covering:
- Happy Path scenarios
- Negative / error cases
- Boundary Value Analysis (BVA)
- Equivalence Partitioning
- Security & Non-Functional tests

The user can **refine the plan** via chat before proceeding.

### Phase 3 — Test Case Generation
The AI generates **detailed test cases** for every scenario in the plan. Each test case includes:
- `ID` (TC-N), `Type`, `Priority`, `Automation` candidate flag
- `Preconditions`, numbered `Test Steps`, `Expected Result`, `Failure Signature`

Test cases are generated using an **auto-loop** that continues until all scenarios are covered, then displayed as a unified document.

---

## 📤 Export Formats

Once test cases are generated, you can export them on demand:

| Format | Content |
|--------|---------|
| **Markdown** | Full formatted test cases document |
| **Text** | Plain text version |
| **JSON** | Structured array of test case objects |
| **CSV** | Spreadsheet-ready format for test management tools |

> 💡 JSON & CSV are generated **on demand** (click "⚙️ Generate JSON & CSV exports") to save API quota.

---

## ⚙️ Function Reference

### LLM Providers

#### `call_gemini(messages, system_prompt, user_message, max_tokens)`
Sends a request to the **Google Gemini** API. Accepts a conversation history (`messages`), a system instruction, and the current user message. Returns the raw text response.

#### `call_openai(messages, system_prompt, user_message, max_tokens)`
Sends a request to the **OpenAI** API (GPT models). Same interface as `call_gemini`. Uses the `openai` Python SDK.

#### `call_llm(messages, system_prompt, user_message, max_tokens)`
**Router function** — dispatches to `call_gemini` or `call_openai` based on the provider selected in the sidebar. All phases use this function for text generation.

#### `call_llm_structured(system_prompt, user_message, max_tokens)`
Calls the LLM with a **JSON schema** response constraint, forcing the model to return a structured array of test case objects. Used for JSON/CSV export generation.

---

### Generation Logic

#### `generate_until_complete(system_prompt, history, initial_prompt, max_iterations, max_tokens)`
**Core auto-loop function.** Calls the LLM iteratively until it detects the `[[GENERATION_COMPLETE]]` signal token at the end of the response — meaning all test cases have been generated. Concatenates all partial responses into a single clean document.
- `max_iterations=2` by default to respect API rate limits
- Adds a `2s` pause between iterations to avoid RPM throttling
- Returns the full merged markdown + updated message history

#### `generate_test_cases_in_batches(system_prompt, plan_ctx, scenario_titles, batch_size)`
Used when the test plan contains **more than 6 scenarios**. Splits the scenario list into batches of `batch_size` (default: 6) and calls `generate_until_complete` for each batch. Shows a **progress bar** during generation. Maintains continuous TC numbering across batches (`TC-1…TC-6`, `TC-7…TC-12`, etc.).

#### `extract_scenario_titles(plan_text)`
Parses the Phase 2 test plan text using regex to extract individual **scenario titles** (lines matching `- TC: ...`). Used to determine whether to use single-call or batch mode. Falls back to any bullet-point line if no `TC:` markers are found.

---

### Utilities

#### `extract_text(file)`
Extracts plain text content from uploaded files. Supports `.txt`, `.pdf`, `.md`, `.json`, `.csv`, and image files. Used to enrich the Phase 1 context with attachment content.

#### `is_image(file)`
Returns `True` if the uploaded file is an image (`png`, `jpg`, `jpeg`, `gif`, `webp`). Used to route image files to vision-capable LLM calls.

#### `file_icon(filename)`
Returns an emoji icon based on file extension (e.g., 📄 for PDF, 🖼️ for images). Used for display in the attachments preview section.

#### `handle_error(e)`
Centralized error handler. Catches API errors (rate limits, auth failures, network issues) and displays a user-friendly `st.error()` message with actionable guidance (e.g., "Check your API key", "Rate limit reached — wait 60s").

#### `build_csv(test_cases)`
Converts the structured list of test case objects (from `call_llm_structured`) into a **CSV string** using Python's `csv` module. Each row represents one test case with all fields as columns.

#### `render_chat(messages, key_prefix)`
Renders a **chat history** using `st.chat_message()` for each message in the list. Used in all three phases to display the conversation between the user and the AI.

#### `render_tab_bar(tabs, active_key)`
Renders a **horizontal tab navigation bar** using Streamlit buttons styled as tabs. Used in Phase 3 to switch between the Markdown view and the individual TC detail view.

---

## 🔑 API Rate Limits

QAForge is optimized to minimize API calls:

| Action | API calls |
|--------|-----------|
| Phase 1 analysis | 1 call |
| Phase 2 plan generation | 1 call |
| Phase 3 TC generation (≤6 scenarios) | 1–2 calls |
| Phase 3 TC generation (>6 scenarios) | 1–2 calls × N batches |
| JSON/CSV export (on demand) | 1 call |

> ⚠️ **Gemini free tier**: 5 RPM (requests/minute). If you hit the limit, wait 60 seconds before retrying.

---

## 🛠️ Tech Stack

| Library | Usage |
|---------|-------|
| `streamlit` | Web UI framework |
| `google-generativeai` | Gemini API client |
| `openai` | OpenAI API client |
| `pandas` | CSV data handling |
| `Pillow` | Image file processing |

---

## 📁 Project Structure

```
app.py              # Main application (all logic + UI)
requirements.txt    # Python dependencies
README.md           # This file
```

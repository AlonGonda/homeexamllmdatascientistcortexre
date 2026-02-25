# Cortex RE · Asset Manager AI

A production-grade multi-agent real estate asset management assistant built with **LangGraph**, **LangChain (Anthropic / Claude)**, and **Streamlit**. It answers natural-language queries about a real property portfolio backed by live ledger data — with strict guarantees against hallucination and silent data substitution.

---

## 📐 Architecture

### 7-Node LangGraph Pipeline

```
User Query (Streamlit)
        │
        ▼
  ┌─────────────┐
  │   Router    │  ← Hybrid: deterministic regex + LLM classification
  └──────┬──────┘
         │
  ┌──────▼────────────────────────────────────────────────────────────┐
  │  Intent?                                                          │
  │  COMPARISON / PL_REPORT / DETAILS / TENANT → [DataRetriever]      │
  │                                                 │                 │
  │                                           [Reasoner] (CoT)       │
  │                                                 │                 │
  │                                           [Analyst] (few-shot)   │
  │                                                 │                 │
  │                                           [Verifier] (self-refl) │
  │                                                 │                 │
  │  GENERAL ─────────────────────────────────► [Analyst]             │
  │  CLARIFY ─────────────────────────────────► [Clarifier]           │
  │  OUT_OF_SCOPE / error ────────────────────► [ErrorHandler]        │
  └───────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                    Final Response
                   (Streamlit chat)
```

### Advanced GenAI Patterns

| # | Pattern | Description |
|---|---------|-------------|
| 1 | **Hybrid Deterministic + LLM Routing** | Regex extracts parameters (year, properties, tenants) with 100% reliability. LLM only classifies intent and relation. A deterministic correction layer overrides misclassifications. |
| 2 | **Chain-of-Thought (CoT) Reasoning** | Dedicated `Reasoner` node performs 4-step structured reasoning (understand → extract → calculate → conclude) before the Analyst generates a response. |
| 3 | **Self-Reflection / Verification** | `Verifier` node checks the Analyst's output against source data for numerical accuracy, year correctness, and ranking consistency. Can trigger one auto-retry. |
| 4 | **Few-Shot Dynamic Prompting** | Intent-specific response examples are injected into the Analyst prompt (e.g. P&L table format, comparison layout). |
| 5 | **Confidence Scoring** | Each response is tagged with a confidence badge (🟢 ≥ 85%, 🟡 ≥ 60%, 🔴 < 60%) from the Verifier. |
| 6 | **History Isolation** | Standard reports receive a "clean slate" (no chat history) to prevent year/entity anchoring. Advisory queries receive full history for sequential reasoning. |

### Type System

The codebase uses production-grade Python type abstractions:

| Type | Kind | Purpose |
|------|------|---------|
| `Intent` | `Enum` | Type-safe intent constants (`PL_REPORT`, `COMPARISON`, etc.) |
| `Relation` | `Enum` | Conversation relation types (`FOLLOW_UP`, `NEW_TOPIC`, `INTERRUPT`) |
| `ExtractionResult` | `@dataclass(frozen=True)` | Immutable regex extraction output |
| `ClassificationResult` | `@dataclass` | LLM classification output (intent + relation) |
| `ActiveContext` | `@dataclass` | Persistent conversational context across turns |

### Agent Responsibilities

| Agent | Role |
|---|---|
| **Router** | 4-phase pipeline: (1) regex extraction, (2) LLM classification, (3) deterministic correction, (4) state-machine context merge. Uses `ContextManager` class for all parameter logic. |
| **DataRetriever** | Queries `data_manager.py` and assembles the exact data slice. Returns errors if entities or years are not in the dataset. |
| **Reasoner** | Chain-of-Thought node — performs structured 4-step reasoning on data before response generation. |
| **Analyst** | Formats retrieved data into professional markdown with CoT trace consumption and few-shot examples. |
| **Verifier** | Self-reflection node — checks response accuracy against source data. Assigns confidence scores and can trigger auto-retry. |
| **Clarifier** | Asks targeted follow-up questions for ambiguous or incomplete queries. |
| **ErrorHandler** | Provides actionable responses for all failure modes (missing property, unavailable year, out-of-scope, system errors). |

---

## 🔀 LangGraph State Machine

### State Definition

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    intent: str
    properties: List[str]
    tenant: Optional[str]
    year: Optional[str]
    raw_data: Dict
    final_output: Optional[str]
    error: Optional[str]
    clarification_needed: Optional[str]
    active_entities: Dict
    # Advanced GenAI fields
    reasoning_trace: Optional[str]      # CoT output from Reasoner
    confidence_score: Optional[float]   # 0.0–1.0 from Verifier
    verification_status: Optional[str]  # PASS | FAIL | RETRY
    retry_count: int                    # Max 1 retry
```

### Graph Edges

```python
# Router branching
router → data_retriever  (COMPARISON, PL_REPORT, DETAILS, TENANT)
router → analyst          (GENERAL)
router → clarifier        (CLARIFY)
router → error_handler    (OUT_OF_SCOPE, error)

# Data pipeline with CoT + Verification
data_retriever → reasoner → analyst → verifier
verifier → END            (PASS)
verifier → analyst         (RETRY — max 1)
```

### Intent Routing Table

| User says | Intent | Route | Data fetched |
|---|---|---|---|
| "Compare Building 17 and Building 120" | `COMPARISON` | retrieve → reason → analyse → verify | `compare_properties()` |
| "P&L for Building 17 in 2024" | `PL_REPORT` | retrieve → reason → analyse → verify | `get_property_pl()` |
| "Total P&L" | `PL_REPORT` | retrieve → reason → analyse → verify | `get_total_pl()` |
| "Details on Building 17" | `DETAILS` | retrieve → reason → analyse → verify | `get_property_details()` |
| "Tenant 12 revenue" | `TENANT` | retrieve → reason → analyse → verify | `get_tenant_details()` |
| "Which property is the best?" | `PL_REPORT` | retrieve → reason → analyse → verify | `get_total_pl()` (advisory) |
| "What is cap rate?" | `GENERAL` | analyst (direct) | None (LLM knowledge) |
| "Compare" (no properties) | `CLARIFY` | clarifier → END | None |
| "Who is Leo Messi?" | `OUT_OF_SCOPE` | error_handler → END | None |

---

## 🧠 Context Manager

The `ContextManager` class handles all parameter extraction, intent correction, and context merging with **zero LLM dependency**:

```python
class ContextManager:
    extract()  → ExtractionResult     # Regex-based parameter extraction
    correct()  → ClassificationResult # Deterministic intent override rules
    merge()    → ActiveContext         # State-machine context transitions
```

### Extraction Strategy
- **Year**: `\b20\d{2}\b` regex anchored to known years
- **Properties / Tenants**: Word-boundary regex (`\b`), sorted longest-first to avoid substring collisions (e.g. "Tenant 1" inside "Tenant 12")
- **Portfolio**: Keyword detection against `PORTFOLIO_KEYWORDS` frozenset

### Correction Rules (Phase 2.5)
| Rule | Trigger | Action |
|------|---------|--------|
| 1 | Advisory language (`best`, `worst`, `sell`, ...) + COMPARISON/DETAILS | → `PL_REPORT` |
| 2 | Portfolio keyword + COMPARISON | → `PL_REPORT` |
| 3 | Regex-detected tenant + non-TENANT intent | → `TENANT` |
| 4 | Domain entities found + OUT_OF_SCOPE | → `TENANT` or `PL_REPORT` |

### Context Merge (Phase 3)
| Relation | Behaviour |
|----------|-----------|
| `INTERRUPT` | Freeze previous context, redirect to ErrorHandler |
| `FOLLOW_UP` | Overlay new params onto previous (regex wins if present) |
| `NEW_TOPIC` | Hard reset to current extraction |

---

## 📁 Project Structure

```
cortex-re/
├── app.py             # Streamlit UI – dark mode, KPI cards, quick actions
├── agents_logic.py    # LangGraph 7-node state machine + ContextManager
├── data_manager.py    # Parquet loading, aggregation helpers
├── cortex.parquet     # Real ledger dataset (3,924 rows × 12 columns)
├── requirements.txt   # Pinned dependencies
└── README.md
```

---

## 📊 Dataset

`cortex.parquet` – PropCo real estate ledger (3,924 rows)

| Column | Type | Description |
|---|---|---|
| `entity_name` | str | Always "PropCo" |
| `property_name` | str | 5 unique properties (Building 17, 120, 140, 160, 180) |
| `tenant_name` | str | 18 unique tenants |
| `ledger_type` | str | `revenue` or `expenses` |
| `ledger_group` | str | 5 groups (rental_income, management_fees, …) |
| `ledger_category` | str | 29 sub-categories |
| `ledger_code` | int | Numeric ledger code |
| `ledger_description` | str | Human-readable label |
| `month` | str | e.g. `2025-M01` |
| `quarter` | str | e.g. `2025-Q1` |
| `year` | str | `2024` or `2025` |
| `profit` | float | Positive = income, negative = expense |

---

## 🚀 Setup

### Option A · Conda (recommended)

```bash
# 1. Create and activate environment
conda env create -f environment.yml

# 2. Run the app
conda run -n cortex-re python -m streamlit run app.py
```

### Option B · pip

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

### 3 · Enter your Anthropic API key

Paste your `sk-ant-...` key into the sidebar. Select a Claude model (Haiku is fastest and most accessible). The agents will initialise and be ready immediately.

---

## 💬 Example Queries

| Type | Example |
|---|---|
| All-time portfolio P&L | *"What is the total P&L?"* |
| Portfolio P&L by year | *"What is the total P&L for all properties in 2024?"* |
| Property P&L | *"Show me the profit and loss for Building 17"* |
| Comparison | *"Compare Building 17 and Building 120 in 2025"* |
| Compare all | *"Compare all my properties"* |
| Property details | *"Give me full details on Building 17"* |
| Tenant info | *"How much revenue does Tenant 12 generate?"* |
| Advisory (best) | *"Which property is the best earner?"* |
| Advisory (sell) | *"Which building should I sell?"* |
| Sequential | *"Which one follows?"* (after a ranking query) |
| General knowledge | *"What is a good cap rate for commercial real estate?"* |
| Self-introduction | *"Who are you?"* → assistant introduces itself |
| Missing property | *"P&L for 999 Fake Street"* → error + list of real properties |
| Missing year | *"P&L for 2023"* → error listing available years |
| Out-of-scope | *"Who is Leo Messi?"* → professional scope redirect |
| Vague comparison | *"Compare"* → clarification asking which properties |

---

## ⚙️ Design Decisions

### No Silent Substitutions
The system **never substitutes** data silently. If a property name doesn't exactly match, a year has no data, or an intent is ambiguous, the agent asks for clarification rather than guessing.

### Hybrid Routing (Deterministic + LLM)
Parameter extraction is fully deterministic (regex). The LLM only classifies intent and relation. A deterministic correction layer (`ContextManager.correct()`) overrides known misclassification patterns. This eliminates the year-switching bug and entity confusion.

### Chain-of-Thought Before Generation
The Reasoner node forces structured analysis (understand → extract → calculate → conclude) before the Analyst writes. This grounds responses in verified calculations rather than pattern-matching.

### Self-Reflection Loop
The Verifier checks every data-driven response against source data. If accuracy is low, it triggers one auto-retry. This implements the Reflection pattern from modern agentic AI research.

### Anti-Hallucination Guards
1. **Router** validates all property/tenant names against the dataset
2. **ContextManager** uses word-boundary regex to prevent substring collisions
3. **DataRetriever** validates years and required parameters before any data fetch
4. **Analyst** system prompt includes `CRITICAL: ONLY use figures from retrieved JSON`
5. **Analyst** refuses to run if `raw_data` is empty for non-GENERAL intents
6. **Verifier** catches fabricated numbers or wrong years post-generation

---

## 🧗 Challenges & Solutions

| Challenge | Solution |
|---|---|
| LLM invents property names | Router validates against real dataset; rejects with suggestions |
| "Tenant 1" matches inside "Tenant 12" | Word-boundary regex (`\b`) + longest-first sorting |
| "which is best?" classified as COMPARISON | Deterministic advisory detection in Phase 2.5 → PL_REPORT |
| Year from history bleeds into new query | Deterministic regex extraction overrides LLM; context merge state machine |
| LLM confused by long conversation history | Router sends only last 2 messages to LLM for classification |
| Analyst fabricates numbers | CoT reasoning pre-generates verified calculations; Verifier double-checks |
| Inconsistent response quality | Few-shot examples ensure consistent formatting per intent |
| `$` signs render as LaTeX in Streamlit | Post-process responses to escape `$` before `st.markdown()` |
| LLM wraps JSON in markdown fences | Regex fallback extracts the JSON object regardless |

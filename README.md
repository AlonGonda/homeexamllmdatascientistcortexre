# Cortex RE · Asset Manager AI

A production-grade multi-agent real estate asset management assistant built with **LangGraph**, **LangChain (Anthropic / Claude)**, and **Streamlit**. It answers natural-language queries about a real property portfolio backed by live ledger data — with strict guarantees against hallucination and silent data substitution.

---

## 📐 Architecture

```
User Query (Streamlit)
        │
        ▼
  ┌─────────────┐
  │   Router    │  ← Claude classifies intent + validates entities (JSON prompt)
  └──────┬──────┘
         │
  ┌──────▼─────────────────────────────────────────┐
  │  Intent?                                       │
  │  COMPARISON / PL_REPORT / DETAILS / TENANT ───►│ [DataRetriever] ──► [Analyst]
  │  GENERAL ─────────────────────────────────────►│ [Analyst] (no data fetch)
  │  CLARIFY ─────────────────────────────────────►│ [Clarifier] (asks follow-up)
  │  error / unknown ─────────────────────────────►│ [ErrorHandler]
  └────────────────────────────────────────────────┘
                          │
                          ▼
                    Final Response
                   (Streamlit chat)
```

### Agent Responsibilities

| Agent | Role |
|---|---|
| **Router** | Classifies intent (`COMPARISON`, `PL_REPORT`, `DETAILS`, `TENANT`, `GENERAL`, `CLARIFY`) and extracts property names, tenant, and year via a structured JSON prompt. Only accepts exact matches — no guessing. |
| **DataRetriever** | Queries `data_manager.py` and assembles the exact data slice requested. Returns errors if entities or years are not in the dataset. |
| **Analyst** | Formats retrieved data into professional markdown. Opens with a direct one-sentence answer. Uses **only** figures from `raw_data` — never estimates or invents numbers. |
| **Clarifier** | Asks a targeted follow-up when the query is ambiguous, incomplete, or the requested data doesn't exist. Prevents silent substitution. |
| **ErrorHandler** | Provides specific, actionable responses for all failure modes (missing property, unavailable year, unsupported intent, system errors). |

---

## 🔀 LangGraph Workflow

### State Machine

```
                    ┌──────────────────────────────┐
                    │       AgentState (TypedDict)  │
                    │  - messages: List[BaseMessage]│
                    │  - intent: str                │
                    │  - properties: List[str]      │
                    │  - tenant: Optional[str]      │
                    │  - year: Optional[str]        │
                    │  - raw_data: Dict             │
                    │  - final_output: str          │
                    │  - error: Optional[str]       │
                    │  - clarification_needed: str  │
                    └──────────────────────────────┘

  User Input
      │
      ▼
 ┌─────────┐   route_decision()
 │  Router │ ─────────────────────────────────────────────────────┐
 └─────────┘                                                      │
      │                                                           │
  ┌───┴──────┬───────────────┬──────────────────┐                │
  │ retrieve │    general    │     clarify       │    end (error) │
  ▼          ▼               ▼                  │                ▼
┌──────────────┐  ┌──────────┐  ┌───────────┐  │        ┌─────────────┐
│DataRetriever │  │ Analyst  │  │ Clarifier │  │        │ErrorHandler │
└──────┬───────┘  └────┬─────┘  └─────┬─────┘  │        └──────┬──────┘
       │               │              │         │               │
  retriever_decision() │              │         │               │
  ┌────┴────┐          │              │         │               │
  │  ok     │ error    │              │         │               │
  ▼         ▼          │              │         │               │
┌───────┐ ┌──────────┐ │              │         │               │
│Analyst│ │ErrorHandl│ │              │         │               │
└───┬───┘ └────┬─────┘ │              │         │               │
    │          │       │              │         │               │
    └──────────┴───────┴──────────────┴─────────┴───────────────┘
                                      │
                                     END
```

### Intent Routing Table

| User says | Intent | Route | Data fetched |
|---|---|---|---|
| "Compare Building 17 and Building 120" | `COMPARISON` | retrieve → analyst | `compare_properties()` |
| "P&L for Building 17 in 2024" | `PL_REPORT` | retrieve → analyst | `get_property_pl()` |
| "Total P&L" | `PL_REPORT` | retrieve → analyst | `get_total_pl()` |
| "Details on Building 17" | `DETAILS` | retrieve → analyst | `get_property_details()` |
| "Tenant 12 revenue" | `TENANT` | retrieve → analyst | `get_tenant_details()` |
| "What is cap rate?" | `GENERAL` | general → analyst | None (LLM knowledge) |
| "Compare" (no properties) | `CLARIFY` | clarify → END | None |
| Property not in dataset | error | end → ErrorHandler | None |
| Year not in dataset | error | retrieve → end → ErrorHandler | None |

### State Flow

LangGraph passes a single **immutable state dict** between nodes. Each node returns only the keys it modifies — LangGraph merges them back into the accumulated state. The `messages` key uses `operator.add` as its reducer, allowing append-only message history.

```python
# Conditional edge: router → next node
graph.add_conditional_edges(
    "router",
    route_decision,          # function inspects state["intent"] and state["error"]
    {"retrieve": "data_retriever", "general": "analyst",
     "clarify": "clarifier", "end": "error_handler"}
)
```

---

## 📁 Project Structure

```
cortex-re/
├── app.py             # Streamlit UI – dark mode, KPI cards, quick actions
├── agents_logic.py    # LangGraph state machine & all agent definitions
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

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Run the app

```bash
python -m streamlit run app.py
```

### 3 · Enter your Anthropic API key

Paste your `sk-ant-...` key into the sidebar. Select a Claude model (Haiku is fastest). The agents will initialise and be ready immediately.

---

## 💬 Example Queries

| Type | Example |
|---|---|
| Portfolio P&L | *"What is the total P&L for all properties in 2024?"* |
| Property P&L | *"Show me the profit and loss for Building 17"* |
| Comparison | *"Compare Building 17 and Building 120 in 2025"* |
| Compare all | *"Compare all my properties"* |
| Property details | *"Give me full details on Building 17"* |
| Tenant info | *"How much revenue does Tenant 12 generate?"* |
| General | *"What is a good cap rate for commercial real estate?"* |
| Missing property | *"P&L for 999 Fake Street"* → error listing real properties |
| Missing year | *"P&L for 2023"* → error + clarification |

---

## ⚙️ Design Decisions

### No Silent Substitutions
The system **never substitutes** data silently. If a property name doesn't exactly match, a year has no data, or an intent is ambiguous, the agent asks for clarification rather than guessing. This is a core design principle applied at every stage:
- Router: exact case-insensitive property matching only; fuzzy match surfaced as a suggestion, not silently applied
- DataRetriever: year not in dataset → clarification error
- DataRetriever: COMPARISON requires ≥ 2 explicitly named properties
- DataRetriever: TENANT requires exact tenant name
- Analyst: only uses figures present in retrieved JSON — refuses to generate output if `raw_data` is empty

### Why LangGraph?
LangGraph's state machine enforces clean separation of concerns — each agent owns a single responsibility, routing logic is explicit and inspectable, and new nodes (e.g. `Clarifier`) can be added independently.

### Why `data_manager.py`?
All pandas/parquet logic is isolated in one module, making it trivial to swap the data source (SQL, REST API) without touching agent code. `@lru_cache` ensures the parquet is read only once per session.

### Structured JSON routing
A JSON-only LLM prompt for the Router ensures reliable structured extraction with a regex fallback for edge cases where the LLM adds markdown fences.

### Anti-Hallucination Guards
1. **Router** validates all property names — invented names are rejected immediately
2. **DataRetriever** validates years and required parameters before any data fetch
3. **Analyst** system prompt includes an explicit `CRITICAL: ONLY use figures from retrieved JSON` instruction
4. **Analyst** refuses to run if `raw_data` is empty for non-GENERAL intents

---

## 🧗 Challenges & Solutions

| Challenge | Solution |
|---|---|
| LLM invents plausible-sounding property names | Router validates against real dataset; invented names rejected with suggestions |
| User asks "this year" (2026) but data only goes to 2025 | DataRetriever returns a clear error with available years |
| Ambiguous comparisons ("compare my properties") | CLARIFY intent routes to Clarifier node which asks which properties |
| `$` signs in markdown render as LaTeX in Streamlit | Post-process responses to escape `$` before `st.markdown()` |
| LLM wraps JSON in markdown fences | Regex fallback extracts the JSON object regardless |

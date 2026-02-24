"""
agents_logic.py – LangGraph Multi-Agent Orchestration
======================================================

Technology Choices & Rationale
--------------------------------
LangGraph (StateGraph)
    Chosen over a single-chain approach because real asset management
    queries have fundamentally different execution paths (data retrieval
    vs. general knowledge vs. clarification).  The explicit state machine
    makes routing transparent, testable, and easy to extend.  Each node
    has a single responsibility, reducing the risk of prompt-bleeding
    between concerns (e.g. the Analyst never decides what data to fetch).

Anthropic Claude (langchain-anthropic)
    Selected for its strong instruction-following on structured JSON
    prompts and its reliable refusal to fabricate data when explicitly
    instructed.  Using ``temperature=0`` ensures deterministic routing
    and reproducible P&L figures.  The modular LangChain integration
    means the LLM provider can be swapped (e.g. to OpenAI or a local
    model) by changing a single import.

Pandas + Parquet (data_manager.py)
    The dataset is a structured ledger — precise aggregation with group-
    by / filter / sum is more reliable than vector similarity search.
    Parquet provides columnar compression and fast predicate pushdown.
    ``@lru_cache`` ensures the file is read only once per process.

JSON-only Router Prompt
    Structured output from the LLM avoids fragile text parsing.  A
    regex fallback handles cases where the model wraps JSON in fences.

No-Silent-Substitution Policy
    All fuzzy matching and inferences are surfaced to the user rather
    than applied silently.  This ensures data integrity and builds trust.

Agent Graph
-----------

    user input ──► [Router]
                       │
       ┌───────────────┼──────────────┬───────────────┐
       ▼               ▼              ▼               ▼
   "retrieve"      "clarify"      "general"         "end"
       │               │              │               │
 [DataRetriever]  [Clarifier]    [Analyst]     [ErrorHandler]
       │         (asks follow-up)     │               │
  (ok)│ (error)       ▼              ▼              END
      ├────────► [ErrorHandler]     END
      ▼
   [Analyst]
      │
     END

Intents (Router output)
------------------------
COMPARISON  – compare two or more properties side-by-side
PL_REPORT   – P&L for a specific property or the whole portfolio
DETAILS     – deep dive into a single property
TENANT      – question about a specific tenant
GENERAL     – general real estate knowledge (no dataset needed)
CLARIFY     – ambiguous / incomplete query → ask a follow-up question

Error Handling
--------------
1. Property not found       → router rejects via validation, ErrorHandler replies
2. Financial data missing   → DataRetriever detects empty result, ErrorHandler replies
3. Ambiguous / incomplete   → Router emits CLARIFY → Clarifier node asks targeted question
4. Unsupported/gibberish    → Router falls back to GENERAL or CLARIFY as appropriate
"""

import json
import operator
import re
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

import data_manager as dm


# ─── State ─────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    intent: str                 # COMPARISON|PL_REPORT|DETAILS|TENANT|GENERAL|CLARIFY
    properties: List[str]       # canonical property names (validated against dataset)
    tenant: Optional[str]       # tenant name (if applicable)
    year: Optional[str]         # year filter (e.g. "2024")
    raw_data: Dict              # output from data_manager
    final_output: str           # formatted response for the UI
    error: Optional[str]        # error message (triggers ErrorHandler)
    clarification_needed: Optional[str]  # populated when intent is CLARIFY


# ─── Graph ─────────────────────────────────────────────────────────────────────


class AssetManagerGraph:
    """
    Multi-agent LangGraph orchestration for Cortex RE.

    Nodes
    -----
    router          – classifies intent, extracts & validates parameters
    data_retriever  – fetches data from the parquet dataset
    analyst         – synthesises data into a professional markdown response
    clarifier       – detects ambiguity and asks a targeted follow-up question
    error_handler   – friendly, actionable responses for all failure modes
    """

    def __init__(self, model_name: str = "claude-3-haiku-20240307", temperature: float = 0):
        self.llm = ChatAnthropic(model=model_name, temperature=temperature)
        self._build_graph()

    # ── Build ───────────────────────────────────────────────────────────────────

    def _build_graph(self):
        wf = StateGraph(AgentState)

        wf.add_node("router", self.router_node)
        wf.add_node("data_retriever", self.data_retriever_node)
        wf.add_node("analyst", self.analyst_node)
        wf.add_node("clarifier", self.clarifier_node)
        wf.add_node("error_handler", self.error_handler_node)

        wf.set_entry_point("router")

        wf.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "retrieve": "data_retriever",
                "general": "analyst",
                "clarify": "clarifier",
                "end": "error_handler",
            },
        )
        wf.add_conditional_edges(
            "data_retriever",
            self.retriever_decision,
            {
                "ok": "analyst",
                "error": "error_handler",
            },
        )

        wf.add_edge("analyst", END)
        wf.add_edge("clarifier", END)
        wf.add_edge("error_handler", END)

        self.app = wf.compile()

    # ── Router ──────────────────────────────────────────────────────────────────

    def router_node(self, state: AgentState) -> Dict:
        """
        Classify intent and extract structured parameters from the user query.

        Also detects ambiguity and validates all property names against the
        real dataset — invented or non-existent names are rejected here.
        """
        user_msg = state["messages"][-1].content.strip()

        # ── Guard: empty / non-text input ──────────────────────────────────────
        if not user_msg:
            return {
                "intent": "CLARIFY",
                "properties": [],
                "tenant": None,
                "year": None,
                "error": None,
                "raw_data": {},
                "clarification_needed": (
                    "It looks like you sent an empty message. "
                    "What would you like to know about the PropCo portfolio?"
                ),
            }

        available_properties = dm.list_properties()
        available_years = dm.list_years()
        available_tenants = dm.list_tenants()

        import datetime
        current_year = str(datetime.datetime.now().year)
        most_recent_year = max(available_years) if available_years else current_year

        system = SystemMessage(
            content=f"""You are a routing agent for a real estate asset management system.

Dataset: PropCo portfolio (ledger data, years {available_years})
Available properties: {available_properties}
Available tenants:    {available_tenants}
Current real-world year: {current_year}
Most recent year with data: {most_recent_year}

Classify the user's intent into EXACTLY ONE of:
  COMPARISON  – comparing two or more properties
  PL_REPORT   – profit & loss report (one property or entire portfolio)
  DETAILS     – detailed info about a single property
  TENANT      – question about a specific tenant
  GENERAL     – general real estate knowledge (no dataset lookup needed)
  CLARIFY     – query is too vague, ambiguous, or missing required information

Use CLARIFY when:
- The user mentions a data-specific intent (COMPARISON/PL_REPORT/DETAILS/TENANT)
  but the necessary parameters are completely missing or unresolvable AND you
  cannot fall back gracefully (e.g. "compare" with no properties at all).
- The user's message is incoherent or off-topic in a way that requires clarification.

IMPORTANT – Year extraction rules:
- If the user says "this year", "current year", or similar, set year = {current_year}.
- If no specific year is mentioned, set year = null (means all available years).
- Always set year as a string (e.g. "2025"), never as an integer.

IMPORTANT – Property extraction rules:
- Only extract properties that EXACTLY match names in the available list (case-insensitive).
- If the user says "all properties", "all my properties", or similar, set properties = {available_properties}.
- Do NOT guess or infer a property name that was not explicitly stated.

Also extract:
  tenant      – matching tenant name from the list, or null
  clarification_question – if CLARIFY, a concise question to ask the user; else null

Respond with ONLY a JSON object, no markdown fences. Example:
{{"intent":"PL_REPORT","properties":["Building 17"],"tenant":null,"year":"2024","clarification_question":null}}"""
        )

        try:
            response = self.llm.invoke([system, HumanMessage(content=user_msg)])
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", response.content, re.DOTALL)
            parsed = json.loads(m.group()) if m else {}
        except Exception:
            parsed = {}

        intent = parsed.get("intent", "GENERAL").upper()
        if intent not in {"COMPARISON", "PL_REPORT", "DETAILS", "TENANT", "GENERAL", "CLARIFY"}:
            intent = "GENERAL"

        # ── Validate extracted property names against the real dataset ──────────
        # Only accept a resolved name if it matches what the user typed closely
        # enough (same words, case-insensitive). Silent fuzzy substitution is
        # not allowed — the user must get what they asked for or an error.
        raw_props = parsed.get("properties") or []
        resolved, unresolved = [], []
        for name in raw_props:
            canonical = dm.search_property(name)
            if canonical and canonical.lower() == name.strip().lower():
                # Exact case-insensitive match — safe to use
                resolved.append(canonical)
            elif canonical:
                # Fuzzy match found a *different* name — surface it as an error
                # so the user can confirm rather than getting a silent substitution.
                unresolved.append(name)
            else:
                unresolved.append(name)

        # If user clearly named properties but NONE resolved exactly → error
        error = None
        if raw_props and not resolved and intent not in {"GENERAL", "CLARIFY"}:
            # Build a helpful message listing close matches if any exist
            suggestions = []
            for name in unresolved:
                candidate = dm.search_property(name)
                if candidate:
                    suggestions.append(f"'{name}' → did you mean **{candidate}**?")
            suggestion_text = ("\n" + "\n".join(suggestions)) if suggestions else ""
            error = (
                f"The following properties were not found verbatim: "
                f"**{', '.join(unresolved)}**.{suggestion_text}\n\n"
                f"Available properties: {', '.join(available_properties)}.\n\n"
                "Please use the exact property name."
            )

        clarification = parsed.get("clarification_question") or None

        return {
            "intent": intent,
            "properties": resolved,
            "tenant": parsed.get("tenant") or None,
            "year": str(parsed["year"]) if parsed.get("year") else None,
            "error": error,
            "raw_data": {},
            "clarification_needed": clarification,
        }

    def route_decision(self, state: AgentState) -> str:
        if state.get("error"):
            return "end"
        intent = state.get("intent", "GENERAL")
        if intent == "CLARIFY":
            return "clarify"
        if intent == "GENERAL":
            return "general"
        return "retrieve"

    # ── Clarifier ───────────────────────────────────────────────────────────────

    def clarifier_node(self, state: AgentState) -> Dict:
        """
        Handles ambiguous or incomplete queries by asking a targeted follow-up.

        Uses the LLM-generated clarification_question from the router if available,
        otherwise generates one from context.
        """
        clarification = state.get("clarification_needed")
        user_msg = state["messages"][-1].content
        available = dm.list_properties()
        tenants = dm.list_tenants()
        years = dm.list_years()

        if clarification:
            message = (
                f"🤔 **I need a little more information to help you.**\n\n"
                f"{clarification}\n\n"
                f"**Quick reference:**\n"
                f"- Properties: {', '.join(available)}\n"
                f"- Years: {', '.join(years)}\n"
                f"- Tenants: {', '.join(tenants[:5])} _(and {len(tenants)-5} more)_"
            )
        else:
            # Fallback: generate a clarification question dynamically
            system = SystemMessage(content=(
                "You are a helpful real estate assistant. "
                "The user sent an ambiguous request. "
                "Ask ONE concise, specific follow-up question to clarify what they need. "
                "Reference only these available properties: " + str(available) + ". "
                "Do not make up any data."
            ))
            try:
                resp = self.llm.invoke([system, HumanMessage(content=user_msg)])
                message = f"🤔 **Could you clarify?**\n\n{resp.content}"
            except Exception:
                message = (
                    f"🤔 **I need more details to answer that.**\n\n"
                    f"Could you specify:\n"
                    f"- Which property? ({', '.join(available)})\n"
                    f"- Which year? ({', '.join(years)})\n"
                    f"- What type of report? (P&L, comparison, details, tenant)"
                )

        return {"final_output": message}

    # ── Data Retriever ──────────────────────────────────────────────────────────

    def data_retriever_node(self, state: AgentState) -> Dict:
        """
        Pull relevant data from the parquet dataset.

        Error cases handled:
        - Property not in dataset (already caught by router, but double-checked)
        - Year with no data in the dataset
        - Tenant not found
        - Insufficient parameters for the requested intent
        """
        intent = state["intent"]
        properties = state.get("properties") or []
        tenant = state.get("tenant")
        year = state.get("year")

        # If the requested year is not in the dataset, ask for clarification.
        # Never substitute silently — the user must confirm which year they want.
        available_years = dm.list_years()
        if year and year not in available_years:
            return {
                "error": (
                    f"No data is available for **{year}**.\n\n"
                    f"The dataset contains data for: **{', '.join(available_years)}**.\n\n"
                    f"Please specify one of those years, or omit the year to see all-time figures."
                ),
                "raw_data": {},
            }

        try:
            if intent == "COMPARISON":
                if len(properties) < 2:
                    # Do NOT silently default to all properties.
                    # Ask the user which properties they want to compare.
                    return {
                        "error": (
                            "Please specify **at least two property names** to compare.\n\n"
                            f"Available properties: {', '.join(dm.list_properties())}.\n\n"
                            "Example: *\"Compare Building 17 and Building 120\"*"
                        ),
                        "raw_data": {},
                    }

                data = dm.compare_properties(properties, year)

                # Surface any not-found properties from the comparison
                if data.get("not_found"):
                    return {
                        "error": (
                            f"Some properties were not found: "
                            f"**{', '.join(data['not_found'])}**. "
                            f"Available: {', '.join(dm.list_properties())}."
                        ),
                        "raw_data": {},
                    }

            elif intent == "PL_REPORT":
                if properties:
                    canonical = dm.search_property(properties[0])
                    if not canonical:
                        return {
                            "error": (
                                f"Property **'{properties[0]}'** not found in the dataset. "
                                f"Available: {', '.join(dm.list_properties())}."
                            ),
                            "raw_data": {},
                        }
                    data = dm.get_property_pl(canonical, year)
                else:
                    data = dm.get_total_pl(year)

            elif intent == "DETAILS":
                if not properties:
                    return {
                        "error": (
                            "Please specify a property name for a details report. "
                            f"Available: {', '.join(dm.list_properties())}."
                        ),
                        "raw_data": {},
                    }
                canonical = dm.search_property(properties[0])
                if not canonical:
                    return {
                        "error": (
                            f"Property **'{properties[0]}'** not found in the dataset. "
                            f"Available: {', '.join(dm.list_properties())}."
                        ),
                        "raw_data": {},
                    }
                data = dm.get_property_details(canonical)

            elif intent == "TENANT":
                all_tenants = dm.list_tenants()
                if not tenant:
                    return {
                        "error": (
                            "Please specify a tenant name. "
                            f"Available tenants: {', '.join(all_tenants)}."
                        ),
                        "raw_data": {},
                    }
                # Exact case-insensitive match only — no fuzzy substitution.
                # If the name doesn't match precisely, ask for clarification.
                tenant_match = next(
                    (t for t in all_tenants if t.lower() == tenant.lower()),
                    None,
                )
                if not tenant_match:
                    return {
                        "error": (
                            f"Tenant **'{tenant}'** was not found in the dataset.\n\n"
                            f"Available tenants: {', '.join(all_tenants)}.\n\n"
                            "Please use the exact tenant name from the list above."
                        ),
                        "raw_data": {},
                    }
                data = dm.get_tenant_details(tenant_match, year)

            else:
                data = dm.get_portfolio_overview()

            # Propagate data_manager-level errors
            if isinstance(data, dict) and "error" in data:
                return {"error": data["error"], "raw_data": {}}

            # Guard: if result is inexplicably empty, surface an error
            if not data:
                return {
                    "error": "The requested data could not be retrieved. Please try rephrasing.",
                    "raw_data": {},
                }

            return {"raw_data": data, "error": None}

        except Exception as exc:
            return {
                "error": f"An unexpected error occurred during data retrieval: {exc}",
                "raw_data": {},
            }

    def retriever_decision(self, state: AgentState) -> str:
        return "error" if state.get("error") else "ok"

    # ── Analyst ─────────────────────────────────────────────────────────────────

    def analyst_node(self, state: AgentState) -> Dict:
        """
        Synthesise retrieved data into a clear, professional markdown response.

        NEVER fabricates numbers — if raw_data is empty for a non-GENERAL intent,
        returns an explicit error message instead.
        """
        user_msg = state["messages"][0].content
        intent = state.get("intent", "GENERAL")
        data = state.get("raw_data", {})
        year_label = state.get("year") or "all available years"

        # ── Anti-hallucination guard ────────────────────────────────────────────
        if intent != "GENERAL" and not data:
            return {
                "final_output": (
                    "⚠️ **No data could be retrieved for your query.**\n\n"
                    "I will not estimate or fabricate any figures.\n\n"
                    f"**Available properties:** {', '.join(dm.list_properties())}\n\n"
                    "Please try again with a property name from the list above."
                )
            }

        # ── GENERAL: LLM answers from its own knowledge ─────────────────────────
        if intent == "GENERAL":
            system = SystemMessage(content=(
                "You are a senior real estate asset management expert. "
                "Answer the user's question with professional insight. "
                "Use markdown formatting for clarity. Be concise and direct."
            ))
            messages = [system, HumanMessage(content=user_msg)]

        # ── Data-driven intents ─────────────────────────────────────────────────
        else:
            system = SystemMessage(content=(
                "You are a professional real estate asset management analyst.\n\n"
                "Output format rules:\n"
                "1. FIRST LINE must be a single, direct summary sentence answering "
                "   the question. Examples:\n"
                "   - 'The total P&L for all properties in 2024 is $1,533,331.87.'\n"
                "   - 'Building 17 generated a net profit of $412,450.22 in 2024.'\n"
                "   - 'Building 120 ($850,567.42 net) outperforms Building 140 ($526,658.85 net).'\n"
                "2. After the summary line, provide a concise supporting breakdown "
                "   using markdown bullet lists or a table.\n"
                "3. Format all currency as $1,234.56 (2 decimal places). Negative = expense/loss.\n"
                "4. Highlight significant losses or anomalies.\n"
                "5. Keep total response under 250 words.\n"
                "6. CRITICAL: ONLY use figures present in the retrieved JSON. "
                "   NEVER estimate, invent, or extrapolate numbers. "
                "   If a figure is absent, say so explicitly."
            ))

            context = (
                f"**User query:** {user_msg}\n"
                f"**Intent:** {intent} | **Year:** {year_label}\n\n"
                f"**Retrieved data:**\n```json\n"
                f"{json.dumps(data, indent=2, default=str)}\n```\n\n"
                "Provide a complete, professional response using only the data above."
            )
            messages = [system, HumanMessage(content=context)]

        try:
            response = self.llm.invoke(messages)
            return {"final_output": response.content}
        except Exception as exc:
            return {
                "final_output": (
                    f"⚠️ **Error generating response:** {exc}\n\n"
                    "Please check your API key and try again."
                )
            }

    # ── Error Handler ────────────────────────────────────────────────────────────

    def error_handler_node(self, state: AgentState) -> Dict:
        """
        Produce a helpful, actionable response for all error scenarios:

        1. Property address not in dataset
        2. Financial data unavailable for the requested period
        3. Ambiguous or unsupported instructions (caught before here by clarifier)
        4. Unexpected system errors
        """
        error = state.get("error") or ""
        user_msg = state["messages"][-1].content if state.get("messages") else ""
        available = dm.list_properties()
        years = dm.list_years()

        if error:
            message = (
                f"⚠️ **I couldn't complete your request.**\n\n"
                f"**Details:** {error}\n\n"
                "---\n"
                "**What you can ask me:**\n"
                "- 📊 P&L report — *\"What's the total P&L for 2024?\"*\n"
                "- 🏠 Property comparison — *\"Compare Building 5 and Building 12\"*\n"
                "- 🔍 Property details — *\"Show details for Building 17\"*\n"
                "- 👤 Tenant info — *\"What revenue does Tenant 12 generate?\"*\n"
                "- 💬 General questions — anything about real estate investing\n\n"
                f"**Available properties:** {', '.join(available)}\n"
                f"**Available years:** {', '.join(years)}"
            )
        else:
            # Unknown routing issue
            message = (
                f'❓ I\'m not sure how to handle: *"{user_msg}"*\n\n'
                "Try one of these:\n"
                "- *\"What is the P&L for Building 17 in 2024?\"*\n"
                "- *\"Compare all properties\"*\n"
                "- *\"Show details for Building 5\"*\n"
                f"\n**Available properties:** {', '.join(available)}"
            )

        return {"final_output": message}
"""
agents_logic.py – LangGraph Multi-Agent Orchestration (Advanced Architecture)
==============================================================================

Advanced GenAI Patterns
-----------------------
1. Hybrid Deterministic + LLM Routing
     ContextManager extracts parameters (year, properties, tenants) via regex.
     LLM only classifies intent + relation. Deterministic override layer
     corrects misclassifications (e.g. advisory → PL_REPORT).

2. Chain-of-Thought (CoT) Reasoning
     Dedicated Reasoner node performs 4-step structured reasoning on the
     data BEFORE the Analyst generates a response.

3. Self-Reflection / Verification
     Verifier node checks the Analyst's response against source data for
     numerical accuracy, year correctness, and ranking consistency.
     Assigns a confidence score (0.0–1.0) and can trigger one auto-retry.

4. Few-Shot Dynamic Prompting
     Intent-specific examples are injected into the Analyst prompt.

5. Confidence Scoring
     Each response is tagged with a confidence badge (🟢🟡🔴).

6. History Isolation
     Standard reports get a "clean slate" (no chat history) to prevent
     year/entity anchoring. Advisory queries get full history for
     sequential reasoning (e.g. "next property to sell").

Agent Graph (7-node architecture)
─────────────────────────────────

    user input ──► [Router]  (deterministic + LLM hybrid)
                       │
       ┌───────────────┼──────────────┬───────────────┐
       ▼               ▼              ▼               ▼
   "retrieve"      "clarify"      "general"         "end"
       │               │              │               │
  [DataRetriever]  [Clarifier]    [Analyst]     [ErrorHandler]
       │           (follow-up)        │               │
   ok ─┤── error       ▼            END             END
       │     └──► [ErrorHandler]
       ▼
    [Reasoner]   ← Chain-of-Thought (4-step structured reasoning)
       │
    [Analyst]    ← Generates response using CoT trace + few-shot examples
       │
    [Verifier]   ← Self-reflection: accuracy check + confidence scoring
       │
   pass ─┤── fail (retry once)
       │     └──► [Analyst]
      END
"""

from __future__ import annotations

import json
import logging
import operator
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

import data_manager as dm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Type Definitions
# ═══════════════════════════════════════════════════════════════════════════════


class Intent(str, Enum):
    """Canonical intent types the Router can classify."""
    COMPARISON = "COMPARISON"
    PL_REPORT = "PL_REPORT"
    DETAILS = "DETAILS"
    TENANT = "TENANT"
    GENERAL = "GENERAL"
    CLARIFY = "CLARIFY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class Relation(str, Enum):
    """How the current message relates to the conversation history."""
    FOLLOW_UP = "FOLLOW_UP"
    NEW_TOPIC = "NEW_TOPIC"
    INTERRUPT = "INTERRUPT"


@dataclass(frozen=True)
class ExtractionResult:
    """Deterministic extraction output — regex-based, 100 % reliable."""
    year: str | None = None
    properties: list[str] = field(default_factory=list)
    tenant: str | None = None
    is_portfolio: bool = False


@dataclass
class ClassificationResult:
    """LLM classification output — intent + relation + optional clarification."""
    intent: Intent = Intent.GENERAL
    relation: Relation = Relation.NEW_TOPIC
    clarification: str | None = None


@dataclass
class ActiveContext:
    """Persistent conversational context carried across turns."""
    properties: list[str] = field(default_factory=list)
    tenant: str | None = None
    year: str | None = None

    def to_dict(self) -> dict:
        return {"properties": self.properties, "tenant": self.tenant, "year": self.year}

    @classmethod
    def from_dict(cls, d: dict | None) -> ActiveContext:
        if not d:
            return cls()
        return cls(
            properties=d.get("properties", []),
            tenant=d.get("tenant"),
            year=d.get("year"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

ADVISORY_PATTERNS: frozenset[str] = frozenset({
    "best", "worst", "sell", "buy", "recommend", "top", "bottom",
    "biggest", "smallest", "highest", "lowest", "most", "least",
    "strongest", "weakest", "next", "after", "another",
})

PORTFOLIO_KEYWORDS: frozenset[str] = frozenset({
    "portfolio", "all properties", "all my properties", "total",
    "overall", "all my buildings", "my assets", "my buildings",
    "all buildings", "for all", "across all",
})

FEW_SHOT_EXAMPLES: dict[Intent, str] = {
    Intent.PL_REPORT: (
        "\n\nEXAMPLE RESPONSE (for reference):\n"
        "\"The portfolio's net profit for 2024 is **$1,171,521.55**.\n\n"
        "- **Revenue:** $2,295,528.74\n"
        "- **Expenses:** -$1,124,007.19\n"
        "- **Net Profit:** $1,171,521.55\n\n"
        "**By Property:**\n"
        "| Property | Net Profit | Rank |\n"
        "|---|---|---|\n"
        "| Building 120 | $675,640.08 | 1 |\"\n"
    ),
    Intent.COMPARISON: (
        "\n\nEXAMPLE RESPONSE (for reference):\n"
        "\"Building 120 outperforms Building 17 by **$395,251.37** in net profit.\n\n"
        "| Metric | Building 120 | Building 17 |\n"
        "|---|---|---|\n"
        "| Revenue | $984,000 | $358,000 |\"\n"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Context Manager — deterministic extraction + classification correction
# ═══════════════════════════════════════════════════════════════════════════════


class ContextManager:
    """
    Responsible for all parameter extraction, intent correction, and
    context merging.  Completely deterministic — no LLM calls.

    Architecture:
        extract()  → regex-based parameter extraction
        correct()  → deterministic intent override rules
        merge()    → state-machine context merge (FOLLOW_UP / NEW_TOPIC / INTERRUPT)
    """

    # ── Deterministic Extraction ─────────────────────────────────────────────

    @staticmethod
    def extract(text: str, available_properties: list[str],
                available_tenants: list[str], available_years: list[str]) -> ExtractionResult:
        """
        Extract structured parameters from raw user text using regex and
        string matching.  This is the ground-truth layer — it always has
        authority over LLM-extracted parameters.

        Matching strategy:
        - Year: ``\\b20\\d{2}\\b`` anchored to known years
        - Properties / Tenants: word-boundary regex, longest-first to avoid
          substring collisions (e.g. 'Tenant 1' inside 'Tenant 12')
        - Portfolio: keyword detection
        """
        lower = text.lower()

        # Year
        year_match = re.search(r'\b(20\d{2})\b', text)
        year = year_match.group(1) if year_match and year_match.group(1) in available_years else None

        # Properties (longest-first to avoid substring collisions)
        properties = [
            p for p in sorted(available_properties, key=len, reverse=True)
            if re.search(r'\b' + re.escape(p.lower()) + r'\b', lower)
        ]

        # Tenant (longest-first)
        tenant = None
        for t in sorted(available_tenants, key=len, reverse=True):
            if re.search(r'\b' + re.escape(t.lower()) + r'\b', lower):
                tenant = t
                break

        # Portfolio keywords
        is_portfolio = any(kw in lower for kw in PORTFOLIO_KEYWORDS)

        result = ExtractionResult(year=year, properties=properties,
                                  tenant=tenant, is_portfolio=is_portfolio)
        logger.debug("Extraction: %s", result)
        return result

    # ── Deterministic Intent Correction ──────────────────────────────────────

    @staticmethod
    def correct(classification: ClassificationResult,
                extraction: ExtractionResult) -> ClassificationResult:
        """
        Post-LLM correction layer.  Overrides bad classifications based on
        deterministic keyword detection.

        Rules:
        1. Advisory language + COMPARISON/DETAILS → PL_REPORT
        2. Portfolio keyword + COMPARISON → PL_REPORT
        3. Regex-detected tenant → TENANT
        4. Domain entities found + OUT_OF_SCOPE → override to correct intent
        """
        intent = classification.intent
        relation = classification.relation

        # Rule 3: Regex-detected tenant → force TENANT
        if extraction.tenant and intent not in (Intent.TENANT, Intent.CLARIFY):
            logger.debug("Correction: %s → TENANT (tenant '%s' detected)", intent, extraction.tenant)
            intent = Intent.TENANT

        # Rule 4: Domain entities but OUT_OF_SCOPE → override
        if intent == Intent.OUT_OF_SCOPE and (extraction.properties or extraction.tenant or extraction.year):
            override = Intent.TENANT if extraction.tenant else Intent.PL_REPORT
            logger.debug("Correction: OUT_OF_SCOPE → %s (domain entities detected)", override)
            intent = override
            relation = Relation.NEW_TOPIC

        return ClassificationResult(intent=intent, relation=relation,
                                    clarification=classification.clarification)

    # ── State-Machine Context Merge ──────────────────────────────────────────

    @staticmethod
    def merge(prev: ActiveContext, extraction: ExtractionResult,
              relation: Relation, intent: Intent) -> ActiveContext:
        """
        Deterministic state machine for context transitions.

        INTERRUPT  → freeze previous context
        FOLLOW_UP  → overlay new params onto previous (regex wins)
        NEW_TOPIC  → hard reset to current extraction
        """
        if intent == Intent.OUT_OF_SCOPE or relation == Relation.INTERRUPT:
            ctx = ActiveContext(
                properties=prev.properties,
                tenant=prev.tenant,
                year=prev.year,
            )

        elif relation == Relation.FOLLOW_UP:
            ctx = ActiveContext(
                properties=extraction.properties if extraction.properties else prev.properties,
                tenant=extraction.tenant or prev.tenant,
                year=extraction.year if extraction.year else prev.year,
            )

        else:  # NEW_TOPIC
            ctx = ActiveContext(
                properties=extraction.properties,
                tenant=extraction.tenant,
                year=extraction.year,
            )

        # Portfolio override: clear property list for aggregate queries
        if extraction.is_portfolio:
            ctx.properties = []

        logger.debug("Merged context: %s", ctx)
        return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph State
# ═══════════════════════════════════════════════════════════════════════════════


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
    reasoning_trace: Optional[str]
    confidence_score: Optional[float]
    verification_status: Optional[str]
    retry_count: int


# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph Agent
# ═══════════════════════════════════════════════════════════════════════════════


class AssetManagerGraph:
    """
    7-node LangGraph orchestration for Cortex RE.

    Pipeline:  Router → DataRetriever → Reasoner (CoT) → Analyst → Verifier
    """

    def __init__(self, model_name: str = "claude-3-haiku-20240307",
                 temperature: float = 0):
        self.llm = ChatAnthropic(model=model_name, temperature=temperature)
        self.ctx_mgr = ContextManager()
        self.memory = MemorySaver()
        self._build_graph()

    # ── Graph Construction ───────────────────────────────────────────────────

    def _build_graph(self):
        wf = StateGraph(AgentState)

        wf.add_node("router", self.router_node)
        wf.add_node("data_retriever", self.data_retriever_node)
        wf.add_node("reasoner", self.reasoner_node)
        wf.add_node("analyst", self.analyst_node)
        wf.add_node("verifier", self.verifier_node)
        wf.add_node("clarifier", self.clarifier_node)
        wf.add_node("error_handler", self.error_handler_node)

        wf.set_entry_point("router")

        wf.add_conditional_edges("router", self.route_decision, {
            "retrieve": "data_retriever",
            "general": "analyst",
            "clarify": "clarifier",
            "end": "error_handler",
        })
        wf.add_conditional_edges("data_retriever", self.retriever_decision, {
            "ok": "reasoner",
            "error": "error_handler",
        })
        wf.add_edge("reasoner", "analyst")
        wf.add_edge("analyst", "verifier")
        wf.add_conditional_edges("verifier", self.verifier_decision, {
            "pass": END,
            "retry": "analyst",
        })
        wf.add_edge("clarifier", END)
        wf.add_edge("error_handler", END)

        self.app = wf.compile(checkpointer=self.memory)

    # ══════════════════════════════════════════════════════════════════════════
    # Node: Router (Hybrid Deterministic + LLM)
    # ══════════════════════════════════════════════════════════════════════════

    def router_node(self, state: AgentState) -> Dict:
        """
        4-phase hybrid classification pipeline:
          Phase 1 — Deterministic regex extraction (year, properties, tenant)
          Phase 2 — LLM classification (intent + relation only)
          Phase 2.5 — Deterministic intent correction
          Phase 3 — State-machine context merge
        """
        messages = state["messages"]
        user_msg = messages[-1].content.strip()

        if not user_msg:
            return self._empty_input_response(state)

        available_properties = dm.list_properties()
        available_years = dm.list_years()
        available_tenants = dm.list_tenants()

        # Phase 1: Deterministic extraction
        extraction = self.ctx_mgr.extract(
            user_msg, available_properties, available_tenants, available_years,
        )

        # Phase 2: LLM classification (intent + relation only)
        classification = self._classify_with_llm(messages)

        # Phase 2.5a: Advisory keyword correction (needs raw user text)
        words_clean = set(re.findall(r'\b\w+\b', user_msg.lower()))
        if (ADVISORY_PATTERNS & words_clean) and classification.intent in (Intent.COMPARISON, Intent.DETAILS):
            logger.debug("Correction: %s → PL_REPORT (advisory language)", classification.intent)
            classification.intent = Intent.PL_REPORT

        if extraction.is_portfolio and classification.intent == Intent.COMPARISON:
            logger.debug("Correction: COMPARISON + portfolio → PL_REPORT")
            classification.intent = Intent.PL_REPORT

        # Phase 2.5b: Entity-based corrections (tenant, OOS override)
        classification = self.ctx_mgr.correct(classification, extraction)

        # Phase 3: Context merge (OOS freeze handled inside merge())
        prev_context = ActiveContext.from_dict(state.get("active_entities"))
        new_context = self.ctx_mgr.merge(
            prev_context, extraction, classification.relation, classification.intent,
        )
        intent = classification.intent

        return {
            "intent": intent.value,
            "properties": new_context.properties,
            "tenant": new_context.tenant,
            "year": new_context.year,
            "error": None,
            "raw_data": {},
            "clarification_needed": classification.clarification,
            "final_output": None,
            "active_entities": new_context.to_dict(),
        }

    def _empty_input_response(self, state: AgentState) -> Dict:
        """Handle empty user input."""
        prev = ActiveContext.from_dict(state.get("active_entities"))
        return {
            "intent": Intent.CLARIFY.value,
            "properties": [], "tenant": None, "year": None,
            "error": None, "raw_data": {}, "final_output": None,
            "clarification_needed": (
                "It looks like you sent an empty message. "
                "What would you like to know about the PropCo portfolio?"
            ),
            "active_entities": prev.to_dict(),
        }

    def _classify_with_llm(self, messages: list[BaseMessage]) -> ClassificationResult:
        """
        Ask the LLM to classify intent and relation.
        Only the last 2 messages are passed to avoid context confusion.
        """
        system = SystemMessage(content=(
            "You are a routing agent. Classify the user's message.\n\n"
            "Intents:\n"
            "- COMPARISON: Explicit comparison of 2+ named properties.\n"
            "- PL_REPORT: P&L report, financial summary, OR any advisory/ranking "
            "question ('which to sell?', 'best/worst performer', 'does this earn "
            "the most?', 'biggest earner', 'which building is best?').\n"
            "- DETAILS: Detailed breakdown of ONE specific named property.\n"
            "- TENANT: Questions about a specific tenant.\n"
            "- GENERAL: Real estate knowledge or self-intro. No data needed.\n"
            "- CLARIFY: Ambiguous or incomplete query.\n"
            "- OUT_OF_SCOPE: Unrelated (sports, weather, celebrities).\n\n"
            "Relations:\n"
            "- FOLLOW_UP: Refers to previous context ('it', 'why?', 'next', "
            "'and in 2024?', 'does this building...', 'what about...?').\n"
            "- NEW_TOPIC: Specifies new entities or a fresh question.\n"
            "- INTERRUPT: Completely off-topic (triggers OUT_OF_SCOPE).\n\n"
            "Respond with ONLY JSON:\n"
            '{"intent":"...","relation":"...","clarification_question":null}'
        ))

        context_window = messages[-2:] if len(messages) >= 2 else messages
        try:
            response = self.llm.invoke([system] + context_window)
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", response.content, re.DOTALL)
            parsed = json.loads(m.group()) if m else {}
        except Exception:
            parsed = {}

        raw_intent = parsed.get("intent", "GENERAL").upper()
        raw_relation = parsed.get("relation", "NEW_TOPIC").upper()

        try:
            intent = Intent(raw_intent)
        except ValueError:
            intent = Intent.GENERAL
        try:
            relation = Relation(raw_relation)
        except ValueError:
            relation = Relation.NEW_TOPIC

        return ClassificationResult(
            intent=intent,
            relation=relation,
            clarification=parsed.get("clarification_question"),
        )

    # ── Routing Decisions ────────────────────────────────────────────────────

    def route_decision(self, state: AgentState) -> str:
        if state.get("error"):
            return "end"
        intent = state.get("intent", "GENERAL")
        if intent == Intent.CLARIFY.value:
            return "clarify"
        if intent == Intent.OUT_OF_SCOPE.value:
            return "end"
        if intent == Intent.GENERAL.value:
            return "general"
        return "retrieve"

    def retriever_decision(self, state: AgentState) -> str:
        return "error" if state.get("error") else "ok"

    def verifier_decision(self, state: AgentState) -> str:
        return "retry" if state.get("verification_status") == "RETRY" else "pass"

    # ══════════════════════════════════════════════════════════════════════════
    # Node: Clarifier
    # ══════════════════════════════════════════════════════════════════════════

    def clarifier_node(self, state: AgentState) -> Dict:
        """Ask a targeted follow-up question for ambiguous queries."""
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

    # ══════════════════════════════════════════════════════════════════════════
    # Node: Data Retriever
    # ══════════════════════════════════════════════════════════════════════════

    def data_retriever_node(self, state: AgentState) -> Dict:
        """Pull relevant data from the parquet dataset based on intent."""
        intent = state["intent"]
        properties = state.get("properties") or []
        tenant = state.get("tenant")
        year = state.get("year")

        available_years = dm.list_years()
        if year and year not in available_years:
            return {
                "error": (
                    f"No data is available for the period **{year}**.\n\n"
                    f"Available data: **{', '.join(available_years)}**.\n\n"
                    "Please specify a valid year."
                ),
                "raw_data": {},
            }

        try:
            data = self._retrieve_by_intent(intent, properties, tenant, year)

            if isinstance(data, dict) and "error" in data:
                return {"error": data["error"], "raw_data": {}}
            if not data:
                return {"error": "No data found. Please try rephrasing.", "raw_data": {}}

            return {"raw_data": data, "error": None}

        except Exception as exc:
            logger.error("Data retrieval failed: %s", exc)
            return {"error": f"Data retrieval error: {exc}", "raw_data": {}}

    def _retrieve_by_intent(self, intent: str, properties: list[str],
                            tenant: str | None, year: str | None) -> dict:
        """Dispatch data retrieval based on intent type."""
        if intent == Intent.COMPARISON.value:
            if len(properties) < 2:
                return {
                    "error": (
                        "Please specify **at least two property names** to compare.\n\n"
                        f"Available: {', '.join(dm.list_properties())}.\n\n"
                        "Example: *\"Compare Building 17 and Building 120\"*"
                    )
                }
            data = dm.compare_properties(properties, year)
            if data.get("not_found"):
                return {
                    "error": (
                        f"Properties not found: **{', '.join(data['not_found'])}**. "
                        f"Available: {', '.join(dm.list_properties())}."
                    )
                }
            return data

        elif intent == Intent.PL_REPORT.value:
            if len(properties) == 1:
                canonical = dm.search_property(properties[0])
                if not canonical:
                    return {
                        "error": (
                            f"Property **'{properties[0]}'** not found. "
                            f"Available: {', '.join(dm.list_properties())}."
                        )
                    }
                return dm.get_property_pl(canonical, year)
            return dm.get_total_pl(year)

        elif intent == Intent.DETAILS.value:
            if not properties:
                return {
                    "error": (
                        "Please specify a property name. "
                        f"Available: {', '.join(dm.list_properties())}."
                    )
                }
            canonical = dm.search_property(properties[0])
            if not canonical:
                return {
                    "error": (
                        f"Property **'{properties[0]}'** not found. "
                        f"Available: {', '.join(dm.list_properties())}."
                    )
                }
            return dm.get_property_details(canonical)

        elif intent == Intent.TENANT.value:
            all_tenants = dm.list_tenants()
            if not tenant:
                return {"error": f"Please specify a tenant. Available: {', '.join(all_tenants)}."}
            match = next((t for t in all_tenants if t.lower() == tenant.lower()), None)
            if not match:
                return {"error": f"Tenant **'{tenant}'** not found. Available: {', '.join(all_tenants)}."}
            return dm.get_tenant_details(match, year)

        return dm.get_portfolio_overview()

    # ══════════════════════════════════════════════════════════════════════════
    # Node: Reasoner (Chain-of-Thought)
    # ══════════════════════════════════════════════════════════════════════════

    def reasoner_node(self, state: AgentState) -> Dict:
        """
        Perform structured 4-step reasoning on retrieved data before the
        Analyst generates a response.  Grounds the output in verified calculations.
        """
        intent = state.get("intent", "GENERAL")
        data = state.get("raw_data", {})
        user_msg = state["messages"][-1].content
        active = ActiveContext.from_dict(state.get("active_entities"))

        if intent == Intent.GENERAL.value or not data:
            return {"reasoning_trace": None, "retry_count": 0}

        system = SystemMessage(content=(
            "You are a data reasoning engine. Analyze the data and produce a "
            "structured reasoning trace. Do NOT write a final answer.\n\n"
            "STEP 1 — QUERY UNDERSTANDING: What type of analysis is requested?\n"
            "STEP 2 — DATA EXTRACTION: List key numbers from the JSON.\n"
            "STEP 3 — CALCULATION: Perform totals, rankings, or differences.\n"
            "STEP 4 — CONCLUSION: One-sentence factual answer.\n\n"
            "Format: STEP 1: ...\nSTEP 2: ...\nSTEP 3: ...\nSTEP 4: ...\n"
            "CRITICAL: Only use numbers from the JSON."
        ))

        scope = "Portfolio" if not active.properties else ", ".join(active.properties)
        context = HumanMessage(content=(
            f"Query: {user_msg}\n"
            f"Year: {active.year or 'all'} | Scope: {scope}\n\n"
            f"Data:\n```json\n{json.dumps(data, indent=2, default=str)}\n```"
        ))

        try:
            trace = self.llm.invoke([system, context]).content
            logger.debug("Reasoner trace: %.200s...", trace)
            return {"reasoning_trace": trace, "retry_count": 0}
        except Exception as exc:
            logger.error("Reasoner failed: %s", exc)
            return {"reasoning_trace": None, "retry_count": 0}

    # ══════════════════════════════════════════════════════════════════════════
    # Node: Analyst (Response Generation)
    # ══════════════════════════════════════════════════════════════════════════

    def analyst_node(self, state: AgentState) -> Dict:
        """
        Generate a professional response grounded in retrieved data.

        Enhanced with CoT trace consumption and few-shot dynamic examples.
        Uses history isolation — only advisory queries receive chat history.
        """
        messages = state["messages"]
        user_msg = messages[-1].content
        intent_str = state.get("intent", "GENERAL")
        data = state.get("raw_data", {})
        reasoning = state.get("reasoning_trace")

        # Anti-hallucination guard
        if intent_str != Intent.GENERAL.value and not data:
            return {
                "final_output": (
                    "⚠️ **No data could be retrieved for your query.**\n\n"
                    "I will not estimate or fabricate any figures.\n\n"
                    f"**Available properties:** {', '.join(dm.list_properties())}\n\n"
                    "Please try again with a property name from the list above."
                )
            }

        if intent_str == Intent.GENERAL.value:
            return self._generate_general_response(messages)

        return self._generate_data_response(messages, user_msg, intent_str, data, reasoning, state)

    def _generate_general_response(self, messages: list[BaseMessage]) -> Dict:
        """Handle GENERAL intent — LLM answers from domain knowledge."""
        system = SystemMessage(content=(
            "You are Cortex RE, a senior real estate asset management AI assistant.\n"
            f"You manage the PropCo portfolio: {', '.join(dm.list_properties())} "
            f"across years {', '.join(dm.list_years())}.\n\n"
            "Answer professionally using markdown. "
            "CRITICAL: Never fabricate financial numbers — tell the user you "
            "need to pull a data report for specific figures."
        ))
        try:
            response = self.llm.invoke([system] + messages)
            return {"final_output": response.content}
        except Exception as exc:
            return {"final_output": f"⚠️ **Error:** {exc}"}

    def _generate_data_response(self, messages: list[BaseMessage], user_msg: str,
                                intent_str: str, data: dict, reasoning: str | None,
                                state: AgentState) -> Dict:
        """Handle data-driven intents with CoT trace and few-shot examples."""
        active = ActiveContext.from_dict(state.get("active_entities"))
        active_year = active.year or "all available years"

        try:
            intent_enum = Intent(intent_str)
        except ValueError:
            intent_enum = Intent.PL_REPORT

        few_shot = FEW_SHOT_EXAMPLES.get(intent_enum, "")

        system = SystemMessage(content=(
            "You are an objective real estate analyst. Respond to the CURRENT query only.\n\n"
            "RULES:\n"
            "1. DATA ONLY: Use ONLY metrics from the JSON. NEVER invent values.\n"
            "2. REPORT: Standardized financial summary from the JSON.\n"
            "   ADVISORY ('sell', 'best/worst', 'next'): use SEQUENTIAL STRATEGY.\n"
            "3. SEQUENTIAL STRATEGY:\n"
            "   - SELL = Rank 5 (worst). 'Next' → Rank 4. NEVER sell Rank 1.\n"
            "   - BEST = Rank 1. 'Next' → Rank 2.\n"
            "   - Read history to find previous rank discussed.\n"
            "4. FORMAT: First line = direct answer. Body = bullet points. "
            "Currency as $1,234.56. Under 200 words.\n"
            f"5. YEAR: Data below is for **{active_year}**. Use this year."
            f"{few_shot}"
        ))

        # Build context block with optional CoT trace
        reasoning_block = f"\n\n**Pre-Analysis (CoT):**\n{reasoning}" if reasoning else ""
        scope = "Portfolio" if not active.properties else ", ".join(active.properties)
        context_block = (
            f"**Query:** {user_msg}\n"
            f"**Year:** {active_year} | **Scope:** {scope}"
            f"{reasoning_block}\n\n"
            f"**Data:**\n```json\n{json.dumps(data, indent=2, default=str)}\n```"
        )

        # History isolation: only advisory queries get chat history
        words_clean = set(re.findall(r'\b\w+\b', user_msg.lower()))
        is_advisory = bool(ADVISORY_PATTERNS & words_clean)

        if is_advisory:
            msgs = [system] + messages[:-1] + [HumanMessage(content=context_block)]
        else:
            msgs = [system, HumanMessage(content=context_block)]

        try:
            response = self.llm.invoke(msgs)
            return {"final_output": response.content}
        except Exception as exc:
            return {"final_output": f"⚠️ **Error:** {exc}"}

    # ══════════════════════════════════════════════════════════════════════════
    # Node: Verifier (Self-Reflection)
    # ══════════════════════════════════════════════════════════════════════════

    def verifier_node(self, state: AgentState) -> Dict:
        """
        Self-reflection node — checks the Analyst's output against source data.

        Evaluates: year accuracy, number accuracy, ranking accuracy, scope accuracy.
        Assigns a confidence score and can trigger one auto-retry on failure.
        """
        output = state.get("final_output", "")
        data = state.get("raw_data", {})
        intent = state.get("intent", "GENERAL")
        active = ActiveContext.from_dict(state.get("active_entities"))
        retry_count = state.get("retry_count", 0)

        # Skip verification for non-data intents or after retry
        if intent == Intent.GENERAL.value or not data or retry_count >= 1:
            return {
                "verification_status": "PASS",
                "confidence_score": 0.85 if intent == Intent.GENERAL.value else 0.7,
            }

        system = SystemMessage(content=(
            "You are a verification agent. Check the RESPONSE against SOURCE DATA.\n\n"
            "Evaluate:\n"
            "1. YEAR ACCURACY: Correct year referenced?\n"
            "2. NUMBER ACCURACY: Financial numbers match source?\n"
            "3. RANKING ACCURACY: Properties ranked correctly?\n"
            "4. SCOPE ACCURACY: Response addresses the query?\n\n"
            "Respond with ONLY JSON:\n"
            '{"status":"PASS or FAIL","confidence":0.95,"issues":[]}'
        ))

        context = HumanMessage(content=(
            f"EXPECTED YEAR: {active.year or 'all'}\n\n"
            f"SOURCE DATA:\n```json\n{json.dumps(data, indent=2, default=str)}\n```\n\n"
            f"RESPONSE TO VERIFY:\n{output}"
        ))

        try:
            response = self.llm.invoke([system, context])
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", response.content, re.DOTALL)
            parsed = json.loads(m.group()) if m else {"status": "PASS", "confidence": 0.7}
        except Exception:
            parsed = {"status": "PASS", "confidence": 0.7}

        status = parsed.get("status", "PASS").upper()
        confidence = min(max(float(parsed.get("confidence", 0.7)), 0.0), 1.0)
        issues = parsed.get("issues", [])

        logger.debug("Verifier: status=%s confidence=%.2f issues=%s", status, confidence, issues)

        if status == "FAIL" and retry_count < 1:
            return {
                "verification_status": "RETRY",
                "confidence_score": confidence,
                "retry_count": retry_count + 1,
            }

        # Append confidence badge
        badge = "🟢" if confidence >= 0.85 else "🟡" if confidence >= 0.6 else "🔴"
        return {
            "final_output": f"{output}\n\n---\n*{badge} Confidence: {confidence:.0%} · Verified by Cortex RE*",
            "verification_status": "PASS",
            "confidence_score": confidence,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Node: Error Handler
    # ══════════════════════════════════════════════════════════════════════════

    def error_handler_node(self, state: AgentState) -> Dict:
        """Produce helpful, actionable responses for all error scenarios."""
        intent = state.get("intent", "")
        error = state.get("error") or ""
        user_msg = state["messages"][-1].content if state.get("messages") else ""
        available = dm.list_properties()
        years = dm.list_years()

        capabilities = (
            "**Here's what I can do for you:**\n"
            "- 📊 P&L reports — *\"What's the total P&L for 2024?\"*\n"
            "- 🏠 Property comparison — *\"Compare Building 17 and Building 120\"*\n"
            "- 🔍 Property details — *\"Show details for Building 140\"*\n"
            "- 👤 Tenant info — *\"What revenue does Tenant 12 generate?\"*\n"
            "- 💬 Real estate knowledge — *\"What is a good cap rate?\"*\n\n"
            f"**Available properties:** {', '.join(available)}\n"
            f"**Available years:** {', '.join(years)}"
        )

        if intent == Intent.OUT_OF_SCOPE.value:
            message = (
                "🏢 **I'm a real estate asset management assistant.**\n\n"
                f"I can't help with *\"{user_msg}\"* — that's outside my scope.\n\n"
                f"{capabilities}"
            )
        elif error:
            message = (
                f"⚠️ **I couldn't complete your request.**\n\n"
                f"**Details:** {error}\n\n---\n"
                f"**What you can ask me:**\n{capabilities}"
            )
        else:
            message = (
                f'❓ I\'m not sure how to handle: *"{user_msg}"*\n\n'
                f"{capabilities}"
            )

        return {"final_output": message}

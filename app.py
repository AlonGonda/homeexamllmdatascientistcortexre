"""
app.py – Cortex RE Asset Manager · Streamlit UI
Launch with: python -m streamlit run app.py
"""

import os
import re as _re

import streamlit as st
from langchain_core.messages import HumanMessage

import data_manager as dm
from agents_logic import AssetManagerGraph

# ─── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cortex RE · Asset Manager AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (high-contrast dark theme) ─────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #111827; color: #f1f5f9; }

[data-testid="stSidebar"] {
    background: #1e293b;
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] h3 { color: #f8fafc !important; font-weight: 700; }
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.8rem; }

p, li, span { color: #e2e8f0; }
h1, h2, h3, h4 { color: #f8fafc; }

.cortex-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 60%, #2563eb 100%);
    border-radius: 16px;
    padding: 26px 34px;
    margin-bottom: 20px;
    box-shadow: 0 8px 30px rgba(37,99,235,0.4);
    border: 1px solid #3b82f6;
}
.cortex-header h1 { color: #ffffff !important; font-size: 2rem; font-weight: 700; margin: 0; }
.cortex-header p  { color: #bfdbfe !important; font-size: 0.95rem; margin: 6px 0 0; }

.kpi-grid { display: flex; gap: 14px; margin-bottom: 22px; flex-wrap: wrap; }
.kpi-card {
    flex: 1;
    min-width: 135px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.kpi-card .kpi-label {
    font-size: 0.68rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}
.kpi-card .kpi-value { font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin-top: 5px; }
.kpi-card .kpi-value.green { color: #34d399; }
.kpi-card .kpi-value.red   { color: #f87171; }
.kpi-card .kpi-value.blue  { color: #60a5fa; }

[data-testid="stChatMessage"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    margin-bottom: 10px;
}

.stButton > button {
    background: #1e293b;
    border: 1px solid #475569;
    color: #e2e8f0;
    border-radius: 20px;
    padding: 5px 16px;
    font-size: 0.82rem;
    font-weight: 500;
    transition: all 0.18s ease;
}
.stButton > button:hover {
    background: #2563eb;
    border-color: #3b82f6;
    color: #ffffff;
    box-shadow: 0 0 12px rgba(59,130,246,0.4);
}

hr { border-color: #334155; }

.sidebar-metric {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 9px 14px;
    margin-bottom: 7px;
    font-size: 0.83rem;
    color: #cbd5e1;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-metric span { color: #93c5fd; font-weight: 600; }

.prop-chip {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #3b82f6;
    border-radius: 6px;
    padding: 3px 10px;
    margin: 3px;
    font-size: 0.78rem;
    color: #93c5fd;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="cortex-header">
  <h1>🏢 Cortex RE · Asset Manager AI</h1>
  <p>Multi-agent real estate intelligence powered by LangGraph + Claude</p>
</div>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False


def fmt_currency(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def escape_dollars(text: str) -> str:
    """Escape bare $ so Streamlit doesn't treat them as LaTeX delimiters."""
    return _re.sub(r'(?<!\\)\$(?!\$)', r'\\$', text)


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Your key is never stored — it lives only in this session.",
    )

    model_name = st.selectbox(
        "Model",
        [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ],
        index=0,
        help="Haiku is fastest and most widely available.",
    )

    custom_model = st.text_input(
        "Custom model ID (optional)",
        placeholder="e.g. claude-3-haiku-20240307",
        help="Overrides the selector above.",
    )
    if custom_model.strip():
        model_name = custom_model.strip()

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        if not st.session_state.api_key_set or st.session_state.graph is None:
            with st.spinner("Initialising agents…"):
                st.session_state.graph = AssetManagerGraph(model_name=model_name)
                st.session_state.api_key_set = True
        st.success("✅ Agents ready", icon="🤖")
    else:
        st.session_state.api_key_set = False
        st.info("Enter your Anthropic API key to start.", icon="🔑")

    st.divider()

    st.markdown("### 📊 Portfolio Overview")
    try:
        overview = dm.get_portfolio_overview()
        st.markdown(
            f'<div class="sidebar-metric">Properties <span>{overview["property_count"]}</span></div>'
            f'<div class="sidebar-metric">Tenants <span>{overview["tenant_count"]}</span></div>'
            f'<div class="sidebar-metric">Years <span>{", ".join(overview["years"])}</span></div>'
            f'<div class="sidebar-metric">Revenue <span>{fmt_currency(overview["all_time_revenue"])}</span></div>'
            f'<div class="sidebar-metric">Expenses <span>{fmt_currency(overview["all_time_expenses"])}</span></div>',
            unsafe_allow_html=True,
        )
        net = overview["all_time_net"]
        colour = "#34d399" if net >= 0 else "#f87171"
        st.markdown(
            f'<div class="sidebar-metric">Net Profit '
            f'<span style="color:{colour}">{fmt_currency(net)}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Properties**")
        chips = "".join(f'<span class="prop-chip">{p}</span>' for p in overview["properties"])
        st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load overview: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ─── KPI Cards ─────────────────────────────────────────────────────────────────

try:
    ov = dm.get_portfolio_overview()
    by_year = {r["year"]: r["net_profit"] for r in ov.get("by_year", [])}
    latest_year = max(by_year.keys()) if by_year else "N/A"
    latest_net = by_year.get(latest_year, 0)
    colour_class = "green" if latest_net >= 0 else "red"

    st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Properties</div>
    <div class="kpi-value blue">{ov["property_count"]}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Tenants</div>
    <div class="kpi-value blue">{ov["tenant_count"]}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">All-Time Revenue</div>
    <div class="kpi-value green">{fmt_currency(ov["all_time_revenue"])}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">All-Time Expenses</div>
    <div class="kpi-value red">{fmt_currency(ov["all_time_expenses"])}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">All-Time Net</div>
    <div class="kpi-value {("green" if ov["all_time_net"] >= 0 else "red")}">{fmt_currency(ov["all_time_net"])}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">{latest_year} Net</div>
    <div class="kpi-value {colour_class}">{fmt_currency(latest_net)}</div>
  </div>
</div>
""", unsafe_allow_html=True)
except Exception:
    pass

# ─── Quick Actions ──────────────────────────────────────────────────────────────

st.markdown("**💡 Quick actions**")

quick_actions = [
    "📈 Portfolio P&L 2024",
    "📊 Portfolio P&L 2025",
    "🏠 Details: Building 17",
    "⚖️ Compare all properties",
    "👤 Tenant 12 revenue",
    "📋 All properties list",
]

cols = st.columns(len(quick_actions))
query_from_button = None
for col, label in zip(cols, quick_actions):
    with col:
        if st.button(label, key=f"quick_{label}"):
            query_from_button = label.split(" ", 1)[1]

st.divider()

# ─── Chat History ───────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Input & Response ──────────────────────────────────────────────────────────

user_query = query_from_button or st.chat_input(
    "Ask anything – e.g. 'What is the P&L for Building 17 in 2024?'",
    disabled=not st.session_state.api_key_set,
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if not st.session_state.api_key_set or st.session_state.graph is None:
        st.warning("Please enter your Anthropic API key in the sidebar first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Agents collaborating…"):
                try:
                    inputs = {"messages": [HumanMessage(content=user_query)]}
                    result = st.session_state.graph.app.invoke(inputs)
                    response = result.get("final_output", "No response generated.")
                except Exception as exc:
                    response = (
                        f"⚠️ **An unexpected error occurred:**\n\n```\n{exc}\n```\n\n"
                        "Please check your API key and try again."
                    )

            safe_response = escape_dollars(response)
            st.markdown(safe_response)
            st.session_state.messages.append({"role": "assistant", "content": safe_response})
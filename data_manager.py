"""
data_manager.py – Cortex RE Data Access Layer
==============================================
Loads and queries the ``cortex.parquet`` dataset.

Dataset schema
--------------
entity_name      : str   – always "PropCo"
property_name    : str   – e.g. "Building 17" (nullable – entity-level rows)
tenant_name      : str   – nullable
ledger_type      : str   – "revenue" | "expenses"
ledger_group     : str   – e.g. "rental_income", "general_expenses"
ledger_category  : str   – finer classification (29 unique)
ledger_code      : int   – numeric ledger code
ledger_description: str  – human-readable description
month            : str   – "2025-M01"
quarter          : str   – "2025-Q1"
year             : str   – "2024" | "2025"
profit           : float – positive = income, negative = expense
"""

import os
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd

PARQUET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cortex.parquet")


# ─── Loader ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Load and cache the dataset once per process."""
    return pd.read_parquet(PARQUET_PATH)


# ─── Lookups ───────────────────────────────────────────────────────────────────

def list_properties() -> List[str]:
    """All unique property names (NaN rows excluded)."""
    return sorted(load_data()["property_name"].dropna().unique().tolist())


def list_years() -> List[str]:
    """All unique years present in the dataset."""
    return sorted(load_data()["year"].dropna().unique().tolist())


def list_tenants() -> List[str]:
    """All unique tenant names."""
    return sorted(load_data()["tenant_name"].dropna().unique().tolist())


def list_ledger_groups() -> List[str]:
    return sorted(load_data()["ledger_group"].dropna().unique().tolist())


# ─── Fuzzy Property Search ─────────────────────────────────────────────────────

def search_property(query: str) -> Optional[str]:
    """
    Match a user-supplied property reference to a canonical property name.

    Strategy (in order):
    1. Exact match (case-insensitive)
    2. Substring match
    3. Token overlap  (e.g. "building17" → "Building 17")

    Returns ``None`` if no reasonable match is found.
    """
    if not query:
        return None

    props = list_properties()
    q = query.lower().strip()

    # 1 – exact
    for p in props:
        if p.lower() == q:
            return p

    # 2 – substring
    matches = [p for p in props if q in p.lower() or p.lower() in q]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return sorted(matches, key=len)[0]

    # 3 – token overlap
    q_tokens = set(q.replace("-", " ").split())
    for p in props:
        p_tokens = set(p.lower().replace("-", " ").split())
        if q_tokens & p_tokens:
            return p

    return None


def _to_native(data):
    """Recursively convert numpy types to native Python types for serialization."""
    if isinstance(data, dict):
        return {k: _to_native(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_native(i) for i in data]
    if hasattr(data, "item"):  # numpy scalars
        return data.item()
    return data


# ─── P&L helpers ───────────────────────────────────────────────────────────────

def _pl_summary(df: pd.DataFrame) -> Dict:
    """Compute revenue / expenses / net from a pre-filtered DataFrame."""
    revenue = df[df["ledger_type"] == "revenue"]["profit"].sum()
    expenses = df[df["ledger_type"] == "expenses"]["profit"].sum()
    net = df["profit"].sum()

    by_category = (
        df.groupby(["ledger_category", "ledger_type"])["profit"]
        .sum()
        .reset_index()
        .rename(columns={"profit": "total"})
        .sort_values("total", ascending=False)
        .to_dict(orient="records")
    )

    by_quarter = (
        df.groupby("quarter")
        .apply(
            lambda g: pd.Series(
                {
                    "revenue": g[g["ledger_type"] == "revenue"]["profit"].sum(),
                    "expenses": g[g["ledger_type"] == "expenses"]["profit"].sum(),
                    "net": g["profit"].sum(),
                }
            )
        )
        .reset_index()
        .sort_values("quarter")
        .to_dict(orient="records")
    )

    return _to_native({
        "total_revenue": round(revenue, 2),
        "total_expenses": round(expenses, 2),
        "net_profit": round(net, 2),
        "by_category": by_category,
        "by_quarter": by_quarter,
    })


# ─── Public API ────────────────────────────────────────────────────────────────

def get_property_pl(property_name: str, year: Optional[str] = None) -> Dict:
    """P&L summary for a single property, optionally filtered by year."""
    df = load_data()
    mask = df["property_name"] == property_name
    if year:
        mask &= df["year"] == str(year)
    sub = df[mask]

    if sub.empty:
        return {
            "error": f"No data found for property '{property_name}'"
            + (f" in {year}" if year else "")
        }

    result = _pl_summary(sub)
    result.update({"property": property_name, "year": year or "all"})
    return result


def get_total_pl(year: Optional[str] = None) -> Dict:
    """Portfolio-wide P&L, optionally filtered by year."""
    df = load_data()
    if year:
        df = df[df["year"] == str(year)]

    if df.empty:
        return {"error": "No data available" + (f" for year {year}" if year else "")}

    result = _pl_summary(df)

    by_property = (
        df.dropna(subset=["property_name"])
        .groupby("property_name")["profit"]
        .sum()
        .reset_index()
        .rename(columns={"profit": "net_profit"})
        .sort_values("net_profit", ascending=False)
        .to_dict(orient="records")
    )

    ranked = [
        {"property": r["property_name"], "net_profit": r["net_profit"], "rank": i + 1}
        for i, r in enumerate(by_property)
    ]

    result.update({
        "scope": "portfolio",
        "year": year or "all",
        "by_property": by_property,
        "ranked_properties_best_to_worst": ranked,
    })
    return _to_native(result)


def compare_properties(property_names: List[str], year: Optional[str] = None) -> Dict:
    """Side-by-side P&L comparison for multiple properties."""
    results, not_found = [], []
    for name in property_names:
        canonical = search_property(name)
        if canonical:
            results.append(get_property_pl(canonical, year))
        else:
            not_found.append(name)

    return {"comparison": results, "not_found": not_found, "year": year or "all"}


def get_property_details(property_name: str) -> Dict:
    """Rich detail record for a single property."""
    df = load_data()
    sub = df[df["property_name"] == property_name]

    if sub.empty:
        return {"error": f"Property '{property_name}' not found in dataset."}

    tenants = sub["tenant_name"].dropna().unique().tolist()
    years_available = sorted(sub["year"].unique().tolist())
    latest_year = years_available[-1] if years_available else None

    pl_all = get_property_pl(property_name)
    pl_latest = get_property_pl(property_name, latest_year) if latest_year else {}

    ledger_groups = (
        sub.groupby(["ledger_group", "ledger_type"])["profit"]
        .sum()
        .reset_index()
        .rename(columns={"profit": "total"})
        .to_dict(orient="records")
    )

    return _to_native({
        "property": property_name,
        "entity": sub["entity_name"].iloc[0],
        "tenants": tenants,
        "tenant_count": len(tenants),
        "years_available": years_available,
        "ledger_groups": ledger_groups,
        "pl_all_time": pl_all,
        "pl_latest_year": pl_latest,
    })


def get_tenant_details(tenant_name: str, year: Optional[str] = None) -> Dict:
    """Financial summary for a specific tenant."""
    df = load_data()
    mask = df["tenant_name"] == tenant_name
    if year:
        mask &= df["year"] == str(year)
    sub = df[mask]

    if sub.empty:
        return {
            "error": f"No data for tenant '{tenant_name}'"
            + (f" in {year}" if year else "")
        }

    by_property = (
        sub.dropna(subset=["property_name"])
        .groupby("property_name")["profit"]
        .sum()
        .reset_index()
        .rename(columns={"profit": "net_profit"})
        .to_dict(orient="records")
    )

    return _to_native({
        "tenant": tenant_name,
        "year": year or "all",
        "properties": sub["property_name"].dropna().unique().tolist(),
        "total_revenue": round(sub[sub["ledger_type"] == "revenue"]["profit"].sum(), 2),
        "total_expenses": round(sub[sub["ledger_type"] == "expenses"]["profit"].sum(), 2),
        "net": round(sub["profit"].sum(), 2),
        "by_property": by_property,
    })


def get_portfolio_overview() -> Dict:
    """High-level KPIs for the entire portfolio (all years)."""
    df = load_data()
    props = list_properties()
    years = list_years()

    by_year = (
        df.groupby("year")["profit"]
        .sum()
        .reset_index()
        .rename(columns={"profit": "net_profit"})
        .sort_values("year")
        .to_dict(orient="records")
    )

    return _to_native({
        "entity": "PropCo",
        "property_count": len(props),
        "tenant_count": len(list_tenants()),
        "years": years,
        "total_records": len(df),
        "all_time_revenue": round(
            df[df["ledger_type"] == "revenue"]["profit"].sum(), 2
        ),
        "all_time_expenses": round(
            df[df["ledger_type"] == "expenses"]["profit"].sum(), 2
        ),
        "all_time_net": round(df["profit"].sum(), 2),
        "by_year": by_year,
        "properties": props,
    })

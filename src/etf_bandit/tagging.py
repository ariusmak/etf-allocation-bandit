"""ETF tagging from metadata text (category + shortName + longName).

Mirrors the regex rules in `1_ETF_Tagging.ipynb` verbatim so any panel built
here matches the notebook output exactly.
"""

from __future__ import annotations

import pandas as pd

from .config import TAG_COLS

# Primary asset-type keyword sets
GOV_KW = (
    r"\b(treasury|treasuries|t[- ]?bill|t[- ]?note|t[- ]?bond|sovereign|government bond)\b"
    r"|\b(tips|inflation[- ]?protected)\b"
)

CREDIT_KW = (
    r"\b(corporate bond|investment grade|high yield|junk bond)\b"
    r"|\b(credit|spread)\b"
    r"|\b(muni|municipal)\b"
    r"|\b(bank loan|leveraged loan|loan)\b"
    r"|\b(floating[- ]?rate|frn|floating rate note)\b"
    r"|\b(preferred)\b"
    r"|\b(securitized|mortgage[- ]?backed|mbs|abs|cmo|cmbs)\b"
    r"|\b(emerging markets bond|multisector bond)\b"
)

MACRO_KW = (
    r"\b(currency|currencies|forex|fx|foreign exchange|currency hedg(e|ed|ing)|currency strategy|carry)\b"
    r"|\b(commodit(y|ies))\b"
    r"|\b(gold|silver|platinum|palladium)\b"
    r"|\b(crude|oil|wti|brent|natural gas|gasoline)\b"
    r"|\b(agriculture|corn|wheat|soy(bean)?s?|sugar|coffee|cocoa|cotton)\b"
    r"|\b(industrial metals|copper|aluminum|nickel|zinc)\b"
    r"|\b(uranium)\b"
)

BOND_CONTEXT_KW = (
    r"\b(bond|bonds|loan|loans|debt|credit|corporate|treasury|muni|municipal|investment grade|high yield)\b"
    r"|\b(floating[- ]?rate|frn|securitized|mbs|abs|cmo|cmbs)\b"
)

EQUITY_KW = (
    r"\b(equity|equities|stock|stocks)\b"
    r"|\b(s&p|msci|russell|nasdaq|dow)\b"
    r"|\b(dividend|growth|value|momentum|quality|small[- ]?cap|mid[- ]?cap|large[- ]?cap)\b"
    r"|\b(communication services|technology|health care|financials|industrials|energy|materials|utilities|consumer( discretionary| staples)?)\b"
    r"|\b(reit|real estate)\b"
)

INTL_KW = (
    r"\b(international|global|world|foreign)\b"
    r"|\b(ex[- ]?us|non[- ]?us)\b"
    r"|\b(emerging|developed|eafe|frontier)\b"
    r"|\b(china|japan|europe|pacific|asia|latam|eurozone|uk)\b"
)

FX_ONLY_KW = r"\b(currency|currencies|forex|fx|foreign exchange|currency hedg(e|ed|ing)|currency strategy|carry)\b"
COMMODITY_PART_KW = (
    r"\b(commodit(y|ies)|gold|silver|platinum|palladium|crude|oil|wti|brent|"
    r"natural gas|gasoline|agriculture|corn|wheat|soy(bean)?s?|sugar|coffee|"
    r"cocoa|cotton|industrial metals|copper|aluminum|nickel|zinc|uranium)\b"
)


def _prep_text(meta: pd.DataFrame) -> pd.Series:
    m = meta.copy()
    for c in ["category", "shortName", "longName"]:
        if c not in m.columns:
            m[c] = ""
        m[c] = m[c].fillna("").astype(str)
    return (m["category"] + " " + m["shortName"] + " " + m["longName"]).str.lower()


def tag_from_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """Produce auto tags + needs_review flag from a metadata frame.

    Input columns expected: ETF, category, shortName, longName.
    Output columns: ETF + TAG_COLS + needs_review.
    """
    text = _prep_text(meta)

    hit_gov = text.str.contains(GOV_KW, regex=True)
    hit_credit = text.str.contains(CREDIT_KW, regex=True)
    hit_equity = text.str.contains(EQUITY_KW, regex=True)
    hit_intl = text.str.contains(INTL_KW, regex=True)

    hit_macro_raw = text.str.contains(MACRO_KW, regex=True)
    hit_bond_ctx = text.str.contains(BOND_CONTEXT_KW, regex=True)
    hit_fx_only = text.str.contains(FX_ONLY_KW, regex=True)
    hit_commodity = text.str.contains(COMMODITY_PART_KW, regex=True)

    hit_macro = hit_macro_raw & ~(hit_bond_ctx & hit_fx_only & ~hit_commodity)

    out = pd.DataFrame({"ETF": meta["ETF"].values})
    out["is_gov"] = hit_gov.astype(int).values
    out["is_credit"] = hit_credit.astype(int).values
    out["is_macro"] = hit_macro.astype(int).values
    out["is_international"] = hit_intl.astype(int).values
    out["is_equity"] = (hit_equity & ~(hit_gov | hit_credit | hit_macro)).astype(int).values

    no_primary = (out[["is_equity", "is_gov", "is_credit", "is_macro"]].sum(axis=1) == 0)
    equity_bond = (out["is_equity"] == 1) & ((out["is_gov"] == 1) | (out["is_credit"] == 1))
    gov_credit_mixed = (out["is_gov"] == 1) & (out["is_credit"] == 1)
    out["needs_review"] = (no_primary | equity_bond | gov_credit_mixed).astype(int)

    return out[["ETF"] + TAG_COLS + ["needs_review"]]


def apply_manual_overrides(auto_tags: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Merge manual tag overrides over the auto-tagged frame.

    Manual values take precedence wherever provided.
    """
    manual = manual.copy()
    for c in TAG_COLS:
        if c in manual.columns:
            manual[c] = pd.to_numeric(manual[c], errors="coerce").astype("Int64")

    merged = auto_tags.merge(manual, on="ETF", how="left", suffixes=("", "_manual"))
    for c in TAG_COLS:
        mc = f"{c}_manual"
        if mc in merged.columns:
            merged[c] = merged[mc].combine_first(merged[c])
            merged = merged.drop(columns=[mc])

    return merged.sort_values("ETF").reset_index(drop=True)

"""
Phase 5 — RAG Compliance Agent
compliance_agent.py

PURPOSE:
    Given a listing's metadata, determines whether the listing is compliant with
    local short-term rental regulations (AMA for Athens, Loi Le Meur for Paris)
    and returns the specific regulation article(s) that apply.

    This module is designed to be imported and called by the Phase 6 LangGraph
    orchestrator as one of its five specialist nodes. It can also be run
    standalone via the ARIA_RAG_v1.ipynb notebook.

HOW IT WORKS:
    1. Rule-based compliance checks (fast, deterministic):
       - Is there a license number? → If not, primary violation (AMA-Art-2 / LeLMeur-Art-4)
       - Is it a multi-listing professional host? → AMA-Art-6 check
       - Is estimated revenue above the tax threshold? → Tax regime check
       - Are nights implying cap breach? → Night cap check

    2. RAG retrieval (semantic search):
       - Build a natural language query describing the listing's compliance profile
       - Retrieve the top-3 most relevant regulation articles from ChromaDB
       - Return article IDs, titles, and the exact text excerpt as citations

    3. Optional GPT-4o synthesis:
       - If OPENAI_API_KEY is set in the environment, calls GPT-4o to write a
         human-readable compliance explanation suitable for the KPMG presentation
       - If no key is available, returns a structured rule-based explanation instead

OUTPUT FORMAT:
    {
        "listing_id":        int or str,
        "city":              "athens" or "paris",
        "compliant":         bool,
        "risk_level":        "COMPLIANT" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
        "violations":        list of violation dicts,
        "primary_article":   str (article_id of most relevant violation),
        "citation":          str (exact article text excerpt),
        "penalty_summary":   str,
        "explanation":       str (GPT-4o if key available, rule-based otherwise),
        "rag_articles":      list of top retrieved articles,
    }

INTEGRATION WITH LANGGRAPH (Phase 6):
    from rag.compliance_agent import check_compliance, load_collection

    collection = load_collection()          # call once at agent startup
    result = check_compliance(listing_row, collection)
"""

import os
import json
from typing import Optional
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── Configuration ──────────────────────────────────────────────────────────────

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(RAG_DIR, "chroma_db")
COLLECTION_NAME = "aria_regulations"

# Thresholds derived from EDA and regulatory text
ATHENS_NIGHT_CAP = 90         # AMA-Art-3: primary residence cap
PARIS_NIGHT_CAP = 120         # LeLMeur-Art-2: primary residence cap
PARIS_TAX_THRESHOLD = 15_000  # LeLMeur-Art-6: micro-BIC threshold (euros/year)
# estimated_occupancy_l365d is already in DAYS (0–365), not a rate — use directly


# ── ChromaDB Collection Loader ─────────────────────────────────────────────────

def load_collection() -> chromadb.Collection:
    """
    Load the ChromaDB collection from disk.
    Call this ONCE at startup, then pass the collection to check_compliance().

    Raises FileNotFoundError if build_index.py has not been run yet.
    """
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"ChromaDB index not found at {CHROMA_PATH}. "
            "Please run `python rag/build_index.py` first."
        )

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


# ── Rule-Based Compliance Checks ───────────────────────────────────────────────

def _run_rule_checks(listing: dict) -> list[dict]:
    """
    Fast deterministic checks against the listing's data fields.
    Returns a list of violation dicts — empty list means no rule-based violations.

    Each violation has:
      - article_id: which article is violated
      - trigger: which rule fired
      - detail: human-readable description of what was found
      - severity: CRITICAL / HIGH / MEDIUM / LOW
    """
    violations = []
    city = str(listing.get("city", "")).lower()

    # ── Check 1: No license number (applies to both cities) ──────────────────
    # The license field should contain a valid registration string.
    # has_license is False, or license is NaN/empty → primary violation.
    license_val = listing.get("license", None)
    has_license = listing.get("has_license", None)

    license_missing = (
        has_license is False
        or has_license == 0
        or license_val is None
        or str(license_val).strip().lower() in ("", "nan", "none", "false", "0")
    )

    if license_missing:
        if city == "athens":
            violations.append({
                "article_id": "AMA-Art-2",
                "trigger":    "no_license",
                "detail":     "No AADE registration number found. Listing is operating without a valid Greek STR license.",
                "severity":   "CRITICAL",
            })
        elif city == "paris":
            violations.append({
                "article_id": "LeLMeur-Art-4",
                "trigger":    "no_license",
                "detail":     "No municipal registration number found. Listing is operating without a valid French STR registration.",
                "severity":   "CRITICAL",
            })

    # ── Check 2: Night cap breach (estimated occupancy is already in days 0–365) ─
    occupancy_rate = listing.get("estimated_occupancy_l365d", None)
    if occupancy_rate is not None:
        try:
            estimated_nights = float(occupancy_rate)  # already in days — use directly
        except (ValueError, TypeError):
            estimated_nights = None

        if estimated_nights is not None:
            if city == "athens" and estimated_nights > ATHENS_NIGHT_CAP:
                violations.append({
                    "article_id": "AMA-Art-3",
                    "trigger":    "high_availability_secondary",
                    "detail":     (
                        f"Estimated {estimated_nights:.0f} occupied nights/year exceeds the "
                        f"90-night primary-residence cap (AMA-Art-3). "
                        "If this is a secondary property, business registration is required."
                    ),
                    "severity":   "HIGH",
                })
            elif city == "paris" and estimated_nights > PARIS_NIGHT_CAP:
                violations.append({
                    "article_id": "LeLMeur-Art-2",
                    "trigger":    "high_availability_primary",
                    "detail":     (
                        f"Estimated {estimated_nights:.0f} occupied nights/year exceeds the "
                        f"120-night primary-residence cap (Loi Le Meur Art-2). "
                        "If this is a secondary property, a changement d'usage authorisation is required."
                    ),
                    "severity":   "HIGH",
                })

    # ── Check 3: Paris tax threshold (Loi Le Meur Art-6) ─────────────────────
    # estimated_revenue_l365d is annual revenue in euros
    if city == "paris":
        revenue = listing.get("estimated_revenue_l365d", None)
        if revenue is not None:
            try:
                revenue = float(revenue)
            except (ValueError, TypeError):
                revenue = None
            if revenue is not None and revenue > PARIS_TAX_THRESHOLD:
                violations.append({
                    "article_id": "LeLMeur-Art-6",
                    "trigger":    "tax_regime",
                    "detail":     (
                        f"Estimated annual revenue €{revenue:,.0f} exceeds the €15,000 "
                        "micro-BIC threshold. Host must use the réel simplifié accounting "
                        "regime and register for VAT (Loi Le Meur Art-6)."
                    ),
                    "severity":   "MEDIUM",
                })

    # ── Check 4: Athens multi-listing professional threshold (AMA-Art-6) ──────
    # host_total_listings_count >= 3 triggers professional operator classification
    if city == "athens":
        total_listings = listing.get("host_total_listings_count", None)
        if total_listings is not None:
            try:
                total_listings = float(total_listings)
            except (ValueError, TypeError):
                total_listings = None
            if total_listings is not None and total_listings >= 3:
                violations.append({
                    "article_id": "AMA-Art-6",
                    "trigger":    "multi_listing_professional",
                    "detail":     (
                        f"Host manages {total_listings:.0f} listings. "
                        "Under AMA-Art-6, operating 3+ STR properties as a natural person "
                        "requires formal business registration as a professional STR operator."
                    ),
                    "severity":   "MEDIUM",
                })

    return violations


# ── RAG Retrieval ──────────────────────────────────────────────────────────────

def _build_query(listing: dict, violations: list[dict]) -> str:
    """
    Build a natural language query string from listing facts and known violations.
    This query is sent to ChromaDB for semantic search.
    The more precise the query, the better the article retrieval.
    """
    city = listing.get("city", "unknown")
    license_val = listing.get("license", "none")
    room_type = listing.get("room_type", "unknown")
    occupancy = listing.get("estimated_occupancy_l365d", "unknown")
    revenue = listing.get("estimated_revenue_l365d", "unknown")
    total_listings = listing.get("host_total_listings_count", 1)

    # Start with core facts about the listing
    query_parts = [
        f"city: {city}",
        f"license number: {license_val}",
        f"room type: {room_type}",
        f"estimated occupancy rate: {occupancy}",
        f"estimated annual revenue: {revenue} euros",
        f"host total listings: {total_listings}",
    ]

    # Add violation triggers found by rule checks so RAG focuses on relevant articles
    if violations:
        triggers = [v["trigger"] for v in violations]
        query_parts.append(f"compliance issues detected: {', '.join(triggers)}")

    return " | ".join(query_parts)


def _retrieve_articles(query: str, collection: chromadb.Collection, n: int = 3) -> list[dict]:
    """
    Query ChromaDB and return the top-n most semantically relevant regulation articles.
    ChromaDB converts the query to a vector and finds the nearest article vectors.
    """
    results = collection.query(query_texts=[query], n_results=n)

    articles = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        articles.append({
            "article_id":         meta["article_id"],
            "title":              meta["title"],
            "city":               meta["city"],
            "law":                meta["law"],
            "compliance_trigger": meta["compliance_trigger"],
            "penalty":            meta["penalty"],
            "text_excerpt":       doc[:300] + "..." if len(doc) > 300 else doc,
            "relevance_score":    round(1 - distance, 4),  # convert distance to similarity
        })
    return articles


# ── Optional GPT-4o Explanation ────────────────────────────────────────────────

def _gpt4o_explanation(listing: dict, violations: list[dict], rag_articles: list[dict]) -> Optional[str]:
    """
    If OPENAI_API_KEY is set, call GPT-4o to write a human-readable compliance
    summary suitable for the KPMG presentation. Otherwise returns None.

    Phase 6 can also call GPT-4o at the orchestration level — this is optional.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None  # Fall back to rule-based explanation

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        violations_text = json.dumps(violations, indent=2) if violations else "None detected"
        articles_text = "\n".join(
            f"- [{a['article_id']}] {a['title']}: {a['text_excerpt']}"
            for a in rag_articles[:2]
        )

        prompt = f"""You are ARIA, a compliance specialist for short-term rental regulations.
Analyse this listing and produce a concise compliance assessment for a KPMG investment report.

LISTING:
- City: {listing.get('city')}
- Neighbourhood: {listing.get('neighbourhood', 'unknown')}
- License: {listing.get('license', 'none')}
- Room type: {listing.get('room_type', 'unknown')}
- Estimated occupancy: {listing.get('estimated_occupancy_l365d', 'unknown')} (rate 0–1)
- Estimated annual revenue: €{listing.get('estimated_revenue_l365d', 'unknown')}
- Host total listings: {listing.get('host_total_listings_count', 'unknown')}

VIOLATIONS DETECTED:
{violations_text}

MOST RELEVANT REGULATION ARTICLES:
{articles_text}

Write a 3-sentence compliance assessment:
1. Overall compliance status (compliant / non-compliant)
2. The primary regulatory risk and which article it violates
3. The recommended action for the investor or regulator

Be precise, professional, and cite article numbers."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.2,  # low temperature for factual/legal content
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # If OpenAI call fails for any reason, fall back to rule-based explanation
        return f"[GPT-4o unavailable: {str(e)}]"


def _rule_based_explanation(listing: dict, violations: list[dict], rag_articles: list[dict]) -> str:
    """
    Fallback explanation when no OpenAI key is available.
    Constructs a structured compliance summary from rule checks and RAG results.
    """
    city = listing.get("city", "unknown")
    neighbourhood = listing.get("neighbourhood", "unknown")

    if not violations:
        primary = rag_articles[0] if rag_articles else {}
        return (
            f"Listing in {neighbourhood} ({city}) appears compliant based on available data. "
            f"No primary violations detected. "
            f"Most relevant regulation: [{primary.get('article_id', 'N/A')}] {primary.get('title', '')}. "
            "Routine monitoring recommended."
        )

    primary_v = violations[0]  # highest-severity violation (checks are ordered by severity)
    top_article = rag_articles[0] if rag_articles else {}

    return (
        f"NON-COMPLIANT: Listing in {neighbourhood} ({city}) violates "
        f"[{primary_v['article_id']}]: {primary_v['detail']} "
        f"Regulatory reference: {top_article.get('law', '')}. "
        f"Penalty exposure: {top_article.get('penalty', 'See regulation')}."
    )


# ── Risk Level Classifier ──────────────────────────────────────────────────────

def _classify_risk(violations: list[dict]) -> str:
    """
    Assign an overall risk level based on the most severe violation found.
    CRITICAL → CRITICAL | HIGH → HIGH | MEDIUM → MEDIUM | LOW → LOW | none → COMPLIANT
    """
    if not violations:
        return "COMPLIANT"
    severities = [v["severity"] for v in violations]
    if "CRITICAL" in severities:
        return "CRITICAL"
    if "HIGH" in severities:
        return "HIGH"
    if "MEDIUM" in severities:
        return "MEDIUM"
    return "LOW"


# ── Main Public Function ───────────────────────────────────────────────────────

def check_compliance(listing: dict, collection: chromadb.Collection) -> dict:
    """
    Main entry point. Given one listing and the loaded ChromaDB collection,
    returns a full compliance assessment dict.

    Parameters
    ----------
    listing : dict
        A single row from the mega dataset or any output CSV, as a dict.
        Must contain at minimum: listing_id, city, license (or has_license).
        All other fields (occupancy, revenue, etc.) add precision but are optional.

    collection : chromadb.Collection
        The ChromaDB collection loaded by load_collection(). Pass it once at
        startup and reuse across all listings for efficiency.

    Returns
    -------
    dict with keys: listing_id, city, compliant, risk_level, violations,
                    primary_article, citation, penalty_summary, explanation,
                    rag_articles
    """
    listing_id = listing.get("listing_id", "unknown")
    city = str(listing.get("city", "")).lower()

    # Step 1 — Rule-based checks (fast, deterministic)
    violations = _run_rule_checks(listing)

    # Step 2 — Build a natural language query and retrieve relevant articles
    query = _build_query(listing, violations)
    rag_articles = _retrieve_articles(query, collection, n=3)

    # Step 3 — Determine primary article (most severe violation, or top RAG result)
    if violations:
        primary_article_id = violations[0]["article_id"]
        # Find the full article text for the primary violation from RAG results
        primary_match = next(
            (a for a in rag_articles if a["article_id"] == primary_article_id),
            rag_articles[0] if rag_articles else {}
        )
    else:
        primary_match = rag_articles[0] if rag_articles else {}
        primary_article_id = primary_match.get("article_id", "N/A")

    # Step 4 — Determine overall compliance and risk level
    compliant = len(violations) == 0
    risk_level = _classify_risk(violations)

    # Step 5 — Generate explanation (GPT-4o if key available, rule-based otherwise)
    gpt_explanation = _gpt4o_explanation(listing, violations, rag_articles)
    explanation = gpt_explanation or _rule_based_explanation(listing, violations, rag_articles)

    # Step 6 — Build and return the result dict
    return {
        "listing_id":      listing_id,
        "city":            city,
        "compliant":       compliant,
        "risk_level":      risk_level,
        "violations":      violations,
        "primary_article": primary_article_id,
        "citation":        primary_match.get("text_excerpt", ""),
        "penalty_summary": primary_match.get("penalty", ""),
        "explanation":     explanation,
        "rag_articles":    rag_articles,
    }


# ── Batch Runner ───────────────────────────────────────────────────────────────

def run_batch(listings_df, collection: chromadb.Collection) -> list[dict]:
    """
    Run check_compliance on every row of a DataFrame.
    Returns a list of result dicts — pass to pd.DataFrame() to save as CSV.

    Parameters
    ----------
    listings_df : pd.DataFrame
        Any DataFrame with listing rows. Each row must have listing_id and city at minimum.
    collection : chromadb.Collection
        The loaded ChromaDB collection.
    """
    results = []
    total = len(listings_df)
    for i, (_, row) in enumerate(listings_df.iterrows()):
        result = check_compliance(row.to_dict(), collection)
        results.append(result)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Processed {i + 1}/{total} listings...")
    return results


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test with two synthetic listings — one compliant, one not
    print("Loading ChromaDB collection...")
    col = load_collection()

    test_listings = [
        {
            "listing_id": 99001,
            "city": "athens",
            "neighbourhood": "ΕΜΠΟΡΙΚΟ ΤΡΙΓΩΝΟ-ΠΛΑΚΑ",
            "license": None,               # no license → CRITICAL violation
            "has_license": False,
            "room_type": "Entire home/apt",
            "estimated_occupancy_l365d": 0.45,
            "estimated_revenue_l365d": 5200,
            "host_total_listings_count": 1,
        },
        {
            "listing_id": 99002,
            "city": "athens",
            "neighbourhood": "ΚΟΛΩΝΑΚΙ",
            "license": "00000874421",       # has license → start compliant
            "has_license": True,
            "room_type": "Entire home/apt",
            "estimated_occupancy_l365d": 0.20,
            "estimated_revenue_l365d": 2500,
            "host_total_listings_count": 1,
        },
    ]

    for listing in test_listings:
        print(f"\n{'='*60}")
        result = check_compliance(listing, col)
        print(f"Listing {result['listing_id']} | City: {result['city']}")
        print(f"  Compliant:     {result['compliant']}")
        print(f"  Risk level:    {result['risk_level']}")
        print(f"  Primary art.:  {result['primary_article']}")
        print(f"  Violations:    {len(result['violations'])}")
        print(f"  Explanation:   {result['explanation'][:200]}")

"""
SPARQL query templates for the Open Research Knowledge Graph (ORKG).

DESIGN PRINCIPLE: These queries search the graph structure (contributions,
methods, datasets, research problems, fields) rather than naively matching
paper titles. Title matching is only used as a fallback strategy.

ORKG ontology key predicates (verified against https://orkg.org/triplestore):
  orkgp:P26         — doi
  orkgp:P27         — author (legacy; current papers use hasAuthors)
  orkgp:P28         — publication MONTH
  orkgp:P29         — publication YEAR
  orkgp:P30         — research field
  orkgp:P31         — contribution
  orkgp:P32         — research problem
  orkgp:P36         — has subfield (research field hierarchy)
  orkgp:hasAuthors  — authors list (an rdf:Seq with rdf:_1, rdf:_2, ... members)

Contribution values are either resources (with an rdfs:label) or plain
literals (~43% of all contribution values, e.g. "Multi-layer Bidirectional
Transformer Encoder"). Entity queries match both.

Scanning every label with CONTAINS/REGEX cannot use an index, so on the
Virtuoso-backed ORKG endpoint label matches are pre-filtered with the
full-text index (``bif:contains``) and then checked exactly with the
standard FILTER. See ``settings.sparql_fulltext``.

Namespace prefixes:
  orkgr: <http://orkg.org/orkg/resource/>
  orkgp: <http://orkg.org/orkg/predicate/>
  orkgc: <http://orkg.org/orkg/class/>
"""

import re

from backend.config import settings

PREFIXES = """
PREFIX orkgc: <http://orkg.org/orkg/class/>
PREFIX orkgp: <http://orkg.org/orkg/predicate/>
PREFIX orkgr: <http://orkg.org/orkg/resource/>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
""".strip()

# bif:contains is Virtuoso-only; rdflib (local mode) does not support it.
FULLTEXT = settings.sparql_fulltext and not settings.use_local_rdf

# Predicate labels that indicate a contribution value is a method / dataset.
# Used only for ranking — matches under other predicates are still returned.
_METHOD_PRED_RE = "method|model|approach|algorithm|architecture|technique|classifier|baseline"
_DATASET_PRED_RE = "dataset|data set|benchmark|corpus|corpora|evaluation"

# Literal contribution values longer than this are abstracts / free-text
# descriptions, not entity names, and are ignored by entity matching.
_MAX_LITERAL_LEN = 200

# Paper URIs passed to the enrichment queries must be plain absolute IRIs.
_IRI_RE = re.compile(r"https?://[^\s<>\"{}|\\^`]+")


def _sanitize(value: str) -> str:
    """
    Sanitize a user-supplied string before interpolating it into a SPARQL query.

    Removes characters that could break out of a SPARQL string literal
    (double quote, backslash, newlines) and strips leading/trailing whitespace.
    Scientific entity names only need letters, digits, spaces, hyphens,
    dots, plus signs, and underscores — everything else is stripped.
    """
    # Remove chars that can close or escape a SPARQL string literal
    sanitized = value.replace("\\", "").replace('"', "").replace("'", "")
    # Remove SPARQL structural characters
    for ch in ("{", "}", "(", ")", "<", ">", ";", "#"):
        sanitized = sanitized.replace(ch, "")
    # Short SPARQL strings cannot contain raw line breaks; collapse whitespace
    return " ".join(sanitized.split())


def _regex_escape(value: str) -> str:
    """
    Escape regex metacharacters without using backslashes.

    Wrapping a metacharacter in a character class (``+`` → ``[+]``) keeps it
    literal in both Virtuoso and rdflib, and avoids the double escaping a
    backslash needs inside a SPARQL string literal. ``^``, ``[`` and ``]``
    cannot be safely wrapped and are dropped.
    """
    value = value.replace("^", "").replace("[", "").replace("]", "")
    return "".join(f"[{c}]" if c in ".+*?|$" else c for c in value)


def _label_match(var: str, value: str) -> str:
    """
    Return a SPARQL boolean expression that matches an entity label.

    Short entities (≤ 4 chars, typically acronyms like CNN, NER, F1, BERT)
    must appear as a standalone token, so "NER" does not match "mineral".
    The token boundary is an explicit non-alphanumeric class rather than
    ``\\b``, because ``\\b`` fails for terms that start or end with a
    non-word character (``\\bC++\\b`` matches "cytochrome C oxidase").

    Longer entities use a case-insensitive CONTAINS, which is safe for
    multi-word phrases like "convolutional neural network".
    """
    if len(value) <= 4:
        pattern = f"(^|[^A-Za-z0-9]){_regex_escape(value)}([^A-Za-z0-9]|$)"
        return f'REGEX(STR({var}), "{pattern}", "i")'
    return f'CONTAINS(LCASE(STR({var})), LCASE("{value}"))'


def _label_filter(var: str, value: str) -> str:
    """Return a SPARQL FILTER clause for matching an entity label."""
    return f"FILTER({_label_match(var, value)})"


def _fulltext(var: str, value: str) -> str:
    """
    Virtuoso full-text pre-filter for ``var``, or "" when unavailable.

    The phrase is built from the term's alphanumeric tokens (the index
    tokenizes "CIFAR-10" as "cifar 10"). Long terms get a trailing wildcard
    so "transformer" also finds "transformers"; Virtuoso requires at least
    4 leading characters before a wildcard. The exact ``_label_filter`` is
    still applied afterwards, so this only narrows the candidate set.
    """
    if not FULLTEXT:
        return ""
    tokens = re.findall(r"[^\W_]+", value)
    if not tokens:
        return ""
    if len(value) > 4 and len(tokens[-1]) >= 4:
        tokens[-1] += "*"
    return f"""{var} bif:contains "'{' '.join(tokens)}'" ."""


def _contrib_value(
    contrib: str, pred: str, value: str, label: str, term: str | None = None
) -> str:
    """
    Graph pattern binding every labelled value of a contribution.

    Matches both resource values (via their rdfs:label) and short literal
    values. rdf:type and the contribution's own rdfs:label are excluded so
    class names ("Contribution") and "Contribution 1" never count as
    entities; URL literals (code links) are skipped as well.

    If ``term`` is given, the label is pre-filtered with the full-text index
    (Virtuoso requires the bif:contains to follow the pattern binding it).
    """
    ft = _fulltext(label, term) if term else ""
    return f"""{{ {value} rdfs:label {label} . {ft} {contrib} {pred} {value} . }}
        UNION
        {{
            {contrib} {pred} {label} . {ft}
            FILTER(isLiteral({label}) && STRLEN(STR({label})) <= {_MAX_LITERAL_LEN}
                   && !STRSTARTS(STR({label}), "http"))
        }}
        FILTER({pred} NOT IN (rdf:type, rdfs:label))"""


def _relevance(pred: str, label: str, value: str, pred_re: str | None) -> str:
    """
    Expression scoring one match: +2 if the predicate looks like the
    requested role (method / dataset), +1 if the label equals the term.
    """
    exact = f'IF(LCASE(STR({label})) = "{value.lower()}", 1, 0)'
    if pred_re is None:
        return exact
    return (
        f'IF(REGEX(STR(COALESCE({pred}Label, "")), "{pred_re}", "i"), 2, 0) + {exact}'
    )


def _check_iri(uri: str) -> str:
    if not _IRI_RE.fullmatch(uri):
        raise ValueError(f"Not a valid paper IRI: {uri!r}")
    return uri


# Some P29 values are resources rather than year literals; skip those.
_PAPER_META = """OPTIONAL { ?paper orkgp:P26 ?doi . }
    OPTIONAL { ?paper orkgp:P29 ?year . FILTER(isLiteral(?year)) }"""

# Same, for grouped queries that SAMPLE() these into ?doi / ?year.
_PAPER_META_GROUPED = """OPTIONAL { ?paper orkgp:P26 ?d . }
    OPTIONAL { ?paper orkgp:P29 ?y . FILTER(isLiteral(?y)) }"""


# ─── SEMANTIC RETRIEVAL QUERIES ──────────────────────────────────
# These traverse graph relationships, not paper titles.


def _ranked_entity_query(
    value: str, label_var: str, pred_re: str | None, limit: int
) -> str:
    """
    Papers whose contributions have a value matching ``value``.

    The inner query ranks papers by their best match and picks the top
    ``limit`` *distinct* papers; the outer query returns every matching
    (contribution, label) row for just those papers. Without the inner
    GROUP BY, one paper with many matching labels fills the whole LIMIT.
    """
    score = _relevance("?p", "?lbl", value, pred_re)
    pred_label = (
        "OPTIONAL { ?p rdfs:label ?pLabel . }" if pred_re is not None else ""
    )
    return f"""{PREFIXES}

SELECT DISTINCT ?paper ?title ?doi ?year ?{label_var} ?contribLabel WHERE {{
    {{
        SELECT ?paper (MAX(?s) AS ?score) WHERE {{
            {_contrib_value("?c", "?p", "?v", "?lbl", value)}
            {_label_filter("?lbl", value)}
            ?paper orkgp:P31 ?c .
            {pred_label}
            BIND({score} AS ?s)
        }}
        GROUP BY ?paper
        ORDER BY DESC(?score) ?paper
        LIMIT {limit}
    }}
    ?paper rdf:type orkgc:Paper ;
           rdfs:label ?title ;
           orkgp:P31 ?contrib .
    ?contrib rdfs:label ?contribLabel .
    {_contrib_value("?contrib", "?pred", "?val", f"?{label_var}")}
    {_label_filter(f"?{label_var}", value)}

    {_PAPER_META}
}}
ORDER BY DESC(?score) ?paper
LIMIT {limit * 5}
"""


def papers_by_method(method: str, limit: int = 5) -> str:
    """
    Find papers whose contributions mention a specific method.

    Traverses: Paper → Contribution → * → Method entity or literal.
    Values under method-like predicates (method, model, approach, ...)
    and exact label matches are ranked first.
    """
    return _ranked_entity_query(_sanitize(method), "methodLabel", _METHOD_PRED_RE, limit)


def papers_by_dataset(dataset: str, limit: int = 5) -> str:
    """
    Find papers whose contributions reference a specific dataset.

    Traverses: Paper → Contribution → * → Dataset entity or literal.
    Values under dataset-like predicates (dataset, benchmark, corpus, ...)
    and exact label matches are ranked first.
    """
    return _ranked_entity_query(_sanitize(dataset), "datasetLabel", _DATASET_PRED_RE, limit)


def papers_by_research_problem(problem: str, limit: int = 5) -> str:
    """
    Find papers that address a specific research problem/task.

    Traverses: Paper → Contribution → P32 (research problem) → Problem entity
    Exact problem-label matches rank first, then newer papers; one row per paper.
    """
    problem = _sanitize(problem)
    return f"""{PREFIXES}

SELECT ?paper (SAMPLE(?t) AS ?title) (SAMPLE(?d) AS ?doi) (SAMPLE(?y) AS ?year)
       (SAMPLE(?problemLabel) AS ?problemLabel) (MAX(?e) AS ?exact) WHERE {{
    ?problem rdfs:label ?problemLabel .
    {_fulltext("?problemLabel", problem)}
    {_label_filter("?problemLabel", problem)}
    ?contrib orkgp:P32 ?problem .
    ?paper orkgp:P31 ?contrib ;
           rdf:type orkgc:Paper ;
           rdfs:label ?t .
    BIND(IF(LCASE(STR(?problemLabel)) = "{problem.lower()}", 1, 0) AS ?e)

    {_PAPER_META_GROUPED}
}}
GROUP BY ?paper
ORDER BY DESC(?exact) DESC(?year) ?paper
LIMIT {limit}
"""


def papers_by_research_field(field: str, limit: int = 5) -> str:
    """
    Find papers in a specific research field, including its subfields.

    Traverses: Field --P36* (has subfield)--> Field ← P30 (research field) ← Paper
    A paper filed under "Machine Learning" is therefore found for
    "Computer Sciences". Papers directly in the matched field rank first,
    then newer papers. ``?fieldLabel`` is the paper's own (most specific) field.
    """
    field = _sanitize(field)
    return f"""{PREFIXES}

SELECT ?paper (SAMPLE(?t) AS ?title) (SAMPLE(?d) AS ?doi) (SAMPLE(?y) AS ?year)
       (SAMPLE(?paperFieldLabel) AS ?fieldLabel) (MAX(?dm) AS ?direct) WHERE {{
    ?matched rdf:type orkgc:ResearchField ;
             rdfs:label ?matchedLabel .
    {_label_filter("?matchedLabel", field)}
    ?matched orkgp:P36* ?fieldRes .
    ?paper orkgp:P30 ?fieldRes ;
           rdf:type orkgc:Paper ;
           rdfs:label ?t .
    ?fieldRes rdfs:label ?paperFieldLabel .
    BIND(IF(?fieldRes = ?matched, 1, 0) AS ?dm)

    {_PAPER_META_GROUPED}
}}
GROUP BY ?paper
ORDER BY DESC(?direct) DESC(?year) ?paper
LIMIT {limit}
"""


def _matching_values(paper: str, contrib: str, value: str, label: str, term: str) -> str:
    """
    Sub-select of DISTINCT (paper, contribution, value, label) rows whose
    label matches ``term``. ``value`` identifies the matched node: the
    resource itself, or the literal for literal values.
    """
    raw = value + "Raw"
    pred = value + "Pred"
    return f"""{{
        SELECT DISTINCT {paper} {contrib} {value} {label} WHERE {{
            {_contrib_value(contrib, pred, raw, label, term)}
            {_label_filter(label, term)}
            {paper} orkgp:P31 {contrib} .
            BIND(COALESCE({raw}, {label}) AS {value})
        }}
    }}"""


def papers_comparing_methods(method_a: str, method_b: str, limit: int = 5) -> str:
    """
    Find papers whose contributions mention BOTH methods.

    This is the core method-comparison query. Each method is matched in its
    own sub-select, and the two sets are joined on the paper — far cheaper
    than one pattern whose two unrestricted ``?contrib ?pred ?value`` scans
    multiply each other.

    Papers where A and B are *different* values rank first; a single label
    mentioning both (e.g. "BERT-BiLSTM-CRF") still matches, but lower.
    """
    method_a = _sanitize(method_a)
    method_b = _sanitize(method_b)
    return f"""{PREFIXES}

SELECT ?paper (SAMPLE(?t) AS ?title) (SAMPLE(?d) AS ?doi) (SAMPLE(?y) AS ?year)
       (SAMPLE(?la) AS ?methodALabel) (SAMPLE(?lb) AS ?methodBLabel)
       (MAX(?sep) AS ?separate) WHERE {{
    {_matching_values("?paper", "?ca", "?va", "?la", method_a)}
    {_matching_values("?paper", "?cb", "?vb", "?lb", method_b)}
    ?paper rdf:type orkgc:Paper ;
           rdfs:label ?t .
    BIND(IF(?va != ?vb, 1, 0) AS ?sep)

    {_PAPER_META_GROUPED}
}}
GROUP BY ?paper
ORDER BY DESC(?separate) DESC(?year) ?paper
LIMIT {limit}
"""


def papers_by_method_and_dataset(
    method: str, dataset: str, limit: int = 5
) -> str:
    """
    Find papers whose contributions mention BOTH the method AND the dataset.

    Method and dataset are matched in separate sub-selects and joined on the
    paper. Papers where both appear in the SAME contribution (the common
    ORKG pattern) rank first; cross-contribution matches follow.

    The method and dataset must be different values, so one long label that
    happens to contain both terms does not count as a match.
    """
    method = _sanitize(method)
    dataset = _sanitize(dataset)
    return f"""{PREFIXES}

SELECT ?paper (SAMPLE(?t) AS ?title) (SAMPLE(?d) AS ?doi) (SAMPLE(?y) AS ?year)
       (SAMPLE(?lm) AS ?methodLabel) (SAMPLE(?ld) AS ?datasetLabel)
       (MAX(?same) AS ?sameContrib) WHERE {{
    {_matching_values("?paper", "?cm", "?vm", "?lm", method)}
    {_matching_values("?paper", "?cd", "?vd", "?ld", dataset)}
    FILTER(?vm != ?vd)
    ?paper rdf:type orkgc:Paper ;
           rdfs:label ?t .
    BIND(IF(?cm = ?cd, 1, 0) AS ?same)

    {_PAPER_META_GROUPED}
}}
GROUP BY ?paper
ORDER BY DESC(?sameContrib) DESC(?year) ?paper
LIMIT {limit}
"""


def paper_full_contributions(paper_uri: str, limit: int = 200) -> str:
    """
    Retrieve all contribution details for a specific paper.

    Returns every (contribution, predicate, value) triple for the paper,
    including literal values such as scores, sizes and descriptions.
    Used to enrich context after an initial paper is found.

    Raises:
        ValueError: if ``paper_uri`` is not a plain http(s) IRI.
    """
    paper_uri = _check_iri(paper_uri)
    return f"""{PREFIXES}

SELECT ?contribLabel ?predLabel ?valueLabel WHERE {{
    <{paper_uri}> orkgp:P31 ?contrib .
    ?contrib rdfs:label ?contribLabel ;
             ?pred ?value .
    FILTER(?pred NOT IN (rdf:type, rdfs:label))
    OPTIONAL {{ ?value rdfs:label ?vl . }}
    BIND(STR(COALESCE(?vl, ?value)) AS ?valueLabel)
    OPTIONAL {{ ?pred rdfs:label ?pl . }}
    BIND(STR(COALESCE(?pl, ?pred)) AS ?predLabel)
}}
LIMIT {limit}
"""


def papers_details(paper_uris: list[str], limit: int = 1500) -> str:
    """
    Full contribution records for several papers in one query (enrichment).

    Returns one row per (paper, contribution, predicate, value) with the
    paper's title / DOI / year, plus the benchmark scores stored two hops
    down (Contribution → PWC_HAS_BENCHMARK → Benchmark → P34 evaluation),
    as predLabel "<benchmark> result" — the numbers claims are verified on.

    Used after a first ranking pass: retrieval queries only return the one
    value that matched (and grouped queries SAMPLE() a single label), so the
    top candidates are re-scored and shown to the LLM with their full record.

    Raises:
        ValueError: if any URI is not a plain http(s) IRI.
    """
    values = " ".join(f"<{_check_iri(u)}>" for u in paper_uris)
    return f"""{PREFIXES}

SELECT ?paper ?title ?doi ?year ?contribLabel ?predLabel ?valueLabel WHERE {{
    VALUES ?paper {{ {values} }}
    ?paper rdfs:label ?title ;
           orkgp:P31 ?contrib .
    ?contrib rdfs:label ?contribLabel .
    {{
        ?contrib ?pred ?value .
        FILTER(?pred NOT IN (rdf:type, rdfs:label, orkgp:PWC_HAS_BENCHMARK))
        OPTIONAL {{ ?value rdfs:label ?vl . }}
        BIND(STR(COALESCE(?vl, ?value)) AS ?valueLabel)
        FILTER(!STRSTARTS(?valueLabel, "http"))
        OPTIONAL {{ ?pred rdfs:label ?pl . }}
        BIND(STR(COALESCE(?pl, ?pred)) AS ?predLabel)
    }}
    UNION
    {{
        ?contrib orkgp:PWC_HAS_BENCHMARK ?bench .
        ?bench rdfs:label ?benchLabel ;
               orkgp:P34 ?eval .
        ?eval rdfs:label ?evalLabel .
        BIND(CONCAT(STR(?benchLabel), " result") AS ?predLabel)
        BIND(REPLACE(STR(?evalLabel), "^Quantity ", "") AS ?valueLabel)
    }}
    {_PAPER_META}
}}
LIMIT {limit}
"""


def paper_metadata(paper_uri: str) -> str:
    """
    Retrieve metadata (title, DOI, year, field, authors) for a specific paper.

    Authors come from the ``hasAuthors`` rdf:Seq (members may be resources
    or literals), falling back to the legacy ``P27`` predicate.

    Raises:
        ValueError: if ``paper_uri`` is not a plain http(s) IRI.
    """
    paper_uri = _check_iri(paper_uri)
    return f"""{PREFIXES}

SELECT (SAMPLE(?t) AS ?title) (SAMPLE(?d) AS ?doi) (SAMPLE(?f) AS ?fieldLabel)
       (SAMPLE(?y) AS ?year)
       (GROUP_CONCAT(DISTINCT ?authorName; separator=", ") AS ?authors) WHERE {{
    <{paper_uri}> rdfs:label ?t .

    OPTIONAL {{ <{paper_uri}> orkgp:P26 ?d . }}
    OPTIONAL {{
        <{paper_uri}> orkgp:P30 ?field .
        ?field rdfs:label ?f .
    }}
    OPTIONAL {{ <{paper_uri}> orkgp:P29 ?y . FILTER(isLiteral(?y)) }}
    OPTIONAL {{
        {{
            <{paper_uri}> orkgp:hasAuthors ?list .
            ?list ?member ?author .
            FILTER(STRSTARTS(STR(?member), "http://www.w3.org/1999/02/22-rdf-syntax-ns#_"))
        }}
        UNION
        {{ <{paper_uri}> orkgp:P27 ?author . }}
        OPTIONAL {{ ?author rdfs:label ?al . }}
        BIND(STR(COALESCE(?al, ?author)) AS ?authorName)
    }}
}}
"""


def claim_evidence(keywords: list[str], limit: int = 15) -> str:
    """
    Search for evidence related to a claim: contributions whose values
    match the claim's key terms, plus their benchmark results.

    Contributions are ranked by how many DIFFERENT keywords they cover
    across all their values (?hits), so a contribution with method=BERT,
    dataset=SQuAD and problem=QA beats a single free-text label that
    happens to mention two of the terms. OR-matching keeps partial evidence.
    Each keyword is a separate full-text-indexed branch, so the OR does not
    force a scan.

    For the selected contributions the query also returns the scores
    stored two hops down (Contribution → PWC_HAS_BENCHMARK → Benchmark →
    P34 (evaluation) → "Quantity F1_95.0"), as predLabel "<benchmark> result".
    These numbers are the actual evidence for "X outperforms Y" claims.

    Values are concatenated per (contribution, predicate), so a contribution
    with twenty benchmark results is one row instead of filling the LIMIT.
    """
    keywords = list(dict.fromkeys(_sanitize(kw) for kw in keywords if _sanitize(kw)))
    if not keywords:
        return ""
    branches = "\n            UNION\n            ".join(
        f"""{{
                {_contrib_value("?c", "?p", "?v", "?lbl", kw)}
                {_label_filter("?lbl", kw)}
                BIND({i} AS ?kw)
            }}"""
        for i, kw in enumerate(keywords)
    )
    any_match = " || ".join(_label_match("?vl", kw) for kw in keywords)
    return f"""{PREFIXES}

SELECT ?paper (SAMPLE(?t) AS ?title) (SAMPLE(?d) AS ?doi) (SAMPLE(?y) AS ?year)
       ?contribLabel ?predLabel
       (GROUP_CONCAT(DISTINCT ?vl; separator=" | ") AS ?valueLabel) ?hits WHERE {{
    {{
        SELECT ?contrib (COUNT(DISTINCT ?kw) AS ?hits) WHERE {{
            {branches}
            BIND(?c AS ?contrib)
        }}
        GROUP BY ?contrib
        ORDER BY DESC(?hits) ?contrib
        LIMIT {limit}
    }}
    ?paper orkgp:P31 ?contrib ;
           rdf:type orkgc:Paper ;
           rdfs:label ?t .
    ?contrib rdfs:label ?contribLabel .
    {{
        # The values that matched the claim's keywords
        {_contrib_value("?contrib", "?pred", "?value", "?vl")}
        FILTER({any_match})
        ?pred rdfs:label ?predLabel .
    }}
    UNION
    {{
        # Benchmark scores of the same contribution
        ?contrib orkgp:PWC_HAS_BENCHMARK ?bench .
        ?bench rdfs:label ?benchLabel ;
               orkgp:P34 ?eval .
        ?eval rdfs:label ?evalLabel .
        BIND(CONCAT(STR(?benchLabel), " result") AS ?predLabel)
        BIND(REPLACE(STR(?evalLabel), "^Quantity ", "") AS ?vl)
    }}

    {_PAPER_META_GROUPED}
}}
GROUP BY ?paper ?contrib ?contribLabel ?predLabel ?hits
ORDER BY DESC(?hits) ?paper ?contribLabel
LIMIT {limit * 3}
"""


def paper_lookup_by_title(title_fragment: str, limit: int = 5) -> str:
    """
    Look up a paper by title fragment (fallback / paper_lookup queries).

    This is the ONLY query that matches on paper title, and it's used
    specifically when the user is looking up a known paper by name.
    Exact title matches rank first, then shorter (closer) titles, so
    "Attention Is All You Need" beats "Attention is All You Need in
    Speech Separation".
    """
    needle = _sanitize(title_fragment).lower()
    return f"""{PREFIXES}

SELECT ?paper (SAMPLE(?t) AS ?title) (SAMPLE(?d) AS ?doi) (SAMPLE(?y) AS ?year)
       (SAMPLE(?f) AS ?fieldLabel) (MAX(?e) AS ?exact) (MIN(?len) AS ?titleLength) WHERE {{
    ?paper rdf:type orkgc:Paper ;
           rdfs:label ?t .
    FILTER(CONTAINS(LCASE(STR(?t)), "{needle}"))
    BIND(IF(LCASE(STR(?t)) = "{needle}", 1, 0) AS ?e)
    BIND(STRLEN(STR(?t)) AS ?len)

    {_PAPER_META_GROUPED}
    OPTIONAL {{
        ?paper orkgp:P30 ?fieldRes .
        ?fieldRes rdfs:label ?f .
    }}
}}
GROUP BY ?paper
ORDER BY DESC(?exact) ?titleLength ?paper
LIMIT {limit}
"""


# ─── FALLBACK: BROAD ENTITY SEARCH ──────────────────────────────
# Used when semantic queries return no results.


def broad_entity_search(keyword: str, limit: int = 5) -> str:
    """
    Broad fallback: find papers that have ANY contribution value matching
    the keyword, regardless of predicate. Exact label matches rank first.
    """
    return _ranked_entity_query(_sanitize(keyword), "entityLabel", None, limit)


def title_keyword_search(keywords: list[str], limit: int = 5) -> str:
    """
    Last-resort fallback: search paper titles for keywords.

    Only used when graph-based queries return no results.
    Short keywords (≤ 4 chars, e.g. "NER", "F1") must appear as a standalone
    token to prevent substring false positives like "NER" matching "mineral".
    Titles matching more keywords rank first.
    """
    keywords = list(dict.fromkeys(_sanitize(kw) for kw in keywords if _sanitize(kw)))
    if not keywords:
        return ""
    matches = [_label_match("?t", kw) for kw in keywords]
    any_match = " || ".join(matches)
    hits = " + ".join(f"IF({m}, 1, 0)" for m in matches)
    return f"""{PREFIXES}

SELECT ?paper (SAMPLE(?t) AS ?title) (SAMPLE(?d) AS ?doi) (SAMPLE(?y) AS ?year)
       (SAMPLE(?f) AS ?fieldLabel) (MAX(?h) AS ?hits) WHERE {{
    ?paper rdf:type orkgc:Paper ;
           rdfs:label ?t .
    FILTER({any_match})
    BIND(({hits}) AS ?h)

    {_PAPER_META_GROUPED}
    OPTIONAL {{
        ?paper orkgp:P30 ?fieldRes .
        ?fieldRes rdfs:label ?f .
    }}
}}
GROUP BY ?paper
ORDER BY DESC(?hits) ?paper
LIMIT {limit}
"""

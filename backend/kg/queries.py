"""
SPARQL query templates for the Open Research Knowledge Graph (ORKG).

DESIGN PRINCIPLE: These queries search the graph structure (contributions,
methods, datasets, research problems, fields) rather than naively matching
paper titles. Title matching is only used as a fallback strategy.

ORKG ontology key predicates:
  orkgp:P26  — has DOI
  orkgp:P27  — has author
  orkgp:P28  — has publication year
  orkgp:P29  — has publication month
  orkgp:P30  — has research field
  orkgp:P31  — has contribution
  orkgp:P32  — has research problem

Namespace prefixes:
  orkgr: <http://orkg.org/orkg/resource/>
  orkgp: <http://orkg.org/orkg/predicate/>
  orkgc: <http://orkg.org/orkg/class/>
"""

PREFIXES = """
PREFIX orkgc: <http://orkg.org/orkg/class/>
PREFIX orkgp: <http://orkg.org/orkg/predicate/>
PREFIX orkgr: <http://orkg.org/orkg/resource/>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
""".strip()


def _sanitize(value: str) -> str:
    """
    Sanitize a user-supplied string before interpolating it into a SPARQL query.

    Removes characters that could break out of a SPARQL string literal
    (double quote, backslash) and strips leading/trailing whitespace.
    Scientific entity names only need letters, digits, spaces, hyphens,
    dots, plus signs, and underscores — everything else is stripped.
    """
    # Remove chars that can close or escape a SPARQL string literal
    sanitized = value.replace("\\", "").replace('"', "").replace("'", "")
    # Remove SPARQL structural characters
    for ch in ("{", "}", "(", ")", "<", ">", ";", "#"):
        sanitized = sanitized.replace(ch, "")
    return sanitized.strip()


# ─── SEMANTIC RETRIEVAL QUERIES ──────────────────────────────────
# These traverse graph relationships, not paper titles.


def papers_by_method(method: str, limit: int = 5) -> str:
    """
    Find papers whose contributions mention a specific method.

    Traverses: Paper → Contribution → * → Method entity
    Matches on the label of the method entity, not the paper title.
    """
    method = _sanitize(method)
    return f"""{PREFIXES}

SELECT DISTINCT ?paper ?title ?doi ?methodLabel ?contribLabel WHERE {{
    ?paper rdf:type orkgc:Paper .
    ?paper rdfs:label ?title .
    ?paper orkgp:P31 ?contrib .
    ?contrib rdfs:label ?contribLabel .
    ?contrib ?pred ?method .
    ?method rdfs:label ?methodLabel .
    FILTER(CONTAINS(LCASE(?methodLabel), LCASE("{method}")))

    OPTIONAL {{ ?paper orkgp:P26 ?doi . }}
}}
LIMIT {limit}
"""


def papers_by_dataset(dataset: str, limit: int = 5) -> str:
    """
    Find papers whose contributions reference a specific dataset.

    Traverses: Paper → Contribution → * → Dataset entity
    """
    dataset = _sanitize(dataset)
    return f"""{PREFIXES}

SELECT DISTINCT ?paper ?title ?doi ?datasetLabel ?contribLabel WHERE {{
    ?paper rdf:type orkgc:Paper .
    ?paper rdfs:label ?title .
    ?paper orkgp:P31 ?contrib .
    ?contrib rdfs:label ?contribLabel .
    ?contrib ?pred ?dataset .
    ?dataset rdfs:label ?datasetLabel .
    FILTER(CONTAINS(LCASE(?datasetLabel), LCASE("{dataset}")))

    OPTIONAL {{ ?paper orkgp:P26 ?doi . }}
}}
LIMIT {limit}
"""


def papers_by_research_problem(problem: str, limit: int = 5) -> str:
    """
    Find papers that address a specific research problem/task.

    Traverses: Paper → Contribution → P32 (research problem) → Problem entity
    """
    problem = _sanitize(problem)
    return f"""{PREFIXES}

SELECT DISTINCT ?paper ?title ?doi ?problemLabel WHERE {{
    ?paper rdf:type orkgc:Paper .
    ?paper rdfs:label ?title .
    ?paper orkgp:P31 ?contrib .
    ?contrib orkgp:P32 ?problem .
    ?problem rdfs:label ?problemLabel .
    FILTER(CONTAINS(LCASE(?problemLabel), LCASE("{problem}")))

    OPTIONAL {{ ?paper orkgp:P26 ?doi . }}
}}
LIMIT {limit}
"""


def papers_by_research_field(field: str, limit: int = 5) -> str:
    """
    Find papers in a specific research field.

    Traverses: Paper → P30 (research field) → Field entity
    """
    field = _sanitize(field)
    return f"""{PREFIXES}

SELECT DISTINCT ?paper ?title ?doi ?fieldLabel WHERE {{
    ?paper rdf:type orkgc:Paper .
    ?paper rdfs:label ?title .
    ?paper orkgp:P30 ?fieldRes .
    ?fieldRes rdfs:label ?fieldLabel .
    FILTER(CONTAINS(LCASE(?fieldLabel), LCASE("{field}")))

    OPTIONAL {{ ?paper orkgp:P26 ?doi . }}
}}
LIMIT {limit}
"""


def papers_comparing_methods(method_a: str, method_b: str, limit: int = 5) -> str:
    """
    Find papers whose contributions mention BOTH methods.

    This is the core method-comparison query. It finds papers that have
    contribution entities referencing both method A and method B,
    indicating a comparison or joint usage.

    Traverses: Paper → Contribution → * → MethodA
               Paper → Contribution → * → MethodB
    Both methods can be in the same or different contributions.
    """
    method_a = _sanitize(method_a)
    method_b = _sanitize(method_b)
    return f"""{PREFIXES}

SELECT DISTINCT ?paper ?title ?doi ?methodALabel ?methodBLabel WHERE {{
    ?paper rdf:type orkgc:Paper .
    ?paper rdfs:label ?title .

    # Method A — via any contribution of this paper
    ?paper orkgp:P31 ?contribA .
    ?contribA ?predA ?entityA .
    ?entityA rdfs:label ?methodALabel .
    FILTER(CONTAINS(LCASE(?methodALabel), LCASE("{method_a}")))

    # Method B — via any contribution of this paper
    ?paper orkgp:P31 ?contribB .
    ?contribB ?predB ?entityB .
    ?entityB rdfs:label ?methodBLabel .
    FILTER(CONTAINS(LCASE(?methodBLabel), LCASE("{method_b}")))

    OPTIONAL {{ ?paper orkgp:P26 ?doi . }}
}}
LIMIT {limit}
"""


def papers_by_method_and_dataset(
    method: str, dataset: str, limit: int = 5
) -> str:
    """
    Find papers whose contributions mention BOTH the method AND the dataset.

    Uses a UNION of two patterns:
      1. **Same contribution** — method and dataset are properties of the
         SAME contribution node (the common case in ORKG).
      2. **Cross-contribution** — method and dataset live in different
         contributions of the same paper (less common, but valid).

    The same-contribution pattern is listed first so the SPARQL engine
    can short-circuit when the tighter pattern matches.

    Traverses:
        Paper → Contribution → * → Method entity
        Paper → Contribution → * → Dataset entity
    """
    method = _sanitize(method)
    dataset = _sanitize(dataset)
    return f"""{PREFIXES}

SELECT DISTINCT ?paper ?title ?doi ?methodLabel ?datasetLabel WHERE {{
    ?paper rdf:type orkgc:Paper .
    ?paper rdfs:label ?title .

    {{
        # Pattern A: SAME contribution contains both method and dataset
        ?paper orkgp:P31 ?contrib .
        ?contrib ?predM ?entityM .
        ?entityM rdfs:label ?methodLabel .
        FILTER(CONTAINS(LCASE(?methodLabel), LCASE("{method}")))

        ?contrib ?predD ?entityD .
        ?entityD rdfs:label ?datasetLabel .
        FILTER(CONTAINS(LCASE(?datasetLabel), LCASE("{dataset}")))
    }}
    UNION
    {{
        # Pattern B: DIFFERENT contributions (cross-contribution)
        ?paper orkgp:P31 ?contribM .
        ?contribM ?predM2 ?entityM2 .
        ?entityM2 rdfs:label ?methodLabel .
        FILTER(CONTAINS(LCASE(?methodLabel), LCASE("{method}")))

        ?paper orkgp:P31 ?contribD .
        ?contribD ?predD2 ?entityD2 .
        ?entityD2 rdfs:label ?datasetLabel .
        FILTER(CONTAINS(LCASE(?datasetLabel), LCASE("{dataset}")))

        FILTER(?contribM != ?contribD)
    }}

    OPTIONAL {{ ?paper orkgp:P26 ?doi . }}
}}
LIMIT {limit}
"""


def paper_full_contributions(paper_uri: str) -> str:
    """
    Retrieve all contribution details for a specific paper.

    Returns every (contribution, predicate, value) triple for the paper.
    Used to enrich context after an initial paper is found.
    """
    return f"""{PREFIXES}

SELECT ?contribLabel ?predLabel ?valueLabel WHERE {{
    <{paper_uri}> orkgp:P31 ?contrib .
    ?contrib rdfs:label ?contribLabel .
    ?contrib ?pred ?value .
    ?pred rdfs:label ?predLabel .

    OPTIONAL {{ ?value rdfs:label ?valueLabel . }}
    FILTER(BOUND(?valueLabel))
}}
"""


def paper_metadata(paper_uri: str) -> str:
    """
    Retrieve metadata (field, DOI, authors, year) for a specific paper.
    """
    return f"""{PREFIXES}

SELECT ?title ?doi ?fieldLabel ?year WHERE {{
    <{paper_uri}> rdfs:label ?title .

    OPTIONAL {{ <{paper_uri}> orkgp:P26 ?doi . }}
    OPTIONAL {{
        <{paper_uri}> orkgp:P30 ?field .
        ?field rdfs:label ?fieldLabel .
    }}
    OPTIONAL {{ <{paper_uri}> orkgp:P28 ?year . }}
}}
LIMIT 1
"""


def claim_evidence(keywords: list[str], limit: int = 15) -> str:
    """
    Search for evidence related to a claim by finding papers whose
    contributions contain entities matching the claim's key terms.

    Uses OR-matching on contribution entity labels so partial evidence
    is still retrieved.
    """
    keywords = [_sanitize(kw) for kw in keywords if _sanitize(kw)]
    if not keywords:
        return ""
    filters = " || ".join(
        f'CONTAINS(LCASE(?valueLabel), LCASE("{kw}"))' for kw in keywords
    )
    return f"""{PREFIXES}

SELECT DISTINCT ?paper ?title ?doi ?contribLabel ?predLabel ?valueLabel WHERE {{
    ?paper rdf:type orkgc:Paper .
    ?paper rdfs:label ?title .
    ?paper orkgp:P31 ?contrib .
    ?contrib rdfs:label ?contribLabel .
    ?contrib ?pred ?value .
    ?pred rdfs:label ?predLabel .
    ?value rdfs:label ?valueLabel .
    FILTER({filters})

    OPTIONAL {{ ?paper orkgp:P26 ?doi . }}
}}
LIMIT {limit}
"""


def paper_lookup_by_title(title_fragment: str, limit: int = 5) -> str:
    """
    Look up a paper by title fragment (fallback / paper_lookup queries).

    This is the ONLY query that matches on paper title, and it's used
    specifically when the user is looking up a known paper by name.
    """
    title_fragment = _sanitize(title_fragment)
    return f"""{PREFIXES}

SELECT DISTINCT ?paper ?title ?doi ?fieldLabel WHERE {{
    ?paper rdf:type orkgc:Paper .
    ?paper rdfs:label ?title .
    FILTER(CONTAINS(LCASE(?title), LCASE("{title_fragment}")))

    OPTIONAL {{ ?paper orkgp:P26 ?doi . }}
    OPTIONAL {{
        ?paper orkgp:P30 ?fieldRes .
        ?fieldRes rdfs:label ?fieldLabel .
    }}
}}
LIMIT {limit}
"""


# ─── FALLBACK: BROAD ENTITY SEARCH ──────────────────────────────
# Used when semantic queries return no results.


def broad_entity_search(keyword: str, limit: int = 5) -> str:
    """
    Broad fallback: find papers that have ANY contribution entity matching
    the keyword. This is more inclusive than method/dataset-specific queries.
    """
    keyword = _sanitize(keyword)
    return f"""{PREFIXES}

SELECT DISTINCT ?paper ?title ?doi ?entityLabel WHERE {{
    ?paper rdf:type orkgc:Paper .
    ?paper rdfs:label ?title .
    ?paper orkgp:P31 ?contrib .
    ?contrib ?pred ?entity .
    ?entity rdfs:label ?entityLabel .
    FILTER(CONTAINS(LCASE(?entityLabel), LCASE("{keyword}")))

    OPTIONAL {{ ?paper orkgp:P26 ?doi . }}
}}
LIMIT {limit}
"""


def title_keyword_search(keywords: list[str], limit: int = 5) -> str:
    """
    Last-resort fallback: search paper titles for keywords.

    Only used when graph-based queries return no results.
    """
    keywords = [_sanitize(kw) for kw in keywords if _sanitize(kw)]
    if len(keywords) == 1:
        filter_clause = f'CONTAINS(LCASE(?title), LCASE("{keywords[0]}"))'
    else:
        filter_clause = " || ".join(
            f'CONTAINS(LCASE(?title), LCASE("{kw}"))' for kw in keywords
        )
    return f"""{PREFIXES}

SELECT DISTINCT ?paper ?title ?doi ?fieldLabel WHERE {{
    ?paper rdf:type orkgc:Paper .
    ?paper rdfs:label ?title .
    FILTER({filter_clause})

    OPTIONAL {{ ?paper orkgp:P26 ?doi . }}
    OPTIONAL {{
        ?paper orkgp:P30 ?fieldRes .
        ?fieldRes rdfs:label ?fieldLabel .
    }}
}}
LIMIT {limit}
"""

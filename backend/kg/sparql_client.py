"""
SPARQL client for querying the ORKG knowledge graph.

Supports two modes:
  1. Remote: queries the ORKG public SPARQL endpoint via SPARQLWrapper
  2. Local:  queries a local RDF dump via rdflib

The mode is controlled by the USE_LOCAL_RDF setting.

Timeout handling:
  Remote queries use ``settings.sparql_timeout`` (default 10 s).
  Queries that exceed the timeout are retried with a simpler
  title-based fallback query.  If the fallback also fails, [].
"""

from __future__ import annotations

import asyncio
import socket
import threading
import urllib.error
from typing import Any
from loguru import logger
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph

from backend.config import settings


class SPARQLTimeoutError(Exception):
    """Raised when a SPARQL query times out so callers can handle it."""


# Module-level result cache: query string → result rows.
# Shared across all SPARQLClient instances (singleton pattern).
# Capped at 256 entries; when full the oldest entry is evicted.
# Protected by a lock so concurrent threads don't corrupt the dict.
_SPARQL_CACHE: dict[str, list[dict[str, Any]]] = {}
_SPARQL_CACHE_MAX = 256
_SPARQL_CACHE_LOCK = threading.Lock()


class SPARQLClient:
    """
    Unified SPARQL client for both remote ORKG endpoint and local RDF graphs.

    Usage (sync):
        client = SPARQLClient()
        results = client.execute(sparql_query_string)

    Usage (async — wraps sync call in executor):
        results = await client.execute_async(sparql_query_string)
    """

    def __init__(self) -> None:
        self._local_graph: Graph | None = None

        if settings.use_local_rdf:
            logger.info(f"Loading local RDF dump from: {settings.local_rdf_path}")
            self._local_graph = Graph()
            self._local_graph.parse(settings.local_rdf_path)
            logger.info(
                f"Loaded {len(self._local_graph)} triples from local RDF file."
            )
        else:
            logger.info(
                f"Using remote ORKG SPARQL endpoint: {settings.orkg_sparql_endpoint}"
            )

    def execute(self, query: str) -> list[dict[str, Any]]:
        """
        Execute a SPARQL SELECT query and return results as a list of dicts.

        Results are cached by query string to avoid redundant network calls.
        Each dict maps variable names → their string values.
        On timeout or error the query is skipped and an empty list is returned.

        Args:
            query: A valid SPARQL SELECT query string.

        Returns:
            List of result rows, each as {variable_name: value_string}.

        Raises:
            SPARQLTimeoutError: When the remote endpoint times out.
                Callers can catch this to trigger a fallback query.
        """
        with _SPARQL_CACHE_LOCK:
            if query in _SPARQL_CACHE:
                logger.debug("SPARQL cache hit")
                return _SPARQL_CACHE[query]

        # Execute outside the lock so other threads can use the cache concurrently
        if self._local_graph is not None:
            results = self._query_local(query)
        else:
            results = self._query_remote(query)

        with _SPARQL_CACHE_LOCK:
            # Evict oldest entry when cache is full
            if len(_SPARQL_CACHE) >= _SPARQL_CACHE_MAX:
                oldest_key = next(iter(_SPARQL_CACHE))
                del _SPARQL_CACHE[oldest_key]
            _SPARQL_CACHE[query] = results

        return results

    async def execute_async(self, query: str) -> list[dict[str, Any]]:
        """
        Execute a SPARQL query asynchronously.

        Wraps the synchronous ``execute()`` call in a thread-pool executor
        so multiple queries can run concurrently with ``asyncio.gather()``.

        Raises:
            SPARQLTimeoutError: propagated from ``execute()``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.execute, query)

    def _query_remote(self, query: str) -> list[dict[str, Any]]:
        """Query the remote ORKG SPARQL endpoint."""
        logger.debug(f"Executing remote SPARQL query:\n{query[:200]}...")

        try:
            sparql = SPARQLWrapper(settings.orkg_sparql_endpoint)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(settings.sparql_timeout)

            response = sparql.query().convert()
            results = self._parse_sparql_json(response)

            logger.info(f"Remote query returned {len(results)} results.")
            return results

        except (socket.timeout, TimeoutError):
            logger.warning(
                f"SPARQL query timed out after {settings.sparql_timeout}s.\n"
                f"  Query (truncated): {query[:120]}..."
            )
            raise SPARQLTimeoutError(
                f"SPARQL timeout after {settings.sparql_timeout}s"
            )
        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout):
                logger.warning(
                    f"SPARQL query timed out after {settings.sparql_timeout}s.\n"
                    f"  Query (truncated): {query[:120]}..."
                )
                raise SPARQLTimeoutError(
                    f"SPARQL timeout after {settings.sparql_timeout}s"
                )
            logger.error(f"SPARQL remote query failed (URLError): {e}")
            return []
        except SPARQLTimeoutError:
            raise  # re-raise our own exception
        except Exception as e:
            logger.error(f"SPARQL remote query failed: {e}")
            return []

    def _query_local(self, query: str) -> list[dict[str, Any]]:
        """Query a local RDF graph using rdflib."""
        logger.debug(f"Executing local SPARQL query:\n{query[:200]}...")

        try:
            assert self._local_graph is not None
            qres = self._local_graph.query(query)
            results = []

            for row in qres:
                result_dict = {}
                for var in qres.vars:
                    value = getattr(row, str(var), None)
                    result_dict[str(var)] = str(value) if value is not None else None
                results.append(result_dict)

            logger.info(f"Local query returned {len(results)} results.")
            return results

        except Exception as e:
            logger.error(f"SPARQL local query failed: {e}")
            return []

    @staticmethod
    def _parse_sparql_json(response: dict) -> list[dict[str, Any]]:
        """
        Parse a SPARQL JSON response into a list of flat dicts.

        The standard SPARQL JSON format is:
        {
            "results": {
                "bindings": [
                    {"var1": {"type": "uri", "value": "..."}, ...},
                    ...
                ]
            }
        }
        """
        rows = []
        bindings = response.get("results", {}).get("bindings", [])
        for binding in bindings:
            row = {}
            for var, val_obj in binding.items():
                row[var] = val_obj.get("value", "")
            rows.append(row)
        return rows


# Module-level convenience instance
sparql_client = SPARQLClient()

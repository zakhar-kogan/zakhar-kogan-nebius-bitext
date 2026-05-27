"""Dataset loading, validation, search, and aggregation helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from rapidfuzz import fuzz

from bitext_agent.schemas import DatasetRow


EXPECTED_COLUMNS = {"flags", "instruction", "category", "intent", "response"}


def normalize_text(value: str | None) -> str:
    """Normalize free text for matching user phrases to dataset fields."""

    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def normalize_label(value: str | None) -> str:
    """Normalize category and intent labels for exact matching."""

    return normalize_text(value).replace(" ", "_")


class DatasetRepository:
    """Read-only repository for the Bitext customer support CSV."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self._df: pl.DataFrame | None = None
        self._search_cache: dict[str, pl.DataFrame] = {}

    def load(self) -> pl.DataFrame:
        """Load and validate the dataset, caching it for later tool calls."""

        if self._df is not None:
            return self._df
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. Run scripts/download_dataset.py first."
            )
        df = pl.read_csv(self.dataset_path)
        missing = EXPECTED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        df = df.with_row_index("row_id")
        df = df.with_columns(
            pl.col("category").cast(pl.Utf8).str.to_uppercase().alias("category"),
            pl.col("intent").cast(pl.Utf8).alias("intent"),
            pl.col("instruction").cast(pl.Utf8).alias("instruction"),
            pl.col("response").cast(pl.Utf8).alias("response"),
        )
        self._df = df
        return df

    def dataset_status(self) -> dict[str, Any]:
        """Return basic dataset diagnostics for Streamlit and README checks."""

        if not self.dataset_path.exists():
            return {"exists": False, "path": str(self.dataset_path), "rows": 0}
        df = self.load()
        return {
            "exists": True,
            "path": str(self.dataset_path),
            "rows": df.height,
            "categories": len(self.list_categories()),
            "intents": len(self.list_intents()),
        }

    def list_categories(self) -> list[str]:
        """List all categories in stable sorted order."""

        return sorted(self.load().get_column("category").unique().to_list())

    def list_intents(self, category: str | None = None) -> list[str]:
        """List all intents, optionally filtered to a category."""

        df = self._filter(category=category)
        return sorted(df.get_column("intent").unique().to_list())

    def search_rows(
        self,
        category: str | None = None,
        intent: str | None = None,
        query: str | None = None,
        fuzzy: bool = True,
        limit: int = 20,
    ) -> tuple[str, int, list[DatasetRow]]:
        """Search dataset rows and cache the full result under a reusable search ID."""

        df = self._filter(category=category, intent=intent)
        if query:
            df = self._query_filter(df, query=query, fuzzy=fuzzy)
        search_id = self._cache_search(df, {"category": category, "intent": intent, "query": query})
        return search_id, df.height, self._rows(df.head(limit))

    def count_rows(
        self,
        category: str | None = None,
        intent: str | None = None,
        search_id: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Count rows by filters or by a previous search ID."""

        if search_id:
            df = self._search_cache.get(search_id)
            if df is None:
                raise KeyError(f"Unknown search_id: {search_id}")
            return df.height, {"search_id": search_id}
        df = self._filter(category=category, intent=intent)
        return df.height, {"category": category, "intent": intent}

    def show_examples(
        self,
        category: str | None = None,
        intent: str | None = None,
        search_id: str | None = None,
        n: int = 3,
        offset: int = 0,
    ) -> tuple[list[DatasetRow], int | None, int]:
        """Return example rows and paging metadata for follow-up requests."""

        df = self._search_cache[search_id] if search_id else self._filter(category=category, intent=intent)
        page = df.slice(offset, n)
        next_offset = offset + n if offset + n < df.height else None
        return self._rows(page), next_offset, df.height

    def intent_distribution(self, category: str | None = None) -> list[dict[str, int | str]]:
        """Return intent counts using DuckDB over the loaded Polars frame."""

        df = self._filter(category=category)
        con = duckdb.connect(database=":memory:")
        con.register("rows", df)
        rows = con.execute(
            "select intent, count(*) as count from rows group by intent order by count desc, intent"
        ).fetchall()
        return [{"intent": row[0], "count": int(row[1])} for row in rows]

    def sample_for_summary(
        self,
        category: str | None = None,
        intent: str | None = None,
        query: str | None = None,
        limit: int = 25,
    ) -> tuple[int, list[DatasetRow]]:
        """Return a bounded sample for summarization."""

        search_id, total, rows = self.search_rows(
            category=category, intent=intent, query=query, fuzzy=True, limit=limit
        )
        _ = search_id
        return total, rows

    def _filter(self, category: str | None = None, intent: str | None = None) -> pl.DataFrame:
        df = self.load()
        if category:
            wanted = normalize_label(category)
            df = df.filter(pl.col("category").map_elements(normalize_label, return_dtype=pl.Utf8) == wanted)
        if intent:
            wanted = normalize_label(intent)
            df = df.filter(pl.col("intent").map_elements(normalize_label, return_dtype=pl.Utf8) == wanted)
        return df

    def _query_filter(self, df: pl.DataFrame, query: str, fuzzy: bool) -> pl.DataFrame:
        original_norm = normalize_text(query)
        query_norm = normalize_text(_expand_query(query))
        if not query_norm:
            return df
        rows: list[dict[str, Any]] = []
        for row in df.iter_rows(named=True):
            haystack = normalize_text(
                " ".join([row["instruction"], row["response"], row["category"], row["intent"]])
            )
            expanded_terms = [term for term in query_norm.split() if len(term) > 2]
            exact = original_norm in haystack or any(term in haystack for term in expanded_terms)
            score = max(
                [fuzz.partial_ratio(original_norm, haystack)]
                + [fuzz.partial_ratio(term, haystack) for term in expanded_terms]
            ) if fuzzy else 0
            if exact or score >= 78:
                row["_score"] = 100 if exact else score
                rows.append(row)
        if not rows:
            return df.head(0)
        return pl.DataFrame(rows).sort("_score", descending=True).drop("_score")

    def _cache_search(self, df: pl.DataFrame, filters: dict[str, Any]) -> str:
        payload = repr(sorted(filters.items())) + str(df.get_column("row_id").head(100).to_list())
        search_id = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        self._search_cache[search_id] = df
        return search_id

    def _rows(self, df: pl.DataFrame) -> list[DatasetRow]:
        return [
            DatasetRow(
                row_id=int(row["row_id"]),
                category=row["category"],
                intent=row["intent"],
                instruction=row["instruction"],
                response=row["response"],
            )
            for row in df.iter_rows(named=True)
        ]


def _expand_query(query: str) -> str:
    """Add assignment-specific synonyms while keeping search deterministic."""

    normalized = normalize_text(query)
    expansions = {
        "money back": "money back refund reimbursement return payment",
        "refunds": "refund reimbursement money back",
        "refund": "refund reimbursement money back",
        "complaints": "complaint complain issue problem dissatisfied",
    }
    return expansions.get(normalized, query)

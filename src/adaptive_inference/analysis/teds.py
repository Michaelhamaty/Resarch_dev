"""TEDS — Tree-Edit-Distance Similarity for HTML tables.

Reference: Zhong et al., 2019, "Image-based table recognition: data,
model, and evaluation" (PubTabNet). This implementation follows the
structure of the authors' reference at
https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/metric.py
but is a clean rewrite to keep dependencies minimal (lxml + apted only).

TEDS in [0, 1]:
    TEDS(t_pred, t_gold) = 1 - edit_distance(t_pred, t_gold) / max(|t_pred|, |t_gold|)

Per-cell text-content cost is normalized edit distance between cell text
contents; structural cost is 1 per mismatched tag. Whitespace inside cells
is collapsed before comparison (matches the reference).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from apted import APTED, Config
from lxml import html as lxml_html


_WHITESPACE_RE = re.compile(r"\s+")
_STRUCTURAL_TAGS = frozenset(
    {"table", "thead", "tbody", "tfoot", "tr", "th", "td"}
)
_TEXT_BEARING_TAGS = frozenset({"th", "td"})


@dataclass
class _Node:
    """Lightweight tree node for APTED."""

    tag: str
    text: str  # normalized; only meaningful for th/td
    children: list["_Node"]


def teds_score(pred_html: str, gold_html: str) -> float:
    """Return TEDS for the first ``<table>`` in pred vs gold.

    If either side has no parseable ``<table>``, returns 0.0 (the model
    failed to produce a table or the GT is unusable).
    """

    pred_tree = _first_table_tree(pred_html)
    gold_tree = _first_table_tree(gold_html)
    if pred_tree is None or gold_tree is None:
        return 0.0

    distance = APTED(pred_tree, gold_tree, _TedsConfig()).compute_edit_distance()
    size = max(_count_nodes(pred_tree), _count_nodes(gold_tree))
    if size == 0:
        return 0.0
    return max(0.0, 1.0 - distance / size)


def _first_table_tree(raw: str) -> _Node | None:
    if not raw or not raw.strip():
        return None
    try:
        # Wrap in a root so multiple top-level tables / leading text parse cleanly.
        root = lxml_html.fragment_fromstring(
            f"<div>{raw}</div>", create_parent=False
        )
    except Exception:
        return None
    table = root.find(".//table")
    if table is None:
        # Maybe the input already is a <table>.
        if root.tag == "table":
            table = root
        else:
            return None
    return _convert(table)


def _convert(elem) -> _Node:
    tag = (elem.tag or "").lower()
    text = ""
    if tag in _TEXT_BEARING_TAGS:
        text = _normalize_text(elem.text_content())
    children: list[_Node] = []
    for child in elem:
        child_tag = (child.tag or "").lower()
        if child_tag in _STRUCTURAL_TAGS:
            children.append(_convert(child))
    return _Node(tag=tag, text=text, children=children)


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _count_nodes(node: _Node) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


class _TedsConfig(Config):
    def rename(self, n1: _Node, n2: _Node) -> float:
        if n1.tag != n2.tag:
            return 1.0
        if n1.tag in _TEXT_BEARING_TAGS:
            return _normalized_edit_distance(n1.text, n2.text)
        return 0.0

    def children(self, node: _Node) -> list[_Node]:
        return list(node.children)


def _normalized_edit_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if not a and not b:
        return 0.0
    d = _levenshtein(a, b)
    return d / max(len(a), len(b))


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein distance with rolling row buffer."""

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ch_a in enumerate(a, 1):
        curr[0] = i
        for j, ch_b in enumerate(b, 1):
            cost = 0 if ch_a == ch_b else 1
            curr[j] = min(
                prev[j] + 1,         # deletion
                curr[j - 1] + 1,     # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[len(b)]

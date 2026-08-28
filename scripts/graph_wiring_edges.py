#!/usr/bin/env python3
"""Add the wiring edges a Python AST extractor structurally cannot see.

graphify's AST pass reads *symbol references* -- `import x`, `f()`, class
definitions -- so an edge exists only where one parsed symbol names another in
Python syntax. Several real couplings in this repo are expressed as something
else entirely, and every one of them comes out of extraction as an isolated
node:

    cron_pi.sh        runs the whole system weekly, via the shell string
                      `uv run python3 main.py --preset ...`
    pyproject.toml    exposes `tech-analyze` via the TOML string
                      "analyzer.main:main"
    analyzer/__init__ makes `from analyzer.main import ...` resolve, and is
                      an empty file -- its meaning is the import system
    tests/conftest.py puts the repo root on sys.path; pytest finds it by
                      filename convention and nothing ever imports it

None of that is missing from the codebase; it just isn't written in a language
the parser reads. This is the same reason DI containers, Django settings and
plugin registries look disconnected in any call graph.

Every edge emitted here is derived from a parsed declaration and verified
against a node that already exists -- nothing is guessed, so all of them are
EXTRACTED. Where a link would have to be inferred (matching README prose to the
symbols it discusses, say) this script emits nothing: inventing edges is what
the AMBIGUOUS tier exists to prevent, and a fuzzy label match would be doing it
at scale. That one needs a re-extraction with the AST ids supplied as a
vocabulary, not a post-pass.

Idempotent: re-running never duplicates an edge. Safe to run after every
`/graphify` rebuild.

    python3 scripts/graph_wiring_edges.py            # patch graphify-out/graph.json
    python3 scripts/graph_wiring_edges.py --dry-run  # report only, write nothing
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

# `_origin` marks these as post-pass edges so they can be told apart from the
# AST's own, and so a re-run can recognise its own previous output.
ORIGIN = "wiring"

# Any token ending in .py that the shell script mentions. Deliberately loose:
# the token is only accepted once it resolves to a file that exists AND to a
# node already in the graph, which is a far stronger check than the regex.
_PY_TOKEN = re.compile(r"[\w./-]+\.py")


class Graph:
    """Thin index over graphify's node-link JSON."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8"))
        self.nodes = self.data["nodes"]
        self.links = self.data["links"]
        self._by_file: dict[str, list[dict]] = {}
        for n in self.nodes:
            self._by_file.setdefault(n.get("source_file") or "", []).append(n)
        self._existing = {
            (e["source"], e["target"], e.get("relation")) for e in self.links
        }

    def file_node(self, rel_path: str) -> str | None:
        """The node representing a whole file, e.g. `analyzer/main.py`."""
        name = Path(rel_path).name
        for n in self._by_file.get(rel_path, []):
            if n["label"] in (name, rel_path):
                return n["id"]
        return None

    def symbol_node(self, rel_path: str, symbol: str) -> str | None:
        """The node for a callable defined in a file, e.g. `main()`."""
        for n in self._by_file.get(rel_path, []):
            if n["label"] in (symbol, f"{symbol}()"):
                return n["id"]
        return None

    def add(self, source: str, target: str, relation: str, source_file: str,
            note: str) -> bool:
        if source == target:
            return False
        if (source, target, relation) in self._existing:
            return False
        self.links.append({
            "source": source,
            "target": target,
            "relation": relation,
            "_origin": ORIGIN,
            "confidence": "EXTRACTED",
            "confidence_score": 1.0,
            "context": note,
            "source_file": source_file,
            "source_location": None,
            "weight": 1.0,
        })
        self._existing.add((source, target, relation))
        return True

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def entry_point_edges(g: Graph, root: Path) -> list[str]:
    """`[project.scripts]` names its target as "module.path:function"."""
    out: list[str] = []
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return out
    conf = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = conf.get("project", {}).get("scripts", {})
    for name, spec in scripts.items():
        if ":" not in spec:
            continue
        module, _, func = spec.partition(":")
        rel = module.replace(".", "/") + ".py"
        if not (root / rel).exists():
            continue
        src = next((n["id"] for n in g.nodes
                    if n.get("source_file") == "pyproject.toml"
                    and n["label"] == name), None)
        dst = g.symbol_node(rel, func)
        if src and dst and g.add(src, dst, "entry_point", "pyproject.toml",
                                 "project.scripts"):
            out.append(f"{name} --entry_point--> {spec}")
    return out


def shell_invocation_edges(g: Graph, root: Path) -> list[str]:
    """Shell scripts invoke Python by command string, not by import."""
    out: list[str] = []
    for sh in sorted(root.glob("*.sh")):
        rel_sh = sh.relative_to(root).as_posix()
        src = g.file_node(rel_sh)
        if src is None:
            continue
        try:
            body = sh.read_text(encoding="utf-8")
        except OSError:
            continue
        for token in dict.fromkeys(_PY_TOKEN.findall(body)):
            target_rel = token.lstrip("./")
            if not (root / target_rel).exists():
                continue
            dst = g.file_node(target_rel)
            if dst and g.add(src, dst, "invokes", rel_sh, "shell command"):
                out.append(f"{rel_sh} --invokes--> {target_rel}")
    return out


def package_edges(g: Graph, root: Path) -> list[str]:
    """An empty `__init__.py` is what binds sibling modules into a package."""
    out: list[str] = []
    for init in sorted(root.glob("*/__init__.py")):
        rel_init = init.relative_to(root).as_posix()
        src = g.file_node(rel_init)
        if src is None:
            continue
        for mod in sorted(init.parent.glob("*.py")):
            if mod.name == "__init__.py":
                continue
            rel_mod = mod.relative_to(root).as_posix()
            dst = g.file_node(rel_mod)
            if dst and g.add(src, dst, "packages", rel_init, "package marker"):
                out.append(f"{rel_init} --packages--> {rel_mod}")
    return out


def conftest_edges(g: Graph, root: Path) -> list[str]:
    """conftest.py is found by filename convention; nothing imports it.

    Its job here is putting the repo root on sys.path, so the honest edge is to
    the package that import then resolves -- one precise edge, rather than one
    per test module, which would be true but pure noise.
    """
    out: list[str] = []
    packages = [p for p in sorted(root.glob("*/__init__.py"))]
    for conftest in sorted(root.rglob("conftest.py")):
        rel_c = conftest.relative_to(root).as_posix()
        src = g.file_node(rel_c)
        if src is None:
            continue
        for pkg in packages:
            rel_p = pkg.relative_to(root).as_posix()
            dst = g.file_node(rel_p)
            if dst and g.add(src, dst, "bootstraps", rel_c, "sys.path shim"):
                out.append(f"{rel_c} --bootstraps--> {rel_p}")
    return out


def merge_external_stubs(g: Graph) -> list[str]:
    """Collapse duplicate stub nodes for the same external name.

    The AST pass mints a node per *file* for names it cannot resolve to project
    code -- pandas' `DataFrame`, pytest's `parametrize`, `Path`, `Exception` --
    so the same concept appears three or four times with no source_file and no
    way to tell the copies apart. Merging them onto one canonical node per name
    keeps the type information those edges carry (this function returns a
    DataFrame) while removing the duplication.

    Deliberately a merge and not a delete: the edges are real, they just all
    pointed at different copies of the same idea.
    """
    by_label: dict[str, list[str]] = {}
    for n in g.nodes:
        if not n.get("source_file"):
            by_label.setdefault(n["label"], []).append(n["id"])

    remap: dict[str, str] = {}
    out: list[str] = []
    for label, ids in sorted(by_label.items()):
        if len(ids) < 2:
            continue
        canonical, *dupes = ids
        for d in dupes:
            remap[d] = canonical
        out.append(f"{label}: {len(ids)} stubs -> 1")

    if not remap:
        return out

    for e in g.links:
        e["source"] = remap.get(e["source"], e["source"])
        e["target"] = remap.get(e["target"], e["target"])

    # Drop the now-unreferenced duplicates and any self-loop the merge created.
    g.data["nodes"] = [n for n in g.nodes if n["id"] not in remap]
    g.nodes = g.data["nodes"]
    seen = set()
    deduped = []
    for e in g.links:
        if e["source"] == e["target"]:
            continue
        sig = (e["source"], e["target"], e.get("relation"))
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(e)
    g.data["links"] = deduped
    g.links = deduped
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="graphify-out/graph.json",
                    help="path to graphify's graph.json")
    ap.add_argument("--root", default=".", help="repo root the graph was built from")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the edges but do not write graph.json")
    ap.add_argument("--no-package-edges", action="store_true",
                    help="skip the __init__.py --packages--> module edges")
    ap.add_argument("--no-merge-externals", action="store_true",
                    help="leave duplicate stub nodes (DataFrame, parametrize, ...) alone")
    args = ap.parse_args()

    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"error: {graph_path} not found — run /graphify first", file=sys.stderr)
        return 1
    root = Path(args.root).resolve()

    g = Graph(graph_path)
    before = len(g.links)

    groups = [
        ("entry points ([project.scripts])", entry_point_edges(g, root)),
        ("shell invocations (*.sh)", shell_invocation_edges(g, root)),
    ]
    if not args.no_package_edges:
        groups.append(("package markers (__init__.py)", package_edges(g, root)))
    groups.append(("pytest bootstrap (conftest.py)", conftest_edges(g, root)))

    merged: list[str] = []
    if not args.no_merge_externals:
        merged = merge_external_stubs(g)

    added = 0
    for title, edges in groups:
        print(f"\n{title}: {len(edges)} edge(s)")
        for line in edges:
            print(f"  + {line}")
        added += len(edges)

    if merged:
        print(f"\nmerged external stubs: {len(merged)} name(s)")
        for line in merged:
            print(f"  ~ {line}")

    if added == 0 and not merged:
        print("\nNothing to do — graph already reconciled.")
        return 0

    if args.dry_run:
        print(f"\n--dry-run: {added} edge(s) and {len(merged)} merge(s) not written.")
        return 0

    g.save()
    print(f"\n{graph_path}: {before} -> {len(g.links)} edges (+{added}).")
    print("Re-run `graphify export html` to refresh the visualization.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

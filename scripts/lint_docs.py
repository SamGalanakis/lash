#!/usr/bin/env python3
"""Static checks for the hand-authored docs.

Uses only the Python standard library. The checks deliberately target the
static source files; docs.js may enhance pages at runtime, but source links,
anchors, registry entries, and asset versions should already be coherent.
"""

from __future__ import annotations

import html
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS_JS = DOCS / "docs.js"

MOVED_STUBS = {
    "architecture.html",
    "architecture/execution-modes.html",
    "plugins-tools.html",
}


def normalize_href(path: str) -> str:
    path = path.strip()
    path = re.sub(r"^[.]/", "", path)
    path = re.sub(r"^/docs/", "", path)
    path = re.sub(r"^docs/", "", path)
    path = path.split("?", 1)[0].split("#", 1)[0]
    path = path.lstrip("/")
    if not path:
        return "index.html"
    if path.endswith("/"):
        path += "index.html"
    if path == "architecture/":
        return "architecture/index.html"
    return path


def registry_block() -> str:
    text = DOCS_JS.read_text(encoding="utf-8")
    start = text.index("const DOCS = [")
    end = text.index("const MOVED_STUB_HREFS", start)
    return text[start:end]


def registry_hrefs() -> list[str]:
    block = registry_block()
    hrefs = [
        normalize_href(m.group(1))
        for m in re.finditer(r'title:\s*"[^"]+"\s*,\s*href:\s*"([^"]+)"', block)
    ]
    return [href for href in hrefs if href != "index.html"]


def registry_titles() -> set[str]:
    block = registry_block()
    titles = {m.group(1).strip().lower() for m in re.finditer(r'title:\s*"([^"]+)"', block)}
    labels = {m.group(1).strip().lower() for m in re.finditer(r'label:\s*"([^"]+)"', block)}
    return {t for t in titles | labels if len(t) > 2}


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.hrefs: list[tuple[str, str]] = []
        self.assets: list[str] = []
        self.pager_hrefs: list[str] = []
        self.strong_texts: list[tuple[str, bool]] = []
        self._stack: list[tuple[str, dict[str, str]]] = []
        self._strong_buf: list[str] | None = None
        self._strong_in_link = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: v or "" for k, v in attrs}
        self._stack.append((tag, attr))
        if "id" in attr:
            self.ids.add(attr["id"])
        if "name" in attr:
            self.ids.add(attr["name"])
        if tag == "a" and "href" in attr:
            href = html.unescape(attr["href"])
            self.hrefs.append((href, self._context()))
            if self._in_class("pager__main"):
                self.pager_hrefs.append(href)
        if tag in {"script", "link"}:
            asset = attr.get("src") or attr.get("href")
            if asset:
                self.assets.append(html.unescape(asset))
        if tag == "strong":
            self._strong_buf = []
            self._strong_in_link = self._in_tag("a")

    def handle_endtag(self, tag: str) -> None:
        if tag == "strong" and self._strong_buf is not None:
            text = " ".join("".join(self._strong_buf).split())
            if text:
                self.strong_texts.append((text, self._strong_in_link))
            self._strong_buf = None
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break

    def handle_data(self, data: str) -> None:
        if self._strong_buf is not None:
            self._strong_buf.append(data)

    def _in_tag(self, tag: str) -> bool:
        return any(t == tag for t, _ in self._stack)

    def _in_class(self, cls: str) -> bool:
        for _, attr in self._stack:
            classes = set((attr.get("class") or "").split())
            if cls in classes:
                return True
        return False

    def _context(self) -> str:
        classes: list[str] = []
        for tag, attr in self._stack[-4:]:
            cls = attr.get("class")
            if cls:
                classes.append(f"{tag}.{'.'.join(cls.split())}")
            else:
                classes.append(tag)
        return " > ".join(classes)


def parse_pages() -> dict[Path, PageParser]:
    out: dict[Path, PageParser] = {}
    for path in sorted(DOCS.rglob("*.html")):
        parser = PageParser()
        parser.feed(path.read_text(encoding="utf-8"))
        out[path] = parser
    return out


def is_external_href(href: str) -> bool:
    parsed = urlparse(href)
    return parsed.scheme in {"http", "https", "mailto", "tel"} or href.startswith("//")


def resolve_internal(source: Path, href: str) -> tuple[Path, str]:
    if href.startswith("#"):
        return source, href[1:]
    parsed = urlparse(href)
    raw_path = parsed.path
    fragment = parsed.fragment
    if raw_path.startswith("/docs/"):
        rel = raw_path[len("/docs/") :]
        target = DOCS / normalize_href(rel)
    elif raw_path.startswith("/"):
        target = DOCS / normalize_href(raw_path)
    else:
        target = (source.parent / raw_path).resolve()
        if raw_path == "":
            target = source
        elif raw_path.endswith("/"):
            target = target / "index.html"
    try:
        target = target.resolve().relative_to(DOCS.resolve())
        target = DOCS / target
    except ValueError:
        return target, fragment
    return target, fragment


def check_registry(errors: list[str], pages: dict[Path, PageParser]) -> list[str]:
    hrefs = registry_hrefs()
    seen: set[str] = set()
    canonical: list[str] = []
    for href in hrefs:
        if href in seen:
            errors.append(f"docs.js registry duplicate href: {href}")
            continue
        seen.add(href)
        canonical.append(href)
        if href in MOVED_STUBS:
            errors.append(f"docs.js registry includes moved stub: {href}")
        target = DOCS / href
        if not target.exists():
            errors.append(f"docs.js registry href missing file: {href}")

    expected = {
        path.relative_to(DOCS).as_posix()
        for path, parser in pages.items()
        if "docs-page" in (path.read_text(encoding="utf-8").split("<body", 1)[1].split(">", 1)[0])
    }
    expected.difference_update(MOVED_STUBS)
    missing = sorted(expected - set(canonical))
    extra = sorted(set(canonical) - expected)
    for href in missing:
        errors.append(f"docs-page missing from docs.js registry: {href}")
    for href in extra:
        errors.append(f"docs.js registry href is not a docs-page source: {href}")
    return canonical


def check_links(errors: list[str], pages: dict[Path, PageParser]) -> None:
    ids_by_path = {path: parser.ids for path, parser in pages.items()}
    for source, parser in pages.items():
        for href, context in parser.hrefs:
            if not href or is_external_href(href):
                continue
            target, fragment = resolve_internal(source, href)
            rel_source = source.relative_to(ROOT).as_posix()
            try:
                rel_target = target.relative_to(DOCS).as_posix()
            except ValueError:
                continue
            if rel_target in MOVED_STUBS:
                errors.append(f"{rel_source}: link points at moved stub {href!r}")
            if not target.exists():
                errors.append(f"{rel_source}: missing internal href {href!r} ({context})")
                continue
            if fragment and target.suffix == ".html":
                if fragment not in ids_by_path.get(target, set()):
                    errors.append(f"{rel_source}: missing anchor {href!r}")


def check_asset_versions(errors: list[str], pages: dict[Path, PageParser]) -> None:
    versions: dict[str, set[str]] = {}
    for parser in pages.values():
        for asset in parser.assets:
            parsed = urlparse(asset)
            if parsed.scheme or not parsed.query:
                continue
            qs = dict(part.split("=", 1) for part in parsed.query.split("&") if "=" in part)
            if "v" not in qs:
                continue
            name = os.path.basename(parsed.path)
            versions.setdefault(name, set()).add(qs["v"])

    js = DOCS_JS.read_text(encoding="utf-8")
    for const, asset in {
        "SCENE_ASSET_VERSION": "scene.js",
        "MERMAID_LOADER_VERSION": "mermaid.js",
    }.items():
        m = re.search(rf'const\s+{const}\s*=\s*"([^"]+)"', js)
        if m:
            versions.setdefault(asset, set()).add(m.group(1))

    for asset, seen in sorted(versions.items()):
        if len(seen) > 1:
            errors.append(f"asset {asset} has inconsistent ?v= values: {', '.join(sorted(seen))}")


def check_pagers(errors: list[str], canonical: list[str], pages: dict[Path, PageParser]) -> None:
    index = {href: i for i, href in enumerate(canonical)}
    for path, parser in pages.items():
        href = path.relative_to(DOCS).as_posix()
        if href not in index:
            continue
        if not parser.pager_hrefs:
            continue
        expected: list[str] = []
        i = index[href]
        if i > 0:
            expected.append(canonical[i - 1])
        if i + 1 < len(canonical):
            expected.append(canonical[i + 1])
        actual = [normalize_href(resolve_internal(path, h)[0].relative_to(DOCS).as_posix()) for h in parser.pager_hrefs]
        if actual != expected:
            errors.append(f"{path.relative_to(ROOT).as_posix()}: static pager conflicts with docs.js order")


def check_bold_cross_refs(errors: list[str], pages: dict[Path, PageParser]) -> None:
    titles = registry_titles()
    ignored = {"note", "contract", "gotcha", "use when", "avoid"}
    for path, parser in pages.items():
        for text, in_link in parser.strong_texts:
            if in_link:
                continue
            normalized = text.strip().lower().rstrip(".:")
            if normalized in ignored:
                continue
            if normalized in titles:
                errors.append(
                    f"{path.relative_to(ROOT).as_posix()}: strong text looks like an unlinked docs reference: {text!r}"
                )


SNIPPETS_SRC = ROOT / "examples" / "docs-snippets" / "src"
REGION_START = re.compile(r"^(\s*)// docs:start:([\w-]+)\s*$")
PRE_BLOCK = re.compile(r"<pre([^>]*)><code>(.*?)</code></pre>", re.S)


def snippet_regions(errors: list[str]) -> dict[str, dict[str, str]]:
    """Extract `// docs:start:<id>` .. `// docs:end:<id>` regions, dedented by
    the indentation of their start marker, from the docs-snippets crate."""
    regions: dict[str, dict[str, str]] = {}
    for path in sorted(SNIPPETS_SRC.glob("*.rs")):
        module = path.stem
        rel = path.relative_to(ROOT).as_posix()
        current: tuple[str, str, list[str]] | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            started = REGION_START.match(line)
            if current is None:
                if started:
                    current = (started.group(2), started.group(1), [])
                continue
            rid, indent, body = current
            if re.match(rf"^\s*// docs:end:{re.escape(rid)}\s*$", line):
                text = "\n".join(
                    l[len(indent):] if l.startswith(indent) else l for l in body
                )
                if rid in regions.get(module, {}):
                    errors.append(f"{rel}: duplicate snippet region {rid!r}")
                regions.setdefault(module, {})[rid] = text.strip("\n")
                current = None
            elif started:
                errors.append(f"{rel}: region {rid!r} not closed before {started.group(2)!r}")
            else:
                body.append(line)
        if current is not None:
            errors.append(f"{rel}: unterminated snippet region {current[0]!r}")
    return regions


def check_code_snippets(errors: list[str], fix: bool) -> None:
    """Every `<pre>` on a published top-level page declares what it is:
    `data-snippet` blocks mirror a compiled region of the docs-snippets crate
    verbatim (`cargo check -p docs-snippets` keeps them building); `data-lang`
    blocks are display-only (shell transcripts, Lashlang, API-shape excerpts).

    `data-snippet` blocks are compiled-and-synced wherever they appear, including
    the `architecture/` sub-pages; the "declare yourself" rule stays scoped to
    top-level pages so hand-authored architecture prose blocks are left alone."""
    regions = snippet_regions(errors)
    referenced: set[str] = set()
    for path in sorted(DOCS.rglob("*.html")):
        rel = path.relative_to(ROOT).as_posix()
        enforce_declaration = path.parent == DOCS
        text = path.read_text(encoding="utf-8")
        changed = False

        def replace(m: re.Match[str], enforce_declaration: bool = enforce_declaration) -> str:
            nonlocal changed
            attrs, body = m.group(1), m.group(2)
            snippet = re.search(r'data-snippet="([^"]+)"', attrs)
            if snippet is None:
                if enforce_declaration and not re.search(r'data-lang="[^"]+"', attrs):
                    errors.append(
                        f"{rel}: <pre> needs data-snippet (compiled, synced) or data-lang (display-only)"
                    )
                return m.group(0)
            ref = snippet.group(1)
            module, _, rid = ref.partition("#")
            source = regions.get(module, {}).get(rid)
            if source is None:
                errors.append(
                    f"{rel}: data-snippet {ref!r} has no region in examples/docs-snippets/src/{module}.rs"
                )
                return m.group(0)
            referenced.add(ref)
            if html.unescape(body).strip("\n") != source:
                if fix:
                    changed = True
                    return f"<pre{attrs}><code>{html.escape(source, quote=False)}</code></pre>"
                errors.append(
                    f"{rel}: <pre data-snippet={ref!r}> drifted from examples/docs-snippets/src/{module}.rs"
                    " (run scripts/lint_docs.py --fix-snippets)"
                )
            return m.group(0)

        new_text = PRE_BLOCK.sub(replace, text)
        if fix and changed:
            path.write_text(new_text, encoding="utf-8")

    check_readme_snippets(errors, regions, referenced, fix)

    for module, by_id in sorted(regions.items()):
        for rid in sorted(by_id):
            if f"{module}#{rid}" not in referenced:
                errors.append(
                    f"examples/docs-snippets/src/{module}.rs: region {rid!r} is not embedded by any docs page"
                )


# README ```rust fences, in document order, mirror these compiled regions.
README_SNIPPETS = ["index#hello-lash"]


def check_readme_snippets(
    errors: list[str],
    regions: dict[str, dict[str, str]],
    referenced: set[str],
    fix: bool,
) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    fences = list(re.finditer(r"```rust\n(.*?)```\n", text, re.S))
    if len(fences) != len(README_SNIPPETS):
        errors.append(
            f"README.md: expected {len(README_SNIPPETS)} ```rust fences (see README_SNIPPETS in scripts/lint_docs.py), found {len(fences)}"
        )
        return
    changed = False
    for fence, ref in zip(reversed(fences), reversed(README_SNIPPETS)):
        module, _, rid = ref.partition("#")
        source = regions.get(module, {}).get(rid)
        if source is None:
            errors.append(f"README.md: snippet ref {ref!r} has no region in examples/docs-snippets/src/{module}.rs")
            continue
        referenced.add(ref)
        if fence.group(1).strip("\n") != source:
            if fix:
                text = text[: fence.start(1)] + source + "\n" + text[fence.end(1) :]
                changed = True
            else:
                errors.append(
                    f"README.md: ```rust fence drifted from examples/docs-snippets/src/{module}.rs region {rid!r}"
                    " (run scripts/lint_docs.py --fix-snippets)"
                )
    if changed:
        readme.write_text(text, encoding="utf-8")


def main() -> int:
    fix = "--fix-snippets" in sys.argv[1:]
    errors: list[str] = []
    pages = parse_pages()
    canonical = check_registry(errors, pages)
    check_links(errors, pages)
    check_asset_versions(errors, pages)
    check_pagers(errors, canonical, pages)
    check_bold_cross_refs(errors, pages)
    check_code_snippets(errors, fix)
    if errors:
        for err in errors:
            print(f"docs lint: {err}", file=sys.stderr)
        return 1
    print(f"docs lint: ok ({len(pages)} html pages, {len(canonical)} registry pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

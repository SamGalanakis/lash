# lash docs · styleguide

Contributor reference for the docs design system. Read this before adding a new
page or section.

For the **canonical CSS** — see [`styles.css`](./styles.css).

---

## Where things live

| Concern | Location |
|---|---|
| Tokens (color, spacing, type) | `styles.css` `:root` |
| Shared components (`.opener`, `.body`, `.section`, `.panel`, `.chapter-list`, `.pager`, `.diagram`, `.table`, …) | `styles.css` |
| Page-specific components (mocks, exporter demos, landing scene) | Page-local `<style>` or focused JS/CSS assets such as `scene.js` |
| Mermaid loader | `mermaid.js` (conditional — only fetches the lib when `.mermaid` blocks exist) |

All HTML pages link the same stylesheet. Root-level docs pages use
`./styles.css`; architecture subpages use `../styles.css`.

---

## Brand voice

- **Tone**: industrial editorial terminal. More field notebook than dashboard.
- **Concept**: *Warm Iron / Fast Amber* — warm-black surfaces, sodium-orange
  intent markers, ash-grey structure, chalk text.
- **Audience**: engineers reading TUI design docs in dim contexts (long working
  sessions, often at night). Dark theme is correct here — don't second-guess it.

---

## Tokens

### Color

The palette is a warm-iron neutral ramp (hue ~75–90, perceptually uniform
OKLCH) plus four signal accents.

| Token | Use |
|---|---|
| `--form-deep` | page background, deepest field |
| `--form` | history background, dominant surface |
| `--form-raised` | cards, panels, status bar, framed surfaces |
| `--surface-hover` | hover state for `.chapter-list a` and similar list rows |
| `--ash` | structural lines (most borders go through `--line` instead) |
| `--ash-mid` | passive labels, thin chrome |
| `--ash-text` | eyebrows, footer text, chapter-arrow — **WCAG AA compliant against `--form`** |
| `--chalk-dim` | swatch descriptions, history copy, stat-note |
| `--chalk-mid` | body prose, descriptions |
| `--chalk` | assistant prose, key readable text |
| `--sodium` | the active edge — prompts, plan mode, key labels, plan updates |
| `--sodium-soft` | the lighter end of the brand-mark accent bar |
| `--lichen` | completion, quiet success |
| `--info` | ambient metadata, scroll position, expansion level |
| `--error` | failures, dangerous edges |
| `--ghost-bar` | assistant bar, subtle live-work trace |
| `--line` | hairline borders (`oklch(... / 0.12)` of chalk) |
| `--line-strong` | hover/active borders (`oklch(... / 0.32)` of sodium) |
| `--focus-ring` | use as `box-shadow` for visible focus on interactive elements |

**Rules**:

1. Only `--sodium`, `--lichen`, `--info`, and `--error` are accents. Everything
   else is structure. Accents work because they're rare.
2. **Don't introduce new hex literals.** If you need a new color, add a token.
3. **Don't drop ash-text or chalk-dim below their current lightness** — they
   are tuned to clear WCAG AA against `--form`. Use `/audit` to verify if you
   change them.
4. **Don't apply `--sodium` to body text or backgrounds.** It's for labels,
   markers, chapter numbers, and the active state of interactive chrome.

### Spacing

A 4pt scale with semantic names. Tokens are in `rem` so the whole layout
scales with the fluid root `font-size`. Always use the tokens — never raw px.

```
--space-3xs: 0.125rem   --space-md:  1rem      --space-3xl: 4rem
--space-2xs: 0.25rem    --space-lg:  1.5rem    --space-4xl: 6rem
--space-xs:  0.5rem     --space-xl:  2rem      --space-5xl: 9rem
--space-sm:  0.75rem    --space-2xl: 3rem
```

At base 16px these resolve to the same 2/4/8/12/16/24/32/48/64/96/144 px
values as before. On larger viewports the root scales up via `clamp()`, and
spacing scales with it. Use `gap` for sibling spacing, not margins. Vary
spacing for hierarchy — a heading with extra space above reads as more
important.

### Type

| Family | Use |
|---|---|
| `--font-display` (Big Shoulders Display) | h1, h2, h3, brand mark, stat values |
| `--font-body` (Spectral) | paragraphs, descriptions, body copy |
| `--font-mono` (Chivo Mono) | labels, eyebrows, code, mock chrome, terminal specimens, table headers |

**Rules**:

1. Pair display + body + ui. Don't set body copy in mono.
2. Headings are `text-transform: uppercase` only when they're brand statements
   (h1, h2). Subsection h3 stays mixed-case.
3. Don't add a fourth font family.
4. Cap body line length at 64–72ch.

---

## Components

### Page shell

```html
<section class="opener">
  <h1 class="opener__title">flow<span class="slash">/</span>turns</h1>
  <p class="opener__lede">Lead paragraph — start with a sentence, not a comma-stuffed list.</p>
</section>

<main class="body" id="content">
  <div class="section">…</div>
</main>

<section class="pager" aria-label="chapter navigation">…</section>
```

`.opener` provides the editorial title block, `.body` provides the constrained
reading column, and `.pager` carries previous/next navigation.

### Opener

Use the opener for page identity: one h1, one lead paragraph, and optional CTA
links only when the page is a redirect or entry point.

```html
<section class="opener">
  <h1 class="opener__title">runtime<span class="slash">/</span>host</h1>
  <p class="opener__lede">One strong paragraph that tells readers what this page owns.</p>
</section>
```

The slash character in the h1 is sodium via `<span class="slash">/</span>`.

### Sticky chip nav

```html
<nav class="nav" aria-label="Architecture pages">
  <a href="./index.html">Docs Home</a>
  <a class="active" href="./architecture/flow.html">Data Flow</a>
  <a href="./architecture/execution.html">Execution</a>
  …
</nav>
```

Mark the current page with `class="active"`. The chip strip horizontal-scrolls
on mobile — don't replace it with a wrapped or hamburger version. Touch
targets are ≥44px; do not shrink the vertical padding.

### Chapter list (editorial TOC)

For a multi-page index. Use this instead of `.toc-grid` of cards.

```html
<ol class="chapter-list">
  <li>
    <a href="./architecture/overview.html">
      <span class="chapter-num">01</span>
      <div class="chapter-meta">
        <h3>Overview</h3>
        <p>One-line description, max ~62ch.</p>
      </div>
      <span class="chapter-arrow">→</span>
    </a>
  </li>
  …
</ol>
```

### Principle list (editorial numbered rules)

For 5+ named rules or principles. CSS counters auto-number — write the items
in the order you want them displayed.

```html
<ol class="principle-list">
  <li>
    <div>
      <h3>Accent Means Action</h3>
      <p>Sodium is reserved for the active edge — prompts, plan mode…</p>
    </div>
  </li>
  …
</ol>
```

### Section header + content

```html
<div class="section">
  <div class="section-header">
    <h2>Provider details</h2>
    <p>One sentence framing the section.</p>
  </div>
  <div class="diagram">…</div>
  <div class="table">…</div>
  <div class="module-grid">…</div>
</div>
```

`.section-header` is a 2-col grid. h2 left, framing prose right, both
bottom-aligned.

### Cards and panels

- **`.card`** — for hero meta, four-column stat strips, summary tiles. Has
  micro-label + strong heading + body span.
- **`.panel`** — for content blocks inside a section. h3 + paragraph or list.
- **`.stat`** — for metric tiles with a big number (use `.stat-value`).

```html
<article class="panel">
  <small>Pure protocol</small>
  <h3>lash-sansio</h3>
  <p>Owns <code>TurnMachine</code>, <code>Effect</code>, <code>Response</code>.</p>
  <code class="path">crates/lash-sansio/src/sansio/mod.rs</code>
</article>
```

### Diagrams (Mermaid)

Wrap in a `.diagram` container with a title, then `.mermaid` for the source.

```html
<div class="diagram">
  <div class="diagram-title">Workspace topology</div>
  <div class="mermaid">
flowchart TD
  CLI --> Runtime
  …
  </div>
</div>
```

Add the loader script to the `<head>`, with a path relative to the page's
location — `mermaid.js` lives at the docs root. Root-level pages use
`<script src="./mermaid.js" defer></script>`; subpages (e.g. under
`architecture/`) must use `../mermaid.js`, matching how they reference
`styles.css`, `docs.js`, etc. A wrong path 404s the loader silently and the
diagram renders as raw text. The loader only fetches the Mermaid library when
`.mermaid` blocks exist — diagram-free pages pay no JS cost.

### Tables

Wrap in `.table` for horizontal-scroll on narrow viewports.

```html
<div class="table">
  <table>
    <thead><tr><th>Path</th><th>Responsibility</th></tr></thead>
    <tbody>
      <tr><td><code>crates/lash-core/src/runtime/mod.rs</code></td><td>…</td></tr>
    </tbody>
  </table>
</div>
```

`<th>` renders in Chivo Mono (`--font-mono`), sodium, uppercase.

### Pager (prev/next)

End every chapter with prev/next links.

```html
<section class="pager" aria-label="chapter navigation">
  <div class="pager__rail">read on &middot;</div>
  <div class="pager__main">
    <a class="pager__prev" href="./architecture/modules.html"><span>previous</span><strong>modules</strong></a>
    <a class="pager__next" href="./architecture/execution.html"><span>next</span><strong>execution</strong></a>
  </div>
</section>
```

If only one direction exists, omit the absent link and let the remaining link
hold its side of the grid.

---

## Anti-pattern guardrails

These are the bans specific to this project. They are non-negotiable.

1. **No glassmorphism on cards or content panels.** `backdrop-filter: blur(...)`
   is allowed only on `.nav` (sticky overlay over scrolling content). Anywhere
   else, use solid `var(--form-raised)`.
2. **No side-stripe borders.** No `border-left:` or `border-right:` greater
   than 1px as a colored accent. Use chapter numbers, full borders, or
   background tints instead.
3. **No gradient text.** No `background-clip: text` with a gradient. Headings
   are solid `var(--chalk)` or `var(--sodium)`.
4. **No new card grid where editorial layout fits.** If you find yourself
   writing `repeat(N, minmax(0, 1fr))` for the third time on a page, stop.
   Convert one of those sections to `.chapter-list`, `.principle-list`, or a
   hanging-number editorial run.
5. **No new fonts.** Display, body, ui — that's the kit.
6. **No raw hex.** Add a token instead. Exceptions: literal swatch chips in
   dedicated color-spec examples.
7. **No raw px for spacing.** Use `--space-*` tokens.
8. **No `outline: none` without a replacement focus indicator.** Use
   `box-shadow: var(--focus-ring)` on `:focus-visible`.
9. **No auto-redirect (`<meta http-equiv="refresh">`) on stub pages.** Style
   the stub and provide a manual link.
10. **No comma-stuffed list-as-sentence prose.** Five+ commas in one paragraph
    is a list — use `<ul>` or em-dashes.

---

## Adding a new architecture chapter

1. Copy `architecture/overview.html` as the template.
2. Update `<title>`, `.opener__title`, and `.opener__lede`.
3. Add an entry to `architecture/index.html`'s `.chapter-list`.
4. Update prev/next `.pager` links on the surrounding chapters.
5. Run `/audit` to verify contrast, focus, and touch targets pass.

## Adding a page-local demo section

1. Keep the shared shell in `styles.css`; add only the component-specific CSS
   needed by the demo.
2. Put the section inside the normal `.body` / `.section` structure unless the
   page is a purpose-built visual mockup.
3. If the demo needs JavaScript, keep the data and behavior page-local or add a
   focused asset next to the existing docs assets.

## Verifying changes

```bash
# Static contrast check (eyeball — actual ratio)
# Bump the dev server then run /audit
cd docs && python3 -m http.server 8767
```

Then run `/audit` to validate accessibility, performance, theming, responsive,
and anti-pattern compliance against this guide.

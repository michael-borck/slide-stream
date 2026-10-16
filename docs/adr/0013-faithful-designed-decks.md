# 0013 — Designed decks stay faithful: fidelity check and code slides

- Status: Accepted (2026-10); implemented in the working copy after the
  Presenton trial
- Drivers: the Presenton trial (PRESENTON_ANALYSIS.md, "Phase 1 live
  trial") showed designed decks dropping code blocks, inventing content and
  compressing long slides. Narration comes from the source, so voice and
  picture disagree. Presenton's faithfulness `instructions` fixed invented
  people and most titles, but not code: none of its 16 templates has a code
  layout. Its one code-related issue (#219, Carbon snippet images) was closed
  as not planned.
- Related: 0008 (one narration path), 0009 (Presenton), 0011 (stages),
  0012 (fallback levels).

## Decision

### 1. Code slides are SlideStream's own, in the video and the deck

When a slide's code did not survive design, SlideStream builds that slide
itself and uses it in both outputs, so the video and the editable deck
match:

- **Video:** the slide's frame is a SlideStream code card instead of
  Presenton's PDF page.
- **Deck:** the designed `.pptx` keeps the slide (its position, layout,
  background and speaker notes) but its shapes are replaced with the
  title and an editable code text box, as SlideStream already edits the
  deck to add notes. Slide count and order never change.

Which slides: `providers.presenton.code_slides: auto | always | never`.
`auto` (default) replaces a slide only when the fidelity check (part 4)
finds its code missing; `always` replaces every slide with a code block;
`never` keeps Presenton's version. Applies to Presenton only; with
`--preserve-style` and `--slides-pdf` the user owns the design.

### 2. Syntax highlighting with Pygments

Code is coloured by Pygments (BSD-2-Clause, already installed as a
dependency of `rich`), using the language from the fence info string
(```` ```python ````), else Pygments' guess, else plain text.

- In picture frames, Pygments tokens are drawn with Pillow in the same
  monospace font as today's code rendering. This also improves code in
  SlideStream's plain text cards, not only in designed decks.
- In the `.pptx`, each token becomes a text run with its colour, so the
  code stays editable text, not a picture.

### 3. Code slides borrow the template's look

So the code slide does not stand out among designed slides:

- Colours: background and accent sampled from the neighbouring Presenton
  frames (most common edge colour; most frequent saturated colour). The
  Pygments style is picked to suit a light or dark background.
- Title font and size: read from the neighbouring slide's title in the
  designed `.pptx`; fall back to SlideStream's fonts.
- If sampling fails, use SlideStream's plain look rather than abort.

### 4. A fidelity check after design

After Presenton returns, compare each designed slide's text (from the
`.pptx`) with its source slide and report:

| Check | Signal |
|---|---|
| Code kept | Every non-trivial code line's identifiers appear on the slide |
| Title kept | Normalised title match |
| Content kept | Share of the source's content words present |
| Table kept | Share of source table cells present |
| Possibly invented | Names, numbers and dates on the slide that are not in the source |

- The report prints after `presenton` and `create --template`, and is
  written to `<output>.fidelity.json` for later inspection.
- `providers.presenton.fidelity: warn | strict | off` (default `warn`).
  `strict` fails the run when a slide misses its code and `code_slides` is
  `never`, or when content kept falls below a threshold
  (`fidelity_min_kept`, default 0.5).
- The check is heuristic and says so. It flags slides for a human to look
  at; it does not judge design.

## Consequences

- Code-heavy lectures become usable with Presenton; explainer decks are
  unaffected unless they contain code.
- The parser must keep code blocks as structured data (language plus
  text), alongside today's plain `content` and `markdown` body.
- A replaced slide in the `.pptx` loses Presenton's decoration on that
  slide. That is the cost of showing the code.
- The video frame for a code slide comes from SlideStream's renderer, not
  Presenton's PDF, so fonts can differ slightly from the neighbours.
- The fidelity check gives the roadmap's planned "strict check that compares
  slide text before and after design" a concrete form, usable later by any
  design backend.
- Presenton output varies run to run; tests use small synthetic `.pptx`
  files, plus the live trial in `e2e/presenton/`.

# 0014 — Design new slides from the user's own PowerPoint template

- Status: Accepted (2026-10); implemented in 2.18.0
- Drivers: users and institutions already maintain a styled PowerPoint
  (corporate or university template, a Quarto/pandoc `reference.pptx`).
  SlideStream's own deck output is a fixed plain layout (ADR-0008), and
  Presenton (ADR-0009) needs a server, rewrites text and can invent
  content. A template backend gives the user's exact look, deterministic
  output, nothing invented, and no second service.
- Related: 0008 (`.pptx` output), 0009 (`--preserve-style` render route),
  0012 (fallback levels), 0013 (code slides, fidelity).

## Context

Two different requests both sound like "reuse my PowerPoint":

| The user wants | Answer |
|---|---|
| Narrate this deck, looking exactly as it does | `--preserve-style` (exists): LibreOffice → PDF → frames |
| New content in this deck's style | This ADR: the deck is a template |

Pandoc, and so Quarto, already has a convention for the second:
`--reference-doc reference.pptx`. Pandoc ignores the file's slides, takes
its slide masters and layouts, and picks layouts by their standard names:
"Title Slide", "Title and Content", "Section Header", "Two Content",
"Comparison", "Content with Caption", "Blank". These are also the names
of the default Office layouts and of python-pptx's built-in template.

## Decision

**The user's `.pptx` file is the template, used as it is.** SlideStream
does not convert it into its own style format. Converting would lose
masters, backgrounds, logos, placeholder positions and per-layout
differences, and would leave two copies of the style to keep in step.

**Only master and layout styling carries over.** Formatting applied by
hand to individual slides is not part of the template and does not reach
new slides. `template check` says so when a file looks hand-styled.

**Layouts are picked by pandoc's names**, so an existing Quarto or pandoc
reference deck works unchanged:

| SlideStream slide | Layout | Filled with |
|---|---|---|
| First slide with a title and at most one short line | Title Slide | title, subtitle |
| Heading only, no content | Section Header | title |
| Content plus artwork | Two Content | bullets left, picture right |
| Code block | Title Only | title; code box drawn as in ADR-0013, on the layout's own background |
| Table | Title Only | title; a native, editable PowerPoint table |
| Anything else | Title and Content | title, bullets |

When a name is missing, use the first layout with the same placeholder
types (title plus body, title only); failing that, Blank with SlideStream's
own text boxes. Example slides already in the file are removed. Speaker
notes carry the narration, as in ADR-0008.

**Video frames come from the filled deck**, through the existing route:
LibreOffice → PDF → pypdfium2 → PNG. The video therefore matches the deck
exactly. Slides whose shape differs from the video (for example a 4:3
template in a 16:9 video) are fitted with bars in the slide's own
background colour, never cropped or stretched.

**Text overflow.** python-pptx does not measure text. Estimate the lines
each placeholder needs from its size and font size; shrink the body text
down to a floor (about 60% of the layout's size); past the floor, keep the
slide and report it. Never split a slide, which would break narration
alignment.

**File types.** python-pptx refuses `.potx` files by content type; load
them by rewriting that one content type in memory. `.pptx` works as is.

**Helpers, after Quarto's workflow:**

- `slide-stream template init my-template.pptx` writes a starter template
  from python-pptx's built-in deck (MIT): widescreen, standard layout
  names, one example slide per layout. Pandoc's own reference file is not
  copied.
- `slide-stream template check my-template.pptx` lists the layouts found,
  which standard names are missing and what stands in for each, and warns
  about missing title or body placeholders and slides that look styled by
  hand.

**Interface until roadmap Phase 6:** `create --pptx-template FILE` (and
`settings.pptx_template`). It works for both outputs: `.pptx` needs only
python-pptx; `.mp4` also needs LibreOffice. It is separate from
`--template NAME`, which names a Presenton template. Phase 6 folds both
into `--design`.

The plain exporter (ADR-0008) gets the same code-slide and table
handling, so all deck outputs show code properly.

## Consequences

- Users get their institution's look with no service and no AI rewording;
  this becomes the local level of the design fallback (ADR-0012), between
  Presenton and the plain layout.
- Video output with a template needs LibreOffice; deck output does not.
- Quality depends on the template: one with unusual or missing
  placeholders falls back to simpler layouts, and `template check` exists
  to make that visible before a render.
- Fonts the template names but the machine lacks are substituted by
  LibreOffice in the video frames; the `.pptx` keeps the original names.
- Tests use small synthetic templates built with python-pptx (standard
  names, renamed layouts, a 4:3 template, a hand-styled deck); one live
  test renders frames through LibreOffice when it is installed.

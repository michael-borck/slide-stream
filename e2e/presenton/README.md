# Presenton trial (roadmap Phase 1)

Test decks for deciding whether Presenton solves SlideStream's template
problem. See [docs/ROADMAP.md](../../docs/ROADMAP.md) Phase 1 and
[docs/PRESENTON_ANALYSIS.md](../../docs/PRESENTON_ANALYSIS.md).

## The decks

| File | Slides | What it tests |
|---|---|---|
| `lecture.md` + `lecture_script.txt` | 9 | A technical lecture. It has a code block (slide 4), sub-headings (5), a table (6), a numbered list over 100 words (7) and plain bullets. Shows how Presenton lays out dense, structured content. |
| `creative.md` + `creative_script.txt` | 7 | A creative explainer about coffee, with short, evocative lines, a slide that is a single quote (5) and a call to action. Shows how Presenton handles sparse slides that rely on design. |
| `styled.pptx` (from `make_styled_deck.py`) | 6 | An already-designed PowerPoint: navy background, orange accent bar, Georgia text, a diagram drawn from shapes, speaker notes on every slide. Tests `--preserve-style` through LibreOffice. No Presenton involved. |

Narration comes from the script files (and the speaker notes for
`styled.pptx`), so the trial needs no LLM on the SlideStream side and every
run says the same words. `config.yaml` uses gTTS for the voice, with no
images and no avatar, to keep attention on the design.

Regenerate `styled.pptx` with `uv run python e2e/presenton/make_styled_deck.py`.

## Before you start

1. Start Presenton `v0.9.11-beta`, fully local: Ollama for text, no image
   generation, no telemetry. Keep its data on an internal disk; SQLite on an
   exFAT drive can misbehave.

   ```bash
   mkdir -p ~/presenton-test/app_data
   docker run -d --name presenton \
     -p 127.0.0.1:5001:80 \
     -e LLM=ollama \
     -e OLLAMA_URL=http://host.docker.internal:11434 \
     -e OLLAMA_MODEL=qwen3:4b \
     -e DISABLE_IMAGE_GENERATION=true \
     -e DISABLE_ANONYMOUS_TRACKING=true \
     -e PRESENTON_COMMUNITY_ENABLED=false \
     -e AUTH_USERNAME=admin \
     -e AUTH_PASSWORD='choose-a-password' \
     -v ~/presenton-test/app_data:/app_data \
     ghcr.io/presenton/presenton:v0.9.11-beta
   docker logs -f presenton
   ```

2. Open http://localhost:5001, log in as `admin`, and make one deck in the
   browser. If that fails, the problem is Presenton or the model, not
   SlideStream.
3. Create an API key for the admin user in the Admin panel, then
   `export PRESENTON_API_KEY='sk-presenton-...'`.
4. For the styled deck: `brew install --cask libreoffice`.

When finished: `docker rm -f presenton`. Settings stay in
`~/presenton-test/app_data`.

If the designs are poor, run once with a larger model or a cloud key before
blaming Presenton. A 4B model may struggle with its structured output.

## Run

```bash
e2e/presenton/run.sh              # everything
e2e/presenton/run.sh lecture      # one deck: lecture | creative | styled | baseline
TEMPLATE_LECTURE=modern TEMPLATE_CREATIVE=nova e2e/presenton/run.sh
```

Outputs, logs and `timings.txt` go to `e2e/out/presenton/` (git-ignored).
`baseline` renders both Markdown decks in SlideStream's own plain layout for
side-by-side comparison.

## What Presenton receives

Each slide's own Markdown, as written: the code block on slide 4, the `##`
sub-headings on slide 5 and the table on slide 6 arrive intact. (Before
October 2026 SlideStream flattened all three into bullets.) To see the exact
payload:

```bash
uv run python -c "from pathlib import Path; from slide_stream.parser import parse_markdown; from slide_stream.presenton import slides_to_markdown; print('\n\n=====\n\n'.join(slides_to_markdown(parse_markdown(Path('e2e/presenton/lecture.md').read_text()))))"
```

In the `baseline` videos, SlideStream's own text cards ignore line breaks,
so code loses its indentation and table rows run together. That is a limit
of the plain layout, not of the trial.

## Record

First run: 16 October 2026. Presenton `v0.9.11-beta` (Docker, arm64), LLM
`anthropic / claude-haiku-5-5`, image generation off. SlideStream working
copy after the parser fix. Design scores are my judgement from the video
frames; re-score them yourself.

| Deck | Template | Count kept? | Order kept? | Text kept (source words) | Design (1-5) | Time | Errors |
|---|---|---|---|---|---|---|---|
| lecture (deck only) | general | yes, 9/9 | yes | 42-94% per slide; titles rewritten | 4 | 40 s | none |
| lecture (video) | general | yes | yes | as above | 4 | 443 s total, about 50 s of it Presenton | none |
| creative (video) | editorial | yes, 7/7 | yes | 89-100% per slide; titles rewritten | 4 | 274 s total | none |
| styled (`--preserve-style`) | n/a | yes, 6/6 | yes | unchanged | design kept | 190 s total | none |

The rest of each video's time is narration and encoding: the plain-layout
baselines took 411 s and 255 s, and the videos have the same lengths
(208 s, 124 s). Presenton adds about a minute per deck with Haiku.

Slide-level notes for the lecture deck:

| Slide | Content type | What Presenton did |
|---|---|---|
| 4 | Code block | **Dropped the code.** Replaced it with prose about the function. |
| 5 | Sub-headings | Good: a numbered two-column layout, one entry per approach. |
| 6 | Table | Good: a styled table with the right values. **Added an invented "Summary" row.** |
| 7 | Over 100 words | Compressed into three stat cards ("Ratio", "0.5-0.75", "Double"); most of the prose went. |

Other findings:

- **Invented content.** The creative title layout has a person-and-role
  slot, and Haiku filled it with a fictional "Maria Lopez, Farm Owner" (a
  different name, "Mateo R., Head Farmer", on a second run).
- **Rendering bugs in Presenton.** Literal `<b>...</b>` tags appear on
  lecture slide 9, both in the PDF frames and in the `.pptx` text. The
  creative title slide draws two titles on top of each other, and two
  creative slides show a stray white band.
- **Placeholder pictures.** With image generation off, every picture slot
  shows the same dark wave texture. Turn images on (for example
  `IMAGE_PROVIDER=comfyui`) for a fair look.
- **Not repeatable.** Two runs of the same deck gave different titles and
  wording.
- **Health check.** `GET /api/v1/auth/status` answers 200 without login
  and reports `"configured": true`; use it for the adapter's `probe()`.

## Second run: with faithfulness instructions

16 October 2026, same setup. SlideStream now sends Presenton's
`instructions` field by default (`DEFAULT_PRESENTON_INSTRUCTIONS`: keep
code verbatim, don't invent people or data, keep titles). Lecture generated
twice, creative once, as PDFs (`e2e/out/presenton/instr/`).

| Problem from the first run | With instructions |
|---|---|
| Code block replaced by prose | **Not fixed.** Neither lecture run showed a code block. Run 2 quoted the code inline (`return hash(key) % n_buckets`) inside text boxes. |
| Invented person on the creative title | Fixed. No person; the title is the source title. |
| Invented "Summary" table row | Fixed in run 1, back in run 2. |
| Titles rewritten | Mostly fixed: 6 of 9 lecture titles and 4 of 7 creative titles kept word for word. |
| Literal `<b>` tags | Not seen in either run (possibly chance). |
| Overlapping titles | Still there on two lecture slides whose titles wrap to two lines. |

Trade-off: the creative deck came out plainer. With less freedom, Presenton
picked simple text layouts for most slides instead of the icons and
timelines of the first run.

Why code can't be fixed this way: none of Presenton's 16 built-in templates
has a layout for code. An instruction can't create one, so the best it can
do is quote code inside a text box, without line breaks or a monospace font.
Code needs a structural fix (see the analysis doc).

## Third run: ADR-0013 (fidelity check and rebuilt code slides)

16 October 2026, same setup, with instructions on. `presenton lecture.md`
(42 s) and `create lecture.md --template general` (426 s), outputs in
`e2e/out/presenton/adr13/`.

- **Code fixed.** Presenton again dropped the code on slide 4. The fidelity
  check caught it, and SlideStream rebuilt the slide in both the video frame
  and the `.pptx`: syntax-coloured code with its indentation, the
  template's background and purple accent, the original title and the two
  bullets. Slide count, order and speaker notes unchanged.
- **Report.** Flagged slide 3 (title changed), slide 4 (code rebuilt) and
  slide 6 ("possibly invented: Summary Linear", the invented table row
  again). Saved as `lecture.fidelity.json`.
- **Remaining differences.** The rebuilt slide has no Presenton
  decoration, and its title font is SlideStream's in the video frame (the
  `.pptx` borrows the template's title font).

Answers:

- **Does the narration match the slide?** Mostly. Narration follows the
  source, and the slide topics and order match. But the code slide shows no
  code while the narration says "here's the whole idea in two lines of
  Python", and the load-factor narration covers more than the stat cards
  show.
- **Show the designed lecture to a class unedited?** No: the code is gone,
  there's a stray `<b>` and an invented table row. With a quick edit in
  Presenton, yes.
- **Creative video clearly better than `creative-basic.mp4`?** Yes, by a
  wide margin, apart from the invented farmer and the title overlap.
- **Did `styled.mp4` keep the design?** Yes: navy background, orange bar,
  serif text and the shape diagram all survived LibreOffice.
- **Go or no-go?** Recommendation: **go, as an experimental backend**,
  aimed at explainer and creative decks rather than code-heavy lectures.
  See the analysis doc for follow-ups.

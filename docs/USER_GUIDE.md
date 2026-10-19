# User guide

The reference for installing, configuring and running SlideStream. For
step-by-step recipes see [Use cases](USE_CASES.md); for every voice,
picture, LLM, presenter and design option see [Providers](PROVIDERS.md).

## Install

```bash
pip install "slide-stream[all]"
```

Python 3.10 or newer. FFmpeg comes with the package (through moviepy), so a
system install is optional; SlideStream uses one on your `PATH` if present.

`[all]` is everything. Smaller installs:

| Extra | Adds |
|---|---|
| (none) | Markdown and PowerPoint input, text-card slides, gTTS voice, `.mp4` and `.pptx` output |
| `local-tts` | Kokoro, the offline voice |
| `serve` | The web UI (`slide-stream serve`); includes PDF and Word import |
| `import` | PDF and Word input for `draft` |
| `presenton` | Designed slides: Presenton, `--preserve-style`, `--slides-pdf` |
| `openai`, `claude`, `gemini`, `groq`, `elevenlabs` | That provider's SDK |
| `all-ai` | Every cloud provider SDK, without the offline voice or web UI (for lean servers) |

Quote the package name (`"slide-stream[all]"`) so your shell doesn't expand
the brackets. `slide-stream doctor <deck>` names any extra a configured
provider still needs.

Other ways to run it:

- **Desktop app** for macOS, Windows and Linux: a window around the web UI
  that installs and updates the engine itself. See the
  [releases page](https://github.com/michael-borck/slide-stream/releases).
- **Docker** for hosting the web UI for others: see
  [deploy/README.md](https://github.com/michael-borck/slide-stream/blob/main/deploy/README.md).

## Configuration

Settings come from YAML files, merged in layers; later layers win
([ADR-0002](adr/0002-layered-yaml-config.md)):

1. Built-in defaults.
2. `~/.slidestream.yaml` (or `.yml`): personal settings shared by every
   project, such as server URLs and key references.
3. `./slidestream.yaml` (or `.yml`) in the current folder, or the file
   given with `--config`: settings for this deck.
4. Command-line flags, for one run.

```bash
slide-stream init           # a minimal file that works with no keys
slide-stream init --full    # every provider and setting, commented
```

The full reference is also in
[slidestream.example.yaml](https://github.com/michael-borck/slide-stream/blob/main/slidestream.example.yaml).

**Secrets** stay out of the files. A value written as `"${VAR}"` is read
from the environment variable `VAR`, and SlideStream warns if it is unset.
Providers also read their standard variables directly (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `ELEVENLABS_API_KEY`,
`PEXELS_API_KEY`, `UNSPLASH_ACCESS_KEY`, `DID_API_KEY`, `PRESENTON_API_KEY`).
The command line does not read `.env` files.

### Settings

`providers:` chooses engines ([Providers](PROVIDERS.md)). `settings:`
controls the output:

| Setting | Default | Meaning |
|---|---|---|
| `video.resolution` | `[1920, 1080]` | Frame size |
| `video.fps` | `24` | Frames per second |
| `video.transition_seconds` | `0` | Crossfade between slides (`--transition`) |
| `video.slide_duration_padding` | `1.0` | Seconds added after each slide's narration |
| `video.default_slide_duration` | `5.0` | Length of a slide with no narration |
| `video.bullets_overlay` | `true` | Show bullets beside generated pictures |
| `image.bg_color`, `image.font_color` | `black`, `white` | Colours of text-card slides |
| `image.title_font_size`, `image.content_font_size` | `100`, `60` | Text sizes on text cards |
| `narration.target_seconds` | none | Aim for this many seconds of speech per slide (`--narration-seconds`) |
| `narration.wpm` | `150` | Speaking rate used for that estimate |
| `avatar.position` | `bottom-right` | Presenter corner |
| `avatar.size` | `0.28` | Presenter circle, as a share of the frame height |
| `avatar.slides` | `all` | Which slides show the presenter (`--avatar-slides`) |
| `avatar.reuse_clip` | `false` | Render one presenter clip and loop it (`--reuse-avatar`) |
| `voice_floor` | `auto` | Local fallback voice: `auto`, `kokoro` or `gtts` |
| `accessibility.captions` | `both` | Caption files beside the video: `both`, `srt`, `vtt` or `none` |
| `accessibility.burn` | `false` | Draw the captions into the picture |
| `accessibility.transcript` | `true` | Write `<output>.transcript.md` |
| `accessibility.chapters` | `true` | MP4 chapter markers + `<output>.chapters.txt` |
| `reveal` | `auto` | Bullets appear as spoken: `auto` or `off` |
| `pronunciations` | `{}` | Written → spoken map applied to narration |
| `accessibility.captions` | `both` | Caption files beside the video: `both`, `srt`, `vtt` or `none` |
| `accessibility.burn` | `false` | Draw the captions into the picture |
| `accessibility.transcript` | `true` | Write `<output>.transcript.md` |
| `accessibility.chapters` | `true` | MP4 chapter markers + `<output>.chapters.txt` |
| `strict` | `false` | Fail instead of falling back when a chosen provider fails (`--strict`) |
| `temp_dir`, `cleanup` | `temp_files`, `true` | Working folder, and whether to delete it afterwards |

## Decks

### Markdown

Each `#` heading starts a slide. Everything under it until the next `#`
heading belongs to that slide.

````markdown
---
title: optional front matter, ignored
---

# Hash Tables

- Find a value without searching
- **Week 6**, Data Structures

# A Hash Function

```python
def bucket_for(key: str, n: int) -> int:
    return hash(key) % n
```

## Notes on this slide

| Operation | List | Hash table |
|---|---|---|
| Look up | O(n) | O(1) |
````

- **Bullets, paragraphs, quotes and `##` sub-headings** all become slide
  content, in order.
- **Fenced code blocks** keep their indentation and are drawn in a
  monospace font with syntax colouring; give the language after the fence.
  A `#` inside a code block never starts a slide.
- **Tables** are drawn one row per line.
- **Pictures:** a line such as `![](images/slide_1.png)` becomes the slide's
  picture instead of narrated text (this is what `enrich` writes).
- **Separator style:** a deck can instead separate slides with lines of
  `---` (Marp style), one heading per slide.

Markdown slides have no speaker notes. Provide narration with an LLM, or
with a `--script` file.

### PowerPoint

`slide-stream create slides.pptx talk.mp4` reads each slide's title, text
and pictures, and its **speaker notes**, which become the narration source.
By default the video uses SlideStream's own slide layout; add
`--preserve-style` to keep the deck's look (needs LibreOffice).

Notes are used when they read as narration. Placeholders ("Click to add
notes") and presenter cues ("Discuss the plan") are ignored, and bracketed
stage directions such as `[pause]` are removed before speaking.

## Narration

Per slide, SlideStream uses the first of these that applies:

1. A block from `--script FILE`: one block per slide, separated by lines of
   `---`, spoken exactly as written.
2. Speaker notes, spoken exactly as written, with `--verbatim-notes`.
3. With an LLM configured: notes cleaned and fitted to the target length;
   else the slide content turned into what a presenter would say; else a
   picture-only slide described by a vision model; else a short
   introduction of the title.
4. With no LLM: the title, content and usable notes, read as written.

`--narration-seconds 30` aims for about 30 seconds per slide.

## Commands

Every command has `--help`.

### `create IN OUT`

Renders a deck. The output's extension picks the mode:

- `.mp4`: the full pipeline (narration, voice, pictures, presenter, video).
- `.pptx`: an editable PowerPoint with the pictures and the narration in
  the speaker notes. No voice, presenter or video work.

| Option | Does |
|---|---|
| `-c, --config FILE` | Use this config file as the project layer |
| `--llm-provider`, `--llm-model` | Narration writer for this run |
| `--narration-seconds N` | Target speech per slide |
| `--script FILE` | Narration from a script, one block per slide |
| `--verbatim-notes` | Speak speaker notes as written |
| `--tts-provider`, `--tts-base-url`, `--voice` | Voice for this run |
| `--voice-sample FILE` | Clone this recording for this run |
| `--avatar / --no-avatar` | Show the presenter |
| `--avatar-slides SPEC` | `all`, `none`, `first`, `last`, `first,last`, `1,3,5`, `every:3` |
| `--reuse-avatar` | One presenter clip looped across the deck |
| `--transition SECONDS` | Crossfade between slides |
| `--strict` | Fail instead of falling back |
| `--dry-run`, `--fail-on-warn` | Report only, as `doctor` does |
| `--preserve-style` | Keep a `.pptx` input's own design in the video |
| `--slides-pdf FILE` | Use a PDF's pages as the video frames |
| `--template NAME` | Designed slides from Presenton (experimental) |
| `--presenton-url`, `--presenton-api-key`, `--save-deck` | Presenton connection and where to save its deck |
| `--alt-text` | Describe pictures with a vision LLM (`.pptx` output) |
| `--pptx-template FILE` | Build the slides in this template's style (see above) |

## Alt text

`create deck.md deck.pptx --alt-text` and `enrich deck.md out/ --alt-text`
describe each picture with a vision LLM (claude, openai, gemini) and write
the description as the picture's alt text in the PowerPoint, and as the
image text in the enriched Markdown. Needs an LLM provider with vision.

## Lecture-quality narration

- **Bullets appear as they are spoken.** When the narration is built from
  the slide's own text (no LLM, no script), each bullet is voiced as its own
  segment and appears in the video exactly when the voice reaches it
  ([ADR-0019](adr/0019-pauses-reveal-pronunciations-clips.md)). `settings.reveal: off` gives a
  single static frame per slide. LLM-written narration and scripts stay
  static (their text can't be attributed to bullets).
- **Pauses:** `[pause]` or `[pause 2s]` in speaker notes, a script or slide
  content becomes real silence — also in narration you authored. Other
  bracketed directions (`[click]`) are still stripped, and prose cues are
  still only flagged by `doctor`.
- **Pronunciations:** `settings.pronunciations` maps written form to spoken
  form (acronyms, names, symbols); word-like keys match whole words, and
  common code symbols (`%`, `->`) have built-in readings. Captions and the
  transcript show the spoken form.
- **Video clips:** reference a clip in the deck (`![demo](demo.mp4)`) and it
  plays in the slide's picture area; the slide lasts at least as long as the
  clip while the narration keeps playing. The `.pptx` output embeds the
  clip's first frame.

## Your own PowerPoint template

`create deck.md talk.mp4 --pptx-template uni.pptx` builds the new slides in
your template's layouts: its masters, fonts, colours and logos style both
the deck and the video ([ADR-0014](adr/0014-powerpoint-template-design.md)).
`settings.pptx_template` sets it permanently.

- Layouts are matched by the standard names (Title Slide, Title and
  Content, Section Header, Two Content, Title Only, Blank), so a Quarto or
  pandoc reference deck works as is; a similar layout stands in when a name
  is missing, and the run says so.
- Only styling in the slide master and layouts carries over: formatting
  applied to the template's example slides does not, and those slides are
  removed.
- Code keeps its syntax colouring, tables become native PowerPoint tables,
  long text is shrunk to fit (and reported when it can't).
- The video needs LibreOffice; the `.pptx` output does not.

Helpers:

```bash
slide-stream template init my-template.pptx   # starter with the standard names
slide-stream template check my-template.pptx  # what a template offers
```

## Captions, transcript and chapters

By default every video (`.mp4`) comes with accessibility files beside it
([ADR-0018](adr/0018-segmented-narration-timing.md)):

- `talk.srt` and `talk.vtt`: captions, one cue per spoken sentence.
- `talk.transcript.md`: the narration in readable form, slide by slide.
- `talk.chapters.txt`: a YouTube-style chapter list; chapter markers are
  also embedded in the MP4, so players show and skip sections.

Timings are exact: the narration is synthesised sentence by sentence and the
joined sentences are the audio you hear, so captions can never drift from
the voice. Turn any of it off, or burn the captions into the picture, under
`settings.accessibility` (see the settings table above). A `--script` file
or `--verbatim-notes` is captioned the same way.

## Alt text

`create deck.md deck.pptx --alt-text` and `enrich deck.md out/ --alt-text`
describe each picture with a vision LLM (claude, openai, gemini) and write
the description as the picture's alt text in the PowerPoint, and as the
image text in the enriched Markdown. Needs an LLM provider with vision.

### `doctor IN`

A preflight that renders nothing. It reports the slide count, slides
without notes, stage directions that would be spoken, voice-sample length,
picture resolution, presenter and engine mismatches, missing packages or
keys, and estimates video length, cost and render time. `--fail-on-warn`
exits non-zero on any warning, for CI. `create --dry-run` gives the same
report using `create`'s options.

### `draft DOC [OUT.md]`

Turns a `.txt`, `.md`, `.pdf`, `.docx` or `.pptx` file into a Markdown deck
with an LLM. `--slides N` sets the count. It never renders; edit the deck,
then run `create`. PDF and Word need the `import` extra (in `[all]`).

### `enrich IN OUT_DIR`

Adds a picture to every slide and writes a new deck: Markdown plus an
`images/` folder, and a `.pptx` with `--pptx`. `--notes all` writes AI
speaker notes for every slide, `--notes fill` only where notes are missing;
existing notes are kept otherwise. The result round-trips:
`create OUT_DIR/deck.pptx` narrates from those notes.

### `scan DIR`

Renames a folder of pictures to descriptive names with a vision model, so
the `local` picture provider can match them to slides. It previews by
default; `--apply` renames.

### `presenton IN OUT`

Designs a deck with a running Presenton (experimental) and writes a `.pptx`
or `.pdf` with the narration in the speaker notes, plus a fidelity report.
See [Use cases](USE_CASES.md#designed-slides-instead-presenton).

### `serve`

The web UI: upload a deck, an optional voice sample and photo, render and
download. Locally it binds to `127.0.0.1` and opens a browser. For a server,
see [deploy/README.md](https://github.com/michael-borck/slide-stream/blob/main/deploy/README.md).
Uploaded voice samples and photos are deleted after the render.

### Others

- `init [FILE] [--full]`: write a config file.
- `providers`: list providers and whether each is usable.
- `voices`: list voices on the configured voice server.
- `avatars`: list the built-in mascots.

## Troubleshooting

**Start with `slide-stream doctor deck.md`.** It names missing packages,
keys and server URLs, with the fix.

- **A provider "is not available".** Its package, key or `base_url` is
  missing. Without `--strict` SlideStream falls back and says so; check
  `slide-stream providers`.
- **The wrong voice.** A voice server failed and the voice floor took over;
  the end-of-run summary counts such slides. Use `--strict` when the exact
  voice matters.
- **The first Kokoro run is slow.** It downloads its model (about 340 MB)
  once, to `~/.cache/slide-stream/kokoro/`.
- **"No FFmpeg found".** The bundled copy is missing, which means a broken
  install; reinstall `slide-stream`, or install FFmpeg yourself.
- **A slide count mismatch with designed slides.** The designed deck or PDF
  must have the same number of slides as the source, or narration would
  drift; SlideStream stops instead.
- **`--preserve-style` fails.** It needs LibreOffice (`soffice`) on your
  `PATH`, or export the PDF yourself and use `--slides-pdf`.

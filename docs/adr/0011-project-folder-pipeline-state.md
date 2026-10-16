# 0011 — A project folder holds the pipeline's state

- Status: Proposed (2026-10)
- Drivers: users want anything from "one command, no questions" to "let me
  edit every step". Stages must be resumable and cheap to re-run after an
  edit. The CLI, web UI and desktop app should share one model of a job.
- Related: 0002 (layered config: the project layer), 0008 (dual output),
  0009 (positional alignment of designed slides), 0010 (stages call adapters).

## Context

`create` runs every stage inside one function of about 940 lines
(`cli.py:203`). Intermediate results go to a temp directory under
`settings.temp_dir`, and the run deletes it at the end. A user can't stop after
the narration script, fix one sentence and carry on. Each new feature
(Presenton, music, presets) would make that function longer.

The web UI already has a partial answer. Its project workflow
(`serve.py:1610`) keeps a per-session workdir with `deck.md` or `deck.pptx` and
`images/`, and supports draft → edit → enrich → render. It also stores a hash
of the deck (`image_deck_hash`) so enriched images are discarded when the deck
changes. But the server owns all of it. The registry lives in memory, projects
expire, and the CLI can't see them.

Human-editable formats already exist for most stages: Markdown decks
(`parser.py`), narration scripts in the `--script` format (`narration.py`), and
a project-level `slidestream.yaml` (ADR-0002).

## Decision

**A project is a folder on disk.** The CLI, web UI and desktop app all read and
write the same layout:

```text
my-talk/
  slidestream.yaml          project config (the ADR-0002 project layer)
  source/                   inputs: topic.txt, .md, .docx, .pdf, .pptx
  outline.md                outline stage (optional)
  deck.md  or  deck.pptx    content stage: the slide source of truth
  narration.md              script stage, in the --script format
  design/                   design stage: slides.pdf, designed.pptx (absent for basic)
  assets/
    images/                 per-slide artwork
    audio/                  per-slide narration audio
    avatar/                 head clips
    music/                  background track(s)
  build/                    outputs: <name>.mp4, <name>.pptx, check report
  .slidestream/state.json   per-stage records (see below)
```

**Stages.** In order: ingest → outline → content → script → design → assets
(images, voice, avatar, music) → render → check. Each stage is a function in a
new `pipeline/` package with the signature
`run(project, config) -> StageResult`. A stage reads only its declared inputs
and writes only its declared outputs. Stages call services through ADR-0010
adapters.

**Staleness.** `state.json` records, for each stage, a hash of its inputs (the
files it reads plus the config keys it uses), the adapter and service version
that produced the output, and a timestamp. `build` re-runs a stage only when
that hash changes. Editing `narration.md` invalidates audio, avatar and render,
but not images. This generalises `image_deck_hash`. `state.json` carries a
`format_version` for future migrations.

**Files are the truth.** No hidden state overrides a file. If a user edits
`deck.md` by hand, the next build uses the edit. A project folder can be
committed to git; `assets/audio/`, `assets/avatar/` and `build/` are
regenerable and can be ignored.

**Positional alignment.** Slides, narration blocks, design pages and assets
are matched by slide index. If a stage changes the slide count, every later
stage is invalidated. A count mismatch between stages is an error, as in 0009.

**Human control is a stop list, not a separate mode.**

| Command | Behaviour |
|---|---|
| `create SRC OUT` | Unchanged for users. Runs every stage in a temp project and keeps only the output. `--keep-project DIR` keeps the folder. |
| `new SRC DIR` | Creates a project folder from a source (topic, file or deck). |
| `build DIR` | Runs every stale stage. |
| `build DIR --stop-after STAGE` | Stops after a stage, for editing. |
| `build DIR --review STAGE,...` | Pauses after the named stages. An interactive terminal prompts to continue. A non-interactive run exits with a distinct "paused" status that a later `build` resumes from. |
| `build DIR --from STAGE` / `--force STAGE` | Re-runs from, or just, a stage regardless of hashes. |

The web UI and desktop app show the same stages as steps, with a "skip
reviews" switch. The server's project registry points at project folders
instead of its own ad-hoc workdirs. Demo mode stays one-shot and stores
nothing, as it promises today.

## Consequences

- `create` becomes a thin wrapper over `new` + `build`. The web server keeps
  its subprocess isolation, but the subprocess runs `build`.
- This is the largest refactor in the project's history. Do it behind
  characterisation tests that pin today's `create` outputs. Extract one stage
  at a time. Add stage-level unit tests and a few small saved project folders
  under `tests/fixtures/projects/`.
- Projects use disk: audio and video assets stay until removed.
  `slide-stream clean DIR` deletes regenerable assets.
- New features become a stage, a provider (0010), or an option on one of
  them. Anything that fits none of these waits for its own ADR.
- Open questions to settle during implementation: whether `outline.md` is its
  own stage or the first pass of `draft`; how `.pptx` decks with embedded
  pictures map onto `assets/images/`; and whether the review pause should
  open the file in the user's editor.

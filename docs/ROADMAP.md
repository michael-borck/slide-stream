# SlideStream roadmap

**Status:** draft, October 2026. Update it as phases land; record decisions in
ADRs, not here.

## Where this is going

SlideStream takes a source and turns it into a presentation: a slide deck, a
narrated video, or both. The source can be a sentence, a Markdown file, a
Word or PDF document, or an existing PowerPoint. Along the way it can add
pictures, a voice (including your own), a presenter, a designed template and
background music. Users choose how much to steer, from one command with no
questions to reviewing and editing every step. Output ranges from a technical
lecture deck to a creative explainer video.

Three decisions shape how we get there:

- **One tool for the user, many engines behind it.** SlideStream orchestrates
  local and cloud services through adapters and never bundles them
  ([ADR-0010](adr/0010-orchestrator-over-external-services.md)). The engines
  are listed in [SERVICES.md](SERVICES.md).
- **A project folder holds the pipeline's state**, so any step can be
  reviewed, edited and resumed
  ([ADR-0011](adr/0011-project-folder-pipeline-state.md)).
- **Every new feature is a stage or an adapter.** If it is neither, it waits.
- **Every role falls back service → local → bare minimum**, so a render
  always finishes and stays local where it can
  ([ADR-0012](adr/0012-three-level-fallback-kokoro-floor.md)). The voice
  already does: Kokoro, then gTTS.

The CLI stays the main product for power users. The web UI and the Tauri
desktop app sit on the same engine.

## Phase 1: test Presenton against a live server

The integration in this working copy was built from Presenton's source and
docs and has never talked to a real server
([PRESENTON_ANALYSIS.md](PRESENTON_ANALYSIS.md)). Find out whether it solves
the template problem before building on it.

- [x] Run Presenton `v0.9.11-beta` from its Docker image. (Used Claude
      Haiku 5.5 as its LLM rather than Ollama.)
- [x] Generate from three real decks: a technical lecture (`.md`), a creative
      piece (`.md`), and an existing `.pptx`.
- [x] Record for each: slide count kept, slide order kept, how much text
      Presenton rewrote, design quality, time taken, and any API differences
      from what `presenton.py` expects.
- [x] Find Presenton's health or version endpoint for the adapter probe:
      `GET /api/v1/auth/status`.
- [x] Write the result into the analysis doc. Recommendation: go, as an
      experimental backend for explainer and creative decks. **Accepted
      16 Oct 2026.**
- [x] Send Presenton's `instructions` field by default. Fixes invented
      people and most rewritten titles; does not preserve code blocks
      (no template has a code layout).
- [x] Code slides and the fidelity check: Phase 2b.
- [ ] Report the `<b>` tag and overlapping-title bugs upstream.

**Done when** there is a written go or no-go. If no-go, Presenton stays an
experimental option and Phase 6 leans on the `pptx-theme` backend.

## Phase 2: ship the current work as experimental

- [x] Replace PyMuPDF (AGPL-3.0) with pypdfium2 (Apache-2.0/BSD-3) for PDF
      page rendering.
- [ ] Add a live contract test for Presenton under `tests/live/`.
- [x] Refresh `uv.lock`.
- [ ] Bump the version in `pyproject.toml` and
      `src/slide_stream/__init__.py`.
- [ ] Release notes say Presenton is optional, external, experimental and
      CLI-only. `--preserve-style` and `--slides-pdf` ship regardless of the
      Phase 1 result; they work without Presenton.

**Done when** the release is on PyPI.

## Phase 2b: faithful designed decks (ADR-0013)

Make Presenton decks trustworthy for technical content. Each step ships on
its own and is tested before the next.

**Step 1: the fidelity check** (comes first: steps 3 and 4 use it to decide
which slides to fix).
- [x] Parser keeps code blocks as structured data: `slide["code_blocks"] =
      [{"lang", "code"}]`, alongside `content` and `markdown`.
- [x] `fidelity.py`: compare source slides with the designed `.pptx` text;
      code kept, title kept, content kept, table kept, possibly invented.
- [x] Print the report after `presenton` and `create --template`; write
      `<output>.fidelity.json`.
- [x] `providers.presenton.fidelity: warn | strict | off`,
      `fidelity_min_kept`.
- [x] Tests on synthetic decks: dropped code, kept code, rewritten title,
      invented name and number, extra table row.

**Step 2: syntax highlighting.**
- [x] `code_render.py`: Pygments tokens drawn with Pillow in the existing
      monospace font; language from the fence, else guessed, else plain.
- [x] Use it for code in SlideStream's plain text cards and the
      bullets-beside-artwork layout.
- [x] Light and dark styles; tests that colours differ per token type and
      that unknown languages fall back to plain.

**Step 3: code slides in the video.**
- [x] Theme sampling from neighbouring Presenton frames: background,
      accent, light or dark; fall back to SlideStream's look.
- [x] Code card: template-coloured background, title, highlighted code,
      sized to fit (shrink the font for long code, within limits).
- [x] `providers.presenton.code_slides: auto | always | never`; `auto`
      uses the step 1 result.
- [x] In `create --template`, swap those slides' frames before rendering.

**Step 4: code slides in the editable deck.**
- [x] In the designed `.pptx`, replace the chosen slides' shapes with the
      title and an editable code text box with one coloured run per
      Pygments token; keep slide order, background and notes.
- [x] Title font read from a neighbouring slide.
- [x] Applies to both `presenton` and `create --template`.
- [x] Tests: slide count unchanged, notes kept, code text exact, runs
      coloured.

- [x] **Then:** re-run the lecture trial and record the result in the
      trial README. Done 16 Oct 2026: code shown in video and deck.

**Done when** the lecture deck shows its code, coloured and correctly
indented, in both the video and the editable `.pptx`, and the fidelity
report flags slides Presenton changed.

## Phase 3: adapters and service status (ADR-0010)

This comes before the project folder because the pipeline stages will call
these adapters. Settle their interface first.

- [ ] Add `is_configured()`, `probe()` and `ServiceStatus` to the provider
      interfaces. Keep `is_available()` as an alias during the change.
- [ ] Implement `probe()` for each self-hosted service in SERVICES.md; replace
      each † with a confirmed endpoint.
- [ ] Wrap `llm.py` in LLM adapters. Move `presenton.py` behind a `design`
      adapter.
- [ ] Add `slide-stream services [--json]`.
- [ ] `doctor` and `create` preflight use `probe()`: strict mode fails before
      rendering; default mode warns.
- [ ] Apply the three-level fallback (ADR-0012) to pictures and presenters:
      a dead image service tries the local folder before text cards; a dead
      avatar service falls to static or puppet before none.
- [ ] Report which level each role actually used, in `services` and in the
      end-of-run summary.
- [ ] Fill in the "Tested" column of SERVICES.md as live tests land.

**Done when** `slide-stream services` reports every configured service, and a
dead `base_url` is caught before the first slide renders.

## Phase 4: the project folder (ADR-0011)

The biggest refactor so far. Do it in small, safe steps.

- [ ] Characterisation tests that pin today's `create` behaviour for
      Markdown, `.pptx`, `--script`, `.pptx` output and designed slides.
- [ ] Create `pipeline/` and move one stage at a time out of `create`
      (`cli.py:203`), keeping all tests green after each move.
- [ ] Add `state.json` with input hashes and `format_version`.
- [ ] Add `new` and `build`; turn `create` into a wrapper. Add
      `--keep-project`.
- [ ] Point the web server's project workflow (`serve.py:1610`) at project
      folders.
- [ ] Add `clean`.
- [ ] Saved example projects under `tests/fixtures/projects/`.

**Done when** `create` is a thin wrapper, and editing `narration.md` then
running `build` re-runs only audio, avatar and render.

## Phase 5: human control

- [ ] `build --stop-after`, `--review`, `--from`, `--force`.
- [ ] An outline stage with review, so users approve structure before any
      expensive work (an idea taken from Presenton).
- [ ] Tone, verbosity and language options for the outline, content and
      script stages (also from Presenton).
- [ ] Web and desktop UI show the stages as steps with a "skip reviews"
      switch.

**Done when** the same project can run fully automatically or pause at any
stage, from the CLI and from the UI.

## Phase 6: design backends (ADR-0014, to write)

- [ ] Write ADR-0014: one `design` role with interchangeable backends.
- [ ] `basic`: today's fixed layout, kept as the floor.
- [ ] `pptx-theme`: build slides from the layouts in a user-supplied `.potx`
      or styled `.pptx` with python-pptx, then render frames through
      LibreOffice. No second service.
- [ ] `presenton`: the Phase 3 adapter.
- [ ] `preserve`: today's `--preserve-style` and `--slides-pdf`.
- [ ] Generalise the Phase 2b fidelity check so any design backend can
      use it.

**Done when** `--design` selects any of the four, and the video stage does not
care which one produced the frames.

## Phase 7: background music (ADR-0015, to write)

- [ ] `--music track.mp3`: loop or trim to length, fade in and out, and lower
      the music under narration (ffmpeg). No AI needed.
- [ ] A `music` adapter for ACE-Step 1.5 XL as the default (MIT, output
      usable commercially). Decide between its own API server and a ComfyUI
      workflow; the ComfyUI route can reuse the avatar adapter's code.
- [ ] A YuE2 adapter as the alternative for songs with lyrics.
- [ ] Generate a track to the video's length, or loop a shorter one with a
      crossfade. Instrumental by default, since a sung vocal competes with
      the narration.
- [ ] `doctor` shows a one-line licence note when YuE2 or another
      non-commercial engine is configured.
- [ ] Ask Puente to add ACE-Step 1.5 XL as a service.

**Done when** a video can carry a supplied or generated background track at a
level that never masks the narration.

## Phase 8: presets (ADR-0016, to write)

- [ ] Presets as a config layer between built-in defaults and the user's
      file: `lecture`, `technical`, `explainer`, `creative`.
- [ ] Each sets narration style, image style, design backend, music on or off,
      and pacing.
- [ ] `slide-stream presets` lists them; projects can define their own.

**Done when** `--preset creative` and `--preset lecture` produce clearly
different videos from the same source with no other flags.

## Phase 9: desktop

The Tauri app stays. Borrow proven patterns from `lens-desktop` instead of
switching frameworks.

- [ ] A Services panel driven by `slide-stream services --json`, showing
      what is live and how to enable the rest.
- [ ] Ollama detection and model download.
- [ ] API keys in the OS keychain instead of `~/.slidestream.yaml`.
- [ ] Code signing and notarisation (builds are unsigned today).

**Done when** a non-technical user can see which engines are available
without reading a config file.

## Phase 10: lens family bridges

- [ ] `ingest` adapter for `document-analyser` over HTTP.
- [ ] `check` stage with `speech-analyser` (pace, silences) and
      `video-analyser` (narration against on-slide text).

**Done when** `build` can end with a check report in `build/`.

## Requests to other projects

| Project | Request | Why |
|---|---|---|
| Puente | `puente connect --json` | Lets SlideStream discover a local stack's URLs without hand-written config. |
| Puente | A Presenton service | Gives power users a one-command Presenton without SlideStream managing Docker. |
| Puente | An ACE-Step 1.5 XL service | The default music engine, alongside or instead of MusicGen, whose weights are non-commercial. |
| Presenton | Open a discussion once Phase 1 works, showing the workflow | Feedback and credit upstream; small, focused PRs for any bugs found. No feature PRs nobody asked for. |

## Not planned

- Vendoring or forking Presenton code, or depending on the `presenton` PyPI
  placeholder.
- SlideStream starting and stopping Docker containers by default.
- Switching the desktop app to Electron.
- Optional dependencies under copyleft licences such as AGPL (see
  [SERVICES.md](SERVICES.md#python-dependencies)).
- Splitting SlideStream into several PyPI packages. Keep clean module
  boundaries inside the repo instead.
- A template picker in the desktop app that assumes a Presenton server is
  running.

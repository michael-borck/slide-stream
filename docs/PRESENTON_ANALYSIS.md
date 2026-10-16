# Presenton integration analysis

**Status:** Review document; not all recommendations are implemented. Live trial done 16 October 2026 (see "Phase 1 live trial").  
**Date:** 15 October 2026  
**SlideStream version examined:** 2.15.25 plus the unreleased Presenton work in this checkout  
**Presenton version examined:** `v0.9.11-beta`, commit `23d7347` from 13 October 2026  
**Presenton source:** https://github.com/presenton/presenton  
**Presenton licence:** Apache-2.0  
**SlideStream licence:** MIT

This document records the investigation into whether SlideStream should borrow,
use, or integrate ideas and code from Presenton. It covers product boundaries,
technical feasibility, maturity, packaging, open-source obligations, user
experience, and the recommended next steps.

## Executive summary

SlideStream and Presenton are best understood as two specialist applications
that can work in sequence:

```text
Idea / document / Markdown
        |
        v
Presenton designs the slides
        |
        v
Editable PowerPoint (.pptx) or PDF
        |
        v
SlideStream narrates the deck
        |
        v
Video (.mp4) with speech and optional avatar
```

Presenton is the presentation designer. SlideStream is the presentation
narrator and video producer.

The main conclusions are:

1. **Do not submit a mega-PR to Presenton** adding narration, TTS, avatars,
   and video. That is outside its stated product vision and contribution
   policy and would transfer a large maintenance burden upstream.
2. **Integrate at arm's length and acknowledge Presenton.** The public
   integration surface is its HTTP API (and MCP), not a Python library.
3. **Presenton is credible and fairly mature as an application**, but it is
   still pre-1.0, changes quickly, and is immature as an embeddable package.
4. **Do not depend on `presenton` from PyPI.** That package is currently a
   placeholder. `presenton-backend` is not published as a usable library.
5. **Do not wholesale-import or vendor Presenton's code.** Extracting its
   design engine would effectively require SlideStream to fork and maintain
   a second application.
6. **The current SlideStream integration is suitable for CLI power users**:
   it calls an already-running Presenton service over HTTP. It is not yet a
   good non-technical desktop experience.
7. **The best simple CLI direction is an ephemeral managed sidecar**:
   SlideStream starts Presenton only for the current command, uses dynamically
   selected loopback ports, and stops it afterwards. The pinned official
   Docker runtime is the least fragile first implementation of that idea.
8. **Keep SlideStream's generic designed-slide support independent of
   Presenton.** Rendering an existing PowerPoint or a user-supplied slides PDF
   into video is valuable regardless of which designer produced the deck.
9. **Do not expose manually managed Presenton setup in the desktop app.**
   Desktop support should wait for a hosted service, a genuinely managed
   local runtime, or a decision to build simpler native themes.

## Phase 1 live trial (16 October 2026)

The integration was run against a real Presenton for the first time. Full
record: [e2e/presenton/README.md](https://github.com/michael-borck/slide-stream/blob/main/e2e/presenton/README.md).

**Setup.** Presenton `v0.9.11-beta` from the official Docker image (arm64),
bound to `127.0.0.1:5001`, LLM `claude-haiku-5-5`, image generation off,
telemetry off. Admin account and a 7-day API key created through
Presenton's admin API. Three test decks: a technical lecture with code, a
table, sub-headings and a long list; a creative explainer; and a styled
`.pptx` for `--preserve-style`.

**The client worked first time.** `POST /presentation/generate` with
`slides_markdown` and `POST /presentation/{id}/export` behaved as read from
the source. No client changes were needed. Every deck kept its slide count
and order, so narration stayed aligned. Presenton took 40 to 55 seconds per
deck with Haiku; the rest of each render was narration and encoding.

**Design quality is the real gain.** The creative deck came out with
icons, timelines and numbered layouts, far ahead of SlideStream's text
cards. The lecture's table became a styled table and its sub-headings a
two-column layout. `--preserve-style` kept a hand-styled deck's colours,
fonts and shape diagram through LibreOffice.

**Problems found:**

1. Presenton dropped the lecture's code block and wrote prose instead.
2. It invents content to fill layouts: a fictional "Maria Lopez, Farm
   Owner" on the creative title slide, an extra "Summary" row in the table.
3. It trims or compresses long slides (100-word outline limit), so the
   narration can mention things the slide no longer shows.
4. Rendering bugs: literal `<b>` tags on one slide, overlapping titles on
   another, stray white bands on two.
5. Output differs from run to run.

**Decision (accepted 16 October 2026): go, as an experimental, CLI-only
design backend**, aimed at explainer and creative decks. Follow-ups,
smallest first:

- ~~Send Presenton's `instructions` field.~~ Done: SlideStream sends
  faithfulness instructions by default (`providers.presenton.instructions`
  overrides; `""` sends none). Result: titles mostly kept, no invented
  people, the invented table row only sometimes. **Code blocks are still
  lost**, because no built-in template has a code layout. The creative
  deck also came out plainer. Details in the trial README.
- Fix code structurally (planned: [ADR-0013](adr/0013-faithful-designed-decks.md), roadmap Phase 2b, using option (a) plus syntax highlighting, a matching fix in the `.pptx`, and a fidelity check). Options considered, cheapest first: (a) in the video, use
  SlideStream's own frame (which now renders code properly) for slides
  that contain a code block, and keep Presenton's frames for the rest;
  (b) a custom Presenton template with a code layout, through its Template
  V2 import from a `.pptx`; (c) ask upstream whether a code layout would be
  welcome.
- Add the fidelity check (now ADR-0013, Phase 2b step 1): compare source and
  designed slide text and warn on dropped code or large rewrites.
- ~~Fix code structurally.~~ Done (ADR-0013): a fidelity check after
  design, and SlideStream-built code slides in the video and `.pptx`.
  Live-tested on the lecture deck; see the trial README's third run.
- Turn image generation on (Puente's ComfyUI) and re-judge the design.
- Report the `<b>` and overlapping-title bugs upstream as small, focused
  issues, with the trial decks as reproductions.
- Use `GET /api/v1/auth/status` as the adapter's health check.

## Product delineation

| Question | Presenton | SlideStream |
|---|---|---|
| Primary job | Design and edit a presentation | Turn a deck into a narrated video |
| Typical inputs | Prompt, document, Markdown slide content, uploaded files, existing PPTX as a template source | Markdown, PowerPoint, documents/topics through `draft` |
| Main output | Editable `.pptx`, PDF | `.mp4`; optionally a simple editable `.pptx` |
| Design/templates | Core strength: templates, themes, layouts, charts, icons, infographics | Deliberately simple fixed layout |
| Slide editor | Yes | No slide-design editor |
| Speaker notes | Can be present in workflow/decks, but narration is not the product focus | Central narration source |
| Text-to-speech | No | Yes, through multiple providers |
| Voice cloning | No | Yes, through supported providers/services |
| Avatar | No | Yes, static/puppet/GPU/hosted options |
| Video assembly | No | Core strength |
| Main interfaces | Web UI, Electron desktop, REST API, MCP, Docker | CLI, web UI through `serve`, Tauri desktop |
| Distribution | Docker, Electron installers, cloud/enterprise, source | PyPI package/CLI, Docker web UI, desktop installers |

There is overlap in content generation. Both projects can accept source
material, use LLMs, obtain or generate images, and produce slide content. The
boundary is the final artifact and the specialist work around it:

- Presenton answers: "Make this look like a professionally designed deck."
- SlideStream answers: "Present this deck as a narrated video."

### Desktop applications are packaging, not separate products

Presenton's Electron application packages its Next.js editor, FastAPI backend,
export runtime, templates, and related tools as a local desktop product. Its
purpose is to let a user design decks locally without running Docker manually.

SlideStream's desktop application is a Tauri shell. It finds or installs `uv`,
starts `slide-stream serve` in local mode, and displays the SlideStream web
UI. Its purpose is to let a user create narrated videos without using a
terminal.

The desktop shells are analogous packaging layers around two different
engines. They do not erase the product boundary.

## What SlideStream already had

Before the Presenton investigation, SlideStream already provided:

- Markdown and PowerPoint parsing, including speaker notes and embedded
  pictures.
- `draft` from topics and `.txt`, `.md`, `.pdf`, `.docx`, and `.pptx` sources.
- LLM narration, verbatim notes, and external narration scripts.
- Multiple TTS, image, LLM, and avatar providers.
- A simple editable PPTX export with narration in speaker notes.
- `enrich` for adding images to a deck and `scan` for matching local images.
- A FastAPI web interface and a Tauri desktop wrapper.
- Local/self-hosted provider patterns based on configured HTTP `base_url`
  values, including Voicebox, ComfyUI, SwarmUI, and Ollama-compatible
  services.

The important limitation was visual design. SlideStream's PowerPoint parser
extracts titles, text, notes, and images, after which the video pipeline builds
new frames using its own layout. A carefully styled PowerPoint therefore lost
its design in the generated video. Its own PPTX exporter used one deliberate,
plain layout: title, bullets, artwork, and notes.

## Ideas worth borrowing or using from Presenton

| Presenton idea | SlideStream opportunity | Assessment |
|---|---|---|
| Presenton HTTP generation API | Use it as an optional designed-deck backend | Highest value; implemented in this checkout |
| Per-slide `slides_markdown` | Preserve the user's slide structure instead of asking Presenton to outline a new deck | Implemented; keeps narration aligned |
| Built-in layout templates/themes | Give designed frames to video and an editable deck to the user | Use through Presenton; do not copy without building a renderer |
| Template V2 from an existing PPTX | Reuse a user's corporate/personal design for new decks | Valuable future integration through the API; complex to extract |
| Tone, verbosity, and language controls | Add equivalent controls to `draft` and narration generation | Good independent product idea |
| Outline review before generation | Let users approve/edit structure before spending rendering time | Good workflow idea, especially for web/desktop |
| Charts/tables/infographics from structured data | Render Markdown tables as data visuals instead of bullet text | Valuable, but requires renderer work if done natively |
| Semantic icon search | Add relevant icons to bullet/data slides | Possible selective borrowing; current service has embedding dependencies |
| OCR and stronger document extraction | Improve `draft` for scanned/complex documents | Good independent direction; Presenton uses LiteParse and other services |
| Chunking long documents | Replace SlideStream's simple source-size clamp with structured summarisation/chunk selection | Good selective idea |
| MCP tools | Expose SlideStream job creation/status/results to agents | Useful future interface over `serve` |
| Async tasks and webhooks | Make long renders observable and integrable | Useful service design idea |
| More LLM/image providers | Consider a LiteLLM-style provider, Pixabay, ComfyUI image generation, and similar additions | Evaluate individually; already broad overlap |
| Multi-user auth/database/Electron editor | Hosted editor requirements | Not appropriate to absorb into SlideStream |
| Mem0 presentation memory/cloud OAuth | Presenton product features | Out of scope |

The central technical realisation was that feeding a Presenton PPTX back into
the old SlideStream `create` path was not enough. SlideStream extracted the
text and rebuilt its own frames, discarding Presenton's visual design. A
designed-deck video needs this additional step:

```text
designed PPTX -> PDF -> one picture per slide -> video frames
```

Presenton's PDF export makes that possible. For a user's existing PowerPoint,
PowerPoint itself or LibreOffice can provide the PDF.

## Implemented in this checkout

The investigation led to an unreleased, optional integration in SlideStream.
It has not been published to PyPI as part of version 2.15.25.

### New modules

- `src/slide_stream/presenton.py`
  - Resolves Presenton URL, API key, and template from CLI flags, layered
    config, environment, and defaults.
  - Converts parsed SlideStream slides to one `slides_markdown` entry per
    slide.
  - Calls `POST /api/v1/ppt/presentation/generate`.
  - Re-exports the same presentation as PDF through
    `POST /api/v1/ppt/presentation/{id}/export`.
  - Downloads PPTX/PDF results and surfaces Presenton's error detail.
  - Writes narration into a downloaded PPTX's speaker notes in slide order.
- `src/slide_stream/slide_render.py`
  - Uses pypdfium2 to render PDF pages to slide PNGs. (The first version
    used PyMuPDF; it was swapped for pypdfium2 because PyMuPDF is
    AGPL-3.0.)
  - Uses headless LibreOffice, when available, to render an existing PPTX to
    PDF while preserving its design.
  - Gives actionable errors when pypdfium2 or LibreOffice is absent.

### CLI changes

- `slide-stream presenton <deck.md|deck.pptx> <out.pptx|out.pdf>`
  designs a deck through an already-running Presenton service.
- `slide-stream create <deck> <video.mp4> --template NAME`
  asks Presenton to design a deck, uses the exported PDF as full-frame video
  slides, produces the normal narrated video, and saves the designed PPTX
  beside the video with narration in speaker notes.
- `--save-deck PATH` chooses where the designed PPTX is saved.
- `--preserve-style` renders an existing PPTX as designed through
  LibreOffice and PDF rasterisation.
- `--slides-pdf FILE` accepts any user-exported PDF of designed slides.
- `--presenton-url` and `--presenton-api-key` override configuration.
- Designed-slide modes fail when slide/page counts differ, rather than
  allowing narration to drift out of alignment.
- Bullet overlay is disabled for designed full-frame slides because the
  designed slide already contains its text.

### Configuration and packaging

`providers.presenton` was added to SlideStream's layered configuration:

```yaml
providers:
  presenton:
    base_url: http://localhost:5001
    api_key: "${PRESENTON_API_KEY}"
    template: general
```

A new `slide-stream[presenton]` extra installs **pypdfium2 only**. It neither
installs Presenton nor depends on the placeholder `presenton` PyPI package.

### Tests and verification performed

New maintained tests in `tests/test_presenton.py` cover:

- Slide-to-Markdown conversion and placeholder-title handling.
- Config/CLI/environment precedence.
- Presenton request payloads, authentication, downloading, and errors.
- Generating both PPTX and PDF for the same Presenton presentation.
- Speaker-notes injection.
- PDF rasterisation and missing-dependency errors.
- Missing LibreOffice behaviour.
- The `presenton` CLI command.
- Full-slide video creation and slide-count mismatch failure.

At implementation time, the full local verification result was:

- `502 passed, 11 skipped` under pytest.
- Ruff clean.
- basedpyright: zero errors, warnings, or notes.

Important caveats:

- Presenton HTTP calls were faked in tests, consistent with SlideStream's
  hermetic-test policy.
- The integration was not exercised against a live Presenton server during
  the initial implementation. It was on 16 October 2026; see "Phase 1 live
  trial".
- `uv.lock` was not refreshed at first because the development sandbox could
  not reach PyPI. It was refreshed later, when PyMuPDF was swapped for
  pypdfium2.
- The API contract was read from Presenton's source and documentation for
  `v0.9.11-beta`; a live contract test against that pinned version remains
  advisable.

## Open-source citizenship

The direct question was whether SlideStream should send Presenton a pull
request adding narration, text-to-speech, avatars, and video. The answer is
no.

Presenton's `CONTRIBUTING.md` asks contributors to open an issue or discussion
before building a feature. It also asks for small, focused pull requests, says
contributions outside `electron/` may not be accepted at this time, and
requires a contributor licence agreement. A video pipeline would touch the
FastAPI backend, the editor, exports, provider configuration, and the desktop
runtime. That is several products' worth of review and support work, and it
sits outside Presenton's stated aim of generating structured visual
documents.

A better way to work with the project is narrower:

1. Use the public API now. Keep the dependency optional.
2. Credit Presenton in the README, user docs, CLI help, and ADR. Link to the
   project. Say plainly that SlideStream does not bundle it.
3. Retain Apache-2.0 licence and NOTICE material if a future change copies
   Presenton code or assets. Copied files do not become MIT code just because
   SlideStream uses the MIT licence.
4. Open a discussion only after there is a working integration to show. A
   short post with a real workflow gives maintainers something concrete to
   react to.
5. Send upstream only small changes that help Presenton users in general.
   Examples are API bugs found during integration, speaker-note handling, or
   documentation for downstream tools.
6. Do not use Presenton's name in a way that suggests endorsement. Also do
   not silently bundle its code inside SlideStream's wheel or Docker image.

This route respects maintainer time. It also gives Presenton visible credit
and feedback from a real downstream user, which is worth more than a large
pull request that nobody asked for.

## How mature is Presenton?

The short version: Presenton is a serious late-beta application and a poor
library candidate. Those are different judgements.

Evidence collected on 15 October 2026:

- Repository created 10 May 2025.
- 2,863 commits and 34 contributors in the local clone.
- Latest examined commit dated 13 October 2026.
- Latest tag examined: `v0.9.11-beta`.
- Regular beta tags before that, going back through the 0.8 series.
- GitHub reported 11,115 stars and 1,694 forks. Popularity is not proof of
  quality, but it does show real use and attention.
- The FastAPI test tree contains hundreds of test functions. CI runs the
  repository tooling tests, the FastAPI suite, Next.js tests, lint, a build,
  and Cypress component tests.
- The project ships Docker images, desktop installers, hosted cloud and
  enterprise offerings, API documentation, and an MCP server.
- Recent work includes security fixes, template changes, desktop runtime
  fixes, and removal of older template machinery.

That is a lot of engineering. It is well ahead of a weekend prototype, and it
is reasonable to build an optional integration on its public API.

The caution signs are also specific:

- It is still version 0.9 beta, not 1.0.
- The backend package itself remains version 0.1.0.
- The database has accumulated many migrations, and Template V2 replaced older
  template approaches. Internal schemas are still moving.
- Recent commits delete older template routes, compilers, and models. Code
  copied today can be orphaned quickly.
- The contribution rules restrict where outside code is likely to be accepted.
- There is no supported Python library release.

So the safe boundary is the versioned HTTP API, with a pinned version and a
live contract test. The unsafe boundary is Presenton's internal Python
modules, database models, template JSON internals, and export code.

## Packaging and distribution

### SlideStream

SlideStream is CLI first. The installed `slide-stream` command is the main
product. The web UI and desktop app sit on top of the same Python engine.

The package is on PyPI as `slide-stream`. On 15 October 2026, PyPI reported
version 2.15.25, matching this checkout's released version. That release does
not contain the Presenton commands or the `slide-stream[presenton]` extra.
Both exist only in this working copy until a later release.

### Presenton

Presenton ships as an application, not as a library:

- Docker image and Docker Compose profiles.
- Electron desktop installers.
- Hosted cloud and enterprise deployments.
- Source repository with FastAPI, Next.js, templates, scripts, and an export
  runtime.
- REST API and MCP server.

The `presenton` name on PyPI resolves to a small placeholder release whose
description says "Coming soon..." and whose wheel is about 1.4 kB. There is no
published `presenton-backend` package on PyPI.

### Can we build a package from the GitHub repository?

We can probably build a wheel from `servers/fastapi`. It has a setuptools
`pyproject.toml` naming the project `presenton-backend`. That does not give us
a supported library.

Problems visible in the source:

- No `[project.scripts]` entry point, so installing the wheel installs no
  Presenton command.
- Python is pinned to `>=3.11,<3.12`. SlideStream supports Python 3.10 to
  3.13 in one environment.
- The import packages use generic top-level names: `api`, `constants`,
  `enums`, `models`, `services`, `templates`, and `utils`. Those names can
  collide with other packages and make the code awkward to embed.
- Startup assumes the process runs with the FastAPI directory as its working
  directory. Resource paths use `os.getcwd()`.
- Default templates live at the repository root in `templates/`, outside the
  backend package. The default-template code resolves them relative to the
  source tree.
- Startup runs database migrations, creates tables, imports default
  templates, loads provider settings, and checks model availability.
- Exports need the separate `@presenton/export-core` Node runtime and a
  Chromium browser.
- The export flow renders a Next.js page. FastAPI on its own is not the whole
  product.

Building the wheel is therefore the easy part. Making it run correctly still
requires reproducing most of Presenton's source layout and runtime. At that
point we have not avoided the service. We have only chosen a more fragile way
to install it.

Running from a pinned Git checkout is more honest. Presenton's own tooling
does this in development: Electron runs `uv run python server.py --port ...`
inside `servers/fastapi`, and the Docker build installs the backend into a
Python 3.11 virtual environment. A SlideStream-managed checkout is possible,
but it still needs the Next.js build, the export runtime, Chromium, fonts,
templates, and writable data directories.

## Why Presenton is an HTTP service

Presentation generation does not have to use HTTP in principle. Presenton uses
HTTP because the current code is built as a multi-process application:

1. FastAPI owns users, provider settings, templates, presentation records,
   files, generation, and exports. It stores state in a database.
2. Generation writes slide records and assets under an application-data
   directory.
3. Export builds a `/pdf-maker` URL. That route lives in the Next.js app.
4. A Node export process opens the page in Chromium. The page asks FastAPI for
   the presentation and its assets.
5. The export process writes the `.pptx` or `.pdf` file and returns its path.

Calling Python functions directly would not remove the database, files,
Next.js route, Node process, or Chromium. It would remove one local HTTP call
and tie SlideStream to private functions that expect database sessions,
request ownership, background-task handling, and environment state.

HTTP also gives us clean isolation. Presenton keeps its Python 3.11
environment, heavy dependencies, database, and browser runtime. SlideStream
keeps its own environment. If Presenton crashes, the SlideStream command can
report the failure and clean up the child process.

For a local sidecar, HTTPS adds little. The service should bind to
`127.0.0.1` only. We can also require an API key or a random local token so
another local process cannot use the service during the short run.

## The user-experience problem

The implemented CLI path expects Presenton to be running already. That is a
reasonable contract for a power user. SlideStream already asks that group to
configure Voicebox, ComfyUI, SwarmUI, Ollama, provider keys, and server URLs.

It is not a reasonable contract for the desktop audience. A non-technical user
should not have to install Docker, start a second application, create an
admin account or API key, choose a port, and then return to SlideStream. Each
step is another place to give up.

There are three audiences to keep separate:

1. CLI users can manage a service, a container, a checkout, and config files.
   They value scriptability and control.
2. Self-hosted web users interact through a browser. The operator can run
   Presenton beside `slide-stream serve`; the end user sees one website.
3. Desktop users expect one installed app. They should not see ports,
   containers, local servers, or a second provider setup flow.

The current integration serves the first group. It does not yet serve the
third group. That limitation should be stated plainly in release notes and
help text.

## Deployment options

### Option A: user runs Presenton, SlideStream calls its URL

This is what the checkout does now.

Advantages:

- Uses Presenton's supported Docker, desktop, or source runtime.
- No bundled code and no process-management responsibility for SlideStream.
- Works when Presenton runs on another machine.
- Easy for a team to share one Presenton service.

Costs:

- The service runs in the background.
- The user handles installation, startup, authentication, provider keys, and
  ports.
- Failure messages can be clear, but the setup remains technical.

Verdict: keep this forever as the external-service mode. It is the simplest
and most honest integration.

### Option B: SlideStream starts the official Docker image for one command

The CLI starts the pinned Presenton container, waits for health, calls the API,
and stops the container in a `finally` block. The all-in-one image already
contains FastAPI, Next.js, nginx, the export runtime, Chromium, fonts, and
templates. Only one host port needs to be exposed because nginx routes the
internal services.

Advantages:

- One SlideStream command from the user's point of view.
- Uses the runtime Presenton already tests and publishes.
- Avoids building Python, Node, and browser pieces on the user's machine.
- Container removal is a reliable cleanup boundary.
- Docker can assign the host port, which avoids most port conflicts.

Costs:

- The user still needs Docker installed and running.
- The first image pull is large.
- Container startup adds time to every command.
- SlideStream must manage container names, logs, orphan cleanup, signals,
  provider configuration, and a persistent `/app_data` mount.

Verdict: best first step for a simple CLI experience. It hides Presenton's
lifecycle without pretending Presenton is a library.

### Option C: SlideStream manages a pinned Git checkout

A `slide-stream presenton setup` command could clone a pinned tag into a cache
directory, run `uv sync --locked` for the backend, install and build Next.js,
sync the export runtime, prepare Chromium, and then start both servers for one
command.

Advantages:

- No Docker requirement.
- Runs directly from source at a known commit.
- Power users can inspect or patch the runtime.

Costs:

- Largest setup path. It needs Python 3.11, `uv`, Node, npm, Chromium,
  ImageMagick or related tools, fonts, model caches, and writable cache
  directories.
- The build has many platform-specific failure points.
- Updates mean rebuilding, not just pulling an image.
- SlideStream takes support calls for Presenton's build system.

Verdict: possible, and maybe worth offering later as an advanced mode. Do not
make it the first managed runtime.

### Option D: SlideStream launches Presenton's built desktop/runtime files

Presenton's Electron build produces a PyInstaller FastAPI binary, a standalone
Next.js server, templates, and export files. In theory, SlideStream could
download a release, locate those files, and start them without opening the
Electron window.

That is not a supported Presenton interface. File layout, environment
variables, local session tokens, and runtime preparation can change between
releases. It also raises redistribution and update questions. Treat this as a
research idea only. Do not build a product path on another app's internal
bundle layout.

### Option E: host Presenton for users

Add Presenton beside `slide-stream serve` in the deployment stack. The browser
UI gains a template picker. The server operator runs both services, and the
end user uploads a deck and downloads a video or PowerPoint.

Advantages:

- Zero local installation for end users.
- One website, one account story if we choose to build it, and centrally
  managed updates.
- Works for the hosted demo and institutional deployments.

Costs:

- Decks leave the user's machine unless the institution hosts the stack.
- The operator pays for compute and model use, or must design a safe
  bring-your-own-key flow.
- Presenton and SlideStream both need authentication, quotas, cleanup, and
  abuse controls in a public deployment.

Verdict: the best zero-install browser experience. It solves a different
problem from the local desktop app.

### Option F: import or vendor Presenton code

Copying small utilities is possible. Importing the whole design engine is
another matter.

The template certification module alone is about 4,600 lines. Its schema code
is about 1,500 lines. Those sit beside layout models, LLM provider plumbing,
image and icon handling, database records, file storage, the Next.js slide
renderer, and the Node/Chromium exporter. Taking the useful part means taking
the system around it, then tracking a fast-moving upstream project by hand.

Verdict: reject wholesale code integration. Reimplement small ideas inside
SlideStream's own provider interfaces where they stand on their own. Copy code
only in narrow cases, with tests and Apache attribution.

### Option G: build simpler native themes in SlideStream

SlideStream could add a small set of built-in themes to its own renderer:
colours, fonts, title slides, section dividers, bullet layouts, image layouts,
and simple chart slides. That gives desktop users better-looking output with
no second service.

This is real product work, and it will not match Presenton's template library,
editor, custom PPTX templates, charts, or infographics. But it is fully under
SlideStream's control and fits the existing install. It is the safer desktop
path if hosting or managing Presenton turns out to be too much operational
weight.

## Recommended CLI experience

For power users, the command should stay short:

```bash
slide-stream create deck.md video.mp4 --template modern
```

The user should not need to know whether Presenton came from an existing URL
or a temporary container. A config block can make the choice explicit:

```yaml
providers:
  presenton:
    mode: auto          # auto | external | docker | source
    base_url: http://localhost:5001
    api_key: "${PRESENTON_API_KEY}"
    template: general
    image: ghcr.io/presenton/presenton:<pinned-tag>
```

Suggested behaviour for `mode: auto`:

1. If an explicit external URL is configured and reachable, use it and leave
   it running. SlideStream did not start it, so SlideStream must not stop it.
2. If no reachable service exists, start the pinned Docker image.
3. Print a short status line while the image pulls or the service starts.
   Send detailed Presenton logs to a file and name that file only on failure.
4. Wait for a real health endpoint, not just container creation.
5. Send the generation request and download the PPTX and PDF.
6. Stop and remove the container in a `finally` path, including after errors
   and Ctrl-C.
7. Label managed containers so a later SlideStream command can clean up an
   orphan after a crash.

Useful supporting commands:

```bash
slide-stream presenton setup      # pull the image and check configuration
slide-stream presenton doctor     # report Docker, image, keys, health, disk use
slide-stream presenton cleanup    # remove managed containers and stale state
```

An explicit setup command is better than surprising a user with a multi-minute
download during their first video render. The first `--template` run can tell
them to run setup, or ask to do it now in an interactive terminal.

### Ports

Port conflicts are manageable.

With Docker, publish the container's port 80 to a Docker-assigned loopback
port:

```bash
docker run --rm -d -p 127.0.0.1::80 ghcr.io/presenton/presenton:<pinned-tag>
```

Docker chooses a free host port. SlideStream then asks Docker which port it
assigned. There is no need to claim port 5001.

With direct child processes, SlideStream needs two free loopback ports, one
for FastAPI and one for Next.js. Presenton's Electron app scans upward from
40000 by trying to bind each port. That check has a small race: the port is
closed after the test and another process can claim it before Presenton binds.
SlideStream should retry with fresh ports when a child exits during startup or
the health check times out because the port was taken.

Do not bind a managed local runtime to `0.0.0.0`. Loopback keeps it off the
local network.

### Data and provider configuration

Stopping the process should not erase the user's Presenton setup. Use a
persistent SlideStream-managed data directory, mounted as Presenton's
`/app_data` in Docker mode:

```text
~/.local/state/slide-stream/presenton/app_data/
```

That directory can hold Presenton's database, templates, generated decks, and
provider settings. Temporary files belong under the normal SlideStream temp
area for that run.

Provider setup needs thought. A one-shot container still needs an LLM provider
and usually an image provider. Options:

- Reuse environment variables the user already has, such as
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY`.
- Translate SlideStream's configured LLM provider and model to Presenton's
  `LLM` and provider-specific environment variables where the mapping is
  exact.
- Let `presenton setup` open the local Presenton UI once so the user can
  configure providers there, then persist the result in app data.

Do not copy keys silently in ways the user cannot inspect. Setup output should
say which provider Presenton will use and where its state lives.

### Concurrency and cleanup

Two SlideStream commands can run at once. The managed runtime needs a policy:

- Give every managed container a unique name and a SlideStream label.
- Use Docker-assigned ports for each run.
- Either use one shared app-data directory guarded by a file lock, or give
  each run its own copied app-data state. The lock is simpler. It serialises
  Presenton design work but avoids SQLite and file conflicts.
- On startup, remove managed containers whose recorded owner process is gone.
  A crash or `kill -9` will eventually happen; plan for it rather than
  treating it as user error.
- Keep a log file per run. On failure, print the last useful error and the log
  path.

### Startup cost

An ephemeral service trades background resource use for startup delay.
Presenton's Electron code allows up to 120 seconds for the FastAPI health
check. Real startup may be much faster after the first run, but migrations,
template imports, model checks, and browser/runtime preparation still take
time.

For a single command, that trade is fine. For batch work, offer an explicit
opt-in such as `--keep-presenton-running` or accept an external URL. Do not
quietly leave a daemon behind by default. The user asked for a simple CLI,
not for a surprise service.

## Desktop strategy

The SlideStream desktop app should stay focused on narrated video for now. It
already removes the terminal by starting the SlideStream backend itself.
Adding a second heavyweight local application would change its size, startup
time, troubleshooting, and update story.

Three future desktop routes remain open:

1. Point the desktop app at a hosted Presenton that the operator manages. This
   keeps the install small but sends deck content to that server.
2. Teach the Tauri shell to download, start, monitor, and stop a supported
   Presenton runtime. This keeps data local, but Presenton has no small,
   supported headless runtime package. Docker is too much to expect from a
   non-technical user, and extracting files from Presenton's Electron bundle
   is not a stable contract.
3. Add native SlideStream themes. This gives a smaller design improvement
   inside the existing app and avoids a second service entirely.

Until one of those is chosen, the honest manual workflow is two desktop apps:
design and export a deck in Presenton's desktop app, then make the video in
SlideStream's desktop app. That is still two installs, but neither app hides
what it does.

Do not add a template picker to the desktop UI that quietly assumes
`localhost:5001` is running. A control that fails until the user installs a
separate service will read as a bug, not as a feature.

## Risks and open questions

### API stability

Presenton is pre-1.0. The REST paths are versioned, but request fields,
authentication, export behaviour, and template handling can still change.
Pin the tested Presenton version in docs and in any managed Docker mode. Add
an opt-in live contract test that starts that pinned runtime and generates a
small deck.

### Text changes during design

Presenton can rephrase or truncate `slides_markdown` so content fits a chosen
layout. SlideStream narration comes from the source text and speaker notes.
The spoken words can therefore differ from the words printed on a designed
slide. That may be acceptable, but users should be warned. A stricter future
mode could compare source and exported slide text and report large changes.

### Slide count and ordering

Video narration is positional. If Presenton adds, removes, or reorders a
slide, voice and picture will drift apart. The implemented code requests no
added title or table-of-contents slide and aborts on count mismatch. It does
not prove that the order or content stayed identical.

### Authentication and local security

A managed local service must bind to loopback and shut down cleanly. Decide
whether to create a short-lived API key, use Presenton's desktop/local auth
mode, or rely on loopback alone for the short process lifetime. Do not expose
a managed instance on the LAN by default.

### Provider duplication

SlideStream and Presenton both need model providers, but they use them for
different jobs. SlideStream writes narration and speech. Presenton writes
slide content, selects layouts, and may generate images. Reusing environment
keys reduces setup, but model names and provider names do not always match.
Setup must show both sides plainly.

### Disk use and updates

A managed runtime needs a version policy:

- Which Presenton tag is tested with this SlideStream release?
- Does setup upgrade automatically, or only when the user runs an upgrade
  command?
- How much disk do the image, app data, generated decks, browser caches, and
  model caches use?
- Can the user delete the lot with one cleanup command?

Silent upgrades are risky with a beta upstream. Pin first. Upgrade on demand.

### Docker availability

Docker solves the runtime packaging problem, but it is still a dependency.
On macOS and Windows, Docker Desktop itself runs in the background. That is
normal for many CLI power users, but it is not zero setup. `presenton doctor`
should say whether Docker is installed, whether the daemon answers, and what
the user can do next.

### Source runtime support

A source checkout mode looks attractive until the support questions arrive.
Node versions, native Sharp packages, Chromium downloads, fonts, spaCy models,
and Python 3.11 environments all fail in different ways. Build it only if a
real group of CLI users cannot use Docker and wants this path.

### Release work still needed

Before the existing checkout integration ships:

- ~~Refresh `uv.lock`~~ Done: the lock now includes pypdfium2 in the extras
  and dev group.
- Bump the SlideStream version in `pyproject.toml` and
  `src/slide_stream/__init__.py`.
- Run a live Presenton contract test against the pinned version.
- Check the generated PPTX/PDF slide count, order, notes, and download paths
  against that live service.
- Confirm the release notes describe Presenton as optional, external, and
  CLI-oriented.

## Suggested roadmap

### Phase 1: make the current CLI integration safe to release

- Keep external-URL mode.
- Pin the tested Presenton version in the docs.
- Add an opt-in live test under the existing live-test gate.
- Refresh `uv.lock` and run the normal release checks.
- Keep `--preserve-style` and `--slides-pdf`. They are useful with or without
  Presenton.

### Phase 2: add managed Docker mode for power users

- Add `providers.presenton.mode: auto | external | docker`.
- Add `slide-stream presenton setup`, `doctor`, and `cleanup`.
- Start one pinned container per command, on a Docker-assigned loopback port.
- Persist app data outside the container.
- Stop the container on success, failure, and Ctrl-C.
- Test process cleanup with a fake Docker command, and add one opt-in live
  Docker test.

### Phase 3: decide the non-technical route

Pick one primary answer:

- Host Presenton beside the web UI for browser users, or
- build native SlideStream themes for local desktop users, or
- invest in a managed local Presenton runtime only if Presenton later ships a
  supported headless package.

Do not try to make all three the main path at once. Each one creates a
different support burden.

### Phase 4: borrow narrow features on their own merits

Candidates that do not require Presenton at runtime:

- Tone, verbosity, and language controls for `draft`.
- Better long-document chunking for `draft`.
- OCR support for scanned documents.
- An outline review step in the web UI.
- MCP tools over `slide-stream serve` jobs.
- Native chart slides for Markdown tables, if the renderer grows that way.

Each candidate should get its own small design decision. None should be
justified only by the fact that Presenton has it.

## Decisions already made

These decisions came out of the investigation and are recorded in
[ADR-0009](adr/0009-presenton-designed-decks.md):

- Source to designed PPTX/PDF to slide pictures to video is the workflow.
- Presenton stays external. SlideStream does not bundle or fork it.
- The CLI integration uses the public HTTP API.
- Presenton receives credit, and its Apache-2.0 licence applies to any copied
  code or assets.
- Designed-slide video fails on slide-count mismatch.

These questions remain open:

- Should managed Docker mode become the default CLI path?
- Should SlideStream ever manage a source checkout, or is Docker enough?
- Should non-technical users get hosted Presenton, native themes, or both?
- How much design responsibility should SlideStream take on over time?

My current view is conservative. Keep Presenton where it is strong, keep
SlideStream where it is strong, and make the handoff smoother for power users
before promising a one-app design experience to everyone else.

## Sources checked

Presenton sources in the local clone:

- `README.md` for product claims, API examples, Docker and desktop routes.
- `VISION.md` for project direction.
- `CONTRIBUTING.md` for contribution scope, PR size, and CLA requirements.
- `servers/fastapi/pyproject.toml` for backend packaging and Python pin.
- `servers/fastapi/server.py` for the required port argument and loopback bind.
- `servers/fastapi/api/lifespan.py` for database, template, and provider
  startup work.
- `servers/fastapi/models/generate_presentation_request.py` for the
  generation request contract.
- `servers/fastapi/utils/export_utils.py` for the `/pdf-maker` export flow.
- `servers/fastapi/services/export_task_service.py` for the Node export
  runtime and Chromium requirements.
- `scripts/sync-presentation-export.cjs` for the separate export-core
  download.
- `electron/app/utils/servers.ts` for managed startup and shutdown.
- `electron/app/utils/index.ts` for free-port scanning.
- `Dockerfile` and `start.js` for the all-in-one runtime.
- `.github/workflows/test-all.yml` for test coverage in CI.
- `docs/template-v2.md` and `docs/presentation-generation-modes.md`.

External records:

- Presenton GitHub repository: https://github.com/presenton/presenton
- Presenton documentation: https://docs.presenton.ai
- Presenton PyPI placeholder: https://pypi.org/project/presenton/
- SlideStream PyPI release: https://pypi.org/project/slide-stream/

SlideStream files changed or added by the implementation work:

- `src/slide_stream/presenton.py`
- `src/slide_stream/slide_render.py`
- `src/slide_stream/cli.py`
- `src/slide_stream/config_loader.py`
- `tests/test_presenton.py`
- `tests/conftest.py`
- `pyproject.toml`
- `slidestream.example.yaml`
- `README.md`
- `docs/USE_CASES.md`
- `docs/adr/0009-presenton-designed-decks.md`
- `docs/adr/README.md`
- `CLAUDE.md`





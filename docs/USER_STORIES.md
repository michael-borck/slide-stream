# User stories

Who SlideStream is for and what they need from it. Each story names its
status: **done** (in a release), or the roadmap phase that delivers it.
Decisions behind them are in the [ADRs](adr/README.md); the order of work is
in the [roadmap](ROADMAP.md). For step-by-step recipes see
[USE_CASES.md](USE_CASES.md).

## People

| Who | Wants | Uses |
|---|---|---|
| **Lecturer** | Turn lecture notes or an existing deck into a narrated video for students; keep teaching in their own voice and their institution's style | Desktop app or CLI |
| **Technical instructor** | Lecture videos with code, tables and diagrams that stay correct | CLI |
| **Explainer maker** | Short, good-looking videos from an idea or a sentence, with a presenter and music | Desktop app or web UI |
| **Power user** | Script everything, run their own AI servers, control every step | CLI, config files |
| **Institution operator** | Host SlideStream for staff, with their own servers, quotas and privacy rules | Docker, web UI |
| **Viewer** (student, member of the public) | Watch and follow the result, including with a hearing or vision impairment | The video, captions, transcript |
| **Newcomer** | Try it with no account, no keys and no setup | Hosted demo, then desktop app |

## Stories

### Getting started

- As a **newcomer**, I want to make a narrated video from a Markdown file
  with no keys or servers, so I can judge the tool in minutes. **Done**
  (offline Kokoro voice, text cards, bundled ffmpeg).
- As a **newcomer**, I want the website and PyPI page to describe what the
  tool does today and how to install it in one line. **Phase 11**.
- As a **lecturer**, I want a desktop app that installs and updates itself,
  so I never open a terminal. **Done** (unsigned); signing **Phase 9**.
- As a **lecturer**, I want to see which of my AI servers are working before
  I render, so a long render doesn't fail at the end. **Phase 3**.

### Sources and content

- As a **lecturer**, I want to narrate my existing PowerPoint from its
  speaker notes. **Done**.
- As a **lecturer**, I want to start from a topic sentence, a Word document
  or a PDF and get a draft deck. **Done** (`draft`); long and scanned
  documents **Phase 15**.
- As a **technical instructor**, I want code blocks and tables in my
  Markdown to stay intact on the slides and in the deck. **Done** (2.16).
- As a **technical instructor**, I want numeric tables shown as charts.
  **Phase 15**.

### Voice and narration

- As a **lecturer**, I want the narration in my own cloned voice, deleted
  from the server afterwards. **Done** (Voicebox, Chatterbox).
- As a **lecturer**, I want narration to work offline and keep my text on
  my machine. **Done** (Kokoro floor, ADR-0012).
- As a **lecturer**, I want acronyms, names and code read correctly.
  **Phase 14**.
- As a **lecturer**, I want to add pauses where students need time to
  think. **Phase 14**.
- As a **lecturer**, I want to check and edit the narration script before
  the expensive render starts. **Phase 5**.

### Look and design

- As a **lecturer**, I want new slides in my university's PowerPoint
  template, so videos match my other teaching material. **Done** (2.18).
- As a **lecturer**, I want my already-styled deck to keep its look in the
  video. **Done** (`--preserve-style`).
- As an **explainer maker**, I want a professionally designed deck from my
  outline, with a choice of templates. **Done, experimental** (Presenton).
- As a **technical instructor**, I want to be told when a designer dropped
  or changed my content, and to get my code back. **Done** (ADR-0013).
- As a **power user**, I want one switch to choose plain, my template,
  Presenton or my own design. **Phase 6**.

### The video

- As a **lecturer**, I want a presenter in the corner, from a mascot to an
  animated photo of me, at a cost I choose. **Done**.
- As a **lecturer**, I want bullets to appear as I talk about them, so
  students follow along. **Phase 14**.
- As a **technical instructor**, I want to show a screen recording on a
  slide. **Phase 14**.
- As an **explainer maker**, I want background music that never drowns the
  voice. **Phase 7**.
- As an **explainer maker**, I want a vertical version for short-video
  platforms. **Phase 13**.
- As an **explainer maker**, I want presets such as "lecture" or
  "creative" instead of a dozen options. **Phase 8**.

### Accessibility

- As a **viewer** who is deaf or hard of hearing, I want captions on every
  video. **Done** (2.17).
- As a **viewer**, I want a transcript I can read, search and translate.
  **Done** (2.17).
- As a **viewer**, I want chapters so I can jump to the part I need.
  **Done** (2.17).
- As a **viewer** using a screen reader, I want pictures in the deck to
  have alt text. **Done** (2.17, `--alt-text`).
- As a **lecturer**, I want to be warned about low contrast and small text
  before I publish. **Done** (2.17, `doctor`).

### Outputs and sharing

- As a **lecturer**, I want an editable PowerPoint with the narration in
  the speaker notes, to present live. **Done**.
- As a **lecturer**, I want a handout PDF with slides and notes.
  **Phase 13**.
- As a **lecturer**, I want a single web page that plays the presentation
  offline. **Phase 13**.

### Control and automation

- As a **power user**, I want to stop after any step, edit its output and
  carry on without redoing earlier work. **Phases 4 and 5**.
- As a **power user**, I want to see which engine and fallback level was
  actually used for each step. **Phase 3**.
- As a **power user**, I want to render a whole course folder in one
  command. **Future work**.
- As a **power user**, I want AI agents to drive renders through MCP.
  **Future work**.
- As a **lecturer**, I want versions of my video in other languages.
  **Future work**.

### Hosting

- As an **institution operator**, I want to host the web UI with our own
  servers and keys, behind our proxy, with rate limits. **Done**.
- As an **institution operator**, I want to keep API keys out of plain-text
  files on staff machines. **Phase 9** (OS keychain).

## Adding a story

Write it as "As a *person*, I want *something*, so *reason*." Give it a
status. If no phase delivers it, add it to the roadmap's future work rather
than leaving it unowned.

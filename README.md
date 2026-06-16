# cycle-detection

A monitor extension for the **pi** agentic coding harness. It watches the
agent for unproductive loops and nudges it back on track before it burns a
whole budget spinning in place.

## What it detects

Two independent layers, both built on one primitive — *"has the trajectory
revisited a state within a recent window?"*

1. **Step-based** (tool calls) — fires at `tool_result`:
   - `EXACT_REPEAT`: the same action + args produced the same observation on a
     run of *consecutive* steps (e.g. retrying an identical failing command).
   - `OSCILLATION`: the outcome-state sequence cycles with a small period
     (`ABAB`, `ABCABC`, … — e.g. an edit↔revert loop).
2. **Stream-based** (reasoning / answer text) — fires mid-stream on
   `thinking_delta` / `text_delta`:
   - sentence-level `EXACT_REPEAT` and `OSCILLATION` in the model's own output.

Both detectors escalate **warn → hard** as the evidence grows.

## How it intervenes

- **Step cycles** are surfaced on the status bar and, in active mode, inject a
  one-shot *steer* message asking the agent to change approach. It only steers
  when a cycle first appears or escalates — it will not spam every step.
- **Stream cycles** (hard) abort the in-flight message, strip the looping
  reasoning/tool calls, and steer a fresh turn (with a `notify-send` desktop
  notification when available).

Stream detection is scoped to a **single message** (reset at `message_end`),
so a sentence repeated in one message never counts toward the next — and it
ignores short non-prose fragments (filenames, list markers, table cells) via
`minSentenceWords`, so normal repetitive output does not trip it.

It never kills the process — interventions are course corrections, not halts.

## Commands

`/cycle [subcommand]`

| Subcommand | Effect |
| --- | --- |
| `status` (default) | Show config, mode, and fire counts |
| `stats` | Detailed step + stream statistics |
| `enable` / `disable` | Master on/off switch |
| `shadow` | Observe only — detect and report, never intervene |
| `active` | Detect and intervene (default) |

## Tuning

Thresholds live in `DEFAULT_CONFIG` (`src/detector.ts`) for the step detector
and `DEFAULT_STREAM_CONFIG` (`src/stream-detector.ts`) for the stream detector
— window size, warn/hard repeat counts, max oscillation period, and warmup.
Run in `shadow` mode against real sessions to calibrate before enabling
interventions.

## Development

```sh
npm install      # links the local pi-coding-agent + dev tooling
npm test         # node:test suites via tsx
npm run typecheck
```

The detection engine (`src/detector.ts`, `src/canonicalize.ts`,
`src/stream-detector.ts`) is pure and stateless, so verdicts are a
deterministic function of the trajectory and can be replayed against recorded
traces during tuning.

# Groundcheck design system

The tokens live as CSS variables at the top of `src/api/static/index.html`. This file records the
decisions behind them, so a later change can keep to them or knowingly break them.

## Direction

A technical, evaluation-first demo. The page should read like a developer tool showing its
working, not a chat product: the answer, the run trace and the evaluation table get equal weight.
Starting point: the ui-ux-pro-max design system for "developer tool AI evaluation technical
dashboard" (dark developer style, code fonts, green "run" accent). Its suggested FAQ-landing
layout didn't fit, so the layout below replaces it.

## Layout

1. Top bar: wordmark, live health pill (`/health`), theme toggle, GitHub.
2. Hero: one-line claim plus the question box, with golden-set questions as one-click examples.
3. Run: answer card (citations as clickable chips) beside a run trace (retrieve → generate →
   verify citations) with per-run metrics: first token, total time, citations valid, sources cited.
   Retrieved chunks are listed underneath in rank order with their fusion score.
4. Evaluation: three facts plus the experiment table, with retrieval, generation and ops columns
   grouped separately (as CLAUDE.md requires) and the v2 regression row left in.

## Color

Both themes are designed: the page follows the system setting and the toggle overrides it.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | #f6f7f9 | #0b1120 | page |
| `--surface` | #ffffff | #111a2d | cards |
| `--surface-2` | #eef1f5 | #18233a | table heads, chips hover |
| `--border` | #d8dee6 | #273450 | all dividers |
| `--ink` | #0f172a | #e6edf7 | body text |
| `--ink-dim` | #475569 | #9aa8bd | secondary text (≥ 4.5:1 on surface in both) |
| `--accent` | #15803d | #22c55e | run button, verified citations, best values |
| `--danger` | #b91c1c | #f87171 | unverified citations, regressions, errors |
| `--warn` | #a16207 | #fbbf24 | in-progress trace step, indexing |

Green means "verified" and red means "unverified" everywhere. Don't reuse them decoratively.

## Type

- IBM Plex Sans for prose, JetBrains Mono for anything that is data: IDs, metrics, labels, table.
- Body 16px / 1.6; answer 17px / 1.7, capped at 70ch.
- Labels are 12px mono uppercase with letter spacing. Nothing smaller than 12px.
- Numbers use `tabular-nums`.

## Spacing and shape

4px base: 4, 8, 12, 16, 24, 32, 48, 64. Radius 8px for cards and buttons, 4px for chips and tags.

## Motion

Subtle only: 150–200ms hover and focus transitions, a pulsing dot on the active trace step,
a blinking cursor while the answer streams, and a highlight flash on the source a citation jumps to.
`prefers-reduced-motion` turns all of it off.

## Accessibility rules

- Touch targets are at least 44px (36px for the inline example chips).
- Visible focus ring (`--ring`) on everything interactive.
- Icons are inline SVG with `aria-hidden`; icon-only buttons have `aria-label`.
- One polite status region announces run progress; errors use `role="alert"`.
- The results table scrolls horizontally inside its own region on narrow screens.

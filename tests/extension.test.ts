/**
 * Integration tests for the extension wiring in index.ts.
 *
 * Run with: npx tsx tests/extension.test.ts
 *
 * Focus: stream-cycle detection must be scoped to a single assistant message.
 * Regression — a previous message's repeated sentences lingered in the
 * detector window and spuriously aborted the *next*, unrelated stream.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import ext from "../index.js";

// ── Mock pi harness ──────────────────────────────────────────────

type Handler = (event: unknown, ctx: unknown) => unknown;

function mount() {
	const handlers: Record<string, Handler> = {};
	let aborts = 0;
	const ui = {
		setStatus: () => {},
		setWidget: () => {},
		notify: () => {},
		// Theme stub: real Theme.fg/bg wrap text in ANSI; identity is enough here.
		theme: {
			fg: (_color: string, text: string) => text,
			bg: (_color: string, text: string) => text,
		},
	};
	const pi = {
		on: (e: string, h: Handler) => {
			handlers[e] = h;
		},
		registerCommand: () => {},
		sendMessage: () => {},
	};
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	ext(pi as any);
	const ctx = {
		ui,
		abort: () => {
			aborts++;
		},
	};
	const fire = (event: string, payload: unknown) =>
		handlers[event]?.(payload, ctx);
	const thinking = (delta: string) =>
		fire("message_update", { assistantMessageEvent: { type: "thinking_delta", delta } });
	const text = (delta: string) =>
		fire("message_update", { assistantMessageEvent: { type: "text_delta", delta } });
	return {
		handlers,
		ctx,
		fire,
		thinking,
		text,
		get aborts() {
			return aborts;
		},
	};
}

const ASSISTANT = { role: "assistant" };

// ── Tests ────────────────────────────────────────────────────────

describe("stream cycle scoping", () => {
	it("aborts on a genuine reasoning loop within a message", async () => {
		const h = mount();
		await h.fire("session_start", {});
		await h.fire("message_start", { message: ASSISTANT });
		const loop = "I will now reconsider the entire approach again. ";
		for (let i = 0; i < 12; i++) await h.thinking(loop);
		assert.ok(h.aborts >= 1, "expected at least one abort on a real loop");
	});

	it("does NOT abort the next message even if message_start does not fire", async () => {
		const h = mount();
		await h.fire("session_start", {});

		// Message A: a genuine loop that triggers an abort.
		await h.fire("message_start", { message: ASSISTANT });
		const loop = "I will now reconsider the entire approach again. ";
		for (let i = 0; i < 12; i++) await h.thinking(loop);
		assert.ok(h.aborts >= 1);
		await h.fire("message_end", {
			message: { role: "assistant", content: [{ type: "thinking", thinking: "x" }] },
		});

		// Message B (the steered retry): short, non-repetitive answer, and we
		// deliberately do NOT fire message_start to mimic a continuation.
		const before = h.aborts;
		for (const t of [
			"Based on plot.py, ",
			"the current code generates ",
			"these files in the figures directory: ",
			"- `plot1_unified.pdf` ",
			"- `plot2_neigh_sw.pdf` ",
			"- `plot3_unified.pdf`. ",
		]) {
			await h.text(t);
		}
		assert.equal(
			h.aborts - before,
			0,
			"no spurious abort on a non-repetitive follow-up stream",
		);
	});

	it("does not abort on a list of bare filenames", async () => {
		const h = mount();
		await h.fire("session_start", {});
		await h.fire("message_start", { message: ASSISTANT });
		for (const f of [
			"plot1_unified. ",
			"plot2_neigh_sw. ",
			"plot2_no_neigh. ",
			"plot2_no_sw. ",
			"plot3_unified. ",
			"plot_summary_stats. ",
			"plot_edge_usage. ",
			"plot_gap_histogram. ",
		]) {
			await h.text(f);
		}
		assert.equal(h.aborts, 0, "bare filenames are not prose and must not abort");
	});
});

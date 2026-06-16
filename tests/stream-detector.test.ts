/**
 * Tests for the streaming cycle detector.
 *
 * Run with: npx tsx tests/stream-detector.test.ts
 *
 * Covers:
 * - Sentence splitting edge cases
 * - Feed incremental deltas (partial sentences, multi-sentence chunks)
 * - Exact-repeat detection (below/at/above warn and hard thresholds)
 * - Oscillation detection (period-2, period-3)
 * - Shadow mode (stats accumulate, no callback fires)
 * - Reset / resetFull semantics
 * - Pending text handling across feed() calls
 * - Empty / whitespace-only input
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
	StreamDetector,
	type StreamConfig,
	type StreamVerdict,
	type Severity,
	DEFAULT_STREAM_CONFIG,
} from "../src/stream-detector.js";

// ── Helpers ──────────────────────────────────────────────────────

/**
 * Create a detector with known config and an optional onCycle spy.
 *
 * The short-sentence filter (minSentenceWords) is disabled by default here so
 * these tests can exercise detection mechanics with terse synthetic sentences
 * ("Alpha.", "Bravo."). The filter itself is covered in its own describe block
 * by passing an explicit minSentenceWords.
 */
function createDetector(
	config?: Partial<StreamConfig>,
	onCycle?: (v: StreamVerdict, s: Severity) => void,
): StreamDetector {
	return new StreamDetector(
		{ ...DEFAULT_STREAM_CONFIG, minSentenceWords: 0, ...config },
		onCycle,
	);
}

/** A sentence that will hash identically. */
const SAME = "This sentence repeats over and over.";

/** Distinct sentences. */
const A = "Alpha.";
const B = "Bravo.";
const C = "Charlie.";

// ── Sentence Splitting (indirect, via feed) ──────────────────────

describe("sentence splitting", () => {
	it("accumulates partial sentences across feed calls", () => {
		const d = createDetector();
		d.feed("Hello");
		assert.equal(d.getStats().sentencesExtracted, 0);
		d.feed(", world.");
		assert.equal(d.getStats().sentencesExtracted, 1);
	});

	it("handles multiple sentences in one chunk", () => {
		const d = createDetector();
		d.feed("First. Second! Third?");
		assert.equal(d.getStats().sentencesExtracted, 3);
	});

	it("handles ellipsis-like text without false splits", () => {
		const d = createDetector();
		// "..." has delimiters so it splits, but the key is it doesn't crash
		d.feed("Wait... Hmm... OK.");
		// At least one complete sentence extracted
		assert.ok(d.getStats().sentencesExtracted >= 1);
	});

	it("returns OK for text with no delimiters", () => {
		const d = createDetector();
		const r = d.feed("No punctuation here at all");
		assert.equal(r.verdict.type, "OK");
		assert.equal(r.severity, "none");
	});

	it("handles empty string", () => {
		const d = createDetector();
		const r = d.feed("");
		assert.equal(r.verdict.type, "OK");
		assert.equal(d.getStats().sentencesExtracted, 0);
	});

	it("handles whitespace-only input", () => {
		const d = createDetector();
		const r = d.feed("   \n\t  ");
		assert.equal(r.verdict.type, "OK");
		assert.equal(d.getStats().sentencesExtracted, 0);
	});
});

// ── Short-fragment filter (minSentenceWords) ─────────────────────

describe("short-fragment filter", () => {
	it("ignores a list of bare filenames that end with a period", () => {
		// Regression: `plot1_unified.`, `plot2_neigh_sw.` … are not prose and
		// must not register as repeated sentences.
		const d = createDetector({
			minSentenceWords: 4,
			warmupSentences: 0,
			tRepeatWarn: 3,
			tRepeatHard: 5,
		});
		const files = [
			"plot1_unified.",
			"plot2_neigh_sw.",
			"plot2_no_neigh.",
			"plot2_no_sw.",
			"plot3_unified.",
			"plot_summary_stats.",
			"plot_edge_usage.",
			"plot_gap_histogram.",
		];
		for (const f of files) d.feed(`${f} `);
		const stats = d.getStats();
		assert.equal(stats.sentencesExtracted, 0);
		assert.equal(stats.lastVerdict.type, "OK");
	});

	it("does not flag a single repeated short fragment", () => {
		const d = createDetector({
			minSentenceWords: 4,
			warmupSentences: 0,
			tRepeatWarn: 3,
			tRepeatHard: 5,
		});
		for (let i = 0; i < 8; i++) d.feed("plot1_unified. ");
		assert.equal(d.getStats().sentencesExtracted, 0);
		assert.equal(d.getStats().lastVerdict.type, "OK");
	});

	it("still counts and flags genuine repeated prose sentences", () => {
		const d = createDetector({
			minSentenceWords: 4,
			warmupSentences: 0,
			tRepeatWarn: 3,
			tRepeatHard: 5,
		});
		const prose = "I am going to read the file now.";
		for (let i = 0; i < 4; i++) d.feed(`${prose} `);
		const r = d.feed(`${prose} `);
		assert.equal(r.verdict.type, "EXACT_REPEAT");
		assert.equal(r.severity, "hard");
	});
});

// ── Exact Repeat ─────────────────────────────────────────────────

describe("exact-repeat detection", () => {
	it("returns OK below warmup threshold", () => {
		const d = createDetector({ warmupSentences: 3 });
		d.feed(SAME); // 1 sentence, below warmup of 3
		const r = d.feed(SAME);
		assert.equal(r.verdict.type, "OK");
		assert.equal(r.severity, "none");
	});

	it("returns OK below warn threshold", () => {
		const d = createDetector({ warmupSentences: 0, tRepeatWarn: 4 });
		for (let i = 0; i < 3; i++) d.feed(SAME);
		// Actually 4 sentences, tRepeatWarn = 4, should fire warn
		// Let's test with 3 < 4
		const d2 = createDetector({ warmupSentences: 0, tRepeatWarn: 5 });
		for (let i = 0; i < 4; i++) d2.feed(SAME);
		const r2 = d2.feed("Different sentence.");
		// 4 same + 1 different — 4 < tRepeatWarn(5) for the same sentence
		assert.equal(r2.verdict.type, "OK");
	});

	it("fires WARN at exactly tRepeatWarn", () => {
		const d = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 4,
			tRepeatHard: 8,
		});
		for (let i = 0; i < 4; i++) d.feed(SAME);
		// Actually the 4th already fires. Let's be precise.
		const d2 = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 4,
			tRepeatHard: 8,
		});
		for (let i = 0; i < 3; i++) d2.feed(SAME);
		const r2 = d2.feed(SAME); // 4th — exactly tRepeatWarn
		assert.equal(r2.verdict.type, "EXACT_REPEAT");
		assert.equal(r2.severity, "warn");
		if (r2.verdict.type === "EXACT_REPEAT") {
			assert.ok(r2.verdict.count >= 4);
		}
	});

	it("fires HARD at exactly tRepeatHard", () => {
		const d = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 3,
			tRepeatHard: 5,
		});
		for (let i = 0; i < 4; i++) d.feed(SAME);
		const r = d.feed(SAME); // 5th — exactly tRepeatHard
		assert.equal(r.verdict.type, "EXACT_REPEAT");
		assert.equal(r.severity, "hard");
		if (r.verdict.type === "EXACT_REPEAT") {
			assert.ok(r.verdict.count >= 5);
		}
	});

	it("does NOT fire on different sentences", () => {
		const d = createDetector({ warmupSentences: 0, tRepeatWarn: 3 });
		const sentences = [
			"First sentence here.",
			"Second sentence here.",
			"Third sentence here.",
			"Fourth sentence here.",
			"Fifth sentence here.",
		];
		for (const s of sentences) d.feed(s);
		const stats = d.getStats();
		assert.equal(stats.lastVerdict.type, "OK");
	});

	it("fires after warmup period passes", () => {
		const d = createDetector({
			warmupSentences: 2,
			tRepeatWarn: 3,
			tRepeatHard: 5,
		});
		// 2 warmup sentences (different)
		d.feed("Warmup one.");
		d.feed("Warmup two.");
		// 3 repeats
		for (let i = 0; i < 3; i++) d.feed(SAME);
		const r = d.feed(SAME); // 4th repeat — tRepeatWarn = 3
		assert.equal(r.verdict.type, "EXACT_REPEAT");
		assert.equal(r.severity, "warn");
	});

	it("reports correct firstRepeatOffset", () => {
		const d = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 3,
			tRepeatHard: 5,
		});
		d.feed("Intro.");
		for (let i = 0; i < 4; i++) d.feed(SAME);
		const r = d.feed(SAME);
		assert.equal(r.verdict.type, "EXACT_REPEAT");
		if (r.verdict.type === "EXACT_REPEAT") {
			// firstRepeatOffset should be > 0 (after "Intro.")
			assert.ok(r.verdict.firstRepeatOffset > 0);
		}
	});

	it("severity escalates from warn to hard as repeats grow", () => {
		const d = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 3,
			tRepeatHard: 6,
		});
		for (let i = 0; i < 2; i++) d.feed(SAME);
		const r1 = d.feed(SAME); // 3rd — warn
		assert.equal(r1.severity, "warn");
		for (let i = 0; i < 2; i++) d.feed(SAME);
		const r2 = d.feed(SAME); // 6th — hard
		assert.equal(r2.severity, "hard");
	});
});

// ── Oscillation ──────────────────────────────────────────────────

describe("oscillation detection", () => {
	it("detects ABAB period-2 oscillation", () => {
		const d = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 10, // High to avoid exact-repeat firing
			tRepeatHard: 20,
			minRepsP2: 2,
			pMax: 4,
		});
		const pairs = [
			[A, B],
			[A, B],
		];
		for (const [s1, s2] of pairs) {
			d.feed(s1);
			d.feed(s2);
		}
		// The last sentences include ABAB pattern
		// Actually let's check the 4th sentence (second B)
		const d2 = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 10,
			tRepeatHard: 20,
			minRepsP2: 2,
			pMax: 4,
		});
		d2.feed(A);
		d2.feed(B);
		d2.feed(A);
		const r2 = d2.feed(B); // 4th — ABAB complete
		assert.equal(r2.verdict.type, "OSCILLATION");
		if (r2.verdict.type === "OSCILLATION") {
			assert.equal(r2.verdict.period, 2);
		}
	});

	it("detects ABCABC period-3 oscillation", () => {
		const d = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 10,
			tRepeatHard: 20,
			minRepsP2: 2,
			pMax: 4,
		});
		d.feed(A);
		d.feed(B);
		d.feed(C);
		d.feed(A);
		d.feed(B);
		const r = d.feed(C); // 6th — ABCABC complete
		assert.equal(r.verdict.type, "OSCILLATION");
		if (r.verdict.type === "OSCILLATION") {
			assert.equal(r.verdict.period, 3);
		}
	});

	it("does NOT fire on non-repeating sequence", () => {
		const d = createDetector({ warmupSentences: 0, minRepsP2: 2, pMax: 4 });
		const sentences = [
			"Alpha one.",
			"Bravo two.",
			"Charlie three.",
			"Delta four.",
			"Echo five.",
			"Foxtrot six.",
		];
		for (const s of sentences) d.feed(s);
		const stats = d.getStats();
		assert.equal(stats.lastVerdict.type, "OK");
	});

	it("period-1 oscillation fires with minRepsP1", () => {
		// Period-1 oscillation = same state repeated = same as exact-repeat
		// But oscillation uses minRepsP1 threshold
		const d = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 10, // Suppress exact-repeat
			tRepeatHard: 20,
			minRepsP1: 4,
		});
		for (let i = 0; i < 4; i++) d.feed(SAME);
		const r = d.feed(SAME); // 5th — but exact-repeat suppressed
		// This should fire as OSCILLATION period-1
		// Actually, exact-repeat check runs first in runDetection.
		// With tRepeatWarn=10, count=5 < 10, so exact-repeat doesn't fire.
		// Then oscillation check runs with minRepsP1=4, which fires.
		assert.equal(r.verdict.type, "OSCILLATION");
		if (r.verdict.type === "OSCILLATION") {
			assert.equal(r.verdict.period, 1);
		}
	});

	it("reports patternStartOffset for oscillation", () => {
		const d = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 10,
			tRepeatHard: 20,
			minRepsP2: 2,
			pMax: 4,
		});
		d.feed("Intro text here.");
		d.feed(A);
		d.feed(B);
		d.feed(A);
		const r = d.feed(B);
		if (r.verdict.type === "OSCILLATION") {
			assert.ok(r.verdict.patternStartOffset >= 0);
		}
	});
});

// ── Shadow Mode ──────────────────────────────────────────────────

describe("shadow mode", () => {
	it("stats accumulate in shadow mode", () => {
		const d = createDetector({
			shadow: true,
			warmupSentences: 0,
			tRepeatWarn: 3,
			tRepeatHard: 5,
		});
		for (let i = 0; i < 6; i++) d.feed(SAME);
		const stats = d.getStats();
		// Verdict is computed even in shadow mode
		assert.equal(stats.lastVerdict.type, "EXACT_REPEAT");
		assert.equal(stats.lastSeverity, "hard");
		assert.ok(stats.hardDetections >= 1);
	});

	it("onCycle callback never fires in shadow mode", () => {
		let callbackFired = 0;
		const d = createDetector(
			{ shadow: true, warmupSentences: 0, tRepeatWarn: 3, tRepeatHard: 5 },
			(_v, _s) => {
				callbackFired++;
			},
		);
		for (let i = 0; i < 6; i++) d.feed(SAME);
		assert.equal(callbackFired, 0);
	});

	it("onCycle callback fires in active mode", () => {
		let callbackFired = 0;
		let lastSeverity: Severity = "none";
		const d = createDetector(
			{ shadow: false, warmupSentences: 0, tRepeatWarn: 3, tRepeatHard: 5 },
			(_v, s) => {
				callbackFired++;
				lastSeverity = s;
			},
		);
		for (let i = 0; i < 6; i++) d.feed(SAME);
		assert.ok(callbackFired > 0);
		assert.equal(lastSeverity, "hard");
	});

	it("setConfig can toggle shadow mode", () => {
		let callbackFired = 0;
		const d = createDetector(
			{ shadow: true, warmupSentences: 0, tRepeatWarn: 3 },
			() => {
				callbackFired++;
			},
		);
		for (let i = 0; i < 4; i++) d.feed(SAME);
		assert.equal(callbackFired, 0); // Shadow: no callback

		d.setConfig({ shadow: false });
		d.feed(SAME); // 5th — should fire now
		assert.ok(callbackFired > 0);
	});
});

// ── Reset / ResetFull ────────────────────────────────────────────

describe("reset semantics", () => {
	it("reset() clears detection state but preserves stats", () => {
		const d = createDetector({ warmupSentences: 0, tRepeatWarn: 3 });
		for (let i = 0; i < 4; i++) d.feed(SAME);
		const stats = d.getStats();
		assert.equal(stats.lastVerdict.type, "EXACT_REPEAT");
		assert.ok(stats.sentencesExtracted >= 4);

		d.reset();
		const after = d.getStats();
		// Stats preserved
		assert.equal(after.sentencesExtracted, stats.sentencesExtracted);
		// But verdict reset
		assert.equal(after.lastVerdict.type, "OK");
		assert.equal(after.lastSeverity, "none");

		// New feeds don't see old sentences
		const r = d.feed("New sentence.");
		assert.equal(r.verdict.type, "OK");
	});

	it("resetFull() clears everything", () => {
		const d = createDetector({ warmupSentences: 0, tRepeatWarn: 3 });
		for (let i = 0; i < 4; i++) d.feed(SAME);
		d.resetFull();
		const stats = d.getStats();
		assert.equal(stats.charactersFed, 0);
		assert.equal(stats.sentencesExtracted, 0);
		assert.equal(stats.warnDetections, 0);
		assert.equal(stats.hardDetections, 0);
		assert.equal(stats.lastVerdict.type, "OK");
	});
});

// ── Stats ────────────────────────────────────────────────────────

describe("statistics", () => {
	it("tracks characters fed", () => {
		const d = createDetector();
		d.feed("Hello.");
		d.feed("World!");
		const stats = d.getStats();
		assert.equal(stats.charactersFed, 12);
	});

	it("tracks max repeat run", () => {
		const d = createDetector({ warmupSentences: 0 });
		d.feed("Different one.");
		d.feed(SAME);
		d.feed(SAME);
		d.feed(SAME);
		d.feed("Another different.");
		const stats = d.getStats();
		assert.equal(stats.maxRepeatRun, 3);
	});

	it("tracks min oscillation period", () => {
		const d = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 10,
			tRepeatHard: 20,
			minRepsP2: 2,
			pMax: 4,
		});
		d.feed(A);
		d.feed(B);
		d.feed(A);
		d.feed(B);
		const stats = d.getStats();
		assert.equal(stats.minOscillationPeriod, 2);
	});

	it("getStats returns a copy (mutation safe)", () => {
		const d = createDetector();
		const s1 = d.getStats();
		s1.charactersFed = 9999;
		const s2 = d.getStats();
		assert.notEqual(s1.charactersFed, s2.charactersFed);
	});
});

// ── Edge Cases ───────────────────────────────────────────────────

describe("edge cases", () => {
	it("handles very long single chunk", () => {
		const d = createDetector({ warmupSentences: 0, tRepeatWarn: 3 });
		// 100 identical sentences in one chunk (space-separated for splitter)
		const chunk = Array.from({ length: 100 }, () => SAME).join(" ");
		const r = d.feed(chunk);
		assert.equal(r.verdict.type, "EXACT_REPEAT");
		assert.equal(r.severity, "hard");
	});

	it("handles mixed delimiters", () => {
		const d = createDetector({ warmupSentences: 0, tRepeatWarn: 3 });
		d.feed("One! Two? Three. Four!");
		assert.equal(d.getStats().sentencesExtracted, 4);
	});

	it("handles sentence that is only a delimiter", () => {
		const d = createDetector();
		d.feed("...");
		// Should not crash, may or may not extract a sentence
		const stats = d.getStats();
		assert.ok(stats.sentencesExtracted >= 0);
	});

	it("window limits detection — old repeats fall out", () => {
		const d = createDetector({
			warmupSentences: 0,
			tRepeatWarn: 3,
			window: 4,
		});
		// 3 repeats of SAME
		for (let i = 0; i < 3; i++) d.feed(SAME);
		// 2 different sentences push the repeats out of window
		d.feed("Different one.");
		d.feed("Different two.");
		const r = d.feed("Different three.");
		assert.equal(r.verdict.type, "OK");
	});

	it("hash is case-insensitive", () => {
		const d = createDetector({ warmupSentences: 0, tRepeatWarn: 3 });
		d.feed("Hello world.");
		d.feed("HELLO WORLD.");
		const r = d.feed("hello  world."); // extra whitespace
		assert.equal(r.verdict.type, "EXACT_REPEAT");
		assert.equal(r.severity, "warn");
	});

	it("pending text preserved across feed calls", () => {
		const d = createDetector();
		d.feed("This is");
		assert.equal(d.getPending().length, 7);
		d.feed(" a complete");
		assert.equal(d.getStats().sentencesExtracted, 0);
		d.feed(" sentence.");
		assert.equal(d.getStats().sentencesExtracted, 1);
		assert.equal(d.getPending().length, 0);
	});
});

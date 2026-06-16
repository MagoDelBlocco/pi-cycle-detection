/**
 * Comprehensive tests for the cycle detection engine.
 *
 * Run with: npx tsx tests/detector.test.ts
 *
 * Covers:
 * - cycle_period: AAAA, ABAB, ABCABC, ABCAB (no cycle), below-min_reps
 * - detectExactRepeat: fires on repeated action + identical observation;
 *   does NOT fire on repeated action + changing observation
 * - detectOscillation: fires on outcome-state oscillation even though
 *   per-step observations differ
 * - canonicalizeArgs: volatile fields stripped; semantically different args differ
 * - runMonitor: warmup, severity levels, verdict priority
 * - replay: recorded trace → deterministic verdict
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
	cyclePeriod,
	detectExactRepeat,
	detectOscillation,
	runMonitor,
	type StepRecord,
	type MonitorConfig,
	DEFAULT_CONFIG,
} from "../src/detector.js";
import { canonicalizeArgs, hashObservation } from "../src/canonicalize.js";

// ── Helpers ──────────────────────────────────────────────────────

const STATE_A = "state_a";
const STATE_B = "state_b";
const STATE_C = "state_c";

function makeRecord(
	idx: number,
	action: string,
	args: unknown,
	obs: string,
	state: string,
): StepRecord {
	return {
		step_index: idx,
		action_type: action,
		canonical_args: canonicalizeArgs(args),
		observation_hash: hashObservation(obs),
		outcome_state_hash: state,
	};
}

// ── cycle_period ─────────────────────────────────────────────────

describe("cycle_period", () => {
	const minReps = (p: number) => (p === 1 ? 4 : 2);

	it("detects AAAA (period 1, 4 reps)", () => {
		const states = [STATE_A, STATE_A, STATE_A, STATE_A];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, 1);
	});

	it("detects ABAB (period 2, 2 reps)", () => {
		const states = [STATE_A, STATE_B, STATE_A, STATE_B];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, 2);
	});

	it("detects ABCABC (period 3, 2 reps)", () => {
		const states = [STATE_A, STATE_B, STATE_C, STATE_A, STATE_B, STATE_C];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, 3);
	});

	it("returns null for ABCAB (incomplete cycle)", () => {
		const states = [STATE_A, STATE_B, STATE_C, STATE_A, STATE_B];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, null);
	});

	it("returns null when below min_reps for period 1", () => {
		// Only 3 reps of A, but min_reps(1) = 4
		const states = [STATE_A, STATE_A, STATE_A];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, null);
	});

	it("returns null when below min_reps for period 2", () => {
		// Only 1 rep of AB, but min_reps(2) = 2
		const states = [STATE_A, STATE_B];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, null);
	});

	it("detects ABCDABCD (period 4, 2 reps)", () => {
		const states = [
			STATE_A,
			STATE_B,
			STATE_C,
			"state_d",
			STATE_A,
			STATE_B,
			STATE_C,
			"state_d",
		];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, 4);
	});

	it("prefers smallest period (AAAA is period 1 not period 2)", () => {
		const states = [STATE_A, STATE_A, STATE_A, STATE_A];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, 1);
	});

	it("returns null for random sequence", () => {
		const states = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, null);
	});
});

// ── detectExactRepeat ────────────────────────────────────────────

describe("detectExactRepeat", () => {
	it("fires on repeated action + identical observation", () => {
		const records: StepRecord[] = [];
		for (let i = 0; i < 4; i++) {
			records.push(
				makeRecord(i, "bash", { command: "ls" }, "same output", STATE_A),
			);
		}
		const result = detectExactRepeat(records, 20, 3);
		assert.ok(result !== null);
		assert.equal(result!.count, 4);
		assert.equal(result!.sig[0], "bash");
	});

	it("does NOT fire on repeated action + changing observation", () => {
		const records: StepRecord[] = [];
		for (let i = 0; i < 4; i++) {
			records.push(
				makeRecord(i, "bash", { command: "ls" }, `output ${i}`, STATE_A),
			);
		}
		const result = detectExactRepeat(records, 20, 3);
		assert.equal(result, null);
	});

	it("does NOT fire when count < threshold", () => {
		const records: StepRecord[] = [];
		for (let i = 0; i < 2; i++) {
			records.push(
				makeRecord(i, "bash", { command: "ls" }, "same output", STATE_A),
			);
		}
		const result = detectExactRepeat(records, 20, 3);
		assert.equal(result, null);
	});

	it("respects window — old repeats outside window are ignored", () => {
		const records: StepRecord[] = [];
		// 3 repeats at indices 0-2 (outside window of 5)
		for (let i = 0; i < 3; i++) {
			records.push(
				makeRecord(i, "bash", { command: "ls" }, "same output", STATE_A),
			);
		}
		// 5 different actions at indices 3-7 (inside window)
		for (let i = 3; i < 8; i++) {
			records.push(
				makeRecord(
					i,
					"read",
					{ path: `f${i}.ts` },
					`content ${i}`,
					`state_${i}`,
				),
			);
		}
		const result = detectExactRepeat(records, 5, 3);
		assert.equal(result, null);
	});

	it("returns correct step indices", () => {
		const records: StepRecord[] = [];
		for (let i = 10; i < 14; i++) {
			records.push(
				makeRecord(i, "bash", { command: "ls" }, "same output", STATE_A),
			);
		}
		const result = detectExactRepeat(records, 20, 3);
		assert.ok(result !== null);
		assert.deepStrictEqual(result!.stepIndices, [10, 11, 12, 13]);
	});

	it("does NOT fire when identical calls are interleaved with progress", () => {
		// `git status` recurs with identical output, but real work happens
		// between each — not a stuck loop, must not flag.
		const records: StepRecord[] = [
			makeRecord(0, "bash", { command: "git status" }, "clean", STATE_A),
			makeRecord(1, "edit", { path: "a.ts" }, "edited a", STATE_B),
			makeRecord(2, "bash", { command: "git status" }, "clean", STATE_A),
			makeRecord(3, "edit", { path: "b.ts" }, "edited b", STATE_C),
			makeRecord(4, "bash", { command: "git status" }, "clean", STATE_A),
		];
		const result = detectExactRepeat(records, 20, 3);
		assert.equal(result, null);
	});

	it("fires only on the consecutive run, reporting its length", () => {
		const records: StepRecord[] = [
			makeRecord(0, "bash", { command: "ls" }, "out", STATE_A),
			makeRecord(1, "read", { path: "x.ts" }, "content", STATE_B),
			// Consecutive stuck run of 3:
			makeRecord(2, "bash", { command: "npm test" }, "1 failing", STATE_C),
			makeRecord(3, "bash", { command: "npm test" }, "1 failing", STATE_C),
			makeRecord(4, "bash", { command: "npm test" }, "1 failing", STATE_C),
		];
		const result = detectExactRepeat(records, 20, 3);
		assert.ok(result !== null);
		assert.equal(result!.count, 3);
		assert.deepStrictEqual(result!.stepIndices, [2, 3, 4]);
	});
});

// ── detectOscillation ────────────────────────────────────────────

describe("detectOscillation", () => {
	const minReps = (p: number) => (p === 1 ? 4 : 2);

	it("fires on ABAB outcome-state oscillation with differing observations", () => {
		const records: StepRecord[] = [
			makeRecord(0, "edit", { path: "a.ts" }, "edit result 1", STATE_A),
			makeRecord(1, "bash", { command: "test" }, "test result 1", STATE_B),
			makeRecord(2, "edit", { path: "a.ts" }, "edit result 2", STATE_A),
			makeRecord(3, "bash", { command: "test" }, "test result 2", STATE_B),
		];
		const result = detectOscillation(records, 20, 4, minReps, true);
		assert.ok(result !== null);
		assert.equal(result!.period, 2);
		assert.equal(result!.span, 4);
	});

	it("fires on ABCABC oscillation", () => {
		const records: StepRecord[] = [
			makeRecord(0, "edit", { path: "a.ts" }, "obs 1", STATE_A),
			makeRecord(1, "edit", { path: "b.ts" }, "obs 2", STATE_B),
			makeRecord(2, "bash", { command: "test" }, "obs 3", STATE_C),
			makeRecord(3, "edit", { path: "a.ts" }, "obs 4", STATE_A),
			makeRecord(4, "edit", { path: "b.ts" }, "obs 5", STATE_B),
			makeRecord(5, "bash", { command: "test" }, "obs 6", STATE_C),
		];
		const result = detectOscillation(records, 20, 4, minReps, true);
		assert.ok(result !== null);
		assert.equal(result!.period, 3);
		assert.equal(result!.span, 6);
	});

	it("does NOT fire on non-repeating states", () => {
		const records: StepRecord[] = [];
		for (let i = 0; i < 6; i++) {
			records.push(
				makeRecord(i, "bash", { command: `cmd${i}` }, `obs ${i}`, `state_${i}`),
			);
		}
		const result = detectOscillation(records, 20, 4, minReps, true);
		assert.equal(result, null);
	});

	it("falls back to action-pattern when outcome_state_hash unavailable", () => {
		const records: StepRecord[] = [
			makeRecord(0, "edit", { path: "a.ts" }, "obs 1", "x1"),
			makeRecord(1, "bash", { command: "test" }, "obs 2", "x2"),
			makeRecord(2, "edit", { path: "a.ts" }, "obs 3", "x3"),
			makeRecord(3, "bash", { command: "test" }, "obs 4", "x4"),
		];
		// With outcome_state: all different states → no oscillation
		const resultWithState = detectOscillation(records, 20, 4, minReps, true);
		assert.equal(resultWithState, null);

		// Without outcome_state: action-pattern ABAB → oscillation
		const resultWithoutState = detectOscillation(
			records,
			20,
			4,
			minReps,
			false,
		);
		assert.ok(resultWithoutState !== null);
		assert.equal(resultWithoutState!.period, 2);
	});
});

// ── canonicalizeArgs ─────────────────────────────────────────────

describe("canonicalizeArgs", () => {
	it("two args differing only in volatile fields hash equal", () => {
		const a1 = { file: "foo.ts", timestamp: 123, requestId: "abc", pid: 999 };
		const a2 = { file: "foo.ts", timestamp: 456, requestId: "xyz", pid: 1000 };
		assert.equal(canonicalizeArgs(a1), canonicalizeArgs(a2));
	});

	it("two semantically different args hash differently", () => {
		const a1 = { file: "foo.ts" };
		const a2 = { file: "bar.ts" };
		assert.notEqual(canonicalizeArgs(a1), canonicalizeArgs(a2));
	});

	it("key order does not matter", () => {
		const a1 = { z: 1, a: 2, m: 3 };
		const a2 = { a: 2, m: 3, z: 1 };
		assert.equal(canonicalizeArgs(a1), canonicalizeArgs(a2));
	});

	it("normalizes whitespace in string values", () => {
		const a1 = { command: "ls   -la  /tmp" };
		const a2 = { command: "ls -la /tmp" };
		assert.equal(canonicalizeArgs(a1), canonicalizeArgs(a2));
	});

	it("handles nested objects", () => {
		const a1 = { outer: { timestamp: 1, data: "x" }, z: 1 };
		const a2 = { z: 1, outer: { data: "x", timestamp: 999 } };
		assert.equal(canonicalizeArgs(a1), canonicalizeArgs(a2));
	});

	it("handles arrays", () => {
		const a1 = { files: ["b.ts", "a.ts"] };
		const a2 = { files: ["b.ts", "a.ts"] };
		assert.equal(canonicalizeArgs(a1), canonicalizeArgs(a2));
	});
});

// ── runMonitor ───────────────────────────────────────────────────

describe("runMonitor", () => {
	const config: MonitorConfig = { ...DEFAULT_CONFIG };

	it("returns OK during warmup", () => {
		const records: StepRecord[] = [];
		for (let i = 0; i < 3; i++) {
			records.push(makeRecord(i, "bash", { command: "ls" }, "same", STATE_A));
		}
		const { verdict, severity } = runMonitor(records, config);
		assert.equal(verdict.type, "OK");
		assert.equal(severity, "none");
	});

	it("returns EXACT_REPEAT at warn level", () => {
		const records: StepRecord[] = [];
		// 2 warmup records to pass warmup threshold
		records.push(makeRecord(0, "read", { path: "a.ts" }, "content A", "s0"));
		records.push(makeRecord(1, "read", { path: "b.ts" }, "content B", "s1"));
		// 4 repeats: >= tRepeatWarn(3) but < tRepeatHard(5)
		for (let i = 2; i < 6; i++) {
			records.push(
				makeRecord(i, "bash", { command: "ls" }, "same output", STATE_A),
			);
		}
		const { verdict, severity } = runMonitor(records, config);
		assert.equal(verdict.type, "EXACT_REPEAT");
		assert.equal(severity, "warn");
		if (verdict.type === "EXACT_REPEAT") {
			assert.equal(verdict.count, 4);
		}
	});

	it("returns EXACT_REPEAT at hard level", () => {
		const records: StepRecord[] = [];
		for (let i = 0; i < 10; i++) {
			records.push(
				makeRecord(i, "bash", { command: "ls" }, "same output", STATE_A),
			);
		}
		const { verdict, severity } = runMonitor(records, config);
		assert.equal(verdict.type, "EXACT_REPEAT");
		assert.equal(severity, "hard");
	});

	it("returns OK when observations change (progressive retry)", () => {
		const records: StepRecord[] = [];
		// Same action, changing observations, AND changing outcome states = progress
		for (let i = 0; i < 10; i++) {
			records.push(
				makeRecord(
					i,
					"bash",
					{ command: "test" },
					`test output ${i}`,
					`state_${i}`,
				),
			);
		}
		const { verdict, severity } = runMonitor(records, config);
		assert.equal(verdict.type, "OK");
		assert.equal(severity, "none");
	});

	it("returns OSCILLATION for ABAB pattern", () => {
		const records: StepRecord[] = [];
		for (let rep = 0; rep < 3; rep++) {
			records.push(
				makeRecord(
					records.length,
					"edit",
					{ path: "a.ts" },
					`edit obs ${rep}`,
					STATE_A,
				),
			);
			records.push(
				makeRecord(
					records.length,
					"bash",
					{ command: "test" },
					`test obs ${rep}`,
					STATE_B,
				),
			);
		}
		const { verdict, severity } = runMonitor(records, config);
		assert.equal(verdict.type, "OSCILLATION");
		assert.equal(severity, "warn");
		if (verdict.type === "OSCILLATION") {
			assert.equal(verdict.period, 2);
		}
	});

	it("EXACT_REPEAT takes priority over OSCILLATION", () => {
		// Same action, same args, same observation — this is both exact-repeat
		// and period-1 oscillation. Exact-repeat should win.
		const records: StepRecord[] = [];
		for (let i = 0; i < 6; i++) {
			records.push(
				makeRecord(i, "bash", { command: "ls" }, "same output", STATE_A),
			);
		}
		const { verdict } = runMonitor(records, config);
		assert.equal(verdict.type, "EXACT_REPEAT");
	});
});

// ── Replay test ──────────────────────────────────────────────────

describe("replay", () => {
	it("deterministic verdict from recorded trace", () => {
		// Simulate a recorded trace: agent stuck in edit→test→revert loop
		const trace: StepRecord[] = [
			makeRecord(0, "read", { path: "src/main.ts" }, "initial content", "s0"),
			makeRecord(1, "edit", { path: "src/main.ts" }, "edit applied", "s1"),
			makeRecord(2, "bash", { command: "npm test" }, "2 failing", "s2"),
			makeRecord(3, "edit", { path: "src/main.ts" }, "revert applied", "s0"),
			makeRecord(4, "bash", { command: "npm test" }, "0 failing", "s3"),
			// Loop begins:
			makeRecord(5, "edit", { path: "src/main.ts" }, "edit v2", "s1"),
			makeRecord(6, "bash", { command: "npm test" }, "2 failing v2", "s2"),
			makeRecord(7, "edit", { path: "src/main.ts" }, "revert v2", "s0"),
			makeRecord(8, "bash", { command: "npm test" }, "0 failing v2", "s3"),
			makeRecord(9, "edit", { path: "src/main.ts" }, "edit v3", "s1"),
			makeRecord(10, "bash", { command: "npm test" }, "2 failing v3", "s2"),
			makeRecord(11, "edit", { path: "src/main.ts" }, "revert v3", "s0"),
			makeRecord(12, "bash", { command: "npm test" }, "0 failing v3", "s3"),
		];

		const config: MonitorConfig = { ...DEFAULT_CONFIG };
		const { verdict } = runMonitor(trace, config);

		// The outcome states cycle: s1, s2, s0, s3, s1, s2, s0, s3, s1, s2, s0, s3
		// That's period 4 with 3 reps → should fire as OSCILLATION
		assert.equal(verdict.type, "OSCILLATION");
		if (verdict.type === "OSCILLATION") {
			assert.equal(verdict.period, 4);
			assert.ok(verdict.span >= 8);
		}
	});
});

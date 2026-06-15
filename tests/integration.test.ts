/**
 * Integration and edge-case tests for the cycle detection extension.
 *
 * Focus: extractObservationText, extractOutcomeState, canonicalizeArgs edge cases,
 * runMonitor boundary conditions, and false-positive scenarios.
 *
 * Run with: npx tsx tests/integration.test.ts
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
import {
	canonicalizeArgs,
	hashObservation,
	hashOutcomeState,
} from "../src/canonicalize.js";

// ── Helpers ──────────────────────────────────────────────────────

const STATE_A = "state_a";
const STATE_B = "state_b";

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
		outcome_state_hash: hashOutcomeState(state),
	};
}

// ── extractObservationText ───────────────────────────────────────
// We cannot import the function directly from index.ts (it's not exported),
// so we inline the logic here to test it independently.

function extractObservationText(content: unknown[]): string {
	if (!Array.isArray(content)) return "";
	return content
		.map((c) => {
			if (typeof c === "string") return c;
			if (typeof c === "object" && c !== null && "text" in c)
				return String((c as { text: unknown }).text);
			return "";
		})
		.join("\n");
}

describe("extractObservationText", () => {
	it("extracts from plain string array", () => {
		const result = extractObservationText(["line1", "line2"]);
		assert.equal(result, "line1\nline2");
	});

	it("extracts from object-with-text array", () => {
		const result = extractObservationText([
			{ text: "hello" },
			{ text: "world" },
		]);
		assert.equal(result, "hello\nworld");
	});

	it("handles mixed string and object content", () => {
		const result = extractObservationText(["plain", { text: "object" }]);
		assert.equal(result, "plain\nobject");
	});

	it("returns empty string for empty array", () => {
		const result = extractObservationText([]);
		assert.equal(result, "");
	});

	it("handles objects without text key", () => {
		const result = extractObservationText([{ data: "no text here" }]);
		assert.equal(result, "");
	});

	it("handles null in array", () => {
		const result = extractObservationText(["valid", null, "also valid"]);
		assert.equal(result, "valid\n\nalso valid");
	});

	it("handles numbers in array", () => {
		const result = extractObservationText([42, "text"]);
		assert.equal(result, "\ntext");
	});

	it("handles boolean in array", () => {
		const result = extractObservationText([true, false]);
		assert.equal(result, "\n");
	});
});

// ── extractOutcomeState ──────────────────────────────────────────
// Inlined from index.ts for testing.

function extractOutcomeState(toolName: string, input: unknown): string {
	if (toolName === "write" || toolName === "edit") {
		const path = (input as { path?: string })?.path;
		return path ? `file:${path}` : "";
	}
	if (toolName === "bash") {
		const cmd = (input as { command?: string })?.command;
		return cmd ? `cmd:${cmd}` : "";
	}
	return `${toolName}:${canonicalizeArgs(input)}`;
}

describe("extractOutcomeState", () => {
	it("returns file:path for write tool", () => {
		const result = extractOutcomeState("write", {
			path: "/foo/bar.ts",
			content: "x",
		});
		assert.equal(result, "file:/foo/bar.ts");
	});

	it("returns file:path for edit tool", () => {
		const result = extractOutcomeState("edit", {
			path: "/foo/bar.ts",
			edits: [],
		});
		assert.equal(result, "file:/foo/bar.ts");
	});

	it("returns cmd:command for bash tool", () => {
		const result = extractOutcomeState("bash", { command: "ls -la" });
		assert.equal(result, "cmd:ls -la");
	});

	it("returns empty string for write without path", () => {
		const result = extractOutcomeState("write", { content: "no path" });
		assert.equal(result, "");
	});

	it("returns empty string for bash without command", () => {
		const result = extractOutcomeState("bash", { timeout: 30 });
		assert.equal(result, "");
	});

	it("returns tool:hash for unknown tool", () => {
		const result = extractOutcomeState("read", { path: "foo.ts" });
		assert.ok(result.startsWith("read:"));
	});

	// ── FALSE POSITIVE RISK ──
	it("CRITICAL: same file edits produce identical state regardless of content", () => {
		const s1 = extractOutcomeState("edit", {
			path: "foo.ts",
			edits: [{ oldText: "a", newText: "b" }],
		});
		const s2 = extractOutcomeState("edit", {
			path: "foo.ts",
			edits: [{ oldText: "b", newText: "c" }],
		});
		// These are DIFFERENT edits but produce the SAME state hash!
		// This means editing the same file back and forth will trigger false oscillation.
		assert.equal(s1, s2);
		assert.equal(s1, "file:foo.ts");
	});

	it("CRITICAL: same bash command produces identical state regardless of output", () => {
		const s1 = extractOutcomeState("bash", { command: "npm test" });
		const s2 = extractOutcomeState("bash", { command: "npm test" });
		// Same command = same state, even if test results differ
		assert.equal(s1, s2);
	});
});

// ── False-positive oscillation scenario ──────────────────────────

describe("false-positive oscillation from extractOutcomeState", () => {
	it("edit→bash→edit→bash on same file triggers false oscillation", () => {
		// Simulate: agent edits foo.ts, runs tests, edits foo.ts again, runs tests again
		// Each edit is DIFFERENT content, each test has DIFFERENT output
		// But extractOutcomeState collapses them to the same state hashes
		const records: StepRecord[] = [
			{
				step_index: 0,
				action_type: "edit",
				canonical_args: canonicalizeArgs({
					path: "foo.ts",
					edits: [{ oldText: "a", newText: "b" }],
				}),
				observation_hash: hashObservation("edit applied v1"),
				outcome_state_hash: hashOutcomeState(
					extractOutcomeState("edit", { path: "foo.ts" }),
				),
			},
			{
				step_index: 1,
				action_type: "bash",
				canonical_args: canonicalizeArgs({ command: "npm test" }),
				observation_hash: hashObservation("2 tests failing"),
				outcome_state_hash: hashOutcomeState(
					extractOutcomeState("bash", { command: "npm test" }),
				),
			},
			{
				step_index: 2,
				action_type: "edit",
				canonical_args: canonicalizeArgs({
					path: "foo.ts",
					edits: [{ oldText: "b", newText: "c" }],
				}),
				observation_hash: hashObservation("edit applied v2"),
				outcome_state_hash: hashOutcomeState(
					extractOutcomeState("edit", { path: "foo.ts" }),
				),
			},
			{
				step_index: 3,
				action_type: "bash",
				canonical_args: canonicalizeArgs({ command: "npm test" }),
				observation_hash: hashObservation("1 test failing"),
				outcome_state_hash: hashOutcomeState(
					extractOutcomeState("bash", { command: "npm test" }),
				),
			},
		];

		const config: MonitorConfig = { ...DEFAULT_CONFIG, warmupSteps: 0 };
		const { verdict } = runMonitor(records, config);

		// This SHOULD be OK (progressive work), but will likely fire as OSCILLATION
		// because outcome_state_hash collapses to ABAB pattern
		if (verdict.type === "OSCILLATION") {
			console.log("FALSE POSITIVE: Progressive work flagged as oscillation");
			console.log(`  Period: ${verdict.period}, Span: ${verdict.span}`);
		}
		// We assert the current (buggy) behavior to document it
		assert.equal(
			verdict.type,
			"OSCILLATION",
			"Expected false positive — this documents the bug",
		);
	});

	it("repeated edits to same file with different content flagged as oscillation", () => {
		// Agent makes 4 different edits to the same file
		const records: StepRecord[] = [];
		for (let i = 0; i < 4; i++) {
			records.push({
				step_index: i,
				action_type: "edit",
				canonical_args: canonicalizeArgs({
					path: "foo.ts",
					edits: [{ oldText: `v${i}`, newText: `v${i + 1}` }],
				}),
				observation_hash: hashObservation(`edit applied v${i}`),
				outcome_state_hash: hashOutcomeState(
					extractOutcomeState("edit", { path: "foo.ts" }),
				),
			});
		}

		const config: MonitorConfig = { ...DEFAULT_CONFIG, warmupSteps: 0 };
		const { verdict } = runMonitor(records, config);

		// All edits have same outcome_state_hash (file:foo.ts)
		// This is period-1 oscillation — FALSE POSITIVE
		if (verdict.type === "OSCILLATION") {
			assert.equal(verdict.period, 1);
		}
		assert.equal(
			verdict.type,
			"OSCILLATION",
			"Expected false positive for repeated edits to same file",
		);
	});
});

// ── canonicalizeArgs edge cases ──────────────────────────────────

describe("canonicalizeArgs edge cases", () => {
	it("handles undefined value", () => {
		const result = canonicalizeArgs(undefined);
		assert.ok(typeof result === "string" && result.length === 16);
	});

	it("handles null value", () => {
		const result = canonicalizeArgs(null);
		assert.ok(typeof result === "string" && result.length === 16);
	});

	it("handles empty object", () => {
		const result = canonicalizeArgs({});
		assert.ok(typeof result === "string" && result.length === 16);
	});

	it("handles empty array", () => {
		const result = canonicalizeArgs([]);
		assert.ok(typeof result === "string" && result.length === 16);
	});

	it("handles deeply nested volatile keys", () => {
		const a1 = { config: { inner: { timestamp: 1, data: "x" } } };
		const a2 = { config: { inner: { timestamp: 999, data: "x" } } };
		assert.equal(canonicalizeArgs(a1), canonicalizeArgs(a2));
	});

	it("handles volatile keys in arrays of objects", () => {
		const a1 = {
			items: [
				{ id: 1, timestamp: 1 },
				{ id: 2, timestamp: 2 },
			],
		};
		const a2 = {
			items: [
				{ id: 1, timestamp: 999 },
				{ id: 2, timestamp: 999 },
			],
		};
		assert.equal(canonicalizeArgs(a1), canonicalizeArgs(a2));
	});

	it("handles string with only whitespace", () => {
		const a1 = { text: "   " };
		const a2 = { text: "" };
		assert.equal(canonicalizeArgs(a1), canonicalizeArgs(a2));
	});

	it("handles bigint values", () => {
		const result = canonicalizeArgs({ big: BigInt(9007199254740991) });
		assert.ok(typeof result === "string" && result.length === 16);
	});

	it("handles function values (edge case)", () => {
		const fn = () => {};
		const result = canonicalizeArgs({ fn });
		// Functions are objects, will be processed as objects
		assert.ok(typeof result === "string" && result.length === 16);
	});

	it("handles circular reference (should not infinite loop)", () => {
		const obj: any = { a: 1 };
		obj.self = obj;
		// This will likely throw or produce unexpected results
		// We just verify it doesn't hang
		assert.doesNotThrow(() => canonicalizeArgs(obj));
	});

	it("strips all known volatile patterns", () => {
		const volatileKeys = [
			"timestamp",
			"date",
			"uuid",
			"requestId",
			"request_id",
			"reqId",
			"req_id",
			"traceId",
			"trace_id",
			"correlationId",
			"correlation_id",
			"tempPath",
			"temp_path",
			"tmpPath",
			"tmp_path",
			"pid",
			"port",
			"portNumber",
			"port_number",
			"random",
			"nonce",
			"sessionId",
			"session_id",
			"token",
			"expires",
			"attempt",
			"retry",
			"iteration",
		];
		const base = { file: "test.ts" };
		const withVolatile = { ...base };
		for (const key of volatileKeys) {
			(withVolatile as any)[key] = "volatile-value";
		}
		assert.equal(canonicalizeArgs(base), canonicalizeArgs(withVolatile));
	});
});

// ── runMonitor boundary conditions ───────────────────────────────

describe("runMonitor boundary conditions", () => {
	it("handles empty records array", () => {
		const config: MonitorConfig = { ...DEFAULT_CONFIG };
		const { verdict, severity } = runMonitor([], config);
		assert.equal(verdict.type, "OK");
		assert.equal(severity, "none");
	});

	it("handles warmupSteps = 0", () => {
		const config: MonitorConfig = { ...DEFAULT_CONFIG, warmupSteps: 0 };
		const records: StepRecord[] = [];
		for (let i = 0; i < 5; i++) {
			records.push(makeRecord(i, "bash", { command: "ls" }, "same", STATE_A));
		}
		const { verdict } = runMonitor(records, config);
		// Should fire immediately without warmup
		assert.equal(verdict.type, "EXACT_REPEAT");
	});

	it("handles pMax = 0 (no oscillation detection)", () => {
		const config: MonitorConfig = {
			...DEFAULT_CONFIG,
			pMax: 0,
			warmupSteps: 0,
		};
		const records: StepRecord[] = [];
		for (let rep = 0; rep < 3; rep++) {
			records.push(
				makeRecord(
					records.length,
					"edit",
					{ path: "a.ts" },
					`obs ${rep}`,
					STATE_A,
				),
			);
			records.push(
				makeRecord(
					records.length,
					"bash",
					{ command: "test" },
					`obs2 ${rep}`,
					STATE_B,
				),
			);
		}
		const { verdict } = runMonitor(records, config);
		// No oscillation detection, but also no exact repeat (different observations)
		assert.equal(verdict.type, "OK");
	});

	it("handles window = 0 (no records examined)", () => {
		const config: MonitorConfig = {
			...DEFAULT_CONFIG,
			window: 0,
			warmupSteps: 0,
		};
		const records: StepRecord[] = [];
		for (let i = 0; i < 10; i++) {
			records.push(makeRecord(i, "bash", { command: "ls" }, "same", STATE_A));
		}
		const { verdict } = runMonitor(records, config);
		// Window of 0 means no records in the window
		assert.equal(verdict.type, "OK");
	});

	it("handles hasOutcomeState = false with oscillation", () => {
		const config: MonitorConfig = {
			...DEFAULT_CONFIG,
			hasOutcomeState: false,
			warmupSteps: 0,
		};
		const records: StepRecord[] = [];
		for (let rep = 0; rep < 2; rep++) {
			records.push(
				makeRecord(
					records.length,
					"edit",
					{ path: "a.ts" },
					`obs ${rep}`,
					"unique_state_" + rep,
				),
			);
			records.push(
				makeRecord(
					records.length,
					"bash",
					{ command: "test" },
					`obs2 ${rep}`,
					"unique_state_" + (rep + 100),
				),
			);
		}
		const { verdict } = runMonitor(records, config);
		// Without outcome state, falls back to action-pattern → ABAB oscillation
		assert.equal(verdict.type, "OSCILLATION");
	});

	it("exact boundary: exactly tRepeatWarn repeats fires warn", () => {
		const config: MonitorConfig = { ...DEFAULT_CONFIG, warmupSteps: 0 };
		const records: StepRecord[] = [];
		for (let i = 0; i < config.tRepeatWarn; i++) {
			records.push(makeRecord(i, "bash", { command: "ls" }, "same", STATE_A));
		}
		const { verdict, severity } = runMonitor(records, config);
		assert.equal(verdict.type, "EXACT_REPEAT");
		assert.equal(severity, "warn");
	});

	it("exact boundary: one below tRepeatWarn does not fire", () => {
		const config: MonitorConfig = { ...DEFAULT_CONFIG, warmupSteps: 0 };
		const records: StepRecord[] = [];
		for (let i = 0; i < config.tRepeatWarn - 1; i++) {
			records.push(makeRecord(i, "bash", { command: "ls" }, "same", STATE_A));
		}
		const { verdict } = runMonitor(records, config);
		assert.equal(verdict.type, "OK");
	});

	it("exact boundary: exactly tRepeatHard repeats fires hard", () => {
		const config: MonitorConfig = { ...DEFAULT_CONFIG, warmupSteps: 0 };
		const records: StepRecord[] = [];
		for (let i = 0; i < config.tRepeatHard; i++) {
			records.push(makeRecord(i, "bash", { command: "ls" }, "same", STATE_A));
		}
		const { verdict, severity } = runMonitor(records, config);
		assert.equal(verdict.type, "EXACT_REPEAT");
		assert.equal(severity, "hard");
	});

	it("exact boundary: one below tRepeatHard fires warn not hard", () => {
		const config: MonitorConfig = { ...DEFAULT_CONFIG, warmupSteps: 0 };
		const records: StepRecord[] = [];
		for (let i = 0; i < config.tRepeatHard - 1; i++) {
			records.push(makeRecord(i, "bash", { command: "ls" }, "same", STATE_A));
		}
		const { verdict, severity } = runMonitor(records, config);
		assert.equal(verdict.type, "EXACT_REPEAT");
		assert.equal(severity, "warn");
	});

	it("multiple different actions in window, only one repeats", () => {
		const config: MonitorConfig = { ...DEFAULT_CONFIG, warmupSteps: 0 };
		const records: StepRecord[] = [];
		// Add some noise
		records.push(makeRecord(0, "read", { path: "a.ts" }, "content A", "s0"));
		records.push(makeRecord(1, "read", { path: "b.ts" }, "content B", "s1"));
		// Repeated action
		for (let i = 2; i < 6; i++) {
			records.push(makeRecord(i, "bash", { command: "ls" }, "same", STATE_A));
		}
		// More noise
		records.push(makeRecord(6, "read", { path: "c.ts" }, "content C", "s2"));
		const { verdict } = runMonitor(records, config);
		assert.equal(verdict.type, "EXACT_REPEAT");
	});

	it("records trimmed correctly when exceeding maxKeep", () => {
		// This tests the logic in index.ts indirectly
		// maxKeep = window + warmupSteps + 10 = 20 + 5 + 10 = 35
		const config: MonitorConfig = { ...DEFAULT_CONFIG };
		const maxKeep = config.window + config.warmupSteps + 10;
		const records: StepRecord[] = [];
		for (let i = 0; i < maxKeep + 5; i++) {
			records.push(
				makeRecord(i, "bash", { command: `cmd${i}` }, `obs ${i}`, `state_${i}`),
			);
		}
		// Simulate trimming
		const trimmed = records.slice(-maxKeep);
		assert.equal(trimmed.length, maxKeep);
		assert.equal(trimmed[0].step_index, 5);
	});
});

// ── cyclePeriod edge cases ───────────────────────────────────────

describe("cyclePeriod edge cases", () => {
	const minReps = (p: number) => (p === 1 ? 4 : 2);

	it("returns null for empty states array", () => {
		const p = cyclePeriod([], 4, minReps);
		assert.equal(p, null);
	});

	it("returns null for single state", () => {
		const p = cyclePeriod([STATE_A], 4, minReps);
		assert.equal(p, null);
	});

	it("handles pMax larger than array length", () => {
		const states = [STATE_A, STATE_B];
		const p = cyclePeriod(states, 10, minReps);
		assert.equal(p, null);
	});

	it("detects period 1 with exactly minReps", () => {
		const states = [STATE_A, STATE_A, STATE_A, STATE_A];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, 1);
	});

	it("detects period 1 with more than minReps", () => {
		const states = [STATE_A, STATE_A, STATE_A, STATE_A, STATE_A, STATE_A];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, 1);
	});

	it("does not detect period 1 with one below minReps", () => {
		const states = [STATE_A, STATE_A, STATE_A];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, null);
	});

	it("handles alternating states with odd count (ABABA)", () => {
		// ABABA: last 4 states = B,A,B,A = 2 reps of period-2 block BA
		// The algorithm correctly detects period 2 from the trailing complete reps.
		const states = [STATE_A, STATE_B, STATE_A, STATE_B, STATE_A];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, 2);
	});

	it("handles alternating states with even count (ABABAB)", () => {
		// ABABAB = 3 reps of AB → period 2
		const states = [STATE_A, STATE_B, STATE_A, STATE_B, STATE_A, STATE_B];
		const p = cyclePeriod(states, 4, minReps);
		assert.equal(p, 2);
	});
});

// ── detectExactRepeat edge cases ─────────────────────────────────

describe("detectExactRepeat edge cases", () => {
	it("handles empty records", () => {
		const result = detectExactRepeat([], 20, 3);
		assert.equal(result, null);
	});

	it("handles single record", () => {
		const records = [
			makeRecord(0, "bash", { command: "ls" }, "output", STATE_A),
		];
		const result = detectExactRepeat(records, 20, 3);
		assert.equal(result, null);
	});

	it("different actions with same observation do not trigger", () => {
		const records: StepRecord[] = [
			makeRecord(0, "bash", { command: "ls" }, "same output", STATE_A),
			makeRecord(1, "read", { path: "a.ts" }, "same output", STATE_A),
			makeRecord(2, "write", { path: "b.ts" }, "same output", STATE_A),
		];
		const result = detectExactRepeat(records, 20, 3);
		assert.equal(result, null);
	});

	it("same action different args do not trigger", () => {
		const records: StepRecord[] = [
			makeRecord(0, "bash", { command: "ls /a" }, "output", STATE_A),
			makeRecord(1, "bash", { command: "ls /b" }, "output", STATE_A),
			makeRecord(2, "bash", { command: "ls /c" }, "output", STATE_A),
		];
		const result = detectExactRepeat(records, 20, 3);
		assert.equal(result, null);
	});

	it("window smaller than records only checks recent", () => {
		const records: StepRecord[] = [];
		// Old records (outside window)
		for (let i = 0; i < 10; i++) {
			records.push(makeRecord(i, "bash", { command: "ls" }, "same", STATE_A));
		}
		// New records (inside window of 5)
		for (let i = 10; i < 15; i++) {
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
		// Only last 5 records checked — all different
		assert.equal(result, null);
	});
});

// ── detectOscillation edge cases ─────────────────────────────────

describe("detectOscillation edge cases", () => {
	const minReps = (p: number) => (p === 1 ? 4 : 2);

	it("handles empty records", () => {
		const result = detectOscillation([], 20, 4, minReps, true);
		assert.equal(result, null);
	});

	it("handles single record", () => {
		const records = [
			makeRecord(0, "bash", { command: "ls" }, "output", STATE_A),
		];
		const result = detectOscillation(records, 20, 4, minReps, true);
		assert.equal(result, null);
	});

	it("window limits oscillation detection", () => {
		const records: StepRecord[] = [];
		// Old oscillation (outside window)
		for (let rep = 0; rep < 3; rep++) {
			records.push(
				makeRecord(
					records.length,
					"edit",
					{ path: "a.ts" },
					`obs ${rep}`,
					STATE_A,
				),
			);
			records.push(
				makeRecord(
					records.length,
					"bash",
					{ command: "test" },
					`obs2 ${rep}`,
					STATE_B,
				),
			);
		}
		// New non-oscillating records (inside window of 5)
		for (let i = 6; i < 11; i++) {
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
		const result = detectOscillation(records, 5, 4, minReps, true);
		// Only last 5 records — all different states
		assert.equal(result, null);
	});

	it("period-1 oscillation with exactly minRepsP1", () => {
		const records: StepRecord[] = [];
		for (let i = 0; i < 4; i++) {
			records.push(
				makeRecord(i, "bash", { command: "ls" }, `obs ${i}`, STATE_A),
			);
		}
		const result = detectOscillation(records, 20, 4, minReps, true);
		assert.ok(result !== null);
		assert.equal(result!.period, 1);
	});

	it("period-1 oscillation with one below minRepsP1", () => {
		const records: StepRecord[] = [];
		for (let i = 0; i < 3; i++) {
			records.push(
				makeRecord(i, "bash", { command: "ls" }, `obs ${i}`, STATE_A),
			);
		}
		const result = detectOscillation(records, 20, 4, minReps, true);
		assert.equal(result, null);
	});
});

// ── Severity transitions ─────────────────────────────────────────

describe("severity transitions", () => {
	it("oscillation severity: warn at minRepsP2, hard at 2x minRepsP2", () => {
		const config: MonitorConfig = { ...DEFAULT_CONFIG, warmupSteps: 0 };
		// Period-2 oscillation with exactly minRepsP2 (2) reps → warn
		const recordsWarn: StepRecord[] = [];
		for (let rep = 0; rep < 2; rep++) {
			recordsWarn.push(
				makeRecord(
					recordsWarn.length,
					"edit",
					{ path: "a.ts" },
					`obs ${rep}`,
					STATE_A,
				),
			);
			recordsWarn.push(
				makeRecord(
					recordsWarn.length,
					"bash",
					{ command: "test" },
					`obs2 ${rep}`,
					STATE_B,
				),
			);
		}
		const { severity: warnSeverity } = runMonitor(recordsWarn, config);
		assert.equal(warnSeverity, "warn");

		// Period-2 oscillation with 2x minRepsP2 (4) reps → hard
		const recordsHard: StepRecord[] = [];
		for (let rep = 0; rep < 4; rep++) {
			recordsHard.push(
				makeRecord(
					recordsHard.length,
					"edit",
					{ path: "a.ts" },
					`obs ${rep}`,
					STATE_A,
				),
			);
			recordsHard.push(
				makeRecord(
					recordsHard.length,
					"bash",
					{ command: "test" },
					`obs2 ${rep}`,
					STATE_B,
				),
			);
		}
		const { severity: hardSeverity } = runMonitor(recordsHard, config);
		assert.equal(hardSeverity, "hard");
	});

	it("oscillation severity: warn at minRepsP1, hard at 2x minRepsP1", () => {
		const config: MonitorConfig = { ...DEFAULT_CONFIG, warmupSteps: 0 };
		// Period-1 oscillation with exactly minRepsP1 (4) reps → warn
		const recordsWarn: StepRecord[] = [];
		for (let i = 0; i < 4; i++) {
			recordsWarn.push(
				makeRecord(i, "bash", { command: "ls" }, `obs ${i}`, STATE_A),
			);
		}
		const { severity: warnSeverity } = runMonitor(recordsWarn, config);
		assert.equal(warnSeverity, "warn");

		// Period-1 oscillation with 2x minRepsP1 (8) reps → hard
		const recordsHard: StepRecord[] = [];
		for (let i = 0; i < 8; i++) {
			recordsHard.push(
				makeRecord(i, "bash", { command: "ls" }, `obs ${i}`, STATE_A),
			);
		}
		const { severity: hardSeverity } = runMonitor(recordsHard, config);
		assert.equal(hardSeverity, "hard");
	});
});

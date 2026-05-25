/**
 * Cycle detection engine — pure functions over the trajectory log.
 *
 * Two detectors, one primitive: "has the trajectory revisited a state
 * within a recent window?" Exact-repeat is the period-1 special case.
 *
 * Both are stateless — verdict is a pure function of records passed in.
 * Replayable against recorded traces during tuning.
 */

// ── Types ──────────────────────────────────────────────────────

/** Per-step record produced by the harness instrumentation. */
export interface StepRecord {
	step_index: number;
	action_type: string;
	canonical_args: string;
	observation_hash: string;
	outcome_state_hash: string;
}

/** Verdict returned by the monitor. */
export type Verdict =
	| { type: "OK" }
	| {
			type: "EXACT_REPEAT";
			sig: [string, string]; // [action_type, canonical_args]
			count: number;
			stepIndices: number[];
	  }
	| {
			type: "OSCILLATION";
			period: number;
			span: number;
			stepIndices: number[];
	  };

/** Configuration for the monitor. */
export interface MonitorConfig {
	/** Number of recent records examined. */
	window: number;
	/** Exact-repeat count to flag at warn level. */
	tRepeatWarn: number;
	/** Exact-repeat count to flag at hard-stop level. */
	tRepeatHard: number;
	/** Maximum oscillation period to check. */
	pMax: number;
	/** Minimum repetitions for period-1 oscillation (stricter). */
	minRepsP1: number;
	/** Minimum repetitions for period >= 2 oscillation. */
	minRepsP2: number;
	/** No flags during initial exploration. */
	warmupSteps: number;
	/** Whether outcome_state_hash is available. If false, oscillation degrades to action-pattern cycles. */
	hasOutcomeState: boolean;
}

/** Default configuration — tune in shadow mode. */
export const DEFAULT_CONFIG: MonitorConfig = {
	window: 20,
	tRepeatWarn: 3,
	tRepeatHard: 5,
	pMax: 4,
	minRepsP1: 4,
	minRepsP2: 2,
	warmupSteps: 5,
	hasOutcomeState: true,
};

/** Severity of the verdict. */
export type Severity = "warn" | "hard" | "none";

// ── Detector 1: Exact Repeat ──────────────────────────────────
/**
 * Detects when the same action with the same args produces the same
 * observation repeatedly. The observation guard is essential: same action
 * + changing output is progress or transient retry — must not flag.
 */
export function detectExactRepeat(
	records: StepRecord[],
	window: number,
	tRepeat: number,
): { sig: [string, string]; count: number; stepIndices: number[] } | null {
	const recent = records.slice(-window);
	const bySig = new Map<string, StepRecord[]>();

	for (const r of recent) {
		const key = `${r.action_type}::${r.canonical_args}`;
		const group = bySig.get(key) ?? [];
		group.push(r);
		bySig.set(key, group);
	}

	for (const [key, group] of bySig) {
		if (group.length < tRepeat) continue;
		// Observation guard: all observation hashes must be identical.
		const firstHash = group[0].observation_hash;
		const allSame = group.every((r) => r.observation_hash === firstHash);
		if (allSame) {
			const [action_type, canonical_args] = key.split("::") as [string, string];
			return {
				sig: [action_type, canonical_args],
				count: group.length,
				stepIndices: group.map((r) => r.step_index),
			};
		}
	}

	return null;
}

// ── Detector 2: Oscillation ───────────────────────────────────
/**
 * Detects a repeating suffix in the state sequence.
 * For each candidate period p, checks whether the last min_reps(p) * p
 * states form min_reps(p) identical consecutive blocks of length p.
 *
 * Catches ABAB, ABCABC, etc.
 */
export function cyclePeriod(
	states: string[],
	pMax: number,
	minRepsFn: (p: number) => number,
): number | null {
	const n = states.length;
	for (let p = 1; p <= pMax; p++) {
		const reps = minRepsFn(p);
		if (n < p * reps) continue;
		const block = states.slice(-p);
		let matched = true;
		for (let i = 0; i < reps; i++) {
			const start = n - (i + 1) * p;
			const end = n - i * p;
			const segment = states.slice(start, end);
			if (segment.length !== block.length) {
				matched = false;
				break;
			}
			for (let j = 0; j < block.length; j++) {
				if (segment[j] !== block[j]) {
					matched = false;
					break;
				}
			}
			if (!matched) break;
		}
		if (matched) return p;
	}
	return null;
}

export function detectOscillation(
	records: StepRecord[],
	window: number,
	pMax: number,
	minRepsFn: (p: number) => number,
	hasOutcomeState: boolean,
): { period: number; span: number; stepIndices: number[] } | null {
	const recent = records.slice(-window);
	// Use outcome_state_hash if available; fall back to action-pattern.
	const states = hasOutcomeState
		? recent.map((r) => r.outcome_state_hash)
		: recent.map((r) => `${r.action_type}::${r.canonical_args}`);

	const p = cyclePeriod(states, pMax, minRepsFn);
	if (p !== null) {
		const span = p * minRepsFn(p);
		const stepIndices = recent.slice(-span).map((r) => r.step_index);
		return { period: p, span, stepIndices };
	}
	return null;
}

// ── Unified Monitor ───────────────────────────────────────────
/**
 * Run the full monitor over the trajectory.
 * Returns EXACT_REPEAT if Detector 1 fires; else OSCILLATION if Detector 2 fires; else OK.
 */
export function runMonitor(
	records: StepRecord[],
	config: MonitorConfig,
): { verdict: Verdict; severity: Severity } {
	// Warmup: no flags during initial exploration.
	if (records.length < config.warmupSteps) {
		return { verdict: { type: "OK" }, severity: "none" };
	}

	// Detector 1: Exact repeat (check hard threshold first, then warn).
	const hardResult = detectExactRepeat(
		records,
		config.window,
		config.tRepeatHard,
	);
	if (hardResult) {
		return {
			verdict: {
				type: "EXACT_REPEAT",
				sig: hardResult.sig,
				count: hardResult.count,
				stepIndices: hardResult.stepIndices,
			},
			severity: "hard",
		};
	}

	const warnResult = detectExactRepeat(
		records,
		config.window,
		config.tRepeatWarn,
	);
	if (warnResult) {
		return {
			verdict: {
				type: "EXACT_REPEAT",
				sig: warnResult.sig,
				count: warnResult.count,
				stepIndices: warnResult.stepIndices,
			},
			severity: "warn",
		};
	}

	// Detector 2: Oscillation.
	const minRepsFn = (p: number) =>
		p === 1 ? config.minRepsP1 : config.minRepsP2;
	const oscResult = detectOscillation(
		records,
		config.window,
		config.pMax,
		minRepsFn,
		config.hasOutcomeState,
	);
	if (oscResult) {
		// Period-1 oscillation with minRepsP1 reps is warn;
		// longer periods with minRepsP2 reps are also warn.
		// For hard-stop, require double the repetitions.
		const hardMinRepsFn = (p: number) =>
			p === 1 ? config.minRepsP1 * 2 : config.minRepsP2 * 2;
		const hardOscResult = detectOscillation(
			records,
			config.window,
			config.pMax,
			hardMinRepsFn,
			config.hasOutcomeState,
		);
		return {
			verdict: {
				type: "OSCILLATION",
				period: oscResult.period,
				span: oscResult.span,
				stepIndices: oscResult.stepIndices,
			},
			severity: hardOscResult !== null ? "hard" : "warn",
		};
	}

	return { verdict: { type: "OK" }, severity: "none" };
}

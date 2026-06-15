/**
 * Cycle-Detection Monitor Extension
 *
 * Watches agent step-by-step execution and detects loops:
 * - EXACT_REPEAT: same action + same args + same observation, repeatedly
 * - OSCILLATION: cycling through a small set of world-states
 *
 * Runs in active mode by default — intervenes when cycles are detected.
 * Switch to shadow mode via /cycle shadow.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	runMonitor,
	type StepRecord,
	type MonitorConfig,
	type Verdict,
	type Severity,
	DEFAULT_CONFIG,
} from "./src/detector.js";
import {
	canonicalizeArgs,
	hashObservation,
	hashOutcomeState,
} from "./src/canonicalize.js";

// ── State ────────────────────────────────────────────────────────

interface ExtensionState {
	records: StepRecord[];
	stepCounter: number;
	config: MonitorConfig;
	shadowMode: boolean;
	enabled: boolean;
	fires: Array<{ step: number; verdict: Verdict; severity: Severity }>;
}

const state: ExtensionState = {
	records: [],
	stepCounter: 0,
	config: { ...DEFAULT_CONFIG },
	shadowMode: false,
	enabled: true,
	fires: [],
};

// ── Observation extraction ───────────────────────────────────────

/**
 * Extract a string from the tool result content for hashing.
 */
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

/**
 * Extract outcome state proxy. Currently uses a hash of the tool name + args
 * as a cheap proxy. The harness can be enhanced to provide a real world-state
 * hash (e.g., failing-test set, working-dir diff).
 */
function extractOutcomeState(toolName: string, input: unknown): string {
	// For write/edit tools, the outcome state is the file path being modified.
	// For bash, it's the command itself.
	// This is a cheap proxy — not full world-state hashing.
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

// ── Verdict formatting ───────────────────────────────────────────

function formatVerdict(verdict: Verdict, severity: Severity): string {
	const prefix = `[cycle-detection] [${severity.toUpperCase()}]`;
	switch (verdict.type) {
		case "OK":
			return `${prefix} OK — no cycles detected`;
		case "EXACT_REPEAT": {
			return `${prefix} EXACT_REPEAT: Action "${verdict.sig[0]}" issued ${verdict.count}× with unchanged output (steps: ${verdict.stepIndices.join(", ")})`;
		}
		case "OSCILLATION": {
			return `${prefix} OSCILLATION: Cycled through ${verdict.period} state(s) for ${verdict.span} steps (steps: ${verdict.stepIndices.join(", ")})`;
		}
	}
}

function formatInterventionMessage(verdict: Verdict): string {
	switch (verdict.type) {
		case "EXACT_REPEAT": {
			return `⚠ CYCLE DETECTED: Identical command "${verdict.sig[0]}" issued ${verdict.count}× with unchanged output. You are stuck in a loop. Consider a different approach.`;
		}
		case "OSCILLATION": {
			return `⚠ CYCLE DETECTED: Your actions are cycling through ${verdict.period} state(s) over ${verdict.span} steps. You are not making progress. Step back and reconsider your strategy.`;
		}
		default:
			return "";
	}
}

// ── Extension ────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	// Reset state on session start.
	pi.on("session_start", async (_event, ctx) => {
		state.records = [];
		state.stepCounter = 0;
		state.fires = [];
		const modeLabel = state.shadowMode ? "shadow" : "active";
		const modeBg = state.shadowMode ? "customMessageBg" : "toolSuccessBg";
		const modeFg = state.shadowMode ? "accent" : "success";
		ctx.ui.setStatus(
			"cycle-detection",
			`│ ${ctx.ui.theme.bg(modeBg, ctx.ui.theme.fg(modeFg, `cycle-detection: ${modeLabel}`))}`,
		);
	});

	// Collect StepRecords from tool_result events.
	pi.on("tool_result", async (event, ctx) => {
		if (!state.enabled) return;

		// Build the StepRecord.
		const observationText = extractObservationText(event.content);
		const record: StepRecord = {
			step_index: state.stepCounter,
			action_type: event.toolName,
			canonical_args: canonicalizeArgs(event.input),
			observation_hash: hashObservation(observationText),
			outcome_state_hash: hashOutcomeState(
				extractOutcomeState(event.toolName, event.input),
			),
		};

		state.records.push(record);
		state.stepCounter++;

		// Keep only the window + warmup to bound memory.
		const maxKeep = state.config.window + state.config.warmupSteps + 10;
		if (state.records.length > maxKeep) {
			state.records = state.records.slice(-maxKeep);
		}

		// Run the monitor.
		const { verdict, severity } = runMonitor(state.records, state.config);

		if (verdict.type !== "OK") {
			state.fires.push({
				step: state.stepCounter - 1,
				verdict,
				severity,
			});

			const logMsg = formatVerdict(verdict, severity);

			if (state.shadowMode) {
				console.log(`[cycle-detection:SHADOW] ${logMsg}`);
				// Update status bar with fire count.
				ctx.ui.setStatus(
					"cycle-detection",
					`│ ${ctx.ui.theme.bg("customMessageBg", ctx.ui.theme.fg("accent", "cycle-detection: shadow (" + state.fires.length + " fires)"))}`,
				);
			} else if (severity === "warn") {
				// Warn: inject a message into the agent context.
				console.log(`[cycle-detection:WARN] ${logMsg}`);
				pi.sendMessage(
					{
						customType: "cycle-detection",
						content: formatInterventionMessage(verdict),
						display: true,
					},
					{ deliverAs: "steer", triggerTurn: false },
				);
				ctx.ui.setStatus(
					"cycle-detection",
					`│ ${ctx.ui.theme.bg("toolPendingBg", ctx.ui.theme.fg("warning", "cycle-detection: ⚠ WARN (" + state.fires.length + " fires)"))}`,
				);
			} else {
				// Hard-stop: halt the run.
				console.log(`[cycle-detection:HARD] ${logMsg}`);
				ctx.ui.notify("Cycle detected — halting run", "error");
				ctx.shutdown();
			}
		}
	});

	// ── /cycle command ─────────────────────────────────────────────
	pi.registerCommand("cycle", {
		description: "Cycle-detection monitor status and controls",
		getArgumentCompletions: (prefix: string) => {
			const cmds = ["status", "enable", "disable", "shadow", "active"];
			const items = cmds
				.filter((c) => c.startsWith(prefix))
				.map((c) => ({ value: c, label: c }));
			return items.length > 0 ? items : null;
		},
		handler: async (args, ctx) => {
			const cmd = (args || "status").trim().toLowerCase();

			if (cmd === "status") {
				const lines = [
					`Cycle Detection Monitor`,
					`  Enabled: ${state.enabled}`,
					`  Shadow mode: ${state.shadowMode}`,
					`  Steps tracked: ${state.stepCounter}`,
					`  Fires: ${state.fires.length}`,
					`  Window: ${state.config.window}`,
					`  T_repeat (warn/hard): ${state.config.tRepeatWarn}/${state.config.tRepeatHard}`,
					`  P_max: ${state.config.pMax}`,
					`  Min reps (p1/p≥2): ${state.config.minRepsP1}/${state.config.minRepsP2}`,
					`  Warmup: ${state.config.warmupSteps}`,
				];
				if (state.fires.length > 0) {
					lines.push(`  Recent fires:`);
					for (const f of state.fires.slice(-5)) {
						lines.push(`    Step ${f.step}: [${f.severity}] ${f.verdict.type}`);
					}
				}
				ctx.ui.setWidget("cycle-detection", lines);
				return;
			}

			if (cmd === "enable") {
				state.enabled = true;
				ctx.ui.notify("Cycle detection enabled", "info");
				ctx.ui.setStatus(
					"cycle-detection",
					`│ ${ctx.ui.theme.bg("toolSuccessBg", ctx.ui.theme.fg("success", "cycle-detection: active"))}`,
				);
				return;
			}

			if (cmd === "disable") {
				state.enabled = false;
				ctx.ui.notify("Cycle detection disabled", "info");
				ctx.ui.setStatus(
					"cycle-detection",
					`│ ${ctx.ui.theme.bg("customMessageBg", ctx.ui.theme.fg("dim", "cycle-detection: off"))}`,
				);
				return;
			}

			if (cmd === "shadow") {
				state.shadowMode = true;
				ctx.ui.notify("Cycle detection: shadow mode", "info");
				ctx.ui.setStatus(
					"cycle-detection",
					`│ ${ctx.ui.theme.bg("customMessageBg", ctx.ui.theme.fg("accent", "cycle-detection: shadow"))}`,
				);
				return;
			}

			if (cmd === "active") {
				state.shadowMode = false;
				ctx.ui.notify(
					"Cycle detection: active mode (interventions enabled)",
					"warning",
				);
				ctx.ui.setStatus(
					"cycle-detection",
					`│ ${ctx.ui.theme.bg("toolErrorBg", ctx.ui.theme.fg("error", "cycle-detection: ACTIVE"))}`,
				);
				return;
			}

			ctx.ui.notify(
				`Unknown command: ${cmd}. Use: status, enable, disable, shadow, active`,
				"error",
			);
		},
	});

	// ── Session shutdown ───────────────────────────────────────────
	pi.on("session_shutdown", async (_event, _ctx) => {
		if (state.fires.length > 0) {
			console.log(
				`[cycle-detection] Session ended. Total fires: ${state.fires.length}`,
			);
		}
	});
}

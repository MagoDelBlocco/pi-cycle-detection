/**
 * Cycle detection extension — entry point.
 *
 * Wires two detection layers into the pi event lifecycle:
 *
 * 1. **Step-based**: monitors tool-call trajectories for exact-repeat loops
 *    and structural oscillation. Fires at tool_result. Intervenes by steering
 *    the agent with a course-correction message (never kills the process).
 *
 * 2. **Stream-based**: monitors thinking_delta and text_delta character
 *    streams for sentence-level repetition. Fires mid-stream and can abort +
 *    recover the in-flight message. Detection is scoped to a single message
 *    (reset at message_end) and ignores short non-prose fragments, so normal
 *    repetitive output (tables, file lists) does not trip it.
 *
 * Active mode (default): both detectors run and intervene when cycles are
 * detected. Switch to shadow mode (observe only) via `/cycle shadow`.
 */

import { exec } from "node:child_process";
import { promisify } from "node:util";
import type { ExtensionAPI, Theme } from "@earendil-works/pi-coding-agent";
import {
	runMonitor,
	type StepRecord,
	type MonitorConfig,
	type Verdict as StepVerdict,
	type Severity,
	DEFAULT_CONFIG,
} from "./src/detector.js";
import {
	canonicalizeArgs,
	hashObservation,
	hashOutcomeState,
} from "./src/canonicalize.js";
import {
	StreamDetector,
	DEFAULT_STREAM_CONFIG,
	type StreamVerdict,
	type Severity as StreamSeverity,
} from "./src/stream-detector.js";

// ── Extension State ────────────────────────────────────────────

interface CycleState {
	/** Master enable flag — gates ALL detection across both layers. */
	enabled: boolean;
	/** Master shadow flag — when true, detect and report but never intervene. */
	shadow: boolean;

	// Step-based detector
	stepConfig: MonitorConfig;
	stepRecords: StepRecord[];
	stepStepIndex: number;
	/** Severity last surfaced by the step detector, to avoid re-steering every
	 *  tool_result while the agent is stuck in the same cycle. */
	lastStepSeverity: Severity;

	// Stream-based detectors (one per content type, reused across messages)
	thinkingDetector: StreamDetector;
	textDetector: StreamDetector;

	// Recovery coordination
	/** Set when we decide to abort; read by message_end handler. */
	pendingAbort: null | {
		verdict: StreamVerdict;
		severity: StreamSeverity;
		contentType: "thinking" | "text";
	};

	// Stats aggregation
	stepWarnCount: number;
	stepHardCount: number;
}

function createState(): CycleState {
	const shadow = false; // Default: active mode
	return {
		enabled: true,
		shadow,
		stepConfig: { ...DEFAULT_CONFIG },
		stepRecords: [],
		stepStepIndex: 0,
		lastStepSeverity: "none",
		thinkingDetector: new StreamDetector({ ...DEFAULT_STREAM_CONFIG, shadow }),
		textDetector: new StreamDetector({ ...DEFAULT_STREAM_CONFIG, shadow }),
		pendingAbort: null,
		stepWarnCount: 0,
		stepHardCount: 0,
	};
}

// ── Outcome State Extraction ───────────────────────────────────
/**
 * Extract a proxy outcome state from a tool call.
 *
 * The state proxy is the tool identity plus its full canonicalized args.
 * For write/edit this includes the content / edit operations, so progressive
 * edits to a single file produce *distinct* states (no false-positive
 * oscillation on normal iterative editing), while a genuine edit↔revert loop
 * reproduces identical states and is still caught.
 */
function extractOutcomeState(toolName: string, input: unknown): string {
	return `${toolName}:${canonicalizeArgs(input)}`;
}

/** Extract observation text from a tool-result content array. */
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

// ── Recovery: Truncate Aborted Message ─────────────────────────

/**
 * Truncate an aborted assistant message based on the cycle verdict.
 *
 * Tool calls are always stripped: they were generated from cycled reasoning,
 * and a tool call left in the finalized message would (a) dangle without a
 * matching tool result and (b) for extended-thinking models break the
 * thinking-block/tool-use pairing the API requires on the next turn. We keep
 * the message to a clean text-only stop and let the steer message re-drive.
 */
function truncateAbortedMessage(
	message: Record<string, unknown>,
	abortInfo: NonNullable<CycleState["pendingAbort"]>,
): Record<string, unknown> {
	const content = (message.content as unknown[]) ?? [];
	const cleaned: unknown[] = [];

	for (const block of content) {
		if (typeof block !== "object" || block === null) continue;
		const bc = block as Record<string, unknown>;

		// Strip thinking (it led to the cycle) and tool calls (see above).
		if (bc.type === "thinking" || bc.type === "toolCall") continue;

		if (bc.type === "text") {
			const fullText = (bc.text as string) ?? "";
			// For a text cycle, keep only the text before the loop began.
			const cutOffset =
				abortInfo.contentType === "text"
					? "firstRepeatOffset" in abortInfo.verdict
						? (abortInfo.verdict as { firstRepeatOffset: number })
								.firstRepeatOffset
						: "patternStartOffset" in abortInfo.verdict
							? (abortInfo.verdict as { patternStartOffset: number })
									.patternStartOffset
							: 0
					: 0;
			const truncated =
				cutOffset > 0 ? fullText.slice(0, cutOffset) : fullText;
			if (truncated.trim().length > 0) {
				cleaned.push({ ...bc, text: truncated.trimEnd() });
			}
		}
	}

	// Never emit an empty assistant message — leave a breadcrumb instead.
	if (cleaned.length === 0) {
		cleaned.push({
			type: "text",
			text: "[response interrupted: repetition detected]",
		});
	}

	return { ...message, content: cleaned, stopReason: "stop" };
}

// ── Steer Message Templates ────────────────────────────────────

function makeStreamSteer(
	verdict: StreamVerdict,
	contentType: "thinking" | "text",
): string {
	const label = contentType === "thinking" ? "reasoning" : "answer";
	if (verdict.type === "EXACT_REPEAT") {
		return `⚠ Cycle detection: your ${label} repeated the same sentence ${verdict.count} times. Break out of this loop and ${contentType === "thinking" ? "proceed with tool calls or a concrete answer" : "try a different approach or conclude your response"}.`;
	}
	if (verdict.type === "OSCILLATION") {
		return `⚠ Cycle detection: your ${label} oscillated with period ${verdict.period} sentences. ${contentType === "thinking" ? "Shift to a different line of reasoning" : "Break the pattern and provide new content"}.`;
	}
	return `⚠ Cycle detection: your ${label} showed repetitive patterns. Try a different approach.`;
}

function makeStepSteer(verdict: StepVerdict, severity: Severity): string {
	const force =
		severity === "hard"
			? "You are stuck in a loop and must stop repeating it now."
			: "You appear to be looping.";
	if (verdict.type === "EXACT_REPEAT") {
		return `⚠ Cycle detection: the action "${verdict.sig[0]}" was issued ${verdict.count}× in a row with unchanged output. ${force} Step back, re-examine your assumptions, and try a different approach.`;
	}
	if (verdict.type === "OSCILLATION") {
		return `⚠ Cycle detection: your tool calls are cycling through ${verdict.period} state(s) over ${verdict.span} steps with no net progress. ${force} Reconsider your overall strategy rather than continuing the cycle.`;
	}
	return `⚠ Cycle detection: repetitive tool-call pattern detected. ${force}`;
}

// ── Desktop Notification ───────────────────────────────────────

const execAsync = promisify(exec);

/**
 * Fire a desktop notification via notify-send.
 * Non-fatal — silently ignored if notify-send is unavailable.
 */
async function notifyDesktop(): Promise<void> {
	try {
		await execAsync(
			'notify-send -u critical -a "Pi" "A cycle has been detected and broken."',
		);
	} catch {
		// notify-send not available or failed — silently ignore
	}
}

// ── Status Bar ─────────────────────────────────────────────────

function statusText(state: CycleState, theme: Theme): string {
	if (!state.enabled) return theme.fg("dim", "cycle-detection: off");
	const mode = state.shadow ? "shadow" : "active";
	const fires = state.stepWarnCount + state.stepHardCount;
	const label =
		fires > 0
			? `cycle-detection: ${mode} (${fires} fires)`
			: `cycle-detection: ${mode}`;
	// Shadow (observe-only) reads as informational; active as a healthy/armed
	// state. Both are restored from theme colors so the footer is no longer a
	// flat white string.
	return theme.fg(state.shadow ? "accent" : "success", label);
}

/**
 * Colored transient status shown while a cycle is actively being detected.
 * Hard severity is an error (we're aborting/steering); warn is a warning.
 */
function detectionStatus(
	theme: Theme,
	modeLabel: string,
	body: string,
	severity: Severity | StreamSeverity,
): string {
	const color = severity === "hard" ? "error" : "warning";
	return theme.fg(color, `${modeLabel} ${body}`);
}

// ── Extension Factory ──────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	const state = createState();

	// Which info view (if any) is currently pinned in the widget above the
	// editor. The widget is persistent UI — it stays until explicitly cleared —
	// so we track what's shown to support toggle-off and an explicit `hide`.
	let widgetView: "status" | "stats" | null = null;
	const clearWidget = (ctx: { ui: { setWidget: (k: string, c: string[] | undefined) => void } }) => {
		ctx.ui.setWidget("cycle-detection", undefined);
		widgetView = null;
	};

	const syncStreamShadow = () => {
		state.thinkingDetector.setConfig({ shadow: state.shadow });
		state.textDetector.setConfig({ shadow: state.shadow });
	};

	// ── Stream-based: message_update ───────────────────────────

	pi.on("message_update", async (event, ctx) => {
		if (!state.enabled) return;

		const ame = event.assistantMessageEvent;
		if (!ame || typeof ame !== "object") return;
		const ameTyped = ame as Record<string, unknown>;

		// Only process thinking_delta and text_delta
		const ameType = ameTyped.type as string | undefined;
		if (ameType !== "thinking_delta" && ameType !== "text_delta") return;

		const delta = ameTyped.delta as string | undefined;
		if (typeof delta !== "string") return;

		// Don't double-intervene on the same message
		if (state.pendingAbort !== null) return;

		const isThinking = ameType === "thinking_delta";
		const detector = isThinking ? state.thinkingDetector : state.textDetector;
		const contentType: "thinking" | "text" = isThinking ? "thinking" : "text";

		const { verdict, severity } = detector.feed(delta);

		// Always show detection status (shadow or active)
		if (severity !== "none") {
			const modeLabel = state.shadow ? "[shadow]" : "[ACTIVE]";
			ctx.ui.setStatus(
				"cycle-detection",
				detectionStatus(
					ctx.ui.theme,
					modeLabel,
					`${contentType}: ${verdict.type} (${severity})`,
					severity,
				),
			);
		}

		// Active mode: abort on a hard stream cycle so we can recover.
		if (!state.shadow && severity === "hard") {
			state.pendingAbort = { verdict, severity, contentType };
			ctx.abort();
		}
	});

	// ── Stream-based: message_start / message_end ──────────────

	pi.on("message_start", async (event, _ctx) => {
		if (event.message.role === "assistant") {
			state.thinkingDetector.reset();
			state.textDetector.reset();
			state.pendingAbort = null;
		}
	});

	pi.on("message_end", async (event, ctx) => {
		if (event.message.role !== "assistant") return;

		// Stream cycle detection is scoped to a single assistant message: a
		// sentence repeated in this message must not count toward a "repeat" in
		// the next one. Reset here, where the event reliably fires — relying on
		// message_start alone let a previous message's repeats linger in the
		// window and spuriously abort the *next*, unrelated stream.
		const resetDetectors = () => {
			state.thinkingDetector.reset();
			state.textDetector.reset();
		};

		// If we flagged an abort, truncate the message and steer.
		if (state.pendingAbort !== null) {
			const abortInfo = state.pendingAbort;
			const msg = event.message as unknown as Record<string, unknown>;
			const truncated = truncateAbortedMessage(msg, abortInfo);

			const steerText = makeStreamSteer(abortInfo.verdict, abortInfo.contentType);
			pi.sendMessage(
				{
					customType: "cycle-detection",
					content: steerText,
					display: false, // Don't clutter the TUI
				},
				{ triggerTurn: true, deliverAs: "steer" },
			);

			state.pendingAbort = null;
			resetDetectors();
			ctx.ui.setStatus("cycle-detection", statusText(state, ctx.ui.theme));

			void notifyDesktop();

			return { message: truncated as never };
		}

		// Normal end: clear status and reset for the next message.
		resetDetectors();
		ctx.ui.setStatus("cycle-detection", statusText(state, ctx.ui.theme));
	});

	// ── Step-based: tool_result ────────────────────────────────

	pi.on("tool_result", async (event, ctx) => {
		if (!state.enabled) return;

		const obsText = extractObservationText((event.content as unknown[]) ?? []);
		const record: StepRecord = {
			step_index: state.stepStepIndex++,
			action_type: event.toolName,
			canonical_args: canonicalizeArgs(event.input),
			observation_hash: hashObservation(obsText),
			outcome_state_hash: hashOutcomeState(
				extractOutcomeState(event.toolName, event.input),
			),
		};

		state.stepRecords.push(record);

		// Trim to prevent unbounded growth.
		const maxKeep = state.stepConfig.window + state.stepConfig.warmupSteps + 10;
		if (state.stepRecords.length > maxKeep) {
			state.stepRecords = state.stepRecords.slice(-maxKeep);
		}

		const { verdict, severity } = runMonitor(state.stepRecords, state.stepConfig);

		if (verdict.type === "OK") {
			// Cycle cleared — allow the next genuine cycle to re-trigger a steer.
			state.lastStepSeverity = "none";
			return;
		}

		if (severity === "warn") state.stepWarnCount++;
		if (severity === "hard") state.stepHardCount++;

		const modeLabel = state.shadow ? "[shadow]" : "[ACTIVE]";
		ctx.ui.setStatus(
			"cycle-detection",
			detectionStatus(
				ctx.ui.theme,
				modeLabel,
				`step: ${verdict.type} (${severity})`,
				severity,
			),
		);

		// Intervene only on a new or escalating cycle, so we don't steer on
		// every subsequent tool_result while the agent is in the same loop.
		const escalated =
			(state.lastStepSeverity === "none" && severity !== "none") ||
			(state.lastStepSeverity === "warn" && severity === "hard");
		state.lastStepSeverity = severity;

		if (!state.shadow && escalated) {
			pi.sendMessage(
				{
					customType: "cycle-detection",
					content: makeStepSteer(verdict, severity),
					display: true,
				},
				{ deliverAs: "steer", triggerTurn: false },
			);
			if (severity === "hard") void notifyDesktop();
		}
	});

	// ── turn_end: reset transient stream status ────────────────

	pi.on("turn_end", async (_event, ctx) => {
		ctx.ui.setStatus("cycle-detection", statusText(state, ctx.ui.theme));
	});

	// ── Command: /cycle ────────────────────────────────────────

	pi.registerCommand("cycle", {
		description: "Cycle-detection monitor: status, stats, and controls",
		getArgumentCompletions: (prefix: string) => {
			const cmds = [
				"status",
				"stats",
				"hide",
				"enable",
				"disable",
				"shadow",
				"active",
			];
			const items = cmds
				.filter((c) => c.startsWith(prefix))
				.map((c) => ({ value: c, label: c }));
			return items.length > 0 ? items : null;
		},
		handler: async (args, ctx) => {
			const cmd = (args || "status").trim().toLowerCase();

			if (cmd === "hide" || cmd === "clear" || cmd === "close") {
				clearWidget(ctx);
				return;
			}

			if (cmd === "status") {
				// Re-running the visible view toggles it off, so the pinned
				// widget is never a one-way trip.
				if (widgetView === "status") {
					clearWidget(ctx);
					return;
				}
				const lines = [
					"Cycle Detection Monitor",
					`  Enabled: ${state.enabled}`,
					`  Mode: ${state.shadow ? "shadow (observe only)" : "active"}`,
					`  Steps tracked: ${state.stepStepIndex}`,
					`  Step fires (warn/hard): ${state.stepWarnCount}/${state.stepHardCount}`,
					`  Window: ${state.stepConfig.window}`,
					`  T_repeat (warn/hard): ${state.stepConfig.tRepeatWarn}/${state.stepConfig.tRepeatHard}`,
					`  P_max: ${state.stepConfig.pMax}`,
					`  Warmup: ${state.stepConfig.warmupSteps}`,
					"",
					"  (run /cycle hide to dismiss)",
				];
				ctx.ui.setWidget("cycle-detection", lines);
				widgetView = "status";
				return;
			}

			if (cmd === "stats") {
				if (widgetView === "stats") {
					clearWidget(ctx);
					return;
				}
				const tStats = state.thinkingDetector.getStats();
				const txStats = state.textDetector.getStats();
				const lines = [
					"=== Cycle Detection Stats ===",
					"",
					"Step-based (tool calls):",
					`  Records: ${state.stepRecords.length}`,
					`  Warn detections: ${state.stepWarnCount}`,
					`  Hard detections: ${state.stepHardCount}`,
					"",
					"Stream-based (thinking):",
					`  Characters fed: ${tStats.charactersFed}`,
					`  Sentences: ${tStats.sentencesExtracted}`,
					`  Warn / Hard: ${tStats.warnDetections} / ${tStats.hardDetections}`,
					`  Max repeat run: ${tStats.maxRepeatRun}`,
					`  Min oscillation period: ${tStats.minOscillationPeriod ?? "none"}`,
					`  Last: ${tStats.lastVerdict.type} (${tStats.lastSeverity})`,
					"",
					"Stream-based (text):",
					`  Characters fed: ${txStats.charactersFed}`,
					`  Sentences: ${txStats.sentencesExtracted}`,
					`  Warn / Hard: ${txStats.warnDetections} / ${txStats.hardDetections}`,
					`  Max repeat run: ${txStats.maxRepeatRun}`,
					`  Min oscillation period: ${txStats.minOscillationPeriod ?? "none"}`,
					`  Last: ${txStats.lastVerdict.type} (${txStats.lastSeverity})`,
					"",
					"  (run /cycle hide to dismiss)",
				];
				ctx.ui.setWidget("cycle-detection", lines);
				widgetView = "stats";
				return;
			}

			if (cmd === "enable") {
				state.enabled = true;
				ctx.ui.notify("Cycle detection enabled", "info");
				ctx.ui.setStatus("cycle-detection", statusText(state, ctx.ui.theme));
				return;
			}

			if (cmd === "disable") {
				state.enabled = false;
				ctx.ui.notify("Cycle detection disabled", "info");
				ctx.ui.setStatus("cycle-detection", statusText(state, ctx.ui.theme));
				return;
			}

			if (cmd === "shadow") {
				state.shadow = true;
				syncStreamShadow();
				ctx.ui.notify("Cycle detection: shadow mode (observe only)", "info");
				ctx.ui.setStatus("cycle-detection", statusText(state, ctx.ui.theme));
				return;
			}

			if (cmd === "active") {
				state.shadow = false;
				syncStreamShadow();
				ctx.ui.notify(
					"Cycle detection: active mode (interventions enabled)",
					"warning",
				);
				ctx.ui.setStatus("cycle-detection", statusText(state, ctx.ui.theme));
				return;
			}

			ctx.ui.notify(
				`Unknown command: ${cmd}. Use: status, stats, hide, enable, disable, shadow, active`,
				"error",
			);
		},
	});

	// ── session_start / session_shutdown ───────────────────────

	pi.on("session_start", async (_event, ctx) => {
		state.stepRecords = [];
		state.stepStepIndex = 0;
		state.stepWarnCount = 0;
		state.stepHardCount = 0;
		state.lastStepSeverity = "none";
		state.pendingAbort = null;
		state.thinkingDetector.resetFull();
		state.textDetector.resetFull();
		ctx.ui.setStatus("cycle-detection", statusText(state, ctx.ui.theme));
	});

	pi.on("session_shutdown", async (_event, _ctx) => {
		const total = state.stepWarnCount + state.stepHardCount;
		if (total > 0) {
			// Best-effort end-of-session summary (stderr — never the TUI stream).
			process.stderr.write(
				`[cycle-detection] Session ended. Step fires: ${total} (warn ${state.stepWarnCount}, hard ${state.stepHardCount})\n`,
			);
		}
	});
}

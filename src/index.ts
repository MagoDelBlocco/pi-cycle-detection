/**
 * Cycle detection extension — entry point.
 *
 * Wires two detection layers into the pi event lifecycle:
 *
 * 1. **Step-based** (existing): monitors tool-call trajectories for
 *    exact-repeat loops and structural oscillation. Fires at tool_result.
 *
 * 2. **Stream-based** (new): monitors thinking_delta and text_delta
 *    character streams for sentence-level repetition. Fires mid-stream
 *    and can abort + recover.
 *
 * Active mode (default): both detectors run and intervene when cycles
 * are detected. Flip `shadow: true` to observe without intervention.
 */

import { exec } from "node:child_process";
import { promisify } from "node:util";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	runMonitor,
	type StepRecord,
	type MonitorConfig,
	DEFAULT_CONFIG,
} from "./detector.js";
import {
	canonicalizeArgs,
	hashObservation,
	hashOutcomeState,
} from "./canonicalize.js";
import {
	StreamDetector,
	DEFAULT_STREAM_CONFIG,
	type StreamVerdict,
	type Severity as StreamSeverity,
} from "./stream-detector.js";

// ── Extended Config ────────────────────────────────────────────

/** Shadow-mode flag layered on top of MonitorConfig. */
interface StepDetectorConfig extends MonitorConfig {
	/** When true, step detector reports but never intervenes. */
	shadow: boolean;
}

// ── Extension State ────────────────────────────────────────────

interface CycleState {
	// Master shadow flag — gates ALL intervention across both detectors
	shadow: boolean;

	// Step-based detector
	stepConfig: StepDetectorConfig;
	stepRecords: StepRecord[];
	stepStepIndex: number;

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
		shadow,
		stepConfig: { ...DEFAULT_CONFIG, shadow },
		stepRecords: [],
		stepStepIndex: 0,
		thinkingDetector: new StreamDetector({ ...DEFAULT_STREAM_CONFIG, shadow }),
		textDetector: new StreamDetector({ ...DEFAULT_STREAM_CONFIG, shadow }),
		pendingAbort: null,
		stepWarnCount: 0,
		stepHardCount: 0,
	};
}

// ── Outcome State Extraction ───────────────────────────────────
/**
 * Extract a proxy outcome state from tool input.
 *
 * KNOWN ISSUE: same-file edits collapse to identical state regardless
 * of content. This causes false-positive oscillation for progressive
 * edits. Documented in integration tests.
 *
 * Mitigation: when hasOutcomeState is false, the detector falls back
 * to action-pattern matching which includes canonicalized args.
 */
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

/** Extract observation text from content array. */
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
 * For thinking cycles: strip thinking content, keep tool calls.
 * For text cycles: truncate text at cycle boundary, strip tool calls.
 */
function truncateAbortedMessage(
	message: Record<string, unknown>,
	abortInfo: NonNullable<CycleState["pendingAbort"]>,
): Record<string, unknown> {
	const content = (message.content as unknown[]) ?? [];

	if (abortInfo.contentType === "thinking") {
		// Remove thinking blocks, keep text and tool calls
		const cleaned = content.filter(
			(block: unknown) =>
				typeof block !== "object" ||
				(block as Record<string, unknown>).type !== "thinking",
		);
		return {
			...message,
			content: cleaned,
			stopReason: cleaned.some(
				(b: unknown) =>
					typeof b === "object" &&
					(b as Record<string, unknown>).type === "toolCall",
			)
				? "toolUse"
				: "stop",
		};
	}

	// Text cycle: truncate text content
	const cleaned: unknown[] = [];
	for (const block of content) {
		if (typeof block !== "object" || block === null) continue;
		const bc = block as Record<string, unknown>;

		if (bc.type === "thinking") {
			// Strip thinking — it led to the cycle
			continue;
		}

		if (bc.type === "text") {
			// Keep text up to the cycle point if we have an offset
			const fullText = (bc.text as string) ?? "";
			const cutOffset =
				"firstRepeatOffset" in abortInfo.verdict
					? (abortInfo.verdict as { firstRepeatOffset: number })
							.firstRepeatOffset
					: "patternStartOffset" in abortInfo.verdict
						? (abortInfo.verdict as { patternStartOffset: number })
								.patternStartOffset
						: 0;
			// Truncate: keep text before the cycle started
			const truncated = cutOffset > 0 ? fullText.slice(0, cutOffset) : fullText;
			if (truncated.length > 0) {
				cleaned.push({ ...bc, text: truncated.trimEnd() });
			}
		}
		// Strip tool calls — they were generated from cycled reasoning
	}

	return {
		...message,
		content: cleaned,
		stopReason: "stop",
	};
}

// ── Steer Message Templates ────────────────────────────────────

function makeSteerContent(
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
	// Fallback (OK — shouldn't reach here)
	return `⚠ Cycle detection: your ${label} showed repetitive patterns. Try a different approach.`;
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

// ── Extension Factory ──────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	const state = createState();

	// ── Stream-based: message_update ───────────────────────────

	pi.on("message_update", async (event, ctx) => {
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
				"cycle-detect",
				`${modeLabel} ${contentType}: ${verdict.type} (${severity})`,
			);
		}

		// Active mode only: abort on hard severity
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

		// If we flagged an abort, truncate the message
		if (state.pendingAbort !== null) {
			const msg = event.message as unknown as Record<string, unknown>;
			const truncated = truncateAbortedMessage(msg, state.pendingAbort);

			// Send steer message to redirect the model
			const steerText = makeSteerContent(
				state.pendingAbort.verdict,
				state.pendingAbort.contentType,
			);

			pi.sendMessage(
				{
					customType: "cycle-detection",
					content: steerText,
					display: false, // Don't clutter the TUI
				},
				{ triggerTurn: true, deliverAs: "steer" },
			);

			state.pendingAbort = null;
			ctx.ui.setStatus("cycle-detect", "");

			// Desktop notification
			notifyDesktop().catch((err) => {
				// Non-fatal — silently ignore if notify-send is unavailable
				void err;
			});

			return { message: truncated as never };
		}

		// Clear status after normal message end
		ctx.ui.setStatus("cycle-detect", "");
	});

	// ── Step-based: tool_call / tool_result ────────────────────

	// Track tool calls so we can match them with results
	const pendingToolCalls = new Map<
		string,
		{
			toolName: string;
			input: unknown;
			timestamp: number;
		}
	>();

	pi.on("tool_execution_start", async (event, _ctx) => {
		pendingToolCalls.set(event.toolCallId, {
			toolName: event.toolName,
			input: event.args,
			timestamp: Date.now(),
		});
	});

	pi.on("tool_result", async (event, _ctx) => {
		const pending = pendingToolCalls.get(event.toolCallId);
		if (!pending) return;
		pendingToolCalls.delete(event.toolCallId);

		const obsText = extractObservationText((event.content as unknown[]) ?? []);

		const record: StepRecord = {
			step_index: state.stepStepIndex++,
			action_type: event.toolName,
			canonical_args: canonicalizeArgs(pending.input),
			observation_hash: hashObservation(obsText),
			outcome_state_hash: hashOutcomeState(
				extractOutcomeState(event.toolName, pending.input),
			),
		};

		state.stepRecords.push(record);

		// Trim to prevent unbounded growth
		const maxKeep = state.stepConfig.window + state.stepConfig.warmupSteps + 10;
		if (state.stepRecords.length > maxKeep) {
			state.stepRecords = state.stepRecords.slice(-maxKeep);
		}

		// Run step-based detector
		const { verdict, severity } = runMonitor(
			state.stepRecords,
			state.stepConfig,
		);

		if (severity === "warn") state.stepWarnCount++;
		if (severity === "hard") state.stepHardCount++;

		if (verdict.type !== "OK") {
			// Step-based cycles are reported via status bar.
			// Hard cycles could trigger a steer message (future enhancement).
			_ctx.ui.setStatus("cycle-detect", `step: ${verdict.type} (${severity})`);
		}
	});

	// ── turn_end: reset stream status ──────────────────────────

	pi.on("turn_end", async (_event, ctx) => {
		ctx.ui.setStatus("cycle-detect", "");
	});

	// ── Commands ───────────────────────────────────────────────

	pi.registerCommand("cycle-stats", {
		description: "Show cycle detection statistics",
		handler: async (_args, ctx) => {
			const tStats = state.thinkingDetector.getStats();
			const txStats = state.textDetector.getStats();
			const lines = [
				"=== Cycle Detection Stats ===",
				"",
				"Step-based (tool calls):",
				`  Records: ${state.stepRecords.length}`,
				`  Warn detections: ${state.stepWarnCount}`,
				`  Hard detections: ${state.stepHardCount}`,
				`  Shadow: ${state.stepConfig.shadow ? "yes" : "no"}`,
				"",
				"Stream-based (thinking):",
				`  Characters fed: ${tStats.charactersFed}`,
				`  Sentences: ${tStats.sentencesExtracted}`,
				`  Warn: ${tStats.warnDetections}`,
				`  Hard: ${tStats.hardDetections}`,
				`  Max repeat run: ${tStats.maxRepeatRun}`,
				`  Min oscillation period: ${tStats.minOscillationPeriod ?? "none"}`,
				`  Last: ${tStats.lastVerdict.type} (${tStats.lastSeverity})`,
				"",
				"Stream-based (text):",
				`  Characters fed: ${txStats.charactersFed}`,
				`  Sentences: ${txStats.sentencesExtracted}`,
				`  Warn: ${txStats.warnDetections}`,
				`  Hard: ${txStats.hardDetections}`,
				`  Max repeat run: ${txStats.maxRepeatRun}`,
				`  Min oscillation period: ${txStats.minOscillationPeriod ?? "none"}`,
				`  Last: ${txStats.lastVerdict.type} (${txStats.lastSeverity})`,
			];
			ctx.ui.setWidget("cycle-stats", lines);
		},
	});

	pi.registerCommand("cycle-mode", {
		description: "Toggle cycle detection shadow/active mode",
		getArgumentCompletions: (
			prefix: string,
		): Array<{ value: string; label: string }> | null => {
			const modes = ["shadow", "active", "toggle"];
			const items = modes.map((m) => ({ value: m, label: m }));
			const filtered = items.filter((i) => i.value.startsWith(prefix));
			return filtered.length > 0 ? filtered : null;
		},
		handler: async (args, ctx) => {
			const action = args?.toLowerCase() ?? "toggle";

			if (action === "shadow") {
				state.stepConfig.shadow = true;
				state.thinkingDetector.setConfig({ shadow: true });
				state.textDetector.setConfig({ shadow: true });
				ctx.ui.notify("Cycle detection: SHADOW mode (observe only)", "info");
			} else if (action === "active") {
				state.stepConfig.shadow = false;
				state.thinkingDetector.setConfig({ shadow: false });
				state.textDetector.setConfig({ shadow: false });
				ctx.ui.notify(
					"Cycle detection: ACTIVE mode (will intervene)",
					"warning",
				);
			} else {
				// Toggle
				const next = !state.stepConfig.shadow;
				state.stepConfig.shadow = next;
				state.thinkingDetector.setConfig({ shadow: next });
				state.textDetector.setConfig({ shadow: next });
				ctx.ui.notify(
					`Cycle detection: ${next ? "SHADOW" : "ACTIVE"}`,
					next ? "info" : "warning",
				);
			}
		},
	});

	// ── session_start / session_shutdown ───────────────────────

	pi.on("session_start", async (_event, ctx) => {
		state.stepRecords = [];
		state.stepStepIndex = 0;
		state.stepWarnCount = 0;
		state.stepHardCount = 0;
		state.thinkingDetector.resetFull();
		state.textDetector.resetFull();
		ctx.ui.setStatus("cycle-detect", "cycle-detect ready (active)");
	});

	pi.on("session_shutdown", async (_event, _ctx) => {
		// Cleanup if needed
	});
}

/**
 * Streaming cycle detector — incremental repetition detection over
 * thinking_delta and text_delta character streams.
 *
 * Splits incoming text into sentences, hashes each sentence, and runs
 * a sliding-window cycle check over the hash sequence. Stateless between
 * calls to reset() — safe to reuse across messages.
 *
 * Two modes:
 *   - Active: fires onCycle callback when a cycle crosses the threshold
 *   - Shadow: runs detection silently, accumulates statistics
 */

import { createHash } from "node:crypto";
import { cyclePeriod } from "./detector.js";

// ── Types ──────────────────────────────────────────────────────

/** Verdict for a streaming text cycle. */
export type StreamVerdict =
	| { type: "OK" }
	| {
			type: "EXACT_REPEAT";
			sentenceHash: string;
			count: number;
			/** Character offset in the stream where the first repetition started */
			firstRepeatOffset: number;
	  }
	| {
			type: "OSCILLATION";
			period: number;
			span: number;
			/** Character offset where the oscillation pattern began */
			patternStartOffset: number;
	  };

/** Severity mirrors the step-based detector. */
export type Severity = "warn" | "hard" | "none";

/** Configuration for the streaming detector. */
export interface StreamConfig {
	/** Number of recent sentence hashes examined. */
	window: number;
	/** Sentence repeat count to flag at warn level. Higher than tool-call defaults
	 * because thinking naturally revisits ideas. */
	tRepeatWarn: number;
	/** Sentence repeat count to flag at hard-stop level. */
	tRepeatHard: number;
	/** Maximum oscillation period (in sentences) to check. */
	pMax: number;
	/** Minimum repetitions for period-1 oscillation. */
	minRepsP1: number;
	/** Minimum repetitions for period >= 2 oscillation. */
	minRepsP2: number;
	/** No flags until N sentences have accumulated. */
	warmupSentences: number;
	/** When true, detection runs but onCycle is never called. Stats still accumulate. */
	shadow: boolean;
}

/**
 * Conservative defaults — thinking text revisits ideas naturally.
 * Require strong evidence before flagging.
 */
export const DEFAULT_STREAM_CONFIG: StreamConfig = {
	window: 30,
	tRepeatWarn: 5,
	tRepeatHard: 8,
	pMax: 6,
	minRepsP1: 5,
	minRepsP2: 3,
	warmupSentences: 3,
	shadow: false, // Start in active mode
};

/** Accumulated statistics for shadow-mode tuning. */
export interface StreamStats {
	/** Total characters fed. */
	charactersFed: number;
	/** Total sentences extracted. */
	sentencesExtracted: number;
	/** Number of times a warn-level cycle was detected. */
	warnDetections: number;
	/** Number of times a hard-level cycle was detected. */
	hardDetections: number;
	/** Longest repeat run observed (sentence count). */
	maxRepeatRun: number;
	/** Shortest period oscillation observed (null if none). */
	minOscillationPeriod: number | null;
	/** Last verdict (always computed, even in shadow mode). */
	lastVerdict: StreamVerdict;
	/** Last severity. */
	lastSeverity: Severity;
}

/** Callback fired when a cycle crosses a threshold (active mode only). */
export type OnCycleFn = (verdict: StreamVerdict, severity: Severity) => void;

// ── Sentence Splitter ──────────────────────────────────────────

/**
 * Split text into sentences. Splits on .!? followed by whitespace or EOF.
 * Keeps the delimiter attached to the sentence for hashing consistency.
 *
 * Known limitations: abbreviations ("e.g.", "Dr."), decimal numbers ("3.14"),
 * and ellipsis ("...") may cause false splits. This is acceptable — false
 * splits increase sensitivity (fewer false negatives), and higher thresholds
 * compensate for false positives.
 */
function splitSentences(text: string): string[] {
	// Match: one or more non-delimiter chars, then delimiter(s), then space/EOF
	const regex = /[^.!?]+[.!?]+(?:\s|$)/g;
	const result: string[] = [];
	let match: RegExpExecArray | null;

	while ((match = regex.exec(text)) !== null) {
		const sentence = match[0].trim();
		if (sentence.length > 0) {
			result.push(sentence);
		}
	}
	return result;
}

// ── Hashing ────────────────────────────────────────────────────

/**
 * Hash a sentence for cycle comparison. Normalizes whitespace for
 * robustness against minor rephrasing differences.
 */
function hashSentence(sentence: string): string {
	const normalized = sentence.replace(/\s+/g, " ").trim().toLowerCase();
	return createHash("sha256").update(normalized).digest("hex").slice(0, 16);
}

// ── Stream Detector ────────────────────────────────────────────

/**
 * Incremental cycle detector over a character stream.
 *
 * Usage:
 *   const detector = new StreamDetector(config);
 *   for (const delta of stream) {
 *     const result = detector.feed(delta);
 *     if (result.severity === "hard") { /* abort *\/ }
 *   }
 *   detector.reset(); // Before next message
 */
export class StreamDetector {
	private config: StreamConfig;
	private onCycle: OnCycleFn;

	// Accumulated partial text (incomplete sentence at the boundary)
	private pendingText: string = "";

	// Sliding window of sentence hashes
	private hashes: string[] = [];

	// Character offsets for each sentence (for reporting cycle boundaries)
	private offsets: number[] = [];

	// Total characters fed (for offset tracking)
	private totalChars: number = 0;

	// Statistics
	private stats: StreamStats = {
		charactersFed: 0,
		sentencesExtracted: 0,
		warnDetections: 0,
		hardDetections: 0,
		maxRepeatRun: 0,
		minOscillationPeriod: null,
		lastVerdict: { type: "OK" },
		lastSeverity: "none",
	};

	constructor(
		config: StreamConfig = DEFAULT_STREAM_CONFIG,
		onCycle?: OnCycleFn,
	) {
		this.config = config;
		this.onCycle = onCycle ?? (() => {});
	}

	/**
	 * Update configuration. Takes effect immediately.
	 */
	setConfig(config: Partial<StreamConfig>): void {
		this.config = { ...this.config, ...config };
	}

	/**
	 * Feed a chunk of text (e.g., one thinking_delta or text_delta).
	 * Returns the current verdict and severity.
	 *
	 * In shadow mode, onCycle is never called but verdict is still computed.
	 */
	feed(chunk: string): { verdict: StreamVerdict; severity: Severity } {
		this.stats.charactersFed += chunk.length;
		this.totalChars += chunk.length;

		// Append to pending text
		this.pendingText += chunk;

		// Extract complete sentences
		const sentences = splitSentences(this.pendingText);

		if (sentences.length === 0) {
			// No complete sentences yet — return current state
			return {
				verdict: this.stats.lastVerdict,
				severity: this.stats.lastSeverity,
			};
		}

		// Keep the remainder (incomplete sentence at the boundary).
		// Strip matched sentences from the front to find what's left.
		let remainder = this.pendingText;
		for (const s of sentences) {
			const idx = remainder.indexOf(s);
			if (idx === -1) break;
			remainder = remainder.slice(idx + s.length);
		}
		this.pendingText = remainder.trimStart() || "";

		// Process new sentences
		for (const sentence of sentences) {
			const hash = hashSentence(sentence);
			const offset =
				this.totalChars - this.pendingText.length - sentence.length;

			this.hashes.push(hash);
			this.offsets.push(offset);
			this.stats.sentencesExtracted++;

			// Maintain window size
			if (this.hashes.length > this.config.window) {
				this.hashes.shift();
				this.offsets.shift();
			}
		}

		// Run detection on the current window
		const result = this.runDetection();
		this.stats.lastVerdict = result.verdict;
		this.stats.lastSeverity = result.severity;

		// Update stats regardless of mode
		if (result.severity === "warn") this.stats.warnDetections++;
		if (result.severity === "hard") this.stats.hardDetections++;

		// Track max repeat run
		const repeatRun = this.computeMaxRepeatRun();
		if (repeatRun > this.stats.maxRepeatRun) {
			this.stats.maxRepeatRun = repeatRun;
		}

		// Track min oscillation period
		const minRepsFn = (p: number) =>
			p === 1 ? this.config.minRepsP1 : this.config.minRepsP2;
		const oscPeriod = cyclePeriod(this.hashes, this.config.pMax, minRepsFn);
		if (oscPeriod !== null) {
			if (
				this.stats.minOscillationPeriod === null ||
				oscPeriod < this.stats.minOscillationPeriod
			) {
				this.stats.minOscillationPeriod = oscPeriod;
			}
		}

		// Fire callback only in active mode
		if (!this.config.shadow && result.severity !== "none") {
			this.onCycle(result.verdict, result.severity);
		}

		return result;
	}

	/**
	 * Reset state for a new message/turn. Preserves statistics.
	 */
	reset(): void {
		this.pendingText = "";
		this.hashes = [];
		this.offsets = [];
		this.totalChars = 0;
		this.stats.lastVerdict = { type: "OK" };
		this.stats.lastSeverity = "none";
	}

	/**
	 * Reset everything including statistics.
	 */
	resetFull(): void {
		this.reset();
		this.stats = {
			charactersFed: 0,
			sentencesExtracted: 0,
			warnDetections: 0,
			hardDetections: 0,
			maxRepeatRun: 0,
			minOscillationPeriod: null,
			lastVerdict: { type: "OK" },
			lastSeverity: "none",
		};
	}

	/**
	 * Get accumulated statistics. Safe to call at any time.
	 */
	getStats(): StreamStats {
		return { ...this.stats };
	}

	/**
	 * Get the current pending (incomplete) text.
	 */
	getPending(): string {
		return this.pendingText;
	}

	// ── Detection Logic ────────────────────────────────────────

	private runDetection(): { verdict: StreamVerdict; severity: Severity } {
		// Warmup: no flags until enough sentences accumulated
		if (this.hashes.length < this.config.warmupSentences) {
			return { verdict: { type: "OK" }, severity: "none" };
		}

		// Detector 1: Exact repeat (same sentence hash repeated)
		const hardRepeat = this.detectExactRepeat(this.config.tRepeatHard);
		if (hardRepeat) {
			return {
				verdict: {
					type: "EXACT_REPEAT",
					sentenceHash: hardRepeat.hash,
					count: hardRepeat.count,
					firstRepeatOffset: hardRepeat.firstOffset,
				},
				severity: "hard",
			};
		}

		const warnRepeat = this.detectExactRepeat(this.config.tRepeatWarn);
		if (warnRepeat) {
			return {
				verdict: {
					type: "EXACT_REPEAT",
					sentenceHash: warnRepeat.hash,
					count: warnRepeat.count,
					firstRepeatOffset: warnRepeat.firstOffset,
				},
				severity: "warn",
			};
		}

		// Detector 2: Oscillation (repeating pattern of sentences)
		const minRepsFn = (p: number) =>
			p === 1 ? this.config.minRepsP1 : this.config.minRepsP2;
		const period = cyclePeriod(this.hashes, this.config.pMax, minRepsFn);

		if (period !== null) {
			const span = period * minRepsFn(period);
			const patternStartOffset =
				this.offsets[Math.max(0, this.offsets.length - span)];

			// Check hard threshold (double repetitions)
			const hardMinRepsFn = (p: number) =>
				p === 1 ? this.config.minRepsP1 * 2 : this.config.minRepsP2 * 2;
			const hardPeriod = cyclePeriod(
				this.hashes,
				this.config.pMax,
				hardMinRepsFn,
			);

			return {
				verdict: {
					type: "OSCILLATION",
					period,
					span,
					patternStartOffset,
				},
				severity: hardPeriod !== null ? "hard" : "warn",
			};
		}

		return { verdict: { type: "OK" }, severity: "none" };
	}

	private detectExactRepeat(
		threshold: number,
	): { hash: string; count: number; firstOffset: number } | null {
		const byHash = new Map<string, { indices: number[] }>();

		for (let i = 0; i < this.hashes.length; i++) {
			const entry = byHash.get(this.hashes[i]) ?? { indices: [] };
			entry.indices.push(i);
			byHash.set(this.hashes[i], entry);
		}

		for (const [hash, entry] of byHash) {
			if (entry.indices.length >= threshold) {
				const firstIdx = entry.indices[0];
				return {
					hash,
					count: entry.indices.length,
					firstOffset: this.offsets[firstIdx],
				};
			}
		}
		return null;
	}

	private computeMaxRepeatRun(): number {
		if (this.hashes.length === 0) return 0;
		let maxRun = 1;
		let currentRun = 1;
		for (let i = 1; i < this.hashes.length; i++) {
			if (this.hashes[i] === this.hashes[i - 1]) {
				currentRun++;
				if (currentRun > maxRun) maxRun = currentRun;
			} else {
				currentRun = 1;
			}
		}
		return maxRun;
	}
}

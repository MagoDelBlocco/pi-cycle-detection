/**
 * Canonicalization module — reduces tool arguments to a deterministic,
 * hashable string. Strips volatile fields, sorts keys, normalizes whitespace.
 *
 * This is the #1 defense against a detector that silently never fires.
 */

import { createHash } from "node:crypto";

// Patterns that identify volatile fields — strip them before hashing.
const VOLATILE_KEY_PATTERNS = [
	/\btimestamp\b/i,
	/\bdate\b/i,
	/\buuid\b/i,
	/\brequest[_-]?id\b/i,
	/\breq[_-]?id\b/i,
	/\btrace[_-]?id\b/i,
	/\bcorrelation[_-]?id\b/i,
	/\btemp[_-]?path\b/i,
	/\btmp[_-]?path\b/i,
	/\bpid\b/i,
	/\bport\b/i,
	/\bport[_-]?number\b/i,
	/\brandom\b/i,
	/\bnonce\b/i,
	/\bsession[_-]?id\b/i,
	/\btoken\b/i,
	/\bexpires\b/i,
	/\battempt\b/i,
	/\bretry\b/i,
	/\biteration\b/i,
];

/**
 * Recursively strip volatile keys from an object and sort remaining keys.
 * Tracks visited objects to prevent infinite recursion on circular references.
 */
function canonicalizeValue(
	value: unknown,
	visited: WeakSet<object> = new WeakSet(),
): unknown {
	if (value === null || value === undefined) return null;
	if (typeof value === "string") {
		// Normalize whitespace: collapse runs of whitespace to single space, trim.
		return value.replace(/\s+/g, " ").trim();
	}
	if (typeof value === "number" || typeof value === "boolean") return value;
	if (Array.isArray(value))
		return value.map((v) => canonicalizeValue(v, visited));
	if (typeof value === "object") {
		// Circular reference guard: if already visited, return a sentinel.
		if (visited.has(value)) return "[circular]";
		visited.add(value);
		const entries = Object.entries(value as Record<string, unknown>);
		const filtered: [string, unknown][] = entries.filter(([key]) => {
			return !VOLATILE_KEY_PATTERNS.some((pat) => pat.test(key));
		});
		// Sort keys for determinism
		filtered.sort((a, b) => a[0].localeCompare(b[0]));
		const result: Record<string, unknown> = {};
		for (const [k, v] of filtered) {
			result[k] = canonicalizeValue(v, visited);
		}
		return result;
	}
	// For any other type (bigint, symbol, etc.), coerce to string
	return String(value);
}

/**
 * Canonicalize tool arguments into a deterministic string, then hash.
 *
 * @param args - The raw tool arguments (any JSON-serializable value)
 * @returns A short hex hash of the canonicalized form
 */
export function canonicalizeArgs(args: unknown): string {
	const canonical = canonicalizeValue(args);
	const serialized = JSON.stringify(canonical);
	return createHash("sha256").update(serialized).digest("hex").slice(0, 16);
}

/**
 * Hash a raw observation string (stdout/stderr/tool output).
 *
 * @param observation - The raw observation text
 * @returns A short hex hash
 */
export function hashObservation(observation: string): string {
	return createHash("sha256").update(observation).digest("hex").slice(0, 16);
}

/**
 * Hash an outcome state proxy (e.g., working-dir diff, failing-test set).
 *
 * @param state - The raw state string
 * @returns A short hex hash
 */
export function hashOutcomeState(state: string): string {
	return createHash("sha256").update(state).digest("hex").slice(0, 16);
}

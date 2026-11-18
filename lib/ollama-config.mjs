/**
 * ollama-config.mjs — settings and decisions for the local Ollama runner.
 *
 * WHY THIS IS A SEPARATE, PURE MODULE
 * -----------------------------------
 * Everything here is a function of its arguments: no network, no file access,
 * no npm package, no exit. Three callers need the same answers — the runner,
 * the doctor check and the tests — and the doctor check is loaded by
 * doctor.mjs, the script that reports an incomplete install. A module that
 * reaches an npm package would make that check need a complete one.
 *
 * WHERE THE DEFAULTS COME FROM
 * ----------------------------
 * They are measurements, not guesses (Ollama 0.40.16, one 32 GB GPU, the real
 * evaluation prompt built from modes/ plus a CV and a profile):
 *   - The prompt is 38.8-40.7K tokens. At num_ctx 32768 Ollama cut it to 16.4K
 *     without an error, the run still succeeded, and the same job scored
 *     differently once the whole prompt fit. Hence 65536, and hence
 *     fitContext(), which refuses a window the prompt does not fit instead of
 *     sending it.
 *   - A full report is 8.2-10.8K output tokens; a 6000-token cap truncated it.
 *     Thinking tokens count against the same cap, so a thinking model needs
 *     more room (numPredictThinking).
 *   - The models ship temperature 1.0. An evaluation should not change from
 *     one run to the next, so 0.2 is sent on every request.
 *   - The server unloads an idle model after 5 minutes, and that is left to
 *     the server: keep_alive is sent only when OLLAMA_KEEP_ALIVE is set.
 *     Asking for 30 minutes on every request, as this module once did, held
 *     18-24 GB of a desktop GPU for half an hour after each command to save a
 *     reload measured at 13 s. OLLAMA_KEEP_ALIVE is also the server's own
 *     variable, so this way one value means one thing to both.
 *   - Thinking is off unless it is asked for. See resolveThink().
 */

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

export const DEFAULTS = Object.freeze({
  baseUrl: 'http://localhost:11434',
  numCtx: 65536,
  numPredict: 16384,
  numPredictThinking: 28000,
  think: 'auto',
  // No keepAlive on purpose: nothing is sent unless OLLAMA_KEEP_ALIVE is set
  // (see WHERE THE DEFAULTS COME FROM, at the top of this file).
  temperature: 0.2,
  timeoutMs: 900000,
  retries: 1,
});

/**
 * Context sizes tried, in order, when OLLAMA_NUM_CTX is not set and the prompt
 * does not fit. A short fixed ladder rather than "exactly what is needed":
 * every distinct num_ctx is a full model reload, so runs should land on the
 * same few values. It stops at 131072 because the KV cache grows with the
 * window (the default model takes 20389 MiB of GPU memory at 65536 and 24549
 * MiB at 131072); anything larger is the user's explicit choice.
 */
export const NUM_CTX_STEPS = Object.freeze([65536, 98304, 131072]);

/**
 * Output room a context window must leave after the prompt when thinking is
 * off. Not num_predict itself: that is a ceiling, and reserving all of it
 * would refuse windows that comfortably hold a real 8-11K report. With
 * thinking on the whole ceiling is reserved instead (see fitContext): the
 * reasoning comes out of the same room and has been measured filling it.
 */
export const OUTPUT_RESERVE_TOKENS = 12000;

/**
 * Tried in this order when no model is named. The first is the only one that
 * applied the hard-stop logic and produced a complete, well-formed report in
 * the probe. The rest are fallbacks. gemma4 is faster: with thinking off it
 * wrote a complete report in 19-26 s three times out of three, each one
 * without the SCORE_SUMMARY block, and with thinking on it spent its whole
 * output budget thinking in two runs of three. glm missed the blocker in its
 * one run, and qwen3-coder was not probed.
 */
export const DEFAULT_MODEL_PREFERENCE = Object.freeze([
  'qwen3.8:27b',
  'gemma4:26b-a4b-it-q4_K_M',
  'glm-4.7-flash:latest',
  'qwen3-coder:30b',
]);

/**
 * The variables that are this backend's own settings. Their presence, in the
 * shell or in a `.env` the runner reads, is what says "this install uses the
 * runner", which is the question `node doctor.mjs` asks before it contacts a
 * server nobody asked it to check.
 *
 * OLLAMA_HOST and OLLAMA_KEEP_ALIVE are left out on purpose. The runner
 * honours both, but they are the Ollama server's own variables, as is every
 * other OLLAMA_* name (OLLAMA_MODELS, OLLAMA_NUM_PARALLEL, OLLAMA_ORIGINS …):
 * someone who tunes their server through the environment has not thereby
 * configured the runner.
 */
export const RUNNER_ENV_VARS = Object.freeze([
  'OLLAMA_BASE_URL',
  'OLLAMA_ALLOW_REMOTE',
  'OLLAMA_MODEL',
  'OLLAMA_MODEL_PREFERENCE',
  'OLLAMA_NUM_CTX',
  'OLLAMA_NUM_PREDICT',
  'OLLAMA_THINK',
  'OLLAMA_TEMPERATURE',
  'OLLAMA_TIMEOUT_MS',
  'OLLAMA_RETRIES',
  'OLLAMA_SEED',
]);

/**
 * Prompt size assumed when there is nothing to measure: setup is not finished
 * and there is no CV yet. It is about what an evaluation built from modes/
 * plus the example CV and profile is counted as, with room for the longest
 * posting: 44.2K by evaluationPromptTokens() on 2026-11-17. The runner and
 * both doctors measure the owner's own files once they exist, so this number
 * never decides a verdict about a real setup.
 */
export const TYPICAL_PROMPT_TOKENS = 44_000;

/**
 * What the fixed text of an evaluation prompt is counted as: the lines around
 * each file in the system prompt, the operating rules after them and the
 * wording of the user message around the posting. Measured on 2026-11-17 as
 * the count of the real prompt less its files and its posting: 850 tokens with
 * all six files, 811 without modes/_custom.md.
 *
 * Deliberately above that. The window of a run is sized with this allowance
 * before the first posting is read; a prompt that then comes out larger moves
 * the run to the next window size in the middle of it, which is a model
 * reload. A few hundred tokens over costs a slightly earlier step to the next
 * size. The room is for the prompt's wording to grow: it grew by 212 tokens on
 * the day this was written. tests/ollama-runner.test.mjs fails when the real
 * prompt passes this allowance.
 */
const PROMPT_FRAME_TOKENS = 1_200;

/** The longest posting the page reader hands the runner: 16,000 characters. */
const LONGEST_POSTING_TOKENS = 4_000;

/** Names of embedding models, which `ollama list` shows beside the chat models. */
export const EMBEDDING_NAME_RE =/(^|[/:-])(bge|nomic-embed|all-minilm|mxbai|snowflake-arctic-embed|embed)/i;

/** AbortSignal.timeout() fires a longer delay after 1 ms, and throws past 2^32-1. */
const MAX_TIMEOUT_MS = 2_147_483_647;
const MAX_RETRIES = 3;
const DEFAULT_PORT = '11434';

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** A setting that cannot be used: a bad number, a bad URL, a remote host. */
export class OllamaConfigError extends Error {
  constructor(message) {
    super(message);
    this.name = 'OllamaConfigError';
    this.code = 'OLLAMA_CONFIG';
  }
}

/**
 * No usable model. `installed` is the chat models the server has, `suggestion`
 * the one to set OLLAMA_MODEL to ('' when the server has none).
 */
export class OllamaModelError extends Error {
  constructor(message, { installed = [], suggestion = '' } = {}) {
    super(message);
    this.name = 'OllamaModelError';
    this.code = 'OLLAMA_MODEL';
    this.installed = installed;
    this.suggestion = suggestion;
  }
}

// ---------------------------------------------------------------------------
// Endpoint
// ---------------------------------------------------------------------------

const hasScheme = (text) => /^[a-z][a-z0-9+.-]*:\/\//i.test(text);
const isUnset = (raw) => raw === undefined || raw === null || String(raw).trim() === '';

/** What stands in a message for an address that is not repeated. */
const WITHHELD = '(not shown: it may hold a password)';

/**
 * A rejected address as it may be repeated in a message, or null when none of
 * it may be.
 *
 * An address can hold a secret in two places: `user:password@` ahead of the
 * host, and the query string. These values are the ones that did NOT parse as
 * expected, so neither part can be picked out of them with any confidence.
 * Anything with an `@` in it is therefore not repeated at all, and a query or
 * fragment is always cut off. Error text ends up in terminals, in the JSON
 * receipt and in whatever the user pastes into an issue.
 *
 * @param {string} text
 * @returns {string|null}
 */
function repeatable(text) {
  if (text.includes('@')) return null;
  const cut = text.search(/[?#]/);
  return cut === -1 ? text : `${text.slice(0, cut)}…`;
}

/**
 * Normalise a client URL: scheme added when missing, trailing slashes and a
 * trailing /v1 removed.
 *
 * /v1 is the OpenAI-compatible route, which people paste from other tools.
 * This runner talks to the native API under the same host, and the native API
 * is the only one that honours num_ctx and think.
 *
 * @param {string} raw
 * @param {string} label What the value came from, for the error.
 * @returns {string}
 */
function normalizeBaseUrl(raw, label) {
  const text = String(raw).trim();
  const invalid = () => {
    const shown = repeatable(text);
    return new OllamaConfigError(
      `Invalid ${label}: ${shown === null ? WITHHELD : `"${shown}"`} — must be an http(s) URL such as ${DEFAULTS.baseUrl}.`,
    );
  };
  let url;
  try {
    url = new URL(hasScheme(text) ? text : `http://${text}`);
  } catch {
    throw invalid();
  }
  // The value is not echoed here: it holds a password.
  if (url.username || url.password) {
    throw new OllamaConfigError(`Invalid ${label}: a URL with credentials in it is not supported.`);
  }
  if ((url.protocol !== 'http:' && url.protocol !== 'https:') || !url.hostname || url.search || url.hash) {
    throw invalid();
  }
  const path = url.pathname.replace(/\/+$/, '').replace(/\/v1$/i, '').replace(/\/+$/, '');
  return `${url.protocol}//${url.host}${path}`;
}

/**
 * Turn an OLLAMA_HOST value into a URL a client can connect to, or null.
 *
 * OLLAMA_HOST is the SERVER's bind address. `0.0.0.0` and `::` mean "listen on
 * every interface" and are not destinations (connecting to 0.0.0.0 is
 * unreliable on Windows), and a value may name no host at all (`:8080`). What
 * such a value still says is the port, and the server it configures is on
 * this machine. So with a port the answer is localhost on that port, which is
 * what Ollama's own client does with the variable; dropping the port sent
 * every request to whatever listens on 11434. Without a port there is nothing
 * in the value to use, and the result is null.
 *
 * The defaults follow the server's own parser: no scheme means http and port
 * 11434; an explicit scheme keeps that scheme's own default port.
 *
 * @param {string} raw
 * @returns {string|null}
 */
function clientUrlFromBindAddress(raw) {
  const text = String(raw).trim();
  const explicitScheme = hasScheme(text);
  let rest = explicitScheme ? text.slice(text.indexOf('://') + 3) : text;
  const scheme = explicitScheme ? text.slice(0, text.indexOf('://')).toLowerCase() : 'http';
  if (scheme !== 'http' && scheme !== 'https') return null;
  rest = rest.split('/')[0];
  // A bare IPv6 address (`::1`, `::`) has no brackets to tell host from port.
  if (!rest.startsWith('[') && (rest.match(/:/g) ?? []).length >= 2) rest = `[${rest}]`;
  // The URL parser refuses `:8080`. To the server it means every interface,
  // so it is parsed with that address written out.
  const hostless = rest.startsWith(':');
  let url;
  try {
    url = new URL(`${scheme}://${hostless ? '0.0.0.0' : ''}${rest}`);
  } catch {
    return null;
  }
  if (!url.hostname) return null;
  // The parser drops a port that is its scheme's own default (`:80` under
  // http), so whether one was written is read from the text. Taking the empty
  // `url.port` for "no port" turned `host:80` into `host:11434`.
  const portWritten = /:\d+$/.test(rest.slice(rest.lastIndexOf(']') + 1));
  const port = portWritten || explicitScheme ? url.port : DEFAULT_PORT;
  const everyInterface = url.hostname === '0.0.0.0' || url.hostname === '[::]';
  // Port 0 asks the system for any free port: it names none to connect to.
  if (everyInterface && (!portWritten || url.port === '0')) return null;
  return `${scheme}://${everyInterface ? 'localhost' : url.hostname}${port ? `:${port}` : ''}`;
}

/**
 * Decide which server to talk to.
 *
 * Order: `--url`, then OLLAMA_BASE_URL, then OLLAMA_HOST when it names a host
 * or a port to connect to, then localhost. `ignoredHost` is set when
 * OLLAMA_HOST was present and gave neither (`0.0.0.0`, `::`), so a caller can
 * say why it was not used.
 *
 * @param {Record<string, string|undefined>} [env]
 * @param {string} [flagUrl] The `--url` value, when given.
 * @returns {{ url: string, source: 'flag'|'OLLAMA_BASE_URL'|'OLLAMA_HOST'|'default', ignoredHost?: string }}
 * @throws {OllamaConfigError} When the chosen value is not an http(s) URL.
 */
export function resolveBaseUrl(env = process.env, flagUrl) {
  if (!isUnset(flagUrl)) return { url: normalizeBaseUrl(flagUrl, '--url'), source: 'flag' };
  if (!isUnset(env?.OLLAMA_BASE_URL)) {
    return { url: normalizeBaseUrl(env.OLLAMA_BASE_URL, 'OLLAMA_BASE_URL'), source: 'OLLAMA_BASE_URL' };
  }
  if (!isUnset(env?.OLLAMA_HOST)) {
    const fromHost = clientUrlFromBindAddress(env.OLLAMA_HOST);
    if (fromHost) return { url: fromHost, source: 'OLLAMA_HOST' };
    // Callers put this into a sentence, so it gets the same care as a
    // rejected URL.
    return { url: DEFAULTS.baseUrl, source: 'default', ignoredHost: repeatable(String(env.OLLAMA_HOST).trim()) ?? WITHHELD };
  }
  return { url: DEFAULTS.baseUrl, source: 'default' };
}

/**
 * Is this URL on the local machine?
 *
 * `new URL('http://[::1]:11434').hostname` is `[::1]`, brackets included, so a
 * comparison against `::1` alone treats IPv6 loopback as a remote host. Both
 * spellings are listed.
 *
 * @param {string} url
 * @returns {boolean}
 */
export function isLoopbackUrl(url) {
  let hostname;
  try {
    hostname = new URL(String(url)).hostname.toLowerCase();
  } catch {
    return false;
  }
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1' || hostname === '[::1]';
}

/**
 * Refuse a server on another machine unless the user asked for one.
 *
 * Every request carries the CV and the job description. A typo in
 * OLLAMA_BASE_URL, or a value left over from another tool, would send them to
 * a host the user never chose, with nothing on screen to show it.
 *
 * @param {string} url
 * @param {Record<string, string|undefined>} [env]
 * @returns {void}
 * @throws {OllamaConfigError}
 */
export function assertEndpointAllowed(url, env = process.env) {
  if (isLoopbackUrl(url)) return;
  if (String(env?.OLLAMA_ALLOW_REMOTE ?? '').trim() === '1') return;
  throw new OllamaConfigError(
    `Remote Ollama endpoint: ${url}. Your CV and the job description would be sent to that host. `
    + 'Set OLLAMA_ALLOW_REMOTE=1 if that is what you want, or point OLLAMA_BASE_URL at localhost.',
  );
}

// ---------------------------------------------------------------------------
// Value parsing
// ---------------------------------------------------------------------------

/**
 * Read a positive integer setting.
 *
 * Digits only. parseInt read "32k" as 32, a 32-token context, and "5m" as a
 * 5 ms timeout; both ran, and failed somewhere far from the typo.
 *
 * @param {string} name The variable or flag, for the error.
 * @param {unknown} raw Unset, empty and whitespace all mean "use the default".
 * @param {number} def
 * @param {number} [max]
 * @returns {number}
 * @throws {OllamaConfigError}
 */
export function parsePositiveInt(name, raw, def, max) {
  if (isUnset(raw)) return def;
  const text = String(raw).trim();
  const value = /^\d+$/.test(text) ? Number(text) : NaN;
  if (!Number.isSafeInteger(value) || value <= 0 || (max !== undefined && value > max)) {
    const bound = max !== undefined ? ` of at most ${max}` : '';
    throw new OllamaConfigError(`Invalid ${name}: "${text}" — must be a positive integer${bound}.`);
  }
  return value;
}

/** As parsePositiveInt, for the settings where 0 is a real value (retries, seed). */
function parseWholeNumber(name, raw, def, max) {
  if (isUnset(raw)) return def;
  const text = String(raw).trim();
  const value = /^\d+$/.test(text) ? Number(text) : NaN;
  if (!Number.isSafeInteger(value) || (max !== undefined && value > max)) {
    const range = max !== undefined ? `an integer from 0 to ${max}` : 'a non-negative integer';
    throw new OllamaConfigError(`Invalid ${name}: "${text}" — must be ${range}.`);
  }
  return value;
}

function parseTemperature(name, raw, def) {
  if (isUnset(raw)) return def;
  const text = String(raw).trim();
  const value = /^(\d+(\.\d*)?|\.\d+)$/.test(text) ? Number(text) : NaN;
  if (!Number.isFinite(value) || value > 2) {
    throw new OllamaConfigError(`Invalid ${name}: "${text}" — must be a number from 0 to 2.`);
  }
  return value;
}

/**
 * The server reads keep_alive as a JSON number of seconds or as a duration
 * string WITH a unit. "300" as a string is neither: Ollama 0.40.16 answers it,
 * and "-1" (the usual way to say "never unload"), with a 400 `missing unit in
 * duration`. An environment variable is always a string, so a bare number is
 * passed on as a number.
 *
 * Unset is `undefined`, not a default: no keep_alive is sent then, and the
 * server's own setting decides.
 */
function parseKeepAlive(name, raw) {
  if (isUnset(raw)) return undefined;
  const text = String(raw).trim();
  if (/^-?\d+(\.\d+)?$/.test(text)) return Number(text);
  if (/^-?(\d+(\.\d+)?(ns|us|ms|s|m|h))+$/.test(text)) return text;
  throw new OllamaConfigError(`Invalid ${name}: "${text}" — must be a duration such as 30m or 1h, or a number of seconds.`);
}

/**
 * @param {string} name
 * @param {unknown} raw
 * @returns {'auto'|'on'|'off'}
 */
function parseThink(name, raw) {
  if (isUnset(raw)) return DEFAULTS.think;
  const text = String(raw).trim().toLowerCase();
  if (text === 'auto') return 'auto';
  if (text === 'on' || text === 'true' || text === '1' || text === 'yes') return 'on';
  if (text === 'off' || text === 'false' || text === '0' || text === 'no') return 'off';
  throw new OllamaConfigError(`Invalid ${name}: "${String(raw).trim()}" — must be auto, on or off.`);
}

/** The first value that is set, with the name it came under. */
function firstSet(candidates) {
  for (const [name, value] of candidates) {
    if (!isUnset(value)) return { name, value };
  }
  return { name: candidates[0][0], value: undefined };
}

/**
 * Read every Ollama setting, validated, in one place.
 *
 * Precedence is flag, then environment, then default. The environment is the
 * caller's: a script loads its .env files into process.env first (dotenv never
 * overrides the shell), or passes a merged object.
 *
 * The endpoint is checked here as well as by assertEndpointAllowed(), so no
 * caller can hold a configuration that points at another machine without
 * OLLAMA_ALLOW_REMOTE=1.
 *
 * `numCtxExplicit` and `numPredictExplicit` say whether the user chose the
 * value. A chosen num_ctx is never raised (fitContext), and a chosen
 * num_predict is never replaced by the larger thinking budget
 * (resolveNumPredict).
 *
 * `keepAlive` is `undefined` unless OLLAMA_KEEP_ALIVE is set. The key is
 * always there, so a caller can pass it straight on: the client leaves
 * keep_alive out of a request when it is given no value.
 *
 * @param {Record<string, string|undefined>} [env]
 * @param {object} [flags] CLI overrides.
 * @param {string} [flags.url] `--url` (also read as `baseUrl`).
 * @param {string} [flags.model] `--model`.
 * @param {string|number} [flags.numCtx] `--num-ctx` (also read as `num-ctx`, `num_ctx`).
 * @param {string|boolean} [flags.think] `--think=on|off|auto`.
 * @param {string|number} [flags.numPredict]
 * @returns {{
 *   baseUrl: string, baseUrlSource: 'flag'|'OLLAMA_BASE_URL'|'OLLAMA_HOST'|'default',
 *   model: string|null, preference: string[],
 *   numCtx: number, numCtxExplicit: boolean, numPredict: number, numPredictExplicit: boolean,
 *   think: 'auto'|'on'|'off', keepAlive: string|number|undefined, temperature: number,
 *   timeoutMs: number, retries: number, seed: number|undefined, allowRemote: boolean,
 * }}
 * @throws {OllamaConfigError} Naming the setting and the value that is wrong.
 */
export function readOllamaConfig(env = process.env, flags = {}) {
  const e = env ?? {};
  const f = flags ?? {};

  const { url: baseUrl, source: baseUrlSource } = resolveBaseUrl(e, f.url ?? f.baseUrl);
  assertEndpointAllowed(baseUrl, e);

  const model = [f.model, e.OLLAMA_MODEL].find((value) => !isUnset(value));
  const preference = String(e.OLLAMA_MODEL_PREFERENCE ?? '').split(',').map((name) => name.trim()).filter(Boolean);

  const numCtx = firstSet([
    ['--num-ctx', f.numCtx ?? f['num-ctx'] ?? f.num_ctx],
    ['OLLAMA_NUM_CTX', e.OLLAMA_NUM_CTX],
  ]);
  const numPredict = firstSet([
    ['--num-predict', f.numPredict ?? f['num-predict'] ?? f.num_predict],
    ['OLLAMA_NUM_PREDICT', e.OLLAMA_NUM_PREDICT],
  ]);
  const think = firstSet([['--think', f.think], ['OLLAMA_THINK', e.OLLAMA_THINK]]);

  return {
    baseUrl,
    baseUrlSource,
    model: model === undefined ? null : String(model).trim(),
    preference: preference.length > 0 ? preference : [...DEFAULT_MODEL_PREFERENCE],
    numCtx: parsePositiveInt(numCtx.name, numCtx.value, DEFAULTS.numCtx),
    numCtxExplicit: numCtx.value !== undefined,
    numPredict: parsePositiveInt(numPredict.name, numPredict.value, DEFAULTS.numPredict),
    numPredictExplicit: numPredict.value !== undefined,
    think: parseThink(think.name, think.value),
    keepAlive: parseKeepAlive('OLLAMA_KEEP_ALIVE', e.OLLAMA_KEEP_ALIVE),
    temperature: parseTemperature('OLLAMA_TEMPERATURE', e.OLLAMA_TEMPERATURE, DEFAULTS.temperature),
    timeoutMs: parsePositiveInt('OLLAMA_TIMEOUT_MS', e.OLLAMA_TIMEOUT_MS, DEFAULTS.timeoutMs, MAX_TIMEOUT_MS),
    retries: parseWholeNumber('OLLAMA_RETRIES', e.OLLAMA_RETRIES, DEFAULTS.retries, MAX_RETRIES),
    seed: parseWholeNumber('OLLAMA_SEED', e.OLLAMA_SEED, undefined),
    allowRemote: String(e.OLLAMA_ALLOW_REMOTE ?? '').trim() === '1',
  };
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/**
 * Is this an embedding model? They cannot chat, so they are kept out of every
 * list a chat model is picked from or shown in.
 *
 * @param {string} name
 * @param {string} [family] `details.family` from the server, when known. The
 *   BERT families are embedding-only whatever the model is called.
 * @returns {boolean}
 */
export function isEmbeddingModel(name, family) {
  return EMBEDDING_NAME_RE.test(String(name ?? '')) || /bert/i.test(String(family ?? ''));
}

/** A model name with its implicit tag written out, lowercased for comparison. */
function withTag(name) {
  const text = String(name ?? '').trim().toLowerCase();
  if (!text) return '';
  // Only a colon after the last slash is a tag; `host:5000/model` has a port.
  return text.slice(text.lastIndexOf('/') + 1).includes(':') ? text : `${text}:latest`;
}

/**
 * Do two names mean the same model? The server lists `mistral-small:latest`
 * for what a user writes as `mistral-small`.
 *
 * @param {string} a
 * @param {string} b
 * @returns {boolean}
 */
export function sameModel(a, b) {
  const left = withTag(a);
  return left !== '' && left === withTag(b);
}

/** Names of the installed chat models. Accepts names or listModels() entries. */
function chatModelNames(installed) {
  const names = [];
  for (const entry of Array.isArray(installed) ? installed : []) {
    const name = typeof entry === 'string' ? entry : entry?.name;
    if (typeof name !== 'string' || !name.trim()) continue;
    if (typeof entry === 'object' && entry.isEmbedding === true) continue;
    if (isEmbeddingModel(name, typeof entry === 'object' ? entry.family : undefined)) continue;
    names.push(name.trim());
  }
  return names;
}

/**
 * Choose the model to run.
 *
 * A named model is used or refused; there is no "closest match". The model
 * decides the report. The old evaluator's default was a model this machine had
 * never pulled, and the first sign of it was a bare 404 after the whole prompt
 * had been built. Nothing here guesses: when neither the request nor the
 * preference list matches, the error lists what is installed and the user
 * picks.
 *
 * The returned name is the server's own spelling, so `--model mistral-small`
 * sends `mistral-small:latest`.
 *
 * @param {object} args
 * @param {string|null} [args.requested] `--model` or OLLAMA_MODEL.
 * @param {string[]} [args.preference] Defaults to DEFAULT_MODEL_PREFERENCE.
 * @param {Array<string|{name: string, family?: string, isEmbedding?: boolean}>} args.installed
 * @returns {{ model: string, reason: 'requested'|'preferred' }}
 * @throws {OllamaModelError}
 */
export function pickModel({ requested, preference, installed } = {}) {
  const chatModels = chatModelNames(installed);
  const wantedOrder = Array.isArray(preference) && preference.length > 0 ? preference.map(String) : [...DEFAULT_MODEL_PREFERENCE];
  const preferred = chatModelNames(wantedOrder);
  let firstPreferred;
  for (const name of preferred) {
    firstPreferred = chatModels.find((model) => sameModel(model, name));
    if (firstPreferred) break;
  }
  const shown = chatModels.length > 0 ? chatModels.join(', ') : '(none)';

  const wanted = typeof requested === 'string' ? requested.trim() : '';
  if (wanted) {
    const match = chatModels.find((model) => sameModel(model, wanted));
    if (match) return { model: match, reason: 'requested' };
    const suggestion = firstPreferred ?? chatModels[0] ?? '';
    const fix = suggestion ? `Set OLLAMA_MODEL=${suggestion} or run: ollama pull ${wanted}` : `Run: ollama pull ${wanted}`;
    throw new OllamaModelError(
      `Model "${wanted}" is not installed in Ollama. Installed chat models: ${shown}. ${fix}`,
      { installed: chatModels, suggestion },
    );
  }

  if (firstPreferred) return { model: firstPreferred, reason: 'preferred' };
  const fix = chatModels.length > 0 ? 'Set OLLAMA_MODEL to one of them.' : `Run: ollama pull ${preferred[0] ?? DEFAULT_MODEL_PREFERENCE[0]}`;
  throw new OllamaModelError(
    `None of the preferred models are installed (${wantedOrder.join(', ')}). Installed chat models: ${shown}. ${fix}`,
    { installed: chatModels, suggestion: chatModels[0] ?? '' },
  );
}

// ---------------------------------------------------------------------------
// Thinking and output budget
// ---------------------------------------------------------------------------

/**
 * Decide the `think` value for a request: true, false, or undefined to leave
 * the field out.
 *
 * It is left out for a model without the `thinking` capability: there is
 * nothing for it to switch, and the server may refuse the request over it
 * (`does not support thinking`, which the client retries without the field;
 * that refusal was not exercised against the installed server). For a model
 * that has the capability, leaving the field out means thinking ON, so a
 * value is always sent, and `auto` sends false whatever the model is:
 *   - qwen3.8: with thinking on it spent 5-8.5K tokens reasoning and ran into
 *     a 16000-token cap before the summary.
 *   - gemma4: `auto` used to mean on for it, on the strength of one probe
 *     run. Measured again with the runner's own prompt, thinking on used the
 *     whole output budget for thinking in two runs of three (25,520 tokens,
 *     no reply, 144-150 s) and failed the live run after 189 s. Thinking off
 *     gave a complete report in 19-26 s, three times out of three.
 *   - glm and anything else: off, the cheaper and more predictable setting.
 * Nothing measured supports "on" for any model, so only `--think on` or
 * OLLAMA_THINK=on turns it on.
 *
 * @param {object} args
 * @param {'auto'|'on'|'off'} [args.setting]
 * @param {string} [args.model] Accepted, and no longer consulted: the answer
 *   does not depend on which model it is.
 * @param {string[]} [args.capabilities] From /api/show.
 * @returns {boolean|undefined}
 */
export function resolveThink({ setting, capabilities } = {}) {
  const mode = parseThink('think setting', setting);
  const capable = Array.isArray(capabilities) && capabilities.some((cap) => String(cap).toLowerCase() === 'thinking');
  if (!capable) return undefined;
  return mode === 'on';
}

/**
 * The output cap for a request.
 *
 * Thinking tokens are counted against num_predict, so the default that fits a
 * report does not fit a report plus the reasoning before it. A value the user
 * set is used as given.
 *
 * @param {object} args
 * @param {number} [args.numPredict]
 * @param {boolean} [args.numPredictExplicit]
 * @param {boolean|undefined} [args.think] What resolveThink() returned.
 * @returns {number}
 */
export function resolveNumPredict({ numPredict, numPredictExplicit, think } = {}) {
  const base = Number.isSafeInteger(numPredict) && numPredict > 0 ? numPredict : DEFAULTS.numPredict;
  if (think === true && !numPredictExplicit) return DEFAULTS.numPredictThinking;
  return base;
}

// ---------------------------------------------------------------------------
// Prompt size
// ---------------------------------------------------------------------------

/**
 * Tokens a text is counted as: the rule of estimateTokens() in
 * lib/context-budget.mjs, which is what the runner counts a prompt with when
 * it sends one.
 *
 * Repeated here and not imported. This module imports nothing, and it is
 * loaded by `node doctor.mjs`: a rename in an upstream module that only this
 * one line needs would end the whole setup check with "doctor.mjs failed".
 * tests/ollama-doctor.test.mjs holds the two rules to the same answers.
 */
function countTokens(text) {
  if (typeof text !== 'string' || text.length === 0) return 0;
  return Math.ceil(text.replace(/\s+/g, ' ').length / 4);
}

/**
 * The size of an evaluation prompt built from these texts, before its posting
 * is known: the texts, the prompt's own fixed wording, and the longest posting
 * the page reader returns.
 *
 * One rule for everything that sizes a context window ahead of a posting: a
 * pipeline run, `evaluate <url>` before the page is read, `models`, and the
 * two setup checks (the runner's `doctor` and `node doctor.mjs`). They used to
 * count separately, the runner from the prompt it had built and doctor.mjs
 * from the files, about 1% apart: a window pinned at the number the runner
 * named was then refused by doctor.mjs.
 *
 * Never below the count of the prompt that is sent, for a posting the page
 * reader returned (see PROMPT_FRAME_TOKENS).
 *
 * @param {Array<string|null|undefined>} texts What the prompt is built from:
 *   modes/_shared.md, modes/oferta.md, cv.md, the profile, modes/_profile.md
 *   and modes/_custom.md. Order does not matter; a file that is absent is ''.
 * @returns {number}
 */
export function evaluationPromptTokens(texts = []) {
  let tokens = PROMPT_FRAME_TOKENS + LONGEST_POSTING_TOKENS;
  for (const text of texts) tokens += countTokens(text);
  return tokens;
}

// ---------------------------------------------------------------------------
// Context window
// ---------------------------------------------------------------------------

/**
 * Find a context window the prompt fits in, or say that there is none.
 *
 * Ollama does not reject a prompt that is too long. It drops the part that
 * does not fit and answers from the rest, so the only place the problem can be
 * caught is before the request is sent. `need` is the prompt estimate plus 5%
 * plus room for the answer. The 5% is there because the estimate is chars/4,
 * not a tokenizer: the probed models counted between 4.5% fewer tokens than it
 * predicted and 0.1% more.
 *
 * The room for the answer is OUTPUT_RESERVE_TOKENS at most while thinking is
 * off, since a report is 8-11K tokens. With thinking on it is the whole output
 * cap: the reasoning and the report come out of the same room, and a model has
 * been measured spending 25.5K tokens on reasoning alone. A window that leaves
 * less than the cap ends such a reply at the window, before the report.
 *
 * The window is never larger than the model's own limit. When the user did not
 * choose it, it is raised along NUM_CTX_STEPS until the prompt fits; a chosen
 * value is left alone, and `ok: false` tells the caller to stop.
 *
 * @param {object} args
 * @param {number} [args.numCtx]
 * @param {boolean} [args.explicit] The user set num_ctx; never raise it.
 * @param {number|null} [args.modelMaxCtx] From /api/show; null when unknown.
 * @param {number} [args.promptTokens] Estimated prompt size.
 * @param {number} [args.numPredict] The output cap, as resolveNumPredict() gave it.
 * @param {boolean|undefined} [args.think] What resolveThink() returned. Only
 *   `true` changes anything: the whole cap is then reserved.
 * @returns {{ numCtx: number, ok: boolean, need: number, raised: boolean, clamped: boolean, reason?: string }}
 */
export function fitContext({ numCtx, explicit = false, modelMaxCtx = null, promptTokens = 0, numPredict, think } = {}) {
  const start = Number.isSafeInteger(numCtx) && numCtx > 0 ? numCtx : DEFAULTS.numCtx;
  const prompt = Number.isFinite(promptTokens) && promptTokens > 0 ? Math.round(promptTokens) : 0;
  const output = Number.isSafeInteger(numPredict) && numPredict > 0 ? numPredict : DEFAULTS.numPredict;
  const thinking = think === true;
  const reserve = thinking ? output : Math.min(output, OUTPUT_RESERVE_TOKENS);
  const need = Math.ceil(prompt * 1.05) + reserve;
  const max = Number.isFinite(modelMaxCtx) && modelMaxCtx > 0 ? modelMaxCtx : null;
  const withinModel = (value) => (max !== null && value > max ? max : value);

  let wanted = start;
  if (!explicit && need > withinModel(start)) {
    // Past the last step nothing fits; stopping there keeps the number in the
    // error the largest one this function would ever choose.
    wanted = Math.max(start, NUM_CTX_STEPS.find((step) => step >= need) ?? NUM_CTX_STEPS[NUM_CTX_STEPS.length - 1]);
  }
  const chosen = withinModel(wanted);
  const ok = need <= chosen;
  const result = { numCtx: chosen, ok, need, raised: chosen > start, clamped: chosen < wanted };
  if (!ok) {
    // Said in the reason, because here the other way out is to leave thinking off.
    result.reason = `prompt ~${prompt} tokens + ${reserve} output${thinking ? ' (thinking on)' : ''} needs ${need}; num_ctx is ${chosen}`
      + ` (model max ${max ?? 'unknown'})`;
  }
  return result;
}

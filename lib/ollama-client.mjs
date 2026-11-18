/**
 * ollama-client.mjs — the one module that sends HTTP to an Ollama server.
 *
 * WHY ONE MODULE
 * --------------
 * Each request to a model carries the CV and a job description, so each one
 * has to pass the privacy gate (lib/llm-outbound.mjs) first. Keeping every
 * request in this file means there is one place to check that it does, and one
 * place that knows how Ollama fails.
 *
 * WHAT IT TALKS TO
 * ----------------
 * The native chat endpoint, streamed. Not the OpenAI-compatible /v1 route:
 * that one ignores num_ctx and think:false, so the window is whatever the
 * server's default happens to be and a thinking model thinks regardless.
 * Not the generate or embed endpoints either: on a host or port other than the
 * default, the ambient guard (lib/pii-interceptor.mjs) does not recognise them
 * and a payload sent there would be inspected by nothing but this file.
 * Streaming is not a nicety. Without it the server sends no response headers
 * until the whole reply exists, and Node's fetch gives up after 300 s of that.
 * And only the address it was given: a redirect is reported as a failure and
 * never followed, because the address it leads to was checked by nothing.
 *
 * WHAT IT REFUSES TO RETURN
 * -------------------------
 * A reply is returned only when the stream ended with done:true, the model
 * stopped on its own, there is text in it, and the server processed the whole
 * prompt. Everything else is an OllamaError with a code, because the failures
 * that matter here are quiet ones: a prompt the server truncated, a reply cut
 * at the output cap, a model that only thought. Each of those arrives as HTTP
 * 200 with plausible text.
 *
 * WHAT IT DOES NOT IMPORT
 * -----------------------
 * The privacy gate is loaded on the first chat request, not at the top of this
 * file. The gate reaches js-yaml, and whatever imports this module inherits
 * what it imports at the top. The setup check (lib/ollama-doctor.mjs) is one
 * such importer, and doctor.mjs loads that check: doctor is the script that
 * tells a user to run `npm install`, so nothing it loads may need the install
 * to be complete, and it reads a module that cannot be found at that step as
 * a copy without the backend. The check never makes a chat request, so it
 * never needs the gate. If the gate cannot be loaded, the request is not sent.
 *
 * Nothing here ends the process. On Windows, Node 24.x before 24.20.0 can
 * abort when a process is told to exit after a request (nodejs/node#56645),
 * so failures are thrown and the caller sets an exit code.
 */

import { DEFAULTS, OllamaConfigError, isEmbeddingModel } from './ollama-config.mjs';

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/** Version, model list and loaded-model list: answers that take milliseconds. */
const METADATA_TIMEOUT_MS = 5_000;
/** Model details are read from the model file's header, which can be cold. */
const SHOW_TIMEOUT_MS = 15_000;
/** One decoded NDJSON record, in UTF-16 code units. Not the whole reply. */
const RECORD_LIMIT = 1024 * 1024;
/** How much of a server's error text is repeated to the user. */
const ERROR_TEXT_LIMIT = 300;
/**
 * A prompt counts as truncated when the server processed less than this share
 * of the estimate. The estimate is chars/4 and the probed models came in at
 * 95-100% of it; the one truncated run came in at about 42%.
 */
const TRUNCATION_RATIO = 0.85;
/** AbortSignal.timeout() fires a longer delay after 1 ms. */
const MAX_TIMEOUT_MS = 2_147_483_647;
/** A mistyped retry count must not become minutes of repeated generations. */
const MAX_RETRIES = 10;

const JSON_HEADERS = Object.freeze({ 'Content-Type': 'application/json' });
const THINK_UNSUPPORTED_RE = /does not support thinking/i;
const MODEL_NOT_FOUND_RE = /model .* not found/i;
/** Where lib/pii-interceptor.mjs records the wrapper it put over fetch. */
const INTERCEPTOR = Symbol.for('career-ops.pii-interceptor.installed');
/** Node's fetch ends a connection that stays silent for 300 s with these. */
const SILENT_TIMEOUT_CODES = new Set(['UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT']);
/**
 * The answers fetch would follow to another address. None is followed here.
 * The loopback rule and the privacy gate look at the configured address only,
 * and on a 307 or 308 fetch repeats the POST, CV included, at whatever host
 * the answer names. Ollama itself never redirects, so one of these means the
 * address belongs to something else: a proxy, another program on the port.
 */
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

const CODES = new Set([
  'UNREACHABLE', 'HTTP', 'MODEL_NOT_FOUND', 'TIMEOUT', 'INCOMPLETE',
  'EMPTY', 'TOOL_CALL', 'TRUNCATED_PROMPT', 'STREAM',
]);

/**
 * A request to Ollama that did not produce a usable answer.
 *
 * `code` says which way, so a caller can decide without reading the message:
 *   UNREACHABLE       no connection, or (`midStream: true`) the connection
 *                     dropped after the reply had started
 *   HTTP              a non-2xx answer; `status` has the number. For a redirect,
 *                     which is never followed, `location` is where it pointed
 *                     (no credentials, query or fragment), when it named one
 *   MODEL_NOT_FOUND   the server does not have the model; `status` is 404
 *   TIMEOUT           no complete reply in time
 *   INCOMPLETE        the model stopped for a reason other than being done
 *   EMPTY             no text, or text the caller's validate() rejected
 *   TOOL_CALL         the model called a tool instead of answering
 *   TRUNCATED_PROMPT  the server processed far fewer tokens than were sent
 *   STREAM            the stream was malformed, errored or ended early
 * `partialChars` is how much reply text had arrived when it failed.
 *
 * Written `new OllamaError(code, message, props)`. `new OllamaError(message,
 * props)` is accepted as well, for a caller that sets `code` in `props`.
 */
export class OllamaError extends Error {
  constructor(first, second, third) {
    const coded = CODES.has(first);
    const message = coded ? (typeof second === 'string' ? second : first) : first;
    const props = (coded && typeof second === 'string' ? third : second) ?? {};
    const detail = typeof props === 'object' ? props : {};
    super(String(message ?? ''), detail.cause !== undefined ? { cause: detail.cause } : undefined);
    this.name = 'OllamaError';
    this.code = coded ? first : detail.code;
    if (detail.status !== undefined) this.status = detail.status;
    if (typeof detail.location === 'string' && detail.location) this.location = detail.location;
    if (detail.partialChars !== undefined) this.partialChars = detail.partialChars;
    if (detail.midStream === true) this.midStream = true;
  }
}

/**
 * A refusal from the privacy guard, or a privacy config it could not read.
 * Recognised by code, not by class: the class lives in the module this file
 * loads late. These pass through every handler here untouched — reworded as
 * "not reachable" the user would go looking for a server problem.
 */
function isPrivacyError(err) {
  return err?.code === 'PRIVACY_GUARD_BLOCKED' || err?.code === 'PRIVACY_GUARD_CONFIG';
}

// ---------------------------------------------------------------------------
// Reply text
// ---------------------------------------------------------------------------

/**
 * Remove inline reasoning from a reply.
 *
 * Ollama returns a model's thinking in a separate field for the models it has
 * a parser for, and the client never mixes that into the text. Others
 * (deepseek-r1 on some templates) write it into the answer between think
 * tags, where it would be saved into the report and could carry a draft score
 * summary ahead of the real one.
 *
 * Three shapes are removed: a complete block anywhere; an opening tag at the
 * start with no closing tag, which is a reply cut off while still reasoning,
 * so all of it goes; and text up to a closing tag that has no opening tag,
 * which is what a template that opens the block itself leaves behind.
 *
 * @param {string} text
 * @returns {string} The text without reasoning, trimmed.
 */
export function stripThinkTags(text) {
  if (typeof text !== 'string' || text === '') return '';
  let out = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
  if (/^\s*<think>/i.test(out)) return '';
  const orphan = /<\/think>/i.exec(out);
  if (orphan) out = out.slice(orphan.index + orphan[0].length);
  return out.trim();
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

/** A count the server reported, or null when it is absent or not a count. */
function reportedCount(value) {
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function isOk(res) {
  if (typeof res?.ok === 'boolean') return res.ok;
  const status = res?.status;
  return status === undefined || (status >= 200 && status < 300);
}

/** The body as text. Reads `json()` when a stand-in response has no `text()`. */
async function readText(res) {
  if (typeof res?.text === 'function') return String(await res.text());
  if (typeof res?.json === 'function') return JSON.stringify(await res.json());
  return '';
}

function isAbortTimeout(err, signal) {
  return Boolean(signal?.aborted) || err?.name === 'TimeoutError' || err?.cause?.name === 'TimeoutError';
}

function isSilentTimeout(err) {
  return SILENT_TIMEOUT_CODES.has(err?.code) || SILENT_TIMEOUT_CODES.has(err?.cause?.code);
}

/**
 * A keep-alive a caller actually has: a number of seconds or a duration.
 * Anything else, `undefined` above all, means no keep_alive key is sent and
 * the server's own OLLAMA_KEEP_ALIVE decides how long the model stays loaded.
 */
function isKeepAlive(value) {
  return (typeof value === 'number' && Number.isFinite(value)) || (typeof value === 'string' && value.trim() !== '');
}

/**
 * Where a redirect points, as it may be shown: resolved against the address
 * that was asked, and without credentials, query or fragment. Every character
 * of it was chosen by whatever answered, and those parts can hold a token.
 * Serialising it through URL also leaves nothing a terminal would act on.
 *
 * @param {{headers?: {get?: (name: string) => string|null}}} res
 * @param {string} requestUrl
 * @returns {string} '' when the answer names no address that can be read.
 */
function redirectTarget(res, requestUrl) {
  let location;
  try {
    location = res?.headers?.get?.('location');
  } catch {
    return '';
  }
  if (typeof location !== 'string' || !location.trim()) return '';
  try {
    const target = new URL(location.trim(), requestUrl);
    return `${target.protocol}//${target.host}${target.pathname}`.slice(0, ERROR_TEXT_LIMIT);
  } catch {
    return '';
  }
}

let gateModule;
/** Load the privacy gate once. A failed load is retried on the next request. */
function loadGate() {
  gateModule ??= import('./llm-outbound.mjs').catch((err) => {
    gateModule = undefined;
    throw err;
  });
  return gateModule;
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/**
 * @typedef {object} ChatRequest
 * @property {string} model
 * @property {string} [system] System prompt, sent as one plain string.
 * @property {string} user User message.
 * @property {number} [numCtx] Context window. Sent on every request: without
 *   it the server uses its own default and cuts the prompt to fit.
 * @property {number} [numPredict] Output cap, thinking included.
 * @property {number} [temperature]
 * @property {boolean} [think] From resolveThink(). Left out of the request
 *   when undefined.
 * @property {string|number} [keepAlive] Sent as keep_alive only when given.
 *   Left out, the server's own setting decides (5 minutes unless changed).
 * @property {number} [seed]
 * @property {number} [estimatedPromptTokens] The caller's estimate of the
 *   prompt. When given, a reply whose prompt was truncated is refused.
 * @property {(text: string) => void} [onToken] Called with each piece of the
 *   answer as it arrives.
 * @property {(text: string) => void} [onThinking] Called with each piece of
 *   the model's thinking. Thinking is never part of the returned content.
 *
 * @typedef {object} ChatResult
 * @property {string} content The answer, without reasoning, trimmed.
 * @property {number} thinkingChars Characters of thinking the model produced.
 * @property {{prompt_tokens: number, completion_tokens: number, total_tokens: number, cached_tokens: number}} usage
 *   What the server counted. All zero when it reported no counts.
 *   `cached_tokens` is the share of `prompt_tokens` that came from the
 *   server's cache, not an amount on top of it.
 * @property {string} doneReason
 */

/**
 * Create a client for one Ollama server.
 *
 * @param {object} [options]
 * @param {string} [options.baseUrl] Server root, without /v1 or a trailing slash.
 * @param {number} [options.timeoutMs] Limit for one chat request, stream included.
 * @param {typeof fetch} [options.fetchImpl] Defaults to the global fetch, read
 *   at each request: a script that installs the privacy interceptor after
 *   creating the client still sends through it.
 * @returns {{
 *   version: () => Promise<string>,
 *   listModels: () => Promise<Array<{name: string, size: number, family: string, parameterSize: string, quant: string, isEmbedding: boolean}>>,
 *   showModel: (name: string) => Promise<{capabilities: string[], contextLength: number|null, parameterSize: string, quant: string}>,
 *   ps: () => Promise<Array<{name: string, size: number, sizeVram: number, contextLength: number|null}>>,
 *   chat: (request: ChatRequest) => Promise<ChatResult>,
 *   preload: (model: string, numCtx?: number, keepAlive?: string|number) => Promise<boolean>,
 * }}
 * @throws {OllamaConfigError} When timeoutMs is not usable.
 */
export function createOllamaClient({ baseUrl = DEFAULTS.baseUrl, timeoutMs = DEFAULTS.timeoutMs, fetchImpl } = {}) {
  const base = String(baseUrl).replace(/\/+$/, '');
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0 || timeoutMs > MAX_TIMEOUT_MS) {
    throw new OllamaConfigError(`Invalid timeoutMs: "${timeoutMs}" — must be a positive integer of at most ${MAX_TIMEOUT_MS}.`);
  }
  if (fetchImpl !== undefined && typeof fetchImpl !== 'function') {
    throw new TypeError('createOllamaClient: fetchImpl must be a function.');
  }
  const currentFetch = () => fetchImpl ?? globalThis.fetch;
  const chatEndpoint = `${base}/api/chat`;
  const showCache = new Map();

  const unreachable = (cause) => new OllamaError(
    'UNREACHABLE',
    `Ollama not reachable at ${base}. Start it (ollama serve) or set OLLAMA_BASE_URL.`,
    { cause },
  );

  const httpFailure = (res, text, model, requestUrl) => {
    const { status } = res;
    if (REDIRECT_STATUSES.has(status)) {
      const location = redirectTarget(res, requestUrl);
      return new OllamaError(
        'HTTP',
        `Ollama API error: HTTP ${status}: ${base} answered with a redirect${location ? ` to ${location}` : ''}. `
        + 'Redirects are not followed: only the configured address was checked, and requests to it carry your CV. '
        + 'Set OLLAMA_BASE_URL (or --url) to the Ollama server itself.',
        { status, location },
      );
    }
    if (status === 404 && MODEL_NOT_FOUND_RE.test(text)) {
      return new OllamaError(
        'MODEL_NOT_FOUND',
        `Model "${model}" was not found by Ollama at ${base}. Run: ollama pull ${model}`,
        { status },
      );
    }
    const shown = text.slice(0, ERROR_TEXT_LIMIT).trim();
    return new OllamaError('HTTP', `Ollama API error: HTTP ${status}${shown ? `: ${shown}` : ''}`, { status });
  };

  /**
   * Name a failed chat request. `partialChars` is given once the reply had
   * started, which is what separates a dropped connection (worth one more
   * try) from a server that is not there.
   */
  const chatFailure = (err, signal, partialChars) => {
    if (err instanceof OllamaError || isPrivacyError(err)) return err;
    const started = partialChars !== undefined;
    const props = started ? { cause: err, partialChars } : { cause: err };
    if (isAbortTimeout(err, signal)) {
      return new OllamaError('TIMEOUT', `Request timed out after ${Math.round(timeoutMs / 1000)}s. Raise OLLAMA_TIMEOUT_MS.`, props);
    }
    if (isSilentTimeout(err)) {
      // Not the wording above: OLLAMA_TIMEOUT_MS cannot move this limit, and
      // sending the user to raise it would cost them another five minutes.
      return new OllamaError(
        'TIMEOUT',
        'Ollama sent nothing for 5 minutes, and Node\'s fetch ends a silent connection at that point whatever '
        + 'OLLAMA_TIMEOUT_MS is. The server is probably busy with another request or still loading the model; '
        + 'try again when it is idle.',
        props,
      );
    }
    if (started) {
      return new OllamaError(
        'UNREACHABLE',
        `Connection to Ollama at ${base} was lost after ${partialChars} chars of the reply.`,
        { ...props, midStream: true },
      );
    }
    return unreachable(err);
  };

  /** GET or POST a small JSON answer. No prompt text ever goes through here. */
  async function requestJson(path, { method = 'GET', body, timeout = METADATA_TIMEOUT_MS, model } = {}) {
    const url = `${base}${path}`;
    let res;
    let text;
    try {
      res = await currentFetch()(url, {
        method,
        ...(body === undefined ? {} : { headers: JSON_HEADERS, body }),
        signal: AbortSignal.timeout(timeout),
        redirect: 'manual',
      });
      text = await readText(res);
    } catch (err) {
      if (err instanceof OllamaError || isPrivacyError(err)) throw err;
      // A timeout lands here too. A server that cannot answer one of these in
      // seconds is not one a run can use, and the remedy is the same.
      throw unreachable(err);
    }
    if (!isOk(res)) throw httpFailure(res, text, model, url);
    try {
      return JSON.parse(text);
    } catch {
      throw new OllamaError('HTTP', `Unexpected reply from ${url}: not JSON. Is this an Ollama server?`, { status: res.status });
    }
  }

  async function version() {
    const data = await requestJson('/api/version');
    return typeof data?.version === 'string' && data.version ? data.version : 'unknown';
  }

  async function listModels() {
    const data = await requestJson('/api/tags');
    const models = [];
    for (const entry of Array.isArray(data?.models) ? data.models : []) {
      const name = entry?.name ?? entry?.model;
      if (typeof name !== 'string' || !name) continue;
      const details = entry.details ?? {};
      const family = typeof details.family === 'string' ? details.family : '';
      // Newer servers say what a model can do; the name is the fallback.
      const caps = Array.isArray(entry.capabilities) ? entry.capabilities : null;
      models.push({
        name,
        size: reportedCount(entry.size) ?? 0,
        family,
        parameterSize: typeof details.parameter_size === 'string' ? details.parameter_size : '',
        quant: typeof details.quantization_level === 'string' ? details.quantization_level : '',
        isEmbedding: caps
          ? caps.includes('embedding') && !caps.includes('completion')
          : isEmbeddingModel(name, family),
      });
    }
    return models;
  }

  async function fetchShow(name) {
    const data = await requestJson('/api/show', {
      method: 'POST',
      body: JSON.stringify({ model: name }),
      timeout: SHOW_TIMEOUT_MS,
      model: name,
    });
    // The limit is stored under the architecture's own key
    // (`qwen35.context_length`, `llama.context_length`), named by
    // `general.architecture`. It is the model's maximum, not the server's
    // default window, which is what a num_ctx has to be checked against.
    const info = data?.model_info && typeof data.model_info === 'object' ? data.model_info : {};
    const details = data?.details ?? {};
    const positive = (value) => Number.isFinite(value) && value > 0;
    let contextLength = info[`${info['general.architecture']}.context_length`];
    if (!positive(contextLength)) {
      contextLength = Object.entries(info).find(([key, value]) => key.endsWith('.context_length') && positive(value))?.[1];
    }
    // The model list carries the same limit under `details`; a server that
    // puts it there in this answer as well is taken at its word.
    if (!positive(contextLength)) contextLength = details.context_length;
    return {
      capabilities: Array.isArray(data?.capabilities) ? data.capabilities.map(String) : [],
      contextLength: positive(contextLength) ? contextLength : null,
      parameterSize: typeof details.parameter_size === 'string' ? details.parameter_size : '',
      quant: typeof details.quantization_level === 'string' ? details.quantization_level : '',
    };
  }

  /**
   * Details of one model. Asked once per name: a pipeline run needs the same
   * answer for every job, and each call makes the server read the model file.
   * A failure is not remembered, so a model pulled meanwhile is found.
   */
  async function showModel(name) {
    const key = String(name ?? '');
    if (!showCache.has(key)) {
      showCache.set(key, fetchShow(key).catch((err) => {
        showCache.delete(key);
        throw err;
      }));
    }
    const shown = await showCache.get(key);
    return { ...shown, capabilities: [...shown.capabilities] };
  }

  async function ps() {
    const data = await requestJson('/api/ps');
    const models = [];
    for (const entry of Array.isArray(data?.models) ? data.models : []) {
      const name = entry?.name ?? entry?.model;
      if (typeof name !== 'string' || !name) continue;
      models.push({
        name,
        size: reportedCount(entry.size) ?? 0,
        sizeVram: reportedCount(entry.size_vram) ?? 0,
        contextLength: reportedCount(entry.context_length) || null,
      });
    }
    return models;
  }

  /**
   * Send one request to the chat endpoint, through the privacy gate.
   *
   * The string that is inspected is the string that is sent. A refusal
   * (PrivacyGuardError) leaves here as it is: nothing was sent, and the
   * caller must not retry or relabel it.
   */
  async function sendChat(payload, signal) {
    const body = JSON.stringify(payload);
    const gate = await loadGate();
    // Read after the await, so the function checked below is the one called.
    const send = currentFetch();
    // With PRIVACY_REDACT=1 the gate lets configured text through, because the
    // interceptor over fetch replaces it on the way out. That holds only if
    // this request goes through the interceptor. When it does not (an injected
    // fetch, a script that never installed it), nothing would replace the text,
    // so it is treated as a finding and the request is refused.
    const scrubbed = globalThis[INTERCEPTOR]?.wrapped === send;
    gate.assertLlmRequestAllowed(chatEndpoint, body, gate.redactModeEnabled() && !scrubbed ? { allowRedact: false } : undefined);
    const init = { method: 'POST', headers: JSON_HEADERS, body, signal, redirect: 'manual' };
    try {
      return await send(chatEndpoint, init);
    } catch (err) {
      // A refusal, a time limit and anything this file raised itself are
      // final. What is left is a connection that failed.
      if (err instanceof OllamaError || isPrivacyError(err) || isAbortTimeout(err, signal) || isSilentTimeout(err)) {
        throw chatFailure(err, signal);
      }
    }
    // The connection failed before any of the reply arrived: nothing was
    // generated, so sending the same request again costs nothing. The usual
    // cause is a kept-alive connection that the far end closed while this
    // process sat in a synchronous step (a PDF render, a tracker merge) and
    // could not notice; the second send opens a new one. Once only, and with
    // the same time limit: a server that is not there fails this one as well.
    try {
      return await send(chatEndpoint, init);
    } catch (err) {
      throw chatFailure(err, signal);
    }
  }

  /**
   * Ask the model one question and return its complete answer.
   *
   * @param {ChatRequest} request
   * @returns {Promise<ChatResult>}
   * @throws {OllamaError} With a code, for every reply that is not complete.
   * @throws {Error} A PrivacyGuardError, unchanged, when the gate refuses the
   *   request. Nothing was sent.
   */
  async function chat(request) {
    const {
      model, system, user, numCtx, numPredict, temperature, think, keepAlive, seed,
      estimatedPromptTokens, onToken, onThinking,
    } = request ?? {};
    if (typeof model !== 'string' || !model.trim()) throw new TypeError('chat: a model name is required.');
    if (typeof user !== 'string' || !user) throw new TypeError('chat: a user message is required.');

    const messages = [];
    if (typeof system === 'string' && system) messages.push({ role: 'system', content: system });
    messages.push({ role: 'user', content: user });

    // The native endpoint reads generation settings from `options` only. A
    // setting placed beside it is ignored without an error, and the request
    // then runs at the model's own temperature (1.0 for the default model)
    // and the server's own context size.
    const options = {
      temperature: temperature ?? DEFAULTS.temperature,
      num_ctx: numCtx ?? DEFAULTS.numCtx,
      num_predict: numPredict ?? DEFAULTS.numPredict,
    };
    if (Number.isSafeInteger(seed)) options.seed = seed;

    let payload = { model, messages, stream: true };
    // Only a value the caller has. Holding a model in GPU memory is the
    // owner's decision, made once, on the server; a default sent from here
    // would override it on every request.
    if (isKeepAlive(keepAlive)) payload.keep_alive = keepAlive;
    // Only a decision is sent. Left out, the server decides, and for a model
    // that can think that means thinking on.
    if (think === true || think === false) payload.think = think;
    payload.options = options;

    // One limit for the whole exchange, stream included.
    const signal = AbortSignal.timeout(timeoutMs);
    const bodyText = async (response) => {
      try {
        return await readText(response);
      } catch (err) {
        throw chatFailure(err, signal);
      }
    };

    let res = await sendChat(payload, signal);
    if (!isOk(res)) {
      let text = await bodyText(res);
      if (res.status === 400 && 'think' in payload && THINK_UNSUPPORTED_RE.test(text)) {
        // The capability list said the model can think and the server says it
        // cannot. The setting is advice, not part of the question, so the
        // same request goes once more without it.
        payload = { ...payload };
        delete payload.think;
        res = await sendChat(payload, signal);
        if (!isOk(res)) text = await bodyText(res);
      }
      if (!isOk(res)) throw httpFailure(res, text, model, chatEndpoint);
    }

    // -- Read the stream: newline-delimited JSON, one object per token, the
    //    last one carrying done:true and the counts. --------------------------
    let content = '';
    let thinkingChars = 0;
    let final = null;
    let buffered = '';
    const decoder = new TextDecoder('utf-8', { fatal: true });
    const streamFailure = (message) => new OllamaError('STREAM', message, { partialChars: content.length });

    const finish = (record) => {
      // Checked before the stop reason. A reply can be both, and for a prompt
      // that did not fit, "raise num_predict" is the wrong half of the advice.
      const processed = reportedCount(record.prompt_eval_count);
      if (processed !== null && Number.isFinite(estimatedPromptTokens) && estimatedPromptTokens > 0) {
        // prompt_eval_count is the whole prompt as the server took it, cached
        // or not: the same prompt reported 39.1K cold and 39,118 warm, with
        // 39,114 of those under prompt_eval_cached_count. The cached count is
        // a share of it, so adding the two counted a warm prompt twice, and a
        // truncated one then passed: 16.4K of 40K read as 32.8K.
        if (processed < TRUNCATION_RATIO * estimatedPromptTokens) {
          throw new OllamaError(
            'TRUNCATED_PROMPT',
            `Ollama processed ${processed} prompt tokens but ~${Math.round(estimatedPromptTokens)} were sent: `
            + 'the prompt was truncated. Raise OLLAMA_NUM_CTX.',
            { partialChars: content.length },
          );
        }
      }
      // A missing reason is a normal finish; older servers do not send one.
      if (record.done_reason && record.done_reason !== 'stop') {
        // Reasoning comes out of the same output room as the answer. When the
        // room ran out before the answer began, more room is the wrong advice:
        // gemma4 spent 25.5K tokens this way without starting its report.
        // Reasoning written inline, between think tags, counts the same.
        const answered = stripThinkTags(content) !== '';
        const thought = thinkingChars + (answered ? 0 : content.length);
        if (record.done_reason === 'length' && !answered && thought > 0) {
          throw new OllamaError(
            'INCOMPLETE',
            'Reply stopped with done_reason "length" before the answer began: '
            + `the model spent its output budget thinking (${thought} chars of it). `
            + (payload.think === true
              ? 'Leave thinking off: drop --think on or OLLAMA_THINK=on. Off is the default.'
              : 'This request did not switch thinking on, so the model thinks whatever it is asked: '
                + `use another one (--model), or raise OLLAMA_NUM_PREDICT (now ${options.num_predict}).`),
            { partialChars: content.length },
          );
        }
        throw new OllamaError(
          'INCOMPLETE',
          `Reply stopped with done_reason "${record.done_reason}" after ${content.length} chars. `
          + `Raise OLLAMA_NUM_PREDICT (now ${options.num_predict}) or OLLAMA_NUM_CTX (now ${options.num_ctx}).`,
          { partialChars: content.length },
        );
      }
      final = record;
    };

    const consumeLine = (line) => {
      if (line.length > RECORD_LIMIT) throw streamFailure('Stream record exceeds the 1 Mi code-unit limit.');
      if (!line.trim()) return;
      let record;
      try {
        record = JSON.parse(line);
      } catch {
        throw streamFailure('Malformed stream record: not JSON.');
      }
      if (record === null || typeof record !== 'object') throw streamFailure('Malformed stream record: not an object.');
      if (record.error) {
        const reported = typeof record.error === 'string' ? record.error : JSON.stringify(record.error);
        throw streamFailure(`Ollama reported an error: ${reported.slice(0, ERROR_TEXT_LIMIT)}`);
      }
      if (final) throw streamFailure('Unexpected data after the completed reply.');
      const message = record.message;
      if (message?.tool_calls?.length) {
        throw new OllamaError('TOOL_CALL', 'Ollama returned a tool call instead of a text reply.', { partialChars: content.length });
      }
      if (typeof message?.thinking === 'string' && message.thinking) {
        thinkingChars += message.thinking.length;
        if (typeof onThinking === 'function') onThinking(message.thinking);
      }
      if (typeof message?.content === 'string' && message.content) {
        content += message.content;
        if (typeof onToken === 'function') onToken(message.content);
      }
      if (record.done === true) finish(record);
    };

    const feed = (text) => {
      buffered += text;
      let newline;
      while ((newline = buffered.indexOf('\n')) !== -1) {
        const line = buffered.slice(0, newline);
        buffered = buffered.slice(newline + 1);
        consumeLine(line);
      }
      if (buffered.length > RECORD_LIMIT) throw streamFailure('Stream record exceeds the 1 Mi code-unit limit.');
    };
    const decode = (chunk) => {
      try {
        return chunk === undefined ? decoder.decode() : decoder.decode(chunk, { stream: true });
      } catch {
        throw streamFailure('Stream is not valid UTF-8.');
      }
    };

    // Only a failed read is a transport failure. A record this code refuses,
    // or an exception from the caller's own callback, is thrown from feed()
    // and must not come out relabelled as a dropped connection.
    const body = res.body;
    const iterator = body?.[Symbol.asyncIterator]?.() ?? body?.[Symbol.iterator]?.();
    if (iterator) {
      try {
        for (;;) {
          let step;
          try {
            step = await iterator.next();
          } catch (err) {
            throw chatFailure(err, signal, content.length);
          }
          if (step.done) break;
          feed(typeof step.value === 'string' ? step.value : decode(step.value));
        }
      } finally {
        // Stop reading on every way out. The server stops generating when the
        // connection closes, so a reply refused halfway does not keep the GPU
        // busy. Not awaited: the error on its way out must not wait on it.
        try {
          Promise.resolve(iterator.return?.()).catch(() => {});
        } catch { /* nothing left to release */ }
      }
    } else {
      feed(await bodyText(res));
    }
    // The last record need not end in a newline.
    consumeLine(buffered + decode());
    if (!final) throw streamFailure(`Stream ended before done:true, after ${content.length} chars.`);

    const answer = stripThinkTags(content);
    if (!answer) {
      throw new OllamaError(
        'EMPTY',
        thinkingChars > 0
          ? `Ollama returned no answer, only ${thinkingChars} chars of thinking.`
          : 'Ollama returned an empty reply.',
        { partialChars: 0 },
      );
    }

    const promptTokens = reportedCount(final.prompt_eval_count) ?? 0;
    const completionTokens = reportedCount(final.eval_count) ?? 0;
    return {
      content: answer,
      thinkingChars,
      usage: {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: promptTokens + completionTokens,
        cached_tokens: reportedCount(final.prompt_eval_cached_count) ?? 0,
      },
      doneReason: final.done_reason ?? 'stop',
    };
  }

  /**
   * Ask the server to load a model before the first job needs it.
   *
   * An empty message list loads the model and returns. The window is the one
   * the jobs will use, because a different num_ctx on the next request would
   * reload the model and undo this. A keep-alive is sent only when the caller
   * has one, as in chat(): otherwise the server's own setting holds the model.
   *
   * Best effort: the first chat request loads the model anyway, and reports
   * whatever is wrong in full.
   *
   * @returns {Promise<boolean>} Whether the server accepted the request.
   */
  async function preload(model, numCtx = DEFAULTS.numCtx, keepAlive) {
    try {
      const payload = { model, messages: [] };
      if (isKeepAlive(keepAlive)) payload.keep_alive = keepAlive;
      payload.options = { num_ctx: numCtx };
      const res = await sendChat(payload, AbortSignal.timeout(timeoutMs));
      await readText(res);
      return isOk(res);
    } catch {
      return false;
    }
  }

  return { version, listModels, showModel, ps, chat, preload };
}

// ---------------------------------------------------------------------------
// Retry
// ---------------------------------------------------------------------------

/**
 * Is another attempt at the same request likely to end differently?
 *
 * Yes for a reply with no text, for a tool call and for a connection that
 * dropped mid-reply. A tool call is a misfire of the sampling, not an answer
 * the model would give every time: the request offers no tools, the default
 * model was still seen calling `Read` 13 tokens into a reply, and it takes
 * about a second to find out whether the next sample does better.
 * No for everything that would fail the same way again, or cost minutes to
 * find out: a reply cut at the output cap, a truncated prompt, a missing
 * model, a timeout, a server that is not there. A privacy refusal has none of
 * these codes, so it is never retried.
 */
function isRetryable(err) {
  return err?.code === 'EMPTY' || err?.code === 'TOOL_CALL' || (err?.code === 'UNREACHABLE' && err.midStream === true);
}

/**
 * chat(), with another attempt when the reply is unusable.
 *
 * In the probe a model answered a 31K-token prompt with one line ("I'll start
 * by reading the candidate's profile and configuration files") and stopped.
 * To the server that is a finished reply, so the caller's validate() is what
 * names it, and the request goes again.
 *
 * A request with a fixed seed gives the same sample every time, so each retry
 * moves a seed the caller set on by one. A request with no seed stays without
 * one: the server draws its own for every request, so the retry is a new
 * sample already. Giving it seed 1, as this once did, made the retry one
 * fixed reply per prompt, the same on every run of a command; for the
 * tailoring prompt both seed-1 replies on record were the tool-call misfire.
 *
 * @param {{ chat: (request: ChatRequest) => Promise<ChatResult> }} client
 * @param {ChatRequest} req
 * @param {object} [options]
 * @param {number} [options.retries] Extra attempts after the first. Default 1.
 * @param {(content: string) => (string|null|undefined|Promise<string|null|undefined>)} [options.validate]
 *   Returns why the reply is unusable, or nothing when it is fine.
 * @param {(info: {attempt: number, reason: string, error?: Error, nextSeed: number|undefined}) => void} [options.onRetry]
 *   Called before each retry. `attempt` is the one that just failed, and
 *   `nextSeed` the seed the next one carries: undefined when the request has
 *   none.
 * @returns {Promise<ChatResult & {attempts: number}>}
 * @throws {OllamaError} Code EMPTY, with `reason`, when the last attempt's
 *   reply was still rejected by validate().
 */
export async function chatWithRetry(client, req, { retries = DEFAULTS.retries, validate, onRetry } = {}) {
  const extra = Number.isFinite(retries) ? Math.min(Math.max(Math.trunc(retries), 0), MAX_RETRIES) : DEFAULTS.retries;
  const firstSeed = Number.isSafeInteger(req?.seed) ? req.seed : undefined;
  // One rule for the request that is sent and for what onRetry is told.
  const seedFor = (attempt) => (firstSeed === undefined ? undefined : firstSeed + attempt - 1);

  for (let attempt = 1; ; attempt += 1) {
    const seed = seedFor(attempt);
    const request = seed === undefined ? req : { ...req, seed };

    let result;
    let error;
    try {
      result = await client.chat(request);
    } catch (err) {
      if (!isRetryable(err)) throw err;
      error = err;
    }

    const content = typeof result?.content === 'string' ? result.content : '';
    let reason;
    if (error) {
      reason = error.message;
    } else {
      const verdict = typeof validate === 'function' ? await validate(content) : null;
      if (!verdict) return { ...result, attempts: attempt };
      reason = String(verdict);
    }

    if (attempt > extra) {
      if (error) {
        error.attempts = attempt;
        throw error;
      }
      const rejected = new OllamaError(
        'EMPTY',
        `Reply rejected after ${attempt} attempt${attempt === 1 ? '' : 's'}: ${reason}.`,
        { partialChars: content.length },
      );
      rejected.reason = reason;
      rejected.attempts = attempt;
      // For a caller that wants to show what came back. Not enumerable, so a
      // logged error does not print a whole report.
      Object.defineProperty(rejected, 'content', { value: content, enumerable: false });
      throw rejected;
    }

    if (typeof onRetry === 'function') {
      onRetry({ attempt, reason, error, nextSeed: seedFor(attempt + 1) });
    }
  }
}

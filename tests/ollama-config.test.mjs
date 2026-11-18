// tests/ollama-config.test.mjs — the settings and decisions behind the local
// Ollama runner (lib/ollama-config.mjs).
//
// Every function here is pure, so every case is a call and a comparison: no
// server, no files, no process.env. What they pin is the set of mistakes the
// old evaluator made quietly on a real machine:
//
//   - OLLAMA_HOST=0.0.0.0 is the server's bind address. Used as a client URL
//     it is not a destination at all.
//   - `new URL('http://[::1]:11434').hostname` is `[::1]`, so a comparison
//     against `::1` called IPv6 loopback a remote host.
//   - parseInt read "32k" as a 32-token context.
//   - The default model was one nobody had pulled, and the first sign of it
//     was a bare 404 after the prompt had been built.
//   - A prompt too long for num_ctx was cut by the server without an error.
//
// And the ones the first version of this backend made, found by review and by
// running it against the real model:
//
//   - OLLAMA_HOST=0.0.0.0:8080 lost its port, so every request went to
//     whatever listened on 11434.
//   - `auto` turned thinking on for one model family, which then spent its
//     whole output budget thinking in two runs of three.
//   - Every request asked the server to hold the model for 30 minutes.
//   - A URL that failed to parse was repeated in the error, password included.
//
// Run:  node --test tests/ollama-config.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import {
  DEFAULTS,
  DEFAULT_MODEL_PREFERENCE,
  EMBEDDING_NAME_RE,
  NUM_CTX_STEPS,
  OUTPUT_RESERVE_TOKENS,
  OllamaConfigError,
  OllamaModelError,
  RUNNER_ENV_VARS,
  TYPICAL_PROMPT_TOKENS,
  assertEndpointAllowed,
  evaluationPromptTokens,
  fitContext,
  isEmbeddingModel,
  isLoopbackUrl,
  parsePositiveInt,
  pickModel,
  readOllamaConfig,
  resolveBaseUrl,
  resolveNumPredict,
  resolveThink,
  sameModel,
} from '../lib/ollama-config.mjs';

/** The message of the OllamaConfigError `fn` throws. Fails when it does not. */
function configError(fn) {
  let caught;
  try {
    fn();
  } catch (err) {
    caught = err;
  }
  assert.ok(caught instanceof OllamaConfigError, `expected an OllamaConfigError, got ${caught?.name ?? 'no error'}`);
  assert.equal(caught.code, 'OLLAMA_CONFIG');
  return caught.message;
}

// What this machine has installed, in the order the server lists it.
const INSTALLED = [
  'qwen3.8:27b',
  'gemma4:26b-a4b-it-q4_K_M',
  'bge-m3:latest',
  'nomic-embed-text:latest',
  'all-minilm:latest',
  'glm-4.7-flash:latest',
  'qwen3-coder:30b',
  'mistral-small:latest',
];

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

test('the defaults are the measured ones, and cannot be changed by a caller', () => {
  assert.deepEqual({ ...DEFAULTS }, {
    baseUrl: 'http://localhost:11434',
    numCtx: 65536,
    numPredict: 16384,
    numPredictThinking: 28000,
    think: 'auto',
    temperature: 0.2,
    timeoutMs: 900000,
    retries: 1,
  });
  // There is no keep-alive default: how long a model stays in GPU memory is
  // the server's setting, and nothing is sent unless OLLAMA_KEEP_ALIVE is set.
  assert.equal('keepAlive' in DEFAULTS, false);
  assert.deepEqual([...NUM_CTX_STEPS], [65536, 98304, 131072]);
  assert.equal(OUTPUT_RESERVE_TOKENS, 12000);
  assert.deepEqual([...DEFAULT_MODEL_PREFERENCE], [
    'qwen3.8:27b', 'gemma4:26b-a4b-it-q4_K_M', 'glm-4.7-flash:latest', 'qwen3-coder:30b',
  ]);
  // One module-level object is shared by the runner, the doctor and the client.
  for (const shared of [DEFAULTS, NUM_CTX_STEPS, DEFAULT_MODEL_PREFERENCE, RUNNER_ENV_VARS]) assert.ok(Object.isFrozen(shared));
  assert.ok(EMBEDDING_NAME_RE instanceof RegExp);
});

test('RUNNER_ENV_VARS is the runner’s own settings: every one it reads, and none of the server’s', () => {
  // The list doctor.mjs asks "does this install use the runner?" with. A
  // server variable in it would make a plain `node doctor.mjs` contact a
  // server for everyone who tunes Ollama through the environment.
  assert.deepEqual([...RUNNER_ENV_VARS], [
    'OLLAMA_BASE_URL', 'OLLAMA_ALLOW_REMOTE', 'OLLAMA_MODEL', 'OLLAMA_MODEL_PREFERENCE', 'OLLAMA_NUM_CTX',
    'OLLAMA_NUM_PREDICT', 'OLLAMA_THINK', 'OLLAMA_TEMPERATURE', 'OLLAMA_TIMEOUT_MS', 'OLLAMA_RETRIES', 'OLLAMA_SEED',
  ]);
  // The two the runner honours and the server owns.
  const SERVER_OWNED = ['OLLAMA_HOST', 'OLLAMA_KEEP_ALIVE'];
  for (const name of SERVER_OWNED) assert.equal(RUNNER_ENV_VARS.includes(name), false, name);

  // One list: a setting added to the reader and not to the list, or the other
  // way round, fails here. What the reader reads is taken from its own source.
  const source = readFileSync(fileURLToPath(new URL('../lib/ollama-config.mjs', import.meta.url)), 'utf8');
  const read = new Set([...source.matchAll(/\b(?:e|env)\??\.(OLLAMA_[A-Z0-9_]+)/g)].map((match) => match[1]));
  assert.deepEqual([...read].sort(), [...RUNNER_ENV_VARS, ...SERVER_OWNED].sort());

  // And each one is a real setting: given a usable value it is accepted, and
  // it changes what readOllamaConfig returns.
  const usable = {
    OLLAMA_BASE_URL: 'http://127.0.0.1:11500', OLLAMA_ALLOW_REMOTE: '1', OLLAMA_MODEL: 'some-model:1',
    OLLAMA_MODEL_PREFERENCE: 'a:1,b:2', OLLAMA_NUM_CTX: '98304', OLLAMA_NUM_PREDICT: '9000', OLLAMA_THINK: 'on',
    OLLAMA_TEMPERATURE: '0.7', OLLAMA_TIMEOUT_MS: '1000', OLLAMA_RETRIES: '2', OLLAMA_SEED: '5',
  };
  assert.deepEqual(Object.keys(usable), [...RUNNER_ENV_VARS]);
  for (const name of RUNNER_ENV_VARS) {
    assert.notDeepEqual(readOllamaConfig({ [name]: usable[name] }), readOllamaConfig({}), name);
  }
});

// ---------------------------------------------------------------------------
// resolveBaseUrl
// ---------------------------------------------------------------------------

test('with nothing set the client talks to localhost', () => {
  assert.deepEqual(resolveBaseUrl({}), { url: 'http://localhost:11434', source: 'default' });
  assert.deepEqual(resolveBaseUrl({ OLLAMA_BASE_URL: '   ', OLLAMA_HOST: '' }), { url: 'http://localhost:11434', source: 'default' });
});

test('OLLAMA_HOST=0.0.0.0 is a bind address: localhost is used, and the caller is told what was ignored', () => {
  assert.deepEqual(resolveBaseUrl({ OLLAMA_HOST: '0.0.0.0' }), {
    url: 'http://localhost:11434', source: 'default', ignoredHost: '0.0.0.0',
  });
  // Every other way of writing "all interfaces" or "no host" that names no
  // port either: there is nothing in the value to connect to.
  for (const bind of ['::', '[::]', ' 0.0.0.0 ', '0', 'http://0.0.0.0', 'https://[::]', '0.0.0.0:', ':', '0:0:0:0:0:0:0:0']) {
    assert.deepEqual(resolveBaseUrl({ OLLAMA_HOST: bind }), { url: 'http://localhost:11434', source: 'default', ignoredHost: bind.trim() }, bind);
  }
  // Port 0 means "any free port" to the server, and a port that is not one
  // leaves nothing to keep.
  for (const bind of ['0.0.0.0:0', ':0', '0.0.0.0:99999', ':8080x', '[::]:port']) {
    assert.deepEqual(resolveBaseUrl({ OLLAMA_HOST: bind }), { url: 'http://localhost:11434', source: 'default', ignoredHost: bind }, bind);
  }
});

test('a bind address with a port keeps the port: that is where the server listens, on this machine', () => {
  // OLLAMA_HOST=0.0.0.0:8080 used to resolve to localhost:11434. With another
  // Ollama on 11434 (the desktop app) the CV went to that one, unannounced.
  for (const [bind, url] of [
    ['0.0.0.0:8080', 'http://localhost:8080'],
    ['[::]:8080', 'http://localhost:8080'],
    [':8080', 'http://localhost:8080'],
    ['http://0.0.0.0:8080', 'http://localhost:8080'],
    ['http://:8080', 'http://localhost:8080'],
    ['[0:0:0:0:0:0:0:0]:8080', 'http://localhost:8080'],
    [' 0.0.0.0:8080/ ', 'http://localhost:8080'],
    ['https://0.0.0.0:8443', 'https://localhost:8443'],
    // The default port, written out, is a port like any other.
    ['0.0.0.0:11434', 'http://localhost:11434'],
    [':11434', 'http://localhost:11434'],
    // A port that is its scheme's own default is not printed, and is still that port.
    ['0.0.0.0:80', 'http://localhost'],
    ['https://0.0.0.0:443', 'https://localhost'],
  ]) {
    // No ignoredHost: the value was used, and "not used as the client URL" would be wrong.
    assert.deepEqual(resolveBaseUrl({ OLLAMA_HOST: bind }), { url, source: 'OLLAMA_HOST' }, bind);
  }
  // The result is this machine, so it passes the loopback rule unaided.
  const cfg = readOllamaConfig({ OLLAMA_HOST: '0.0.0.0:8080' });
  assert.equal(cfg.baseUrl, 'http://localhost:8080');
  assert.equal(cfg.baseUrlSource, 'OLLAMA_HOST');
  assert.equal(cfg.allowRemote, false);
  // OLLAMA_BASE_URL still wins over it.
  assert.equal(resolveBaseUrl({ OLLAMA_HOST: '0.0.0.0:8080', OLLAMA_BASE_URL: 'http://127.0.0.1:11500' }).url, 'http://127.0.0.1:11500');
});

test('an OLLAMA_HOST that names a real host becomes a URL', () => {
  const from = (host) => resolveBaseUrl({ OLLAMA_HOST: host });
  assert.deepEqual(from('192.168.1.5:11434'), { url: 'http://192.168.1.5:11434', source: 'OLLAMA_HOST' });
  // No scheme: http and the server's own default port.
  assert.deepEqual(from('192.168.1.5'), { url: 'http://192.168.1.5:11434', source: 'OLLAMA_HOST' });
  assert.deepEqual(from('127.0.0.1:11435'), { url: 'http://127.0.0.1:11435', source: 'OLLAMA_HOST' });
  assert.deepEqual(from('::1'), { url: 'http://[::1]:11434', source: 'OLLAMA_HOST' });
  assert.deepEqual(from('[::1]:11435'), { url: 'http://[::1]:11435', source: 'OLLAMA_HOST' });
  // An explicit scheme keeps that scheme's port, as the server's parser does.
  assert.deepEqual(from('https://ollama.example.test'), { url: 'https://ollama.example.test', source: 'OLLAMA_HOST' });
  assert.deepEqual(from('https://ollama.example.test:8443'), { url: 'https://ollama.example.test:8443', source: 'OLLAMA_HOST' });
  assert.deepEqual(from('192.168.1.5:11434/'), { url: 'http://192.168.1.5:11434', source: 'OLLAMA_HOST' });
  // A written port 80 is port 80. The URL parser reports it as no port at all,
  // which used to be read as "none given" and replaced by 11434.
  assert.deepEqual(from('192.168.1.5:80'), { url: 'http://192.168.1.5', source: 'OLLAMA_HOST' });
  assert.deepEqual(from('localhost:80'), { url: 'http://localhost', source: 'OLLAMA_HOST' });
  assert.deepEqual(from('localhost'), { url: 'http://localhost:11434', source: 'OLLAMA_HOST' });
});

test('OLLAMA_BASE_URL wins over OLLAMA_HOST, and --url wins over both', () => {
  const env = { OLLAMA_BASE_URL: 'http://127.0.0.1:11500', OLLAMA_HOST: '0.0.0.0' };
  // Nothing was ignored: OLLAMA_HOST was never consulted.
  assert.deepEqual(resolveBaseUrl(env), { url: 'http://127.0.0.1:11500', source: 'OLLAMA_BASE_URL' });
  assert.deepEqual(resolveBaseUrl(env, 'http://localhost:11600'), { url: 'http://localhost:11600', source: 'flag' });
  assert.deepEqual(resolveBaseUrl({ OLLAMA_HOST: '192.168.1.5' }, 'http://localhost:11600'), { url: 'http://localhost:11600', source: 'flag' });
});

test('a trailing slash and a trailing /v1 are stripped', () => {
  const url = (value) => resolveBaseUrl({ OLLAMA_BASE_URL: value }).url;
  assert.equal(url('http://localhost:11434/'), 'http://localhost:11434');
  assert.equal(url('http://localhost:11434/v1'), 'http://localhost:11434');
  assert.equal(url('http://localhost:11434/v1/'), 'http://localhost:11434');
  assert.equal(url('http://localhost:11434///'), 'http://localhost:11434');
  // A reverse-proxy prefix stays; only the OpenAI-compatible suffix goes.
  assert.equal(url('https://box.example.test/ollama/v1'), 'https://box.example.test/ollama');
  assert.equal(url('HTTP://LOCALHOST:11434'), 'http://localhost:11434');
  // A value pasted without a scheme is a host, not a URL with scheme "localhost:".
  assert.equal(url('localhost:11434'), 'http://localhost:11434');
  assert.equal(resolveBaseUrl({}, 'http://127.0.0.1:11434/v1').url, 'http://127.0.0.1:11434');
});

test('a value that is not an http(s) URL is refused, naming where it came from', () => {
  assert.match(configError(() => resolveBaseUrl({ OLLAMA_BASE_URL: 'ftp://localhost:11434' })), /^Invalid OLLAMA_BASE_URL: "ftp:\/\/localhost:11434"/);
  assert.match(configError(() => resolveBaseUrl({ OLLAMA_BASE_URL: 'http://' })), /^Invalid OLLAMA_BASE_URL/);
  assert.match(configError(() => resolveBaseUrl({ OLLAMA_BASE_URL: 'http://localhost:11434/?key=1' })), /^Invalid OLLAMA_BASE_URL/);
  assert.match(configError(() => resolveBaseUrl({}, 'http://local host')), /^Invalid --url/);
  // The value is a secret here, so the message must not repeat it.
  const withPassword = configError(() => resolveBaseUrl({ OLLAMA_BASE_URL: 'http://someone:hunter2@localhost:11434' }));
  assert.match(withPassword, /credentials/);
  assert.doesNotMatch(withPassword, /hunter2|someone/);
});

test('a value that cannot be parsed is not repeated when it may hold a password', () => {
  // The credentials check needs a parsed URL. These do not parse, and the
  // message for that used to quote the whole value.
  for (const [env, flag, label] of [
    [{ OLLAMA_BASE_URL: 'http://someone:hunter2@local host:11434' }, undefined, 'OLLAMA_BASE_URL'],
    [{ OLLAMA_BASE_URL: 'someone:hunter2@localhost:99999' }, undefined, 'OLLAMA_BASE_URL'],
    [{ OLLAMA_BASE_URL: 'http://someone:hunt/er2@[::1' }, undefined, 'OLLAMA_BASE_URL'],
    [{}, 'https://someone:hunter2@', '--url'],
  ]) {
    const message = configError(() => resolveBaseUrl(env, flag));
    assert.equal(message, `Invalid ${label}: (not shown: it may hold a password) — must be an http(s) URL such as http://localhost:11434.`);
    assert.doesNotMatch(message, /hunter2|hunt|someone/);
  }
  // A query string is the other place a secret sits: it is why the value is
  // refused, and it is cut from what is shown.
  const withToken = configError(() => resolveBaseUrl({ OLLAMA_BASE_URL: 'http://localhost:11434/?token=s3cr3t#frag' }));
  assert.equal(withToken, 'Invalid OLLAMA_BASE_URL: "http://localhost:11434/…" — must be an http(s) URL such as http://localhost:11434.');
  assert.doesNotMatch(configError(() => resolveBaseUrl({}, 'http://local host/?token=s3cr3t')), /s3cr3t/);
  // A value with nothing to hide is still shown whole: that is what the user has to fix.
  assert.equal(
    configError(() => resolveBaseUrl({}, 'http://local host')),
    'Invalid --url: "http://local host" — must be an http(s) URL such as http://localhost:11434.',
  );

  // An OLLAMA_HOST that was not used is put into a sentence by the runner and
  // the doctor, so it gets the same care.
  assert.deepEqual(resolveBaseUrl({ OLLAMA_HOST: 'http://someone:hunter2@0.0.0.0' }), {
    url: 'http://localhost:11434', source: 'default', ignoredHost: '(not shown: it may hold a password)',
  });
  assert.deepEqual(resolveBaseUrl({ OLLAMA_HOST: 'someone:hunter2@local host' }), {
    url: 'http://localhost:11434', source: 'default', ignoredHost: '(not shown: it may hold a password)',
  });
});

// ---------------------------------------------------------------------------
// Loopback guard
// ---------------------------------------------------------------------------

test('[::1] is loopback, with the brackets the URL parser keeps', () => {
  assert.equal(new URL('http://[::1]:11434').hostname, '[::1]', 'the premise: the hostname keeps its brackets');
  for (const url of ['http://[::1]:11434', 'http://localhost:11434', 'http://127.0.0.1:11434', 'http://LOCALHOST', 'https://127.0.0.1:8443/ollama']) {
    assert.equal(isLoopbackUrl(url), true, url);
  }
  for (const url of ['http://0.0.0.0:11434', 'http://192.168.1.5:11434', 'https://ollama.example.test', 'http://localhost.example.test', 'not a url', '', undefined]) {
    assert.equal(isLoopbackUrl(url), false, String(url));
  }
});

test('a remote endpoint is refused unless OLLAMA_ALLOW_REMOTE=1', () => {
  assert.equal(assertEndpointAllowed('http://localhost:11434', {}), undefined);
  assert.equal(assertEndpointAllowed('http://[::1]:11434', {}), undefined);

  const refused = configError(() => assertEndpointAllowed('http://192.168.1.5:11434', {}));
  assert.match(refused, /192\.168\.1\.5:11434/, 'names the host the data would go to');
  assert.match(refused, /OLLAMA_ALLOW_REMOTE=1/, 'names the way to allow it');

  assert.equal(assertEndpointAllowed('http://192.168.1.5:11434', { OLLAMA_ALLOW_REMOTE: '1' }), undefined);
  // Only the exact value: a stray "true" must not send a CV to another machine.
  for (const value of ['true', 'yes', '0', '']) {
    configError(() => assertEndpointAllowed('http://192.168.1.5:11434', { OLLAMA_ALLOW_REMOTE: value }));
  }
});

// ---------------------------------------------------------------------------
// Number parsing
// ---------------------------------------------------------------------------

test('parsePositiveInt takes digits only', () => {
  assert.equal(parsePositiveInt('OLLAMA_NUM_CTX', undefined, 65536), 65536);
  assert.equal(parsePositiveInt('OLLAMA_NUM_CTX', '', 65536), 65536);
  assert.equal(parsePositiveInt('OLLAMA_NUM_CTX', '   ', 65536), 65536);
  assert.equal(parsePositiveInt('OLLAMA_NUM_CTX', '98304', 65536), 98304);
  assert.equal(parsePositiveInt('OLLAMA_NUM_CTX', ' 8192 ', 65536), 8192);
  assert.equal(parsePositiveInt('--num-ctx', 8192, 65536), 8192, 'a flag parser may hand over a number');

  assert.equal(
    configError(() => parsePositiveInt('OLLAMA_NUM_CTX', '32k', 65536)),
    'Invalid OLLAMA_NUM_CTX: "32k" — must be a positive integer.',
  );
  for (const bad of ['0', '-1', '1.5', '1e5', '0x10', 'abc', '65 536', '9007199254740993']) {
    assert.match(configError(() => parsePositiveInt('OLLAMA_NUM_CTX', bad, 65536)), /^Invalid OLLAMA_NUM_CTX: /, bad);
  }
  assert.equal(parsePositiveInt('OLLAMA_TIMEOUT_MS', '2147483647', 1, 2147483647), 2147483647);
  assert.equal(
    configError(() => parsePositiveInt('OLLAMA_TIMEOUT_MS', '2147483648', 1, 2147483647)),
    'Invalid OLLAMA_TIMEOUT_MS: "2147483648" — must be a positive integer of at most 2147483647.',
  );
});

// ---------------------------------------------------------------------------
// readOllamaConfig
// ---------------------------------------------------------------------------

test('readOllamaConfig returns every setting at its default', () => {
  assert.deepEqual(readOllamaConfig({}), {
    baseUrl: 'http://localhost:11434',
    baseUrlSource: 'default',
    model: null,
    preference: ['qwen3.8:27b', 'gemma4:26b-a4b-it-q4_K_M', 'glm-4.7-flash:latest', 'qwen3-coder:30b'],
    numCtx: 65536,
    numCtxExplicit: false,
    numPredict: 16384,
    numPredictExplicit: false,
    think: 'auto',
    // No value, and the key is still there: the runner passes it straight to
    // the client, which sends no keep_alive for it. The server's own setting
    // then decides how long the model stays loaded.
    keepAlive: undefined,
    temperature: 0.2,
    timeoutMs: 900000,
    retries: 1,
    seed: undefined,
    allowRemote: false,
  });
  assert.equal('keepAlive' in readOllamaConfig({}), true);
  for (const unset of ['', '   ', undefined, null]) {
    assert.equal(readOllamaConfig({ OLLAMA_KEEP_ALIVE: unset }).keepAlive, undefined, JSON.stringify(unset));
  }
  // This machine's shell: the bind address changes nothing.
  assert.equal(readOllamaConfig({ OLLAMA_HOST: '0.0.0.0', OLLAMA_FLASH_ATTENTION: '1' }).baseUrl, 'http://localhost:11434');
  // The default list is a copy, so a caller that edits it does not edit the module's.
  readOllamaConfig({}).preference.push('edited');
  assert.equal(readOllamaConfig({}).preference.length, 4);
});

test('readOllamaConfig reads the environment, and says which values the user chose', () => {
  const cfg = readOllamaConfig({
    OLLAMA_BASE_URL: 'http://127.0.0.1:11500/v1/',
    OLLAMA_MODEL: ' gemma4:26b-a4b-it-q4_K_M ',
    OLLAMA_MODEL_PREFERENCE: 'a:1, b:2,, c:3 ',
    OLLAMA_NUM_CTX: '131072',
    OLLAMA_NUM_PREDICT: '20000',
    OLLAMA_THINK: 'ON',
    OLLAMA_KEEP_ALIVE: '1h30m',
    OLLAMA_TEMPERATURE: '0',
    OLLAMA_TIMEOUT_MS: '60000',
    OLLAMA_RETRIES: '0',
    OLLAMA_SEED: '0',
  });
  assert.deepEqual(cfg, {
    baseUrl: 'http://127.0.0.1:11500',
    baseUrlSource: 'OLLAMA_BASE_URL',
    model: 'gemma4:26b-a4b-it-q4_K_M',
    preference: ['a:1', 'b:2', 'c:3'],
    numCtx: 131072,
    numCtxExplicit: true,
    numPredict: 20000,
    numPredictExplicit: true,
    think: 'on',
    keepAlive: '1h30m',
    temperature: 0,
    timeoutMs: 60000,
    retries: 0,
    seed: 0,
    allowRemote: false,
  });
});

test('flags win over the environment', () => {
  const env = { OLLAMA_MODEL: 'env-model', OLLAMA_NUM_CTX: '131072', OLLAMA_THINK: 'on', OLLAMA_BASE_URL: 'http://127.0.0.1:11500' };
  const cfg = readOllamaConfig(env, { model: 'flag-model', url: 'http://localhost:11600', numCtx: '8192', think: 'off' });
  assert.equal(cfg.model, 'flag-model');
  assert.equal(cfg.baseUrl, 'http://localhost:11600');
  assert.equal(cfg.baseUrlSource, 'flag');
  assert.equal(cfg.numCtx, 8192);
  assert.equal(cfg.numCtxExplicit, true, 'a --num-ctx value is a choice, so it is never raised');
  assert.equal(cfg.think, 'off');

  // The spellings a flag parser may produce for --num-ctx, and a numeric value.
  assert.equal(readOllamaConfig({}, { 'num-ctx': 98304 }).numCtx, 98304);
  assert.equal(readOllamaConfig({}, { num_ctx: '98304' }).numCtx, 98304);
  assert.equal(readOllamaConfig({}, { numCtx: 98304 }).numCtxExplicit, true);
  // A flag that was not given leaves the environment in charge.
  assert.equal(readOllamaConfig(env, { model: undefined, numCtx: undefined, think: undefined, url: undefined }).model, 'env-model');
  assert.equal(readOllamaConfig(env, {}).numCtx, 131072);
  assert.match(configError(() => readOllamaConfig({}, { numCtx: '8k' })), /^Invalid --num-ctx: "8k" — must be a positive integer\.$/);
  assert.match(configError(() => readOllamaConfig({}, { think: 'maybe' })), /^Invalid --think: "maybe" — must be auto, on or off\.$/);
});

test('an invalid value is an error that names the variable and the value', () => {
  const bad = (env) => configError(() => readOllamaConfig(env));
  assert.equal(bad({ OLLAMA_NUM_CTX: '32k' }), 'Invalid OLLAMA_NUM_CTX: "32k" — must be a positive integer.');
  assert.equal(bad({ OLLAMA_NUM_PREDICT: '0' }), 'Invalid OLLAMA_NUM_PREDICT: "0" — must be a positive integer.');
  assert.equal(bad({ OLLAMA_TIMEOUT_MS: '5m' }), 'Invalid OLLAMA_TIMEOUT_MS: "5m" — must be a positive integer of at most 2147483647.');
  assert.match(bad({ OLLAMA_TIMEOUT_MS: '2147483648' }), /^Invalid OLLAMA_TIMEOUT_MS/);
  assert.equal(bad({ OLLAMA_RETRIES: '4' }), 'Invalid OLLAMA_RETRIES: "4" — must be an integer from 0 to 3.');
  assert.match(bad({ OLLAMA_RETRIES: 'twice' }), /^Invalid OLLAMA_RETRIES/);
  assert.equal(bad({ OLLAMA_SEED: '-1' }), 'Invalid OLLAMA_SEED: "-1" — must be a non-negative integer.');
  assert.equal(bad({ OLLAMA_TEMPERATURE: '2.5' }), 'Invalid OLLAMA_TEMPERATURE: "2.5" — must be a number from 0 to 2.');
  assert.match(bad({ OLLAMA_TEMPERATURE: 'hot' }), /^Invalid OLLAMA_TEMPERATURE/);
  assert.match(bad({ OLLAMA_TEMPERATURE: '-0.1' }), /^Invalid OLLAMA_TEMPERATURE/);
  assert.equal(bad({ OLLAMA_THINK: 'maybe' }), 'Invalid OLLAMA_THINK: "maybe" — must be auto, on or off.');
  assert.match(bad({ OLLAMA_KEEP_ALIVE: 'soon' }), /^Invalid OLLAMA_KEEP_ALIVE: "soon"/);
  assert.match(bad({ OLLAMA_BASE_URL: 'ftp://localhost' }), /^Invalid OLLAMA_BASE_URL/);
});

test('the accepted spellings of each setting', () => {
  assert.equal(readOllamaConfig({ OLLAMA_RETRIES: '3' }).retries, 3);
  assert.equal(readOllamaConfig({ OLLAMA_TEMPERATURE: '1.5' }).temperature, 1.5);
  assert.equal(readOllamaConfig({ OLLAMA_TEMPERATURE: '2' }).temperature, 2);
  assert.equal(readOllamaConfig({ OLLAMA_SEED: '42' }).seed, 42);
  for (const [raw, think] of [['auto', 'auto'], ['off', 'off'], ['Off', 'off'], ['false', 'off'], ['on', 'on'], ['true', 'on']]) {
    assert.equal(readOllamaConfig({ OLLAMA_THINK: raw }).think, think, raw);
  }
  assert.equal(readOllamaConfig({}, { think: true }).think, 'on');
  assert.equal(readOllamaConfig({}, { think: false }).think, 'off');
  // The server takes a number of seconds, or a duration WITH a unit. "300" as a
  // string is neither, so a bare number is handed on as a number.
  assert.equal(readOllamaConfig({ OLLAMA_KEEP_ALIVE: '300' }).keepAlive, 300);
  assert.equal(readOllamaConfig({ OLLAMA_KEEP_ALIVE: '-1' }).keepAlive, -1);
  assert.equal(readOllamaConfig({ OLLAMA_KEEP_ALIVE: '0' }).keepAlive, 0);
  assert.equal(readOllamaConfig({ OLLAMA_KEEP_ALIVE: '10m' }).keepAlive, '10m');
});

test('readOllamaConfig itself refuses a remote endpoint, so no caller can forget to', () => {
  const refused = configError(() => readOllamaConfig({ OLLAMA_BASE_URL: 'http://192.168.1.5:11434' }));
  assert.match(refused, /OLLAMA_ALLOW_REMOTE=1/);
  configError(() => readOllamaConfig({ OLLAMA_HOST: '192.168.1.5:11434' }));
  configError(() => readOllamaConfig({}, { url: 'https://ollama.example.test' }));

  const allowed = readOllamaConfig({ OLLAMA_BASE_URL: 'http://192.168.1.5:11434', OLLAMA_ALLOW_REMOTE: '1' });
  assert.equal(allowed.baseUrl, 'http://192.168.1.5:11434');
  assert.equal(allowed.allowRemote, true);
  assert.equal(readOllamaConfig({ OLLAMA_HOST: '192.168.1.5', OLLAMA_ALLOW_REMOTE: '1' }).baseUrlSource, 'OLLAMA_HOST');
});

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

test('embedding models are told apart from chat models', () => {
  for (const name of ['bge-m3:latest', 'nomic-embed-text:latest', 'all-minilm:latest', 'mxbai-embed-large', 'snowflake-arctic-embed:335m', 'qwen3-embedding:8b', 'registry.example.test/team/embed-small']) {
    assert.equal(isEmbeddingModel(name), true, name);
  }
  for (const name of ['qwen3.8:27b', 'gemma4:26b-a4b-it-q4_K_M', 'glm-4.7-flash:latest', 'qwen3-coder:30b', 'mistral-small:latest', 'deepseek-coder:33b-instruct', 'phi4-reasoning:plus']) {
    assert.equal(isEmbeddingModel(name, 'llama'), false, name);
  }
  // The family decides when the name says nothing.
  assert.equal(isEmbeddingModel('paraphrase-multilingual:latest', 'bert'), true);
  assert.equal(isEmbeddingModel('house-model:1', 'nomic-bert'), true);
  assert.equal(isEmbeddingModel('house-model:1', undefined), false);
});

test('sameModel treats a missing tag as :latest', () => {
  assert.equal(sameModel('mistral-small', 'mistral-small:latest'), true);
  assert.equal(sameModel('mistral-small:latest', 'mistral-small'), true);
  assert.equal(sameModel('Gemma4:26B-A4B-IT-Q4_k_m', 'gemma4:26b-a4b-it-q4_K_M'), true);
  assert.equal(sameModel('qwen3.8:27b', 'qwen3.8:latest'), false);
  assert.equal(sameModel('qwen3.8', 'qwen3.8:27b'), false);
  // The colon before the slash is a registry port, not a tag.
  assert.equal(sameModel('registry.example.test:5000/team/model', 'registry.example.test:5000/team/model:latest'), true);
  assert.equal(sameModel('', ''), false);
  assert.equal(sameModel(undefined, 'x'), false);
});

test('pickModel takes the first preferred model that is installed', () => {
  assert.deepEqual(pickModel({ preference: DEFAULT_MODEL_PREFERENCE, installed: INSTALLED }), { model: 'qwen3.8:27b', reason: 'preferred' });
  // Preference order decides, not the order the server lists them in.
  const withoutFirst = INSTALLED.filter((name) => name !== 'qwen3.8:27b');
  assert.deepEqual(pickModel({ preference: DEFAULT_MODEL_PREFERENCE, installed: withoutFirst }), { model: 'gemma4:26b-a4b-it-q4_K_M', reason: 'preferred' });
  assert.deepEqual(
    pickModel({ preference: ['glm-4.7-flash:latest', 'qwen3.8:27b'], installed: INSTALLED }),
    { model: 'glm-4.7-flash:latest', reason: 'preferred' },
  );
  // No list given: the default one.
  assert.deepEqual(pickModel({ installed: ['qwen3-coder:30b', 'tinyllama:latest'] }), { model: 'qwen3-coder:30b', reason: 'preferred' });
  assert.deepEqual(pickModel({ requested: null, preference: [], installed: INSTALLED }), { model: 'qwen3.8:27b', reason: 'preferred' });
});

test('pickModel honours a requested model, in the spelling the server uses', () => {
  assert.deepEqual(pickModel({ requested: 'glm-4.7-flash:latest', preference: DEFAULT_MODEL_PREFERENCE, installed: INSTALLED }), { model: 'glm-4.7-flash:latest', reason: 'requested' });
  // :latest tolerance, both ways round.
  assert.deepEqual(pickModel({ requested: 'mistral-small', installed: INSTALLED }), { model: 'mistral-small:latest', reason: 'requested' });
  assert.deepEqual(pickModel({ requested: 'tinyllama:latest', installed: ['tinyllama'] }), { model: 'tinyllama', reason: 'requested' });
  assert.deepEqual(pickModel({ preference: ['glm-4.7-flash'], installed: INSTALLED }), { model: 'glm-4.7-flash:latest', reason: 'preferred' });
  assert.deepEqual(pickModel({ requested: '  qwen3.8:27b ', installed: INSTALLED }), { model: 'qwen3.8:27b', reason: 'requested' });
});

test('a requested model that is not installed is an error listing what is', () => {
  let caught;
  try {
    pickModel({ requested: 'llama3.3', preference: DEFAULT_MODEL_PREFERENCE, installed: INSTALLED });
  } catch (err) {
    caught = err;
  }
  assert.ok(caught instanceof OllamaModelError);
  assert.equal(caught.code, 'OLLAMA_MODEL');
  assert.equal(
    caught.message,
    'Model "llama3.3" is not installed in Ollama. Installed chat models: qwen3.8:27b, gemma4:26b-a4b-it-q4_K_M, '
    + 'glm-4.7-flash:latest, qwen3-coder:30b, mistral-small:latest. Set OLLAMA_MODEL=qwen3.8:27b or run: ollama pull llama3.3',
  );
  assert.deepEqual(caught.installed, ['qwen3.8:27b', 'gemma4:26b-a4b-it-q4_K_M', 'glm-4.7-flash:latest', 'qwen3-coder:30b', 'mistral-small:latest']);
  assert.equal(caught.suggestion, 'qwen3.8:27b');

  // The suggestion is the first PREFERRED model that is installed, else the first installed.
  const noDefault = ['mistral-small:latest', 'qwen3-coder:30b'];
  assert.throws(() => pickModel({ requested: 'llama3.3', installed: noDefault }), (err) => err.suggestion === 'qwen3-coder:30b');
  assert.throws(() => pickModel({ requested: 'llama3.3', installed: ['mistral-small:latest'] }), (err) => err.suggestion === 'mistral-small:latest'
    && err.message.endsWith('Set OLLAMA_MODEL=mistral-small:latest or run: ollama pull llama3.3'));
  // No near-miss matching: a different tag is a different model.
  assert.throws(() => pickModel({ requested: 'qwen3.8:latest', installed: INSTALLED }), OllamaModelError);
});

test('with no preferred model installed nothing is guessed', () => {
  let caught;
  try {
    pickModel({ preference: DEFAULT_MODEL_PREFERENCE, installed: ['mistral-small:latest', 'bge-m3:latest', 'tinyllama:latest'] });
  } catch (err) {
    caught = err;
  }
  assert.ok(caught instanceof OllamaModelError);
  assert.equal(
    caught.message,
    'None of the preferred models are installed (qwen3.8:27b, gemma4:26b-a4b-it-q4_K_M, glm-4.7-flash:latest, qwen3-coder:30b). '
    + 'Installed chat models: mistral-small:latest, tinyllama:latest. Set OLLAMA_MODEL to one of them.',
  );
  assert.deepEqual(caught.installed, ['mistral-small:latest', 'tinyllama:latest']);
  assert.equal(caught.suggestion, 'mistral-small:latest');
});

test('embedding models are left out of every list', () => {
  // The part of a message that lists what is installed.
  const listed = (err) => err.message.split('Installed chat models: ')[1].split('. ')[0];
  // Not offered, not listed, and not usable even when asked for by name.
  assert.throws(() => pickModel({ requested: 'bge-m3:latest', installed: INSTALLED }), (err) => err instanceof OllamaModelError
    && !err.installed.some((name) => /bge|embed|minilm/.test(name))
    && listed(err) === 'qwen3.8:27b, gemma4:26b-a4b-it-q4_K_M, glm-4.7-flash:latest, qwen3-coder:30b, mistral-small:latest');
  assert.throws(() => pickModel({ preference: ['nomic-embed-text:latest'], installed: INSTALLED }), (err) => err instanceof OllamaModelError
    && err.message.startsWith('None of the preferred models are installed (nomic-embed-text:latest). '));
  assert.deepEqual(pickModel({ preference: ['bge-m3:latest', 'mistral-small'], installed: INSTALLED }), { model: 'mistral-small:latest', reason: 'preferred' });

  // listModels() entries work as well as names. Their flag is honoured where
  // the name says nothing, and so is their family.
  const entries = [
    { name: 'vector-store:1', family: 'llama', isEmbedding: true },
    { name: 'paraphrase-multilingual:latest', family: 'bert', isEmbedding: false },
    { name: 'qwen3-coder:30b', family: 'qwen3moe', isEmbedding: false },
  ];
  assert.deepEqual(pickModel({ installed: entries }), { model: 'qwen3-coder:30b', reason: 'preferred' });
  assert.throws(() => pickModel({ requested: 'vector-store:1', installed: entries }), (err) => err.installed.join() === 'qwen3-coder:30b');
  assert.throws(() => pickModel({ requested: 'paraphrase-multilingual', installed: entries }), (err) => err.installed.join() === 'qwen3-coder:30b');
  // As a bare name the same model is a chat model: there is nothing to say otherwise.
  assert.deepEqual(pickModel({ requested: 'vector-store:1', installed: ['vector-store:1'] }), { model: 'vector-store:1', reason: 'requested' });
});

test('an empty server is reported as such', () => {
  assert.throws(() => pickModel({ requested: 'qwen3.8:27b', installed: [] }), (err) => err instanceof OllamaModelError
    && err.message === 'Model "qwen3.8:27b" is not installed in Ollama. Installed chat models: (none). Run: ollama pull qwen3.8:27b'
    && err.suggestion === '' && err.installed.length === 0);
  assert.throws(() => pickModel({ installed: undefined }), (err) => err instanceof OllamaModelError
    && err.message.endsWith('Installed chat models: (none). Run: ollama pull qwen3.8:27b'));
  // Only embedding models installed is the same as none.
  assert.throws(() => pickModel({ installed: ['bge-m3:latest'] }), /Installed chat models: \(none\)/);
});

// ---------------------------------------------------------------------------
// Thinking and output budget
// ---------------------------------------------------------------------------

test('resolveThink: the whole table', () => {
  const THINKING = ['completion', 'tools', 'thinking'];
  const PLAIN = ['completion', 'tools'];
  for (const [setting, model, capabilities, expected] of [
    // auto is off for every model that can think. gemma4 is the one that used
    // to be on: measured with the runner's prompt it spent its whole output
    // budget thinking in two runs of three, and wrote a complete report in
    // 19-26 s every time with thinking off.
    ['auto', 'qwen3.8:27b', THINKING, false],
    ['auto', 'glm-4.7-flash:latest', THINKING, false],
    ['auto', 'gemma4:26b-a4b-it-q4_K_M', THINKING, false],
    ['auto', 'registry.example.test/team/gemma4:26b', THINKING, false],
    ['auto', 'GEMMA4:latest', THINKING, false],
    ['auto', 'deepseek-r1:32b', THINKING, false],
    ['auto', 'phi4-reasoning:plus', ['thinking', 'completion'], false],
    // auto, on a model without the capability: the field is left out
    ['auto', 'qwen3-coder:30b', PLAIN, undefined],
    ['auto', 'mistral-small:latest', PLAIN, undefined],
    // on / off are honoured only where the server would accept the field
    ['on', 'qwen3.8:27b', THINKING, true],
    ['on', 'gemma4:26b-a4b-it-q4_K_M', THINKING, true],
    ['off', 'gemma4:26b-a4b-it-q4_K_M', THINKING, false],
    ['on', 'mistral-small:latest', PLAIN, undefined],
    ['off', 'mistral-small:latest', PLAIN, undefined],
    // capabilities unknown (an older server): nothing is sent
    ['on', 'qwen3.8:27b', undefined, undefined],
    ['auto', 'gemma4:26b', [], undefined],
    // no setting means auto
    [undefined, 'gemma4:26b', THINKING, false],
    [undefined, 'qwen3.8:27b', THINKING, false],
    // the model's name plays no part, so neither does its absence
    ['auto', undefined, THINKING, false],
    ['on', undefined, THINKING, true],
  ]) {
    assert.equal(resolveThink({ setting, model, capabilities }), expected, `${setting} / ${model} / ${JSON.stringify(capabilities)}`);
  }
  assert.match(configError(() => resolveThink({ setting: 'sometimes', model: 'qwen3.8:27b', capabilities: THINKING })), /must be auto, on or off/);
});

test('with the default settings no model thinks, so none gets the larger output cap', () => {
  // The two decisions as the runner chains them. A thinking default would also
  // have raised the cap to 28000 and, with it, the room the window must leave.
  const THINKING = ['completion', 'tools', 'thinking'];
  const cfg = readOllamaConfig({});
  for (const model of DEFAULT_MODEL_PREFERENCE) {
    const think = resolveThink({ setting: cfg.think, model, capabilities: THINKING });
    assert.equal(think, false, model);
    assert.equal(resolveNumPredict({ numPredict: cfg.numPredict, numPredictExplicit: cfg.numPredictExplicit, think }), 16384, model);
  }
  const forced = readOllamaConfig({ OLLAMA_THINK: 'on' });
  assert.equal(resolveThink({ setting: forced.think, model: 'gemma4:26b-a4b-it-q4_K_M', capabilities: THINKING }), true);
});

test('resolveNumPredict gives a thinking model more room, unless the user chose a cap', () => {
  assert.equal(resolveNumPredict({ numPredict: 16384, numPredictExplicit: false, think: true }), 28000);
  assert.equal(resolveNumPredict({ numPredict: 16384, numPredictExplicit: true, think: true }), 16384);
  assert.equal(resolveNumPredict({ numPredict: 40000, numPredictExplicit: true, think: true }), 40000);
  assert.equal(resolveNumPredict({ numPredict: 16384, numPredictExplicit: false, think: false }), 16384);
  assert.equal(resolveNumPredict({ numPredict: 16384, numPredictExplicit: false, think: undefined }), 16384);
  assert.equal(resolveNumPredict({ think: false }), 16384);
  assert.equal(resolveNumPredict({ think: true }), 28000);
});

// ---------------------------------------------------------------------------
// evaluationPromptTokens
// ---------------------------------------------------------------------------

test('evaluationPromptTokens is the files, the prompt’s own wording and the longest posting', () => {
  // The one rule the runner and both setup checks size a window with before a
  // posting is known. While the runner and doctor.mjs each had their own, a
  // window pinned at the number one of them named was refused by the other.
  const empty = evaluationPromptTokens([]);
  // 4,000 tokens for the 16,000 characters the page reader returns at most,
  // 1,200 for the wording of the prompt around its files.
  assert.equal(empty, 5200);
  assert.equal(evaluationPromptTokens(), empty);
  // Four characters to a token, rounded up for each text, a run of whitespace counted as one space.
  assert.equal(evaluationPromptTokens(['a'.repeat(400), 'b'.repeat(401)]), empty + 100 + 101);
  assert.equal(evaluationPromptTokens(['a  \n\n b']), empty + 1);
  // A file that is absent is passed as nothing, or as an empty text.
  assert.equal(evaluationPromptTokens(['abcd', undefined, null, '']), empty + 1);

  // What is assumed while there is no CV to measure: about the size the
  // example files give, and one the default window holds.
  assert.equal(TYPICAL_PROMPT_TOKENS, 44000);
  assert.deepEqual(
    fitContext({ numCtx: DEFAULTS.numCtx, modelMaxCtx: 262144, promptTokens: TYPICAL_PROMPT_TOKENS, numPredict: DEFAULTS.numPredict }),
    { numCtx: 65536, ok: true, need: 58200, raised: false, clamped: false },
  );
});

// ---------------------------------------------------------------------------
// fitContext
// ---------------------------------------------------------------------------

test('fitContext leaves a window alone when the prompt fits', () => {
  // The measured prompt: 40.6K tokens estimated, on a 262144 model.
  assert.deepEqual(
    fitContext({ numCtx: 65536, explicit: false, modelMaxCtx: 262144, promptTokens: 40600, numPredict: 16384 }),
    { numCtx: 65536, ok: true, need: 54630, raised: false, clamped: false },
  );
  // With thinking off the reserve is min(numPredict, 12000): a large cap alone does not inflate it...
  assert.equal(fitContext({ numCtx: 65536, modelMaxCtx: 262144, promptTokens: 40600, numPredict: 28000 }).need, 54630);
  assert.equal(fitContext({ numCtx: 65536, modelMaxCtx: 262144, promptTokens: 40600, numPredict: 28000, think: false }).need, 54630);
  assert.equal(fitContext({ numCtx: 65536, modelMaxCtx: 262144, promptTokens: 40600, numPredict: 28000, think: undefined }).need, 54630);
  // ...and a small cap shrinks it.
  assert.equal(fitContext({ numCtx: 65536, modelMaxCtx: 262144, promptTokens: 40600, numPredict: 4000 }).need, 46630);
  // need = ceil(prompt * 1.05) + reserve
  assert.equal(fitContext({ promptTokens: 1001, numPredict: 100 }).need, 1052 + 100);
  // A prompt that needs exactly the window fits it; one token more does not.
  assert.deepEqual(
    fitContext({ numCtx: 65536, explicit: false, modelMaxCtx: 262144, promptTokens: 50986, numPredict: 16384 }),
    { numCtx: 65536, ok: true, need: 65536, raised: false, clamped: false },
  );
  assert.deepEqual(
    fitContext({ numCtx: 65536, explicit: false, modelMaxCtx: 262144, promptTokens: 50987, numPredict: 16384 }),
    { numCtx: 98304, ok: true, need: 65537, raised: true, clamped: false },
  );
});

test('with thinking on, fitContext reserves the whole output cap', () => {
  // The reasoning and the report share the cap, and a model has spent 25.5K
  // tokens on the reasoning alone. With 12000 reserved, a 48K prompt kept the
  // 65536 window and left a thinking reply 15K of the 28K it may use.
  const base = { numCtx: 65536, explicit: false, modelMaxCtx: 262144, promptTokens: 48000, numPredict: 28000 };
  assert.deepEqual(fitContext(base), { numCtx: 65536, ok: true, need: 62400, raised: false, clamped: false });
  assert.deepEqual(fitContext({ ...base, think: true }), { numCtx: 98304, ok: true, need: 78400, raised: true, clamped: false });
  // The measured evaluation prompt, with thinking forced on: one step up.
  assert.deepEqual(
    fitContext({ ...base, promptTokens: 40600, think: true }),
    { numCtx: 98304, ok: true, need: 70630, raised: true, clamped: false },
  );
  // A cap the user chose is the cap: it is what the reply can reach, small or large.
  assert.equal(fitContext({ ...base, numPredict: 6000, think: true }).need, 50400 + 6000);
  assert.equal(fitContext({ ...base, numPredict: 40000, think: true }).need, 50400 + 40000);
  // Only `true` is thinking on. A caller that passes the setting's text has not decided anything.
  for (const think of [false, undefined, null, 'on', 1]) {
    assert.equal(fitContext({ ...base, think }).need, 62400, String(think));
  }
  // When it does not fit, the reason says why the room is that large: leaving
  // thinking off is the other way out.
  assert.deepEqual(
    fitContext({ ...base, promptTokens: 40600, explicit: true, think: true }),
    {
      numCtx: 65536, ok: false, need: 70630, raised: false, clamped: false,
      reason: 'prompt ~40600 tokens + 28000 output (thinking on) needs 70630; num_ctx is 65536 (model max 262144)',
    },
  );
});

test('fitContext raises an unset window along the ladder until the prompt fits', () => {
  assert.deepEqual(
    fitContext({ numCtx: 65536, explicit: false, modelMaxCtx: 262144, promptTokens: 60000, numPredict: 16384 }),
    { numCtx: 98304, ok: true, need: 75000, raised: true, clamped: false },
  );
  assert.deepEqual(
    fitContext({ numCtx: 65536, explicit: false, modelMaxCtx: 262144, promptTokens: 100000, numPredict: 16384 }),
    { numCtx: 131072, ok: true, need: 117000, raised: true, clamped: false },
  );
  // Raised, then held to what the model has.
  assert.deepEqual(
    fitContext({ numCtx: 65536, explicit: false, modelMaxCtx: 80000, promptTokens: 60000, numPredict: 16384 }),
    { numCtx: 80000, ok: true, need: 75000, raised: true, clamped: true },
  );
  // Past the last step nothing fits, and the model's larger limit is not used unasked.
  const tooBig = fitContext({ numCtx: 65536, explicit: false, modelMaxCtx: 262144, promptTokens: 130000, numPredict: 16384 });
  assert.equal(tooBig.ok, false);
  assert.equal(tooBig.numCtx, 131072);
  assert.equal(tooBig.reason, 'prompt ~130000 tokens + 12000 output needs 148500; num_ctx is 131072 (model max 262144)');
});

test('fitContext clamps to the model, and says so when the prompt then does not fit', () => {
  // mistral-small and phi4-reasoning on this machine: a 32768 ceiling.
  assert.deepEqual(
    fitContext({ numCtx: 65536, explicit: false, modelMaxCtx: 32768, promptTokens: 40600, numPredict: 16384 }),
    {
      numCtx: 32768, ok: false, need: 54630, raised: false, clamped: true,
      reason: 'prompt ~40600 tokens + 12000 output needs 54630; num_ctx is 32768 (model max 32768)',
    },
  );
  // A small prompt on the same model is fine: the clamp alone is not a failure.
  assert.deepEqual(
    fitContext({ numCtx: 65536, explicit: false, modelMaxCtx: 32768, promptTokens: 10000, numPredict: 16384 }),
    { numCtx: 32768, ok: true, need: 22500, raised: false, clamped: true },
  );
});

test('fitContext never raises a window the user chose', () => {
  assert.deepEqual(
    fitContext({ numCtx: 32768, explicit: true, modelMaxCtx: 262144, promptTokens: 40600, numPredict: 16384 }),
    {
      numCtx: 32768, ok: false, need: 54630, raised: false, clamped: false,
      reason: 'prompt ~40600 tokens + 12000 output needs 54630; num_ctx is 32768 (model max 262144)',
    },
  );
  // A chosen value above the model's limit is still held to it.
  assert.deepEqual(
    fitContext({ numCtx: 300000, explicit: true, modelMaxCtx: 262144, promptTokens: 40600, numPredict: 16384 }),
    { numCtx: 262144, ok: true, need: 54630, raised: false, clamped: true },
  );
  // An unknown limit (the server did not report one) clamps nothing.
  assert.deepEqual(
    fitContext({ numCtx: 8192, explicit: true, modelMaxCtx: null, promptTokens: 40600, numPredict: 16384 }),
    {
      numCtx: 8192, ok: false, need: 54630, raised: false, clamped: false,
      reason: 'prompt ~40600 tokens + 12000 output needs 54630; num_ctx is 8192 (model max unknown)',
    },
  );
});

// ---------------------------------------------------------------------------
// The module stays loadable without node_modules
// ---------------------------------------------------------------------------

test('lib/ollama-config.mjs imports nothing', () => {
  // doctor.mjs loads this file with its Ollama check, and doctor is the script
  // that reports a missing `npm install`: what it loads must not need one.
  const source = readFileSync(fileURLToPath(new URL('../lib/ollama-config.mjs', import.meta.url)), 'utf8');
  assert.doesNotMatch(source, /^\s*import\s/m, 'the module is pure: no imports at all');
  assert.doesNotMatch(source, /\bimport\s*\(/, 'and no dynamic ones');
  assert.doesNotMatch(source, /\bfetch\s*\(/, 'no network');
  assert.doesNotMatch(source, /process\.exit\s*\(/, 'a library never ends the process');
});

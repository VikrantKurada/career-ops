// tests/ollama-doctor.test.mjs — the setup check for the local Ollama backend
// (lib/ollama-doctor.mjs) and the place doctor.mjs runs it from.
//
// Two things can go wrong here that no other suite would see.
//
// The check. It decides from the server's metadata alone whether an evaluation
// can run: which model, whether the prompt fits its window, how thinking is
// set, what is loaded. A wrong "fits" is the expensive mistake, because Ollama
// cuts a prompt that does not fit and answers anyway. A wrong severity is the
// annoying one: doctor is the first thing a new user runs, and most users have
// no Ollama server.
//
// The wiring. Plain `node doctor.mjs` has to stay offline unless the runner
// itself is configured: the Ollama server's own variables and --strict are
// not that. `--json` is read by every agent at the start of every session, so
// it must not change by a byte or make a request. And a run that did talk to
// the server must not end through process.exit: on Windows, Node 24 aborts
// there after a request (nodejs/node#56645), which turns a passing report
// into a crash code.
//
// A third thing can only go wrong between two programs. doctor.mjs and the
// runner each say whether a context window holds the evaluation prompt, and
// the runner is the one that then refuses to send. A number the check
// recommends has to be one the runner accepts, so the last section starts
// both on the same files and compares them.
//
// No Ollama is needed and none is contacted. The check is driven through an
// injected fetch. doctor.mjs is started as a child with a preload that
// replaces fetch, records every request and reports any process.exit call;
// the runs that need real sockets (the crash only happens with them) or that
// start the runner go over real HTTP to a stub on 127.0.0.1 instead.
//
// Run:  node --test tests/ollama-doctor.test.mjs
import { after, test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { isBuiltin } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { estimateTokens } from '../lib/context-budget.mjs';
import { RUNNER_ENV_VARS, evaluationPromptTokens } from '../lib/ollama-config.mjs';
import { checkOllama, measurePromptTokens, ollamaOptIn } from '../lib/ollama-doctor.mjs';
import { linkNodeModules } from './helpers.mjs';
import { describeSpawn } from './helpers/describe-spawn.mjs';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const DOCTOR = join(ROOT, 'doctor.mjs');
const RUNNER = join(ROOT, 'ollama-runner.mjs');
// Replaces the playwright package with a browser that starts at once, so a
// doctor run here launches no Chromium and does not depend on one.
const PLAYWRIGHT_STUB = pathToFileURL(join(ROOT, 'tests', 'fixtures', 'playwright-stub.mjs')).href;
const DEFAULT_URL = 'http://localhost:11434';

// ---------------------------------------------------------------------------
// A stub Ollama
// ---------------------------------------------------------------------------

const QWEN = 'qwen3.8:27b';
const GEMMA = 'gemma4:26b-a4b-it-q4_K_M';
const CODER = 'qwen3-coder:30b';
const SMALL = 'mistral-small:latest';
const EMBED = 'nomic-embed-text:latest';
const GIB = 1024 ** 3;

// The shapes are the ones Ollama 0.40.16 returns; the values are the limits
// and capabilities of the models the defaults were measured on.
const THINKER = ['completion', 'vision', 'tools', 'thinking'];
const MODELS = Object.freeze({
  [QWEN]: { family: 'qwen35', ctx: 262144, caps: THINKER },
  [GEMMA]: { family: 'gemma4', ctx: 262144, caps: THINKER },
  [CODER]: { family: 'qwen3moe', ctx: 262144, caps: ['completion', 'tools'] },
  [SMALL]: { family: 'llama', ctx: 32768, caps: ['completion', 'tools'] },
  [EMBED]: { family: 'nomic-bert', ctx: 2048, caps: ['embedding'] },
});

/** What the stub serves. `down` refuses every connection. */
const server = (overrides = {}) => ({ version: '0.40.16', models: MODELS, ps: [], status: {}, raw: {}, down: false, ...overrides });

/** One entry of /api/ps. */
const loaded = (name, { ctx, size = 17 * GIB, vram = size } = {}) => ({
  name, model: name, size, size_vram: vram, ...(ctx ? { context_length: ctx } : {}),
});

/**
 * The stub's whole behaviour: one request in, `{ status, json | text }` out,
 * or null for a server that is not there.
 *
 * It uses nothing but its arguments, because its source is also written into
 * the preload a child process loads (see preloadSource).
 */
function route(state, method, path, body) {
  if (state.down) return null;
  // `status` forces a reply code for a path and `raw` a body that is not JSON.
  if (state.status[path]) return { status: state.status[path], text: state.raw[path], json: { error: 'stub failure' } };
  if (state.raw[path]) return { status: 200, text: state.raw[path] };
  const details = (model) => ({ family: model.family, parameter_size: '27.3B', quantization_level: 'Q4_K_M' });
  if (method === 'GET' && path === '/api/version') return { status: 200, json: { version: state.version } };
  if (method === 'GET' && path === '/api/tags') {
    return {
      status: 200,
      json: {
        models: Object.entries(state.models).map(([name, model]) => ({
          name, model: name, size: 17000000000, details: details(model), capabilities: model.caps,
        })),
      },
    };
  }
  if (method === 'POST' && path === '/api/show') {
    const model = state.models[body && body.model];
    if (!model) return { status: 404, json: { error: `model '${body && body.model}' not found` } };
    const info = { 'general.architecture': model.family };
    if (model.ctx) info[`${model.family}.context_length`] = model.ctx;
    return { status: 200, json: { capabilities: model.caps, details: details(model), model_info: info } };
  }
  if (method === 'GET' && path === '/api/ps') return { status: 200, json: { models: state.ps } };
  return { status: 404, json: { error: 'not found' } };
}

/** A fetch that answers from the stub and records what was asked. */
function fakeFetch(state) {
  const calls = [];
  const hosts = new Set();
  const fetchImpl = async (input, init = {}) => {
    const url = new URL(String(input));
    const method = init.method ?? 'GET';
    const body = init.body ? JSON.parse(init.body) : null;
    hosts.add(url.origin);
    calls.push(`${method} ${url.pathname}${body?.model ? ` ${body.model}` : ''}`);
    const reply = route(state, method, url.pathname, body);
    if (!reply) throw new TypeError('fetch failed', { cause: Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED' }) });
    return new Response(reply.text ?? JSON.stringify(reply.json), { status: reply.status });
  };
  return { fetchImpl, calls, hosts };
}

/**
 * Run the check against the stub. `env` is always given: the developer's own
 * OLLAMA_* settings must not reach a test.
 */
async function check(env = {}, state = {}, options = {}) {
  const { fetchImpl, calls, hosts } = fakeFetch(server(state));
  const results = await checkOllama({ env, fetchImpl, ...options });
  return { results, calls, hosts: [...hosts] };
}

const METADATA = ['GET /api/version', 'GET /api/tags', `POST /api/show ${QWEN}`, 'GET /api/ps'];
const REACHABLE = { pass: true, label: `Ollama 0.40.16 reachable at ${DEFAULT_URL}` };
const PICKED = { pass: true, label: `Ollama model: ${QWEN} (preferred)` };
// With nothing to measure the check assumes a prompt of 44,000 tokens, the
// size of one built from the example files. NEED is what a window has to hold
// for it: the prompt plus 5%, plus 12,000 tokens for the reply.
const NEED = 58200;
const FITS = { pass: true, label: 'num_ctx 65536 fits ~44K prompt + output (model max 262144)' };
const TOO_SMALL_FOR = 'num_ctx too small: prompt ~44000 tokens + 12000 output needs 58200';
const NOT_THINKING = { pass: true, label: 'thinking: off (auto)' };
const HEALTHY = [REACHABLE, PICKED, FITS, NOT_THINKING];
const SMALLER_MODEL = 'a smaller model (node ollama-runner.mjs models)';

// The Ollama SERVER reads these. A machine with a tuned Ollama install has
// them in its environment whether or not career-ops ever talks to it; the
// first two were set on the machine this backend was built on before anything
// about the runner was configured.
const SERVER_VARIABLES = Object.freeze({
  OLLAMA_HOST: '0.0.0.0',
  OLLAMA_FLASH_ATTENTION: '1',
  OLLAMA_KEEP_ALIVE: '10m',
  OLLAMA_MODELS: 'D:\\ollama\\models',
  OLLAMA_KV_CACHE_TYPE: 'q8_0',
  OLLAMA_NUM_PARALLEL: '1',
  OLLAMA_MAX_LOADED_MODELS: '1',
  OLLAMA_ORIGINS: '*',
  OLLAMA_CONTEXT_LENGTH: '8192',
});
// The runner's own settings, written out: lib/ollama-config.mjs holds the list
// the code uses, and a name dropped from it or added to it has to show here.
const RUNNER_VARIABLES = [
  'OLLAMA_BASE_URL', 'OLLAMA_ALLOW_REMOTE', 'OLLAMA_MODEL', 'OLLAMA_MODEL_PREFERENCE', 'OLLAMA_NUM_CTX',
  'OLLAMA_NUM_PREDICT', 'OLLAMA_THINK', 'OLLAMA_TEMPERATURE', 'OLLAMA_TIMEOUT_MS', 'OLLAMA_RETRIES', 'OLLAMA_SEED',
];

// ---------------------------------------------------------------------------
// ollamaOptIn
// ---------------------------------------------------------------------------

test('ollamaOptIn: the variables of the Ollama server do not switch the check on', () => {
  assert.equal(ollamaOptIn({}), false);
  assert.equal(ollamaOptIn({ PATH: '/usr/bin', HOME: '/home/someone' }), false);
  assert.equal(ollamaOptIn(SERVER_VARIABLES), false);
  assert.equal(ollamaOptIn({}, SERVER_VARIABLES), false);
  for (const [name, value] of Object.entries(SERVER_VARIABLES)) {
    assert.equal(ollamaOptIn({ [name]: value }), false, `${name} in the shell`);
    assert.equal(ollamaOptIn({}, { [name]: value }), false, `${name} in the .env`);
  }
  // The prefix is Ollama's, and its server gains names with each release: one
  // that nobody has listed anywhere is not the runner's either.
  assert.equal(ollamaOptIn({ OLLAMA_SOMETHING_NEW: '1' }, { OLLAMA_DEBUG: '1' }), false);
});

test('ollamaOptIn: each of the runner\'s own settings switches it on, from the shell or from a .env', () => {
  assert.deepEqual([...RUNNER_ENV_VARS].sort(), [...RUNNER_VARIABLES].sort());
  for (const name of RUNNER_VARIABLES) {
    assert.equal(ollamaOptIn({ [name]: '1' }), true, `${name} in the shell`);
    assert.equal(ollamaOptIn({}, { [name]: '1' }), true, `${name} in the .env`);
    assert.equal(ollamaOptIn({ ...SERVER_VARIABLES, [name]: '1' }), true, `${name} beside the server's variables`);
  }
});

test('ollamaOptIn: a name set to nothing, or a name that only looks similar, does not count', () => {
  // `OLLAMA_MODEL=` is how a variable is cleared, and the settings reader
  // treats it as unset.
  assert.equal(ollamaOptIn({ OLLAMA_MODEL: '' }), false);
  assert.equal(ollamaOptIn({ OLLAMA_MODEL: '   ' }, { OLLAMA_NUM_CTX: '' }), false);
  assert.equal(ollamaOptIn({ OLLAMA_MODEL: undefined }), false);
  assert.equal(ollamaOptIn({ MY_OLLAMA_MODEL: 'x', OLLAMAX_MODEL: 'x', OLLAMA: 'x', OLLAMA_MODEL_: 'x' }), false);
  // A caller with no .env passes nothing for it.
  assert.equal(ollamaOptIn({ OLLAMA_MODEL: QWEN }, undefined), true);
  assert.equal(ollamaOptIn({}, undefined), false);
  assert.equal(ollamaOptIn(null, null), false);
});

// On Windows a variable is found whatever case it was typed in, and the runner
// reads the environment that way. Everywhere else `ollama_model` is another
// variable, which the runner does not read.
const IGNORES_CASE = process.platform === 'win32';

test('ollamaOptIn: a name typed in another case counts where the runner would find it, and only there', () => {
  assert.equal(ollamaOptIn({ ollama_model: QWEN }), IGNORES_CASE);
  assert.equal(ollamaOptIn({}, { Ollama_Num_Ctx: '65536' }), IGNORES_CASE);
  // The case of a name does not make a server variable the runner's.
  assert.equal(ollamaOptIn({ ollama_host: '0.0.0.0', ollama_keep_alive: '10m' }), false);
});

// ---------------------------------------------------------------------------
// checkOllama: the healthy case and what it sends
// ---------------------------------------------------------------------------

test('a healthy server passes four checks, in order', async () => {
  const { results, calls, hosts } = await check();
  assert.deepEqual(results, HEALTHY);
  // Version, model list, one model's details, loaded models: nothing that
  // loads a model or generates a token.
  assert.deepEqual(calls, METADATA);
  assert.deepEqual(hosts, [DEFAULT_URL]);
});

test('the check never asks the server to generate, and every result has a shape doctor prints', async () => {
  const runs = await Promise.all([
    check(),
    check({}, { down: true }),
    check({}, { down: true }, { soft: true }),
    check({ OLLAMA_MODEL: 'llama3.3' }),
    check({ OLLAMA_MODEL: SMALL }),
    check({ OLLAMA_MODEL: GEMMA, OLLAMA_NUM_PREDICT: '16384' }),
    check({ OLLAMA_NUM_CTX: '8k' }),
    check({ OLLAMA_NUM_CTX: '300000' }),
    check({}, { ps: [loaded(QWEN, { ctx: 131072, size: 35 * GIB, vram: 29 * GIB })] }),
    check({}, { ps: [loaded(QWEN, { ctx: 65536, size: 35 * GIB, vram: 29 * GIB })] }),
    check({}, { ps: [loaded(QWEN, { ctx: 4096, size: 35 * GIB, vram: 29 * GIB })] }),
    check({}, { ps: [loaded(QWEN, { size: 35 * GIB, vram: 29 * GIB })] }),
    check({ OLLAMA_NUM_CTX: '45000' }, { ps: [loaded(QWEN, { ctx: 65536, size: 35 * GIB, vram: 29 * GIB })] }),
    check({}, { status: { '/api/show': 500 } }),
    check({}, { status: { '/api/show': 500 } }, { soft: true }),
    check({}, { raw: { '/api/tags': '<html>' } }),
    check({}, { raw: { '/api/tags': '<html>' } }, { soft: true }),
    check({ OLLAMA_MODEL: 'llama3.3' }, {}, { soft: true }),
    check({ OLLAMA_THINK: 'on' }),
    check({}, {}, { promptTokens: 60000 }),
  ]);
  const allowed = new Set(['GET /api/version', 'GET /api/tags', 'POST /api/show', 'GET /api/ps']);
  for (const { results, calls } of runs) {
    for (const call of calls) assert.ok(allowed.has(call.split(' ').slice(0, 2).join(' ')), `unexpected request: ${call}`);
    assert.ok(results.length >= 1);
    for (const result of results) {
      const keys = Object.keys(result).sort().join(',');
      if (keys === 'label,pass') assert.equal(result.pass, true, JSON.stringify(result));
      else if (keys === 'fix,label,warn') assert.equal(result.warn, true, JSON.stringify(result));
      else if (keys === 'fix,label,pass') assert.equal(result.pass, false, JSON.stringify(result));
      else assert.fail(`not a doctor result: ${JSON.stringify(result)}`);
      // One line each: doctor prints a label after its mark and a fix after an
      // arrow, and the runner prints the same two strings.
      for (const text of [result.label, ...('fix' in result ? [result.fix] : [])]) {
        assert.equal(typeof text, 'string', JSON.stringify(result));
        assert.ok(text.trim() === text && text.length > 0 && !text.includes('\n'), JSON.stringify(result));
      }
    }
  }
});

// ---------------------------------------------------------------------------
// checkOllama: settings
// ---------------------------------------------------------------------------

test('a setting the runner would refuse is one failure, before any request', async () => {
  const bad = await check({ OLLAMA_NUM_CTX: '8k' });
  assert.equal(bad.results.length, 1);
  assert.equal(bad.results[0].pass, false);
  assert.equal(bad.results[0].label, 'Ollama settings cannot be used');
  // The runner's own message, which names the variable and the value.
  assert.match(bad.results[0].fix, /^Invalid OLLAMA_NUM_CTX: "8k"/);
  assert.deepEqual(bad.calls, []);

  // Not softened. A server that is off may be on tomorrow; a bad value stays
  // bad, and the runner refuses to start until it is corrected.
  const soft = await check({ OLLAMA_NUM_CTX: '8k' }, {}, { soft: true });
  assert.deepEqual(soft.results, bad.results);
});

test('a server on another machine is refused without being contacted, unless it was allowed', async () => {
  const refused = await check({ OLLAMA_BASE_URL: 'http://192.168.1.5:11434' });
  assert.equal(refused.results.length, 1);
  assert.equal(refused.results[0].pass, false);
  assert.match(refused.results[0].fix, /^Remote Ollama endpoint: http:\/\/192\.168\.1\.5:11434\./);
  assert.match(refused.results[0].fix, /OLLAMA_ALLOW_REMOTE=1/);
  assert.deepEqual(refused.calls, []);

  const allowed = await check({ OLLAMA_BASE_URL: 'http://192.168.1.5:11434', OLLAMA_ALLOW_REMOTE: '1' });
  assert.deepEqual(allowed.hosts, ['http://192.168.1.5:11434']);
  assert.equal(allowed.results[0].label, 'Ollama 0.40.16 reachable at http://192.168.1.5:11434');
});

test('the server that is checked is the one the runner would use', async () => {
  const byUrl = await check({ OLLAMA_BASE_URL: 'http://127.0.0.1:11500/v1/' });
  assert.deepEqual(byUrl.hosts, ['http://127.0.0.1:11500']);
  assert.equal(byUrl.results[0].label, 'Ollama 0.40.16 reachable at http://127.0.0.1:11500');
  assert.deepEqual(byUrl.calls, METADATA);

  // OLLAMA_HOST naming a real host is a destination; a bind address is not.
  const byHost = await check({ OLLAMA_HOST: '127.0.0.1:11435' });
  assert.deepEqual(byHost.hosts, ['http://127.0.0.1:11435']);
  const bindAddress = await check({ OLLAMA_HOST: '0.0.0.0' });
  assert.deepEqual(bindAddress.hosts, [DEFAULT_URL]);
  assert.deepEqual(bindAddress.results, HEALTHY);
});

test('a setting typed in another case is checked where the runner would use it, and only there', async () => {
  // PowerShell stores `$env:ollama_model = 'llama3.3'` under the name as
  // typed. The runner reads process.env, which finds it on Windows whatever
  // its case; the check is handed a copy, which does not. So the runner used
  // llama3.3 and the report certified the preferred model.
  const lower = await check({ ollama_model: 'llama3.3' });
  if (IGNORES_CASE) {
    assert.equal(lower.results[1].pass, false);
    assert.equal(lower.results[1].label, 'Ollama model: llama3.3 is not installed');
  } else {
    assert.deepEqual(lower.results, HEALTHY);
  }
  const mixed = await check({ Ollama_Num_Ctx: '131072', ollama_base_url: 'http://127.0.0.1:11500' });
  assert.deepEqual(mixed.hosts, [IGNORES_CASE ? 'http://127.0.0.1:11500' : DEFAULT_URL]);
  assert.match(mixed.results[2].label, IGNORES_CASE ? /^num_ctx 131072 fits / : /^num_ctx 65536 fits /);

  // Callers put the weaker source first: the .env, then the shell, then
  // flags. When two spellings of one name meet, the later one decides, as it
  // does for the runner.
  const both = await check({ OLLAMA_MODEL: 'llama3.3', ollama_model: GEMMA });
  assert.equal(both.results[1].label, IGNORES_CASE ? `Ollama model: ${GEMMA} (requested)` : 'Ollama model: llama3.3 is not installed');
});

// ---------------------------------------------------------------------------
// checkOllama: 1. reachable
// ---------------------------------------------------------------------------

test('a server that is not running is a failure, and the check stops there', async () => {
  const { results, calls } = await check({}, { down: true });
  assert.deepEqual(results, [{
    pass: false,
    label: `Ollama not reachable at ${DEFAULT_URL}`,
    fix: 'Start Ollama (ollama serve) or set OLLAMA_BASE_URL',
  }]);
  assert.deepEqual(calls, ['GET /api/version']);
});

test('soft: a server that is not running is a warning', async () => {
  const { results, calls } = await check({}, { down: true }, { soft: true });
  // Exactly the warning shape: a `pass: false` left beside `warn` would still
  // print as a warning in doctor, and as a failure anywhere that reads `pass`.
  assert.deepEqual(results, [{
    warn: true,
    label: `Ollama not reachable at ${DEFAULT_URL}`,
    fix: 'Start Ollama (ollama serve) or set OLLAMA_BASE_URL',
  }]);
  assert.deepEqual(calls, ['GET /api/version']);
});

test('an OLLAMA_HOST that was not used is explained when the server does not answer', async () => {
  const { results } = await check({ OLLAMA_HOST: '0.0.0.0' }, { down: true });
  assert.deepEqual(results, [{
    pass: false,
    label: `Ollama not reachable at ${DEFAULT_URL}`,
    fix: 'Start Ollama (ollama serve) or set OLLAMA_BASE_URL. '
      + 'OLLAMA_HOST=0.0.0.0 is the server bind address and is not used as the client URL',
  }]);
  // Nothing to explain when OLLAMA_BASE_URL decided the address.
  const overridden = await check({ OLLAMA_HOST: '0.0.0.0', OLLAMA_BASE_URL: 'http://127.0.0.1:11500' }, { down: true });
  assert.equal(overridden.results[0].fix, 'Start Ollama (ollama serve) or set OLLAMA_BASE_URL');
});

test('something else answering on the port is reported as that, not as a server that is off', async () => {
  const notFound = await check({}, { status: { '/api/version': 404 } });
  assert.equal(notFound.results.length, 1);
  assert.equal(notFound.results[0].pass, false);
  assert.equal(notFound.results[0].label, `Ollama at ${DEFAULT_URL} could not be checked`);
  assert.match(notFound.results[0].fix, /HTTP 404/);
  assert.deepEqual(notFound.calls, ['GET /api/version']);

  const html = await check({ OLLAMA_HOST: '0.0.0.0' }, { raw: { '/api/version': '<html>router login</html>' } }, { soft: true });
  assert.equal(html.results.length, 1);
  assert.deepEqual(Object.keys(html.results[0]).sort(), ['fix', 'label', 'warn']);
  assert.equal(html.results[0].label, `Ollama at ${DEFAULT_URL} could not be checked`);
  assert.match(html.results[0].fix, /not JSON/);

  // The bind-address note follows what the server said as a sentence of its
  // own, whether or not that ended in a full stop.
  const note = /: gone\. OLLAMA_HOST=0\.0\.0\.0 is the server bind address and is not used as the client URL$/;
  for (const body of ['gone', 'gone.']) {
    const { results } = await check({ OLLAMA_HOST: '0.0.0.0' }, { status: { '/api/version': 410 }, raw: { '/api/version': body } });
    assert.match(results[0].fix, note, `body ${JSON.stringify(body)}`);
  }
  const noNote = await check({}, { status: { '/api/version': 410 }, raw: { '/api/version': 'gone' } });
  assert.match(noNote.results[0].fix, /: gone$/);
});

test('a server that reports no version is still a server', async () => {
  const { results } = await check({}, { version: undefined });
  assert.equal(results[0].label, `Ollama (version unknown) reachable at ${DEFAULT_URL}`);
  assert.deepEqual(results.slice(1), HEALTHY.slice(1));
});

// ---------------------------------------------------------------------------
// checkOllama: 2. model
// ---------------------------------------------------------------------------

test('a requested model that is not installed fails with the installed models, and the check stops', async () => {
  const { results, calls } = await check({ OLLAMA_MODEL: 'llama3.3' });
  assert.equal(results.length, 2);
  assert.deepEqual(results[0], REACHABLE);
  assert.equal(results[1].pass, false);
  assert.equal(results[1].label, 'Ollama model: llama3.3 is not installed');
  for (const name of [QWEN, GEMMA, CODER, SMALL]) assert.ok(results[1].fix.includes(name), results[1].fix);
  // An embedding model cannot chat, so it is not offered.
  assert.ok(!results[1].fix.includes(EMBED), results[1].fix);
  assert.match(results[1].fix, new RegExp(`Set OLLAMA_MODEL=${QWEN.replace('.', '\\.')} or run: ollama pull llama3\\.3$`));
  // No model, so nothing to ask details of.
  assert.deepEqual(calls, ['GET /api/version', 'GET /api/tags']);
});

test('soft: what the server holds is a warning, in the same words', async () => {
  // A run that was not asked for by name. Which models a server has, and
  // whether it can describe them, is the state of that server and not a
  // mistake in this install, so none of it may turn the setup check red.
  const softened = async (env, state) => {
    const [hard, soft] = [await check(env, state), await check(env, state, { soft: true })];
    const last = hard.results.at(-1);
    assert.equal(last.pass, false, JSON.stringify(last));
    assert.deepEqual(soft.results.slice(0, -1), hard.results.slice(0, -1));
    // Exactly the warning shape: a `pass: false` left beside `warn` would
    // still print as a warning in doctor, and as a failure anywhere that
    // reads `pass`.
    assert.deepEqual(soft.results.at(-1), { warn: true, label: last.label, fix: last.fix });
    assert.deepEqual(soft.calls, hard.calls);
    return last.label;
  };
  const onlyLlama = { models: { 'llama3.2:latest': { family: 'llama', ctx: 131072, caps: ['completion', 'tools'] } } };
  assert.equal(await softened({ OLLAMA_MODEL: 'llama3.3' }, {}), 'Ollama model: llama3.3 is not installed');
  assert.equal(await softened({}, onlyLlama), 'Ollama model: none of the preferred models is installed');
  assert.equal(await softened({}, { raw: { '/api/tags': '<html>' } }), 'Ollama model list could not be read');
  assert.equal(await softened({}, { status: { '/api/show': 500 } }), `Ollama model details could not be read for ${QWEN}`);
});

test('soft: a setting the runner refuses, and a window the prompt does not fit, stay failures', async () => {
  // These are wrong until someone edits them, whatever the server is doing,
  // and the runner stops on both however the check was started.
  for (const [env, label] of [
    [{ OLLAMA_NUM_CTX: '8k' }, 'Ollama settings cannot be used'],
    [{ OLLAMA_BASE_URL: 'http://192.168.1.5:11434' }, 'Ollama settings cannot be used'],
    [{ OLLAMA_NUM_CTX: '32768' }, `${TOO_SMALL_FOR}; num_ctx is 32768 (model max 262144)`],
    [{ OLLAMA_MODEL: SMALL }, `${TOO_SMALL_FOR}; num_ctx is 32768 (model max 32768)`],
  ]) {
    const [hard, soft] = [await check(env), await check(env, {}, { soft: true })];
    assert.deepEqual(soft.results, hard.results, JSON.stringify(env));
    const failures = soft.results.filter((result) => result.pass === false);
    assert.deepEqual(failures.map((result) => result.label), [label], JSON.stringify(env));
  }
});

test('no preferred model installed fails and says what is installed', async () => {
  const models = { 'llama3.2:latest': { family: 'llama', ctx: 131072, caps: ['completion', 'tools'] }, [EMBED]: MODELS[EMBED] };
  const { results, calls } = await check({}, { models });
  assert.equal(results.length, 2);
  assert.equal(results[1].pass, false);
  assert.equal(results[1].label, 'Ollama model: none of the preferred models is installed');
  assert.match(results[1].fix, /Installed chat models: llama3\.2:latest\. Set OLLAMA_MODEL to one of them\.$/);
  assert.deepEqual(calls, ['GET /api/version', 'GET /api/tags']);

  // The user's own preference list is honoured the same way as the default.
  const preferred = await check({ OLLAMA_MODEL_PREFERENCE: 'missing-model, llama3.2' }, { models });
  assert.deepEqual(preferred.results[1], { pass: true, label: 'Ollama model: llama3.2:latest (preferred)' });
});

test('a requested model is reported, and asked about, under the name the server uses', async () => {
  const { results, calls } = await check({ OLLAMA_MODEL: 'qwen3-coder:30b' });
  assert.deepEqual(results[1], { pass: true, label: `Ollama model: ${CODER} (requested)` });
  const untagged = await check({ OLLAMA_MODEL: 'mistral-small' });
  assert.deepEqual(untagged.results[1], { pass: true, label: `Ollama model: ${SMALL} (requested)` });
  assert.equal(untagged.calls[2], `POST /api/show ${SMALL}`);
  assert.equal(calls[2], `POST /api/show ${CODER}`);
});

test('a model list that is not one becomes a failure, not an exception', async () => {
  const { results, calls } = await check({}, { raw: { '/api/tags': '<html>' } });
  assert.equal(results.length, 2);
  assert.equal(results[1].pass, false);
  assert.equal(results[1].label, 'Ollama model list could not be read');
  assert.match(results[1].fix, /not JSON/);
  assert.deepEqual(calls, ['GET /api/version', 'GET /api/tags']);

  // An error page has many lines, and a fix is printed as one: the first.
  const page = await check({}, { status: { '/api/tags': 502 }, raw: { '/api/tags': '<html>\n<body>Bad Gateway</body>\n</html>' } });
  assert.deepEqual(page.results[1], {
    pass: false,
    label: 'Ollama model list could not be read',
    fix: 'Ollama API error: HTTP 502: <html>',
  });
});

// ---------------------------------------------------------------------------
// checkOllama: 3. context window
// ---------------------------------------------------------------------------

test('a model whose window cannot hold the prompt fails, and the remaining checks still run', async () => {
  const { results, calls } = await check({ OLLAMA_MODEL: SMALL });
  assert.deepEqual(results, [
    REACHABLE,
    { pass: true, label: `Ollama model: ${SMALL} (requested)` },
    {
      pass: false,
      label: `${TOO_SMALL_FOR}; num_ctx is 32768 (model max 32768)`,
      // Not "raise OLLAMA_NUM_CTX": this model cannot honour a larger one.
      fix: `Pick a model with a larger context (e.g. ${QWEN})`,
    },
    { pass: true, label: 'thinking: n/a (auto)' },
  ]);
  assert.equal(calls.length, 4);
});

test('a window the user pinned too small fails, and the advice is about the setting', async () => {
  const { results } = await check({ OLLAMA_NUM_CTX: '32768' });
  assert.deepEqual(results[2], {
    pass: false,
    label: `${TOO_SMALL_FOR}; num_ctx is 32768 (model max 262144)`,
    // Not "pick a larger model": this one holds 262144.
    fix: `Raise OLLAMA_NUM_CTX to ${NEED} or more, or unset it so the runner sizes the window`,
  });
  assert.deepEqual(results[3], NOT_THINKING);

  // A model whose limit is exactly what the prompt needs is large enough.
  const exact = await check({ OLLAMA_NUM_CTX: '32768' }, { models: { [QWEN]: { ...MODELS[QWEN], ctx: NEED } } });
  assert.equal(exact.results[2].fix, `Raise OLLAMA_NUM_CTX to ${NEED} or more, or unset it so the runner sizes the window`);
  const oneShort = await check({ OLLAMA_NUM_CTX: '32768' }, { models: { [QWEN]: { ...MODELS[QWEN], ctx: NEED - 1 } } });
  assert.equal(oneShort.results[2].fix, `Pick a model with a larger context (e.g. ${QWEN})`);

  // The number in the advice is itself enough, and one less is not.
  const advised = await check({ OLLAMA_NUM_CTX: String(NEED) });
  assert.deepEqual(advised.results[2], { pass: true, label: `num_ctx ${NEED} fits ~44K prompt + output (model max 262144)` });
  const below = await check({ OLLAMA_NUM_CTX: String(NEED - 1) });
  assert.equal(below.results[2].pass, false);
});

test('the room kept for the reply follows the output cap', async () => {
  // 56000 holds the prompt and a small reply, not the prompt and a full report.
  const full = await check({ OLLAMA_NUM_CTX: '56000' });
  assert.equal(full.results[2].pass, false);
  assert.equal(full.results[2].label, `${TOO_SMALL_FOR}; num_ctx is 56000 (model max 262144)`);
  const short = await check({ OLLAMA_NUM_CTX: '56000', OLLAMA_NUM_PREDICT: '4000' });
  assert.deepEqual(short.results[2], { pass: true, label: 'num_ctx 56000 fits ~44K prompt + output (model max 262144)' });
});

test('with thinking on, the window keeps room for the whole output cap, as the runner\'s does', async () => {
  // Reasoning and report come out of the same cap (28,000 with thinking on),
  // so the default window no longer holds a 44K prompt and the runner moves to
  // the next size. A check that kept 12,000 would report the smaller window,
  // and then compare the loaded copy with a size the runner does not use.
  const { results } = await check({ OLLAMA_THINK: 'on' });
  assert.deepEqual(results[2], {
    warn: true,
    label: 'num_ctx raised from 65536 to 98304 to fit ~44K prompt + output (model max 262144)',
    fix: 'Nothing to do: the runner does this by itself while OLLAMA_NUM_CTX is unset. A larger window takes more GPU memory',
  });
  assert.deepEqual(results[3], { pass: true, label: 'thinking: on (forced)' });
  // A model that cannot think is sent no setting, and keeps the smaller reserve.
  const cannot = await check({ OLLAMA_MODEL: CODER, OLLAMA_THINK: 'on' });
  assert.deepEqual(cannot.results[2], FITS);
  // A window set by hand is not raised. The fix names both ways out, the
  // second in the words of the runner's own refusal: raising the window is
  // the expensive one, and thinking is on only because it was asked for.
  const pinned = await check({ OLLAMA_THINK: 'on', OLLAMA_NUM_CTX: '65536' });
  assert.deepEqual(pinned.results[2], {
    pass: false,
    label: 'num_ctx too small: prompt ~44000 tokens + 28000 output (thinking on) needs 74200; num_ctx is 65536 (model max 262144)',
    fix: 'Raise OLLAMA_NUM_CTX to 74200 or more, or unset it so the runner sizes the window. Or leave thinking off, which is the '
      + 'default: the window then has to hold 12000 tokens of reply, not the whole output cap of 28000',
  });
  // With a cap no larger than a reply gets anyway, thinking costs the window nothing, and the fix says nothing about it.
  const smallCap = await check({ OLLAMA_THINK: 'on', OLLAMA_NUM_PREDICT: '8000', OLLAMA_NUM_CTX: '50000' });
  assert.equal(smallCap.results[2].fix, 'Raise OLLAMA_NUM_CTX to 54200 or more, or unset it so the runner sizes the window');
});

test('a prompt past the last automatic step fails with the size to set', async () => {
  const { results } = await check({}, {}, { promptTokens: 120000 });
  assert.deepEqual(results[2], {
    pass: false,
    label: 'num_ctx too small: prompt ~120000 tokens + 12000 output needs 138000; num_ctx is 131072 (model max 262144)',
    fix: 'Set OLLAMA_NUM_CTX to 138000 or more: the runner does not go past 131072 by itself',
  });
});

test('a prompt larger than the default window raises it, with a warning', async () => {
  const { results } = await check({}, {}, { promptTokens: 60000 });
  assert.equal(results.length, 4);
  assert.deepEqual(results[2], {
    warn: true,
    label: 'num_ctx raised from 65536 to 98304 to fit ~60K prompt + output (model max 262144)',
    fix: 'Nothing to do: the runner does this by itself while OLLAMA_NUM_CTX is unset. A larger window takes more GPU memory',
  });
});

test('a window above what the model accepts is cut to it, with a warning', async () => {
  const pinned = await check({ OLLAMA_NUM_CTX: '300000' });
  assert.deepEqual(pinned.results[2], {
    warn: true,
    label: 'num_ctx 300000 is more than this model accepts; using 262144, which fits ~44K prompt + output (model max 262144)',
    fix: 'Set OLLAMA_NUM_CTX to 262144 or less, or leave it unset',
  });

  // The default window on a model with a smaller limit that still holds the prompt.
  const models = { [QWEN]: { ...MODELS[QWEN], ctx: 49152 } };
  const byDefault = await check({}, { models }, { promptTokens: 30000 });
  assert.deepEqual(byDefault.results[2], {
    warn: true,
    label: 'num_ctx 65536 is more than this model accepts; using 49152, which fits ~30K prompt + output (model max 49152)',
    fix: 'Nothing to do: the runner uses the model\'s own limit',
  });

  // Raised for the prompt and stopped at the model's limit, in one sentence.
  const both = await check({}, { models: { [QWEN]: { ...MODELS[QWEN], ctx: 90000 } } }, { promptTokens: 70000 });
  assert.equal(both.results[2].warn, true);
  assert.equal(
    both.results[2].label,
    'num_ctx raised from 65536 to 90000, the most this model accepts, to fit ~70K prompt + output (model max 90000)',
  );
});

test('a model that does not state its limit is checked against the window alone', async () => {
  const { results } = await check({}, { models: { [QWEN]: { ...MODELS[QWEN], ctx: null } } });
  assert.deepEqual(results[2], { pass: true, label: 'num_ctx 65536 fits ~44K prompt + output (model max unknown)' });
});

test('without a usable prompt size, the size of a prompt built from the example files is assumed', async () => {
  // null is what measurePromptTokens() returns while there is no CV to measure.
  for (const promptTokens of [undefined, NaN, 0, -5, '40000', null]) {
    const { results } = await check({}, {}, { promptTokens });
    assert.deepEqual(results[2], FITS, `promptTokens: ${String(promptTokens)}`);
  }
  // Any positive number is an estimate and is used as given, to the nearest
  // thousand in the label.
  for (const [promptTokens, shown] of [[38800, '~39K'], [41500, '~42K'], [40647.25, '~41K'], [1, '~0K']]) {
    const { results } = await check({}, {}, { promptTokens });
    assert.equal(results[2].label, `num_ctx 65536 fits ${shown} prompt + output (model max 262144)`);
  }
});

test('model details that cannot be read are a failure, not an exception', async () => {
  const { results, calls } = await check({}, { status: { '/api/show': 500 } });
  assert.equal(results.length, 3);
  assert.equal(results[2].pass, false);
  assert.equal(results[2].label, `Ollama model details could not be read for ${QWEN}`);
  assert.match(results[2].fix, /HTTP 500/);
  assert.deepEqual(calls, METADATA.slice(0, 3));
});

// ---------------------------------------------------------------------------
// checkOllama: 4. thinking
// ---------------------------------------------------------------------------

test('thinking is reported as the request will carry it, with who decided', async () => {
  const cases = [
    // Left to the runner, a model that can think does not, whichever it is.
    [{}, 'thinking: off (auto)'],
    [{ OLLAMA_MODEL: GEMMA }, 'thinking: off (auto)'],
    [{ OLLAMA_THINK: 'on' }, 'thinking: on (forced)'],
    [{ OLLAMA_MODEL: GEMMA, OLLAMA_THINK: 'on' }, 'thinking: on (forced)'],
    [{ OLLAMA_MODEL: GEMMA, OLLAMA_THINK: 'off' }, 'thinking: off (forced)'],
    // A model without the capability: the request carries no setting at all.
    [{ OLLAMA_MODEL: CODER }, 'thinking: n/a (auto)'],
    [{ OLLAMA_MODEL: CODER, OLLAMA_THINK: 'on' }, 'thinking: n/a (forced)'],
  ];
  for (const [env, label] of cases) {
    const { results } = await check(env);
    assert.deepEqual(results[3], { pass: true, label }, JSON.stringify(env));
  }
});

test('thinking with an output cap too small for it is a warning', async () => {
  const { results } = await check({ OLLAMA_THINK: 'on', OLLAMA_NUM_PREDICT: '16384' });
  assert.deepEqual(results[3], {
    warn: true,
    label: 'thinking: on (forced), but num_predict is 16384',
    fix: 'Thinking counts against the output cap, and a report needs 8-11K tokens after it. '
      + 'Raise OLLAMA_NUM_PREDICT to 24000 or more, or unset it so the runner sizes the cap itself',
  });

  const atTheLine = await check({ OLLAMA_THINK: 'on', OLLAMA_NUM_PREDICT: '24000' });
  assert.deepEqual(atTheLine.results[3], { pass: true, label: 'thinking: on (forced)' });
  // The same small cap is fine for a model that is not thinking.
  const notThinking = await check({ OLLAMA_NUM_PREDICT: '16384' });
  assert.deepEqual(notThinking.results[3], NOT_THINKING);
});

// ---------------------------------------------------------------------------
// checkOllama: 5. what is loaded
// ---------------------------------------------------------------------------

test('a model loaded at another window size is a warning about the reload', async () => {
  const { results } = await check({}, { ps: [loaded(QWEN, { ctx: 131072 })] });
  assert.deepEqual(results, [...HEALTHY, {
    warn: true,
    label: `${QWEN} is loaded at 131072, runner uses 65536: the next call reloads the model`,
    fix: 'Accept one reload, or set OLLAMA_NUM_CTX=131072 so the runner asks for the size that is loaded',
  }]);

  // A window another program chose is not worth matching when the prompt does
  // not fit in it.
  const tiny = await check({}, { ps: [loaded(QWEN, { ctx: 4096 })] });
  assert.equal(tiny.results.length, 5);
  assert.equal(tiny.results[4].label, `${QWEN} is loaded at 4096, runner uses 65536: the next call reloads the model`);
  assert.equal(tiny.results[4].fix, 'Nothing to do: 4096 cannot hold the evaluation prompt, so the reload is needed');
  // The line between the two is what the prompt needs.
  const enough = await check({}, { ps: [loaded(QWEN, { ctx: NEED })] });
  assert.equal(enough.results[4].fix, `Accept one reload, or set OLLAMA_NUM_CTX=${NEED} so the runner asks for the size that is loaded`);
  const oneShort = await check({}, { ps: [loaded(QWEN, { ctx: NEED - 1 })] });
  assert.equal(oneShort.results[4].fix, `Nothing to do: ${NEED - 1} cannot hold the evaluation prompt, so the reload is needed`);
});

test('nothing is said about a model that is loaded the way the runner will use it', async () => {
  const same = await check({}, { ps: [loaded(QWEN, { ctx: 65536 })] });
  assert.deepEqual(same.results, HEALTHY);
  const matched = await check({ OLLAMA_NUM_CTX: '131072' }, { ps: [loaded(QWEN, { ctx: 131072 })] });
  assert.equal(matched.results.length, 4);
  assert.equal(matched.results[2].label, 'num_ctx 131072 fits ~44K prompt + output (model max 262144)');
  // Another model, or a server that does not say what size it loaded.
  const other = await check({}, { ps: [loaded(GEMMA, { ctx: 4096, size: 20 * GIB, vram: 10 * GIB })] });
  assert.deepEqual(other.results, HEALTHY);
  const sizeless = await check({}, { ps: [loaded(QWEN)] });
  assert.deepEqual(sizeless.results, HEALTHY);
});

test('no reload is announced for a prompt the runner will refuse to send', async () => {
  const { results } = await check({ OLLAMA_NUM_CTX: '32768' }, { ps: [loaded(QWEN, { ctx: 131072 })] });
  assert.equal(results.length, 4);
  assert.equal(results[2].pass, false);
});

// A copy that is partly in system memory: the figures /api/ps gave for the one
// model that spilled in the probe.
const SPILLED = { size: 34.95 * GIB, vram: 29.10 * GIB };
const NO_SMALLER_WINDOW = `Use ${SMALLER_MODEL}: this prompt needs a window of at least ${NEED} tokens`;

test('a model partly in system memory, at the runner\'s own window, is told how far the window can go down', async () => {
  const { results } = await check({}, { ps: [loaded(QWEN, { ctx: 65536, ...SPILLED })] });
  assert.deepEqual(results, [...HEALTHY, {
    warn: true,
    label: `${QWEN} is 83% on GPU at num_ctx 65536: CPU spill`,
    // The floor is part of the advice. "Lower OLLAMA_NUM_CTX" alone sent the
    // owner to a window the model fits in and the prompt does not, which the
    // runner then refused.
    fix: `Set OLLAMA_NUM_CTX lower, but not below ${NEED}, or use ${SMALLER_MODEL}`,
  }]);

  // Rounded down: "100% on GPU: CPU spill" would contradict itself.
  const nearly = await check({}, { ps: [loaded(QWEN, { ctx: 65536, size: 1000, vram: 996 })] });
  assert.equal(nearly.results[4].label, `${QWEN} is 99% on GPU at num_ctx 65536: CPU spill`);

  // The floor follows the prompt: here the runner raised the window for a
  // larger one, and the copy is at the raised size.
  const raised = await check({}, { ps: [loaded(QWEN, { ctx: 98304, ...SPILLED })] }, { promptTokens: 60000 });
  assert.equal(raised.results.length, 5);
  assert.equal(raised.results[4].fix, `Set OLLAMA_NUM_CTX lower, but not below 75000, or use ${SMALLER_MODEL}`);
  // And the number it names is one the check itself accepts.
  const lowered = await check({ OLLAMA_NUM_CTX: '75000' }, {}, { promptTokens: 60000 });
  assert.deepEqual(lowered.results[2], { pass: true, label: 'num_ctx 75000 fits ~60K prompt + output (model max 262144)' });
});

test('a spilled model whose window cannot go lower is told to use a smaller model', async () => {
  // Pinned at exactly what the prompt needs: there is nothing left to lower.
  const atTheFloor = await check({ OLLAMA_NUM_CTX: String(NEED) }, { ps: [loaded(QWEN, { ctx: NEED, ...SPILLED })] });
  assert.equal(atTheFloor.results.length, 5);
  assert.equal(atTheFloor.results[4].fix, NO_SMALLER_WINDOW);

  // Loaded by another program at a smaller window than the runner's, and it
  // does not fit even there: the runner's own copy takes more memory still.
  const smaller = await check({}, { ps: [loaded(QWEN, { ctx: 4096, ...SPILLED })] });
  assert.equal(smaller.results.length, 6);
  assert.equal(smaller.results[4].fix, 'Nothing to do: 4096 cannot hold the evaluation prompt, so the reload is needed');
  assert.deepEqual(smaller.results[5], {
    warn: true, label: `${QWEN} is 83% on GPU at num_ctx 4096: CPU spill`, fix: NO_SMALLER_WINDOW,
  });

  // A server that does not say what size it loaded: whose copy it is cannot
  // be told, so no window is suggested.
  const cpuOnly = await check({}, { ps: [loaded(QWEN, { size: 17 * GIB, vram: 0 })] });
  assert.deepEqual(cpuOnly.results, [...HEALTHY, {
    warn: true, label: `${QWEN} is 0% on GPU: CPU spill`, fix: NO_SMALLER_WINDOW,
  }]);
});

test('a spilled copy another program loaded at a larger window is not blamed on the runner\'s setting', async () => {
  // OLLAMA_NUM_CTX is unset here and played no part: the desktop app loaded
  // the model at its own 262144, and the runner's 65536 copy may well fit.
  const { results } = await check({}, { ps: [loaded(QWEN, { ctx: 262144, size: 17613654587, vram: 9000000000 })] });
  assert.deepEqual(results, [...HEALTHY, {
    warn: true,
    label: `${QWEN} is loaded at 262144, runner uses 65536: the next call reloads the model`,
    // Not "set OLLAMA_NUM_CTX=262144": that would pin the size that spilled.
    fix: 'Nothing to do: the reload at 65536 replaces a copy that is partly in system memory',
  }, {
    warn: true,
    label: `${QWEN} is 51% on GPU at num_ctx 262144: CPU spill`,
    fix: 'Nothing to do yet: the runner\'s next call reloads the model at 65536, which takes less memory. '
      + 'Check again after the first run',
  }]);
  for (const { fix } of results.slice(4)) assert.doesNotMatch(fix, /OLLAMA_NUM_CTX/);

  // Without the spill, the loaded size is still one worth matching.
  const fits = await check({}, { ps: [loaded(QWEN, { ctx: 262144 })] });
  assert.equal(fits.results[4].fix, 'Accept one reload, or set OLLAMA_NUM_CTX=262144 so the runner asks for the size that is loaded');
});

test('a spilled model beside a window that is too small is not told to raise and to lower the same setting', async () => {
  // The owner who read "lower OLLAMA_NUM_CTX" and set 45000 got this report
  // next: one line said raise it, the other still said lower it.
  const { results } = await check({ OLLAMA_NUM_CTX: '45000' }, { ps: [loaded(QWEN, { ctx: 65536, ...SPILLED })] });
  assert.deepEqual(results, [
    REACHABLE,
    PICKED,
    {
      pass: false,
      label: `${TOO_SMALL_FOR}; num_ctx is 45000 (model max 262144)`,
      fix: `Raise OLLAMA_NUM_CTX to ${NEED} or more, or unset it so the runner sizes the window`,
    },
    NOT_THINKING,
    {
      warn: true,
      label: `${QWEN} is 83% on GPU at num_ctx 65536: CPU spill`,
      fix: 'Nothing to do yet: this copy was not loaded with the current settings, which cannot run an evaluation '
        + '(see the window above). Check again after the first run',
    },
  ]);
  assert.doesNotMatch(results.map((result) => result.fix ?? '').join('\n'), /lower/i);

  // A copy at a window the prompt does not fit in either: no window the
  // runner can use is smaller than this one, so the model is what has to give.
  const tooSmallToo = await check({ OLLAMA_NUM_CTX: '45000' }, { ps: [loaded(QWEN, { ctx: 45000, ...SPILLED })] });
  assert.equal(tooSmallToo.results[2].pass, false);
  assert.equal(tooSmallToo.results[4].fix, NO_SMALLER_WINDOW);
  assert.doesNotMatch(tooSmallToo.results.map((result) => result.fix ?? '').join('\n'), /lower/i);
});

test('a server that cannot list its loaded models does not change the result', async () => {
  const { results, calls } = await check({}, { status: { '/api/ps': 500 } });
  assert.deepEqual(results, HEALTHY);
  assert.deepEqual(calls, METADATA);
});

// ---------------------------------------------------------------------------
// checkOllama: never throws
// ---------------------------------------------------------------------------

test('whatever goes wrong, the check returns results instead of throwing', async () => {
  // doctor collects every check before it prints one. A throw from this check
  // would replace the whole report with "doctor.mjs failed".
  const hard = await checkOllama({ env: {}, fetchImpl: 'not a function' });
  assert.equal(hard.length, 1);
  assert.equal(hard[0].pass, false);
  assert.equal(hard[0].label, 'Ollama check could not finish');
  assert.match(hard[0].fix, /fetchImpl must be a function/);
  const soft = await checkOllama({ env: {}, fetchImpl: 'not a function', soft: true });
  assert.deepEqual(Object.keys(soft[0]).sort(), ['fix', 'label', 'warn']);

  // Settings that cannot even be read.
  const env = { get OLLAMA_MODEL() { throw new RangeError('synthetic failure'); } };
  const unreadable = await checkOllama({ env, fetchImpl: fakeFetch(server()).fetchImpl });
  assert.deepEqual(unreadable, [{ pass: false, label: 'Ollama check could not finish', fix: 'synthetic failure' }]);

  // A request that fails in a way no server produces, after one that worked:
  // what was established before it is kept.
  let call = 0;
  const fetchImpl = async () => {
    call += 1;
    if (call === 1) return new Response('{"version":"0.40.16"}');
    throw new RangeError('synthetic failure');
  };
  const later = await checkOllama({ env: {}, fetchImpl });
  assert.deepEqual(later[0], REACHABLE);
  assert.equal(later.length, 2);
  assert.equal(later[1].pass, false);
  assert.equal(later[1].label, 'Ollama model list could not be read');
});

// ---------------------------------------------------------------------------
// The size of the prompt
// ---------------------------------------------------------------------------

// What every prompt is counted as before a word of a file is in it: its own
// fixed wording and the longest posting.
const EMPTY_PROMPT = evaluationPromptTokens([]);

test('evaluationPromptTokens counts a text by the rule the runner counts with', () => {
  // The runner counts the prompt it sends with estimateTokens() from
  // lib/context-budget.mjs. lib/ollama-config.mjs repeats that rule instead of
  // importing it (doctor.mjs loads that module, and it may import nothing), so
  // the two are held to the same answers here.
  const texts = [
    '', 'a', 'abcd', 'abcde', '   ', '\n\n# Title\n\n\n- one\t\ttwo   three\r\n', 'héllo wörld — 日本語 ✓',
    'x'.repeat(39_999), `${'word '.repeat(5000)}\n`.repeat(3),
    readFileSync(join(ROOT, 'modes', '_shared.md'), 'utf8'),
  ];
  for (const text of texts) {
    assert.equal(evaluationPromptTokens([text]) - EMPTY_PROMPT, estimateTokens(text), JSON.stringify(text.slice(0, 40)));
  }
  assert.equal(evaluationPromptTokens(texts) - EMPTY_PROMPT, texts.reduce((sum, text) => sum + estimateTokens(text), 0));
  // A file that is absent is passed as nothing, or as an empty text.
  assert.equal(evaluationPromptTokens(['abcd', undefined, null, '']), EMPTY_PROMPT + 1);
  assert.equal(evaluationPromptTokens(), EMPTY_PROMPT);
  // Room for the longest posting the page reader returns (16,000 characters)
  // is part of every size, with the prompt's own wording on top.
  assert.ok(EMPTY_PROMPT > 4000, `an empty prompt is counted as ${EMPTY_PROMPT}`);
});

/** A folder holding these files, given as `{ 'relative/path': 'text' }`. */
function tree(files) {
  const dir = mkdtempSync(join(tmpdir(), 'ollama-doctor-test-'));
  scratch.push(dir);
  for (const [name, text] of Object.entries(files)) {
    mkdirSync(dirname(join(dir, name)), { recursive: true });
    writeFileSync(join(dir, name), text);
  }
  return dir;
}

// 400 characters are 100 tokens, and so on: sizes that can be added by hand.
const CODE_FILES = { 'modes/_shared.md': 's'.repeat(400), 'modes/oferta.md': 'o'.repeat(800) };
const OWNER_FILES = { 'cv.md': 'c'.repeat(40), 'config/profile.yml': 'p'.repeat(80) };

test('measurePromptTokens adds up the files the runner builds an evaluation prompt from', () => {
  const codeRoot = tree(CODE_FILES);
  const required = measurePromptTokens({ codeRoot, dataRoot: tree(OWNER_FILES) });
  assert.equal(required, EMPTY_PROMPT + 100 + 200 + 10 + 20);

  // The two optional files count when they are there.
  const withProfileMode = tree({ ...OWNER_FILES, 'modes/_profile.md': 'm'.repeat(120) });
  assert.equal(measurePromptTokens({ codeRoot, dataRoot: withProfileMode }), required + 30);
  const withBoth = tree({ ...OWNER_FILES, 'modes/_profile.md': 'm'.repeat(120), 'modes/_custom.md': 'h'.repeat(160) });
  assert.equal(measurePromptTokens({ codeRoot, dataRoot: withBoth }), required + 30 + 40);

  // Read the way the runner reads them: a byte-order mark and the blank lines
  // around the text are not part of it.
  const padded = tree({ 'cv.md': `﻿\n\n${'c'.repeat(40)}\n\n\n`, 'config/profile.yml': `${'p'.repeat(80)}\n` });
  assert.equal(measurePromptTokens({ codeRoot, dataRoot: padded }), required);

  // The owner's files only: a modes/_shared.md in the data root is not the
  // one the prompt is built from, and one from the code root is.
  const stray = tree({ ...OWNER_FILES, 'modes/_shared.md': 'x'.repeat(40000), 'modes/oferta.md': 'x'.repeat(40000) });
  assert.equal(measurePromptTokens({ codeRoot, dataRoot: stray }), required);
  const largerModes = tree({ 'modes/_shared.md': 's'.repeat(4000), 'modes/oferta.md': 'o'.repeat(800) });
  assert.equal(measurePromptTokens({ codeRoot: largerModes, dataRoot: tree(OWNER_FILES) }), required + 900);
});

test('measurePromptTokens follows a profile that lives somewhere else', () => {
  // CAREER_OPS_PROFILE names the profile the runner reads; the one in the
  // data root is then not part of the prompt, and need not exist.
  const codeRoot = tree(CODE_FILES);
  const elsewhere = tree({ 'work.yml': 'w'.repeat(4000) });
  const expected = EMPTY_PROMPT + 100 + 200 + 10 + 1000;
  assert.equal(measurePromptTokens({ codeRoot, dataRoot: tree(OWNER_FILES), profilePath: join(elsewhere, 'work.yml') }), expected);
  assert.equal(
    measurePromptTokens({ codeRoot, dataRoot: tree({ 'cv.md': OWNER_FILES['cv.md'] }), profilePath: join(elsewhere, 'work.yml') }),
    expected,
  );
});

test('measurePromptTokens gives null where the runner has no prompt to build', () => {
  const codeRoot = tree(CODE_FILES);
  const dataRoot = tree(OWNER_FILES);
  assert.equal(typeof measurePromptTokens({ codeRoot, dataRoot }), 'number');

  // Setup is not finished: the runner evaluates nothing without a CV and a
  // profile, and an empty cv.md is no CV.
  assert.equal(measurePromptTokens({ codeRoot, dataRoot: tree({ 'config/profile.yml': 'p' }) }), null);
  assert.equal(measurePromptTokens({ codeRoot, dataRoot: tree({ 'cv.md': 'c' }) }), null);
  assert.equal(measurePromptTokens({ codeRoot, dataRoot: tree({ ...OWNER_FILES, 'cv.md': ' \n\n' }) }), null);
  assert.equal(measurePromptTokens({ codeRoot, dataRoot, profilePath: join(dataRoot, 'config', 'absent.yml') }), null);
  // A checkout without the mode files cannot build the prompt either.
  assert.equal(measurePromptTokens({ codeRoot: tree({ 'modes/_shared.md': 's' }), dataRoot }), null);
  assert.equal(measurePromptTokens({ codeRoot: tree({ 'modes/oferta.md': 'o' }), dataRoot }), null);
  // A file that is there and cannot be read as one (here: a folder of that
  // name) stops the runner, so there is no size to report.
  assert.equal(measurePromptTokens({ codeRoot, dataRoot: tree({ ...OWNER_FILES, 'modes/_custom.md/x': '' }) }), null);
  // Nothing to look in.
  for (const where of [undefined, {}, { codeRoot }, { dataRoot }, { codeRoot: '', dataRoot }]) {
    assert.equal(measurePromptTokens(where), null, JSON.stringify(where));
  }
});

// ---------------------------------------------------------------------------
// The module
// ---------------------------------------------------------------------------

test('lib/ollama-doctor.mjs imports nothing that needs an installed package', () => {
  // doctor.mjs loads this file at its Ollama step and takes "module not found"
  // there to mean that the copy does not hold the backend. A package reached
  // from here, at any depth, would turn an incomplete node_modules into that
  // answer: the checks would be skipped without a word, or --ollama would name
  // the wrong thing as missing.
  const seen = new Set();
  const queue = [join(ROOT, 'lib', 'ollama-doctor.mjs')];
  while (queue.length > 0) {
    const file = queue.pop();
    if (seen.has(file)) continue;
    seen.add(file);
    const source = readFileSync(file, 'utf8');
    // Static imports and re-exports only: a dynamic import runs when its
    // function is called, not when the module is linked.
    for (const [, specifier] of source.matchAll(/^\s*(?:import|export)\s(?:[^'"()]*?\sfrom\s*)?['"]([^'"]+)['"]/gm)) {
      if (isBuiltin(specifier)) continue;
      assert.match(specifier, /^\.\.?\//, `${file} statically imports the package "${specifier}"`);
      queue.push(fileURLToPath(new URL(specifier, pathToFileURL(file))));
    }
  }
  const names = [...seen].map((file) => file.slice(ROOT.length).replace(/\\/g, '/')).sort();
  // The three files of the backend, and path-resolver.mjs: the one place a
  // profile path may be built (tests/profile-path-override.test.mjs fails on a
  // tracked module that joins its own). doctor.mjs imports it too, so it is
  // there whenever this file is loaded, and it imports nothing but Node.
  assert.deepEqual(names, ['lib/ollama-client.mjs', 'lib/ollama-config.mjs', 'lib/ollama-doctor.mjs', 'path-resolver.mjs']);
});

test('lib/ollama-doctor.mjs builds no path to the profile by itself', () => {
  // The rule of tests/profile-path-override.test.mjs, which reads tracked
  // files only and so said nothing while this file was new: the profile is
  // found through getProfilePath(), where CAREER_OPS_PROFILE is resolved.
  const lines = readFileSync(join(ROOT, 'lib', 'ollama-doctor.mjs'), 'utf8').split('\n')
    .filter((line) => !/^\s*(\/\/|\*|\/\*)/.test(line));
  const handBuilt = lines.filter((line) => /\b(?:join|resolve)\([^\n]*['"](?:config\/)?profile\.yml['"]/.test(line)
    || /process\.env\.CAREER_OPS_PROFILE/.test(line));
  assert.deepEqual(handBuilt, []);
});

// ---------------------------------------------------------------------------
// doctor.mjs
// ---------------------------------------------------------------------------

const scratch = [];
const started = new Map();
after(async () => {
  // A test that failed may not have waited for every run it started, and on
  // Windows a directory that is still a child's working directory cannot be
  // removed: the first one would throw and leave the rest behind.
  await Promise.allSettled(started.values());
  for (const dir of scratch) rmSync(dir, { recursive: true, force: true, maxRetries: 10 });
});

/** A data root with setup finished, so nothing but the checks under test decides the exit code. */
function dataRoot() {
  const parent = mkdtempSync(join(tmpdir(), 'ollama-doctor-test-'));
  scratch.push(parent);
  const root = join(parent, 'root');
  mkdirSync(join(root, 'config'), { recursive: true });
  mkdirSync(join(root, 'modes'));
  // Fictional, like every value in this file.
  writeFileSync(join(root, 'cv.md'), '# Jordan Rivera\n\n## Experience\n\n### Data Analyst, Globex\n');
  writeFileSync(join(root, 'config', 'profile.yml'), 'candidate:\n  full_name: "Jordan Rivera"\n');
  writeFileSync(join(root, 'modes', '_profile.md'), '# Targeting\n\nSenior data analyst roles.\n');
  writeFileSync(join(root, 'portals.yml'), 'title_filter:\n  positive:\n    - "Data Analyst"\n');
  return { parent, root };
}

/**
 * The environment of a doctor child: the developer's shell without anything
 * that selects a data root, configures Ollama or reroutes requests.
 */
function childEnv(root, extra) {
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    !/^(CAREER_OPS_|OPENAI_|OLLAMA_|DOTENV_|NODE_OPTIONS$|NODE_PATH$|HTTPS?_PROXY$|ALL_PROXY$)/i.test(key)));
  return { ...env, CAREER_OPS_ROOT: root, DOTENV_CONFIG_QUIET: 'true', NO_COLOR: '1', ...extra };
}

// How many doctor children run at a time. All of them at once is quickest on
// an idle machine, but test-all runs as many suites side by side as the
// machine has cores, and most of them start children of their own.
const MAX_CHILDREN = 6;
let running = 0;
const waiting = [];
async function withSlot(start) {
  if (running >= MAX_CHILDREN) await new Promise((go) => { waiting.push(go); });
  running += 1;
  try {
    return await start();
  } finally {
    running -= 1;
    waiting.shift()?.();
  }
}

/** Start node and collect what it printed, in the shape spawnSync returns. */
function runNode(args, { cwd, env, timeout = 60_000 }) {
  return withSlot(() => new Promise((resolve) => {
    const child = spawn(process.execPath, args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    let stdout = '';
    let stderr = '';
    let error = null;
    let settled = false;
    const finish = (status, signal) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({ status, signal, stdout, stderr, error });
    };
    const timer = setTimeout(() => {
      error = Object.assign(new Error(`still running after ${timeout} ms`), { code: 'ETIMEDOUT' });
      child.kill('SIGKILL');
    }, timeout);
    child.stdout.setEncoding('utf8').on('data', (chunk) => { stdout += chunk; });
    child.stderr.setEncoding('utf8').on('data', (chunk) => { stderr += chunk; });
    child.on('error', (err) => { error = err; finish(null, null); });
    child.on('close', finish);
  }));
}

/**
 * The module a doctor child loads first. It replaces fetch with the stub and
 * writes each request to `log` as it is made, so "no request" is a statement
 * about every host, not only about a stub's address. It also reports a
 * process.exit call on stderr; with `trapExit` such a call ends the run with
 * 99 instead of the code it was given, which fails on every platform what
 * only crashes on one.
 */
function preloadSource({ state, log, trapExit, holdOpen }) {
  return `import { appendFileSync, writeSync } from 'node:fs';
${route.toString()}
const STATE = ${JSON.stringify(state)};
const exit = process.exit;
process.exit = (code) => { writeSync(2, 'process.exit called with ' + code + '\\n'); exit(${trapExit ? '99' : 'code'}); };
globalThis.fetch = async (input, init = {}) => {
  const url = new URL(String(input));
  const method = init.method ?? 'GET';
  appendFileSync(${JSON.stringify(log)}, method + ' ' + url.origin + url.pathname + '\\n');
  const reply = route(STATE, method, url.pathname, init.body ? JSON.parse(init.body) : null);
  if (!reply) throw new TypeError('fetch failed');
  return new Response(reply.text ?? JSON.stringify(reply.json), { status: reply.status });
};
${holdOpen ? 'setInterval(() => {}, 1000);' : ''}
`;
}

/**
 * Run doctor.mjs on a data root, with the stub in place of the network.
 *
 * @param {object} options
 * @param {string[]} [options.args]
 * @param {Record<string, string>} [options.env] Added to the child's environment.
 * @param {string} [options.dotenv] Contents of the data root's .env.
 * @param {object} [options.state] Overrides for the stub server.
 * @param {boolean} [options.trapExit] Fail the run if it ends through process.exit.
 * @param {boolean} [options.holdOpen] Keep the child's event loop busy forever.
 * @param {{parent: string, root: string}} [options.at] An existing data root.
 * @param {string} [options.cwd] The folder to start in, when it is not the data root.
 * @param {string} [options.doctor] Another copy of doctor.mjs to run.
 */
async function runDoctor({ args = [], env = {}, dotenv, state = {}, trapExit = false, holdOpen = false, at, cwd, doctor = DOCTOR } = {}) {
  const { parent, root } = at ?? dataRoot();
  if (dotenv !== undefined) writeFileSync(join(root, '.env'), dotenv);
  const log = join(parent, `requests-${Math.random().toString(36).slice(2)}.txt`);
  const preload = `${log}.preload.mjs`;
  writeFileSync(preload, preloadSource({ state: server(state), log, trapExit, holdOpen }));
  const result = await runNode(
    ['--import', PLAYWRIGHT_STUB, '--import', pathToFileURL(preload).href, doctor, ...args],
    { cwd: cwd ?? root, env: childEnv(root, env) },
  );
  let requests = [];
  try { requests = readFileSync(log, 'utf8').split('\n').filter(Boolean); } catch { /* nothing was requested */ }
  return { ...result, requests, root };
}

const STUB_REQUESTS = [
  `GET ${DEFAULT_URL}/api/version`, `GET ${DEFAULT_URL}/api/tags`, `POST ${DEFAULT_URL}/api/show`, `GET ${DEFAULT_URL}/api/ps`,
];
// doctor.mjs measures the prompt from the files of the data root and the mode
// files of this checkout, so the size in its report is theirs. Every data root
// made by dataRoot() holds the same files, and the mode files change with
// every upstream release, so the size is asked for here and not written down.
const ROOT_PROMPT = measurePromptTokens({ codeRoot: ROOT, dataRoot: dataRoot().root });
const fitsLine = (numCtx) => `✓ num_ctx ${numCtx} fits ~${Math.round(ROOT_PROMPT / 1000)}K prompt + output (model max 262144)`;
const HEALTHY_LINES = [
  `✓ Ollama 0.40.16 reachable at ${DEFAULT_URL}`,
  `✓ Ollama model: ${QWEN} (preferred)`,
  fitsLine(65536),
  '✓ thinking: off (auto)',
];
const OLLAMA_LINE = /Ollama|num_ctx|thinking:/;
const lines = (run) => run.stdout.split('\n').filter((line) => OLLAMA_LINE.test(line));

// Every doctor run is started here, before the first test, so they overlap:
// one after another they would take most of the time a suite is given. Each
// test awaits the run it is about.
function scenario(name, start) {
  if (!started.has(name)) {
    const run = start();
    // Reported by the test that awaits it, not as an unhandled rejection.
    run.catch(() => {});
    started.set(name, run);
  }
  return started.get(name);
}
// A server with none of the models the runner would pick. A check that ran
// against it, asked for or not, has something to complain about.
const ONLY_LLAMA = { models: { 'llama3.2:latest': { family: 'llama', ctx: 131072, caps: ['completion', 'tools'] } } };

/** A folder that is neither the checkout nor a data root, holding this .env. */
function folderWithEnv(text) {
  const dir = mkdtempSync(join(tmpdir(), 'ollama-doctor-test-'));
  scratch.push(dir);
  writeFileSync(join(dir, '.env'), text);
  return dir;
}

const SCENARIOS = {
  // First, because each waits five seconds for the backstop: started first,
  // that wait overlaps every other run.
  heldOpen: () => runDoctor({ args: ['--ollama'], holdOpen: true }),
  heldOpenFailing: () => runDoctor({ args: ['--ollama'], env: { OLLAMA_MODEL: 'llama3.3' }, holdOpen: true }),
  // Nothing about Ollama anywhere: what every other run is compared with.
  bare: () => runDoctor(),
  // A machine whose Ollama server is tuned through the environment, and
  // nothing about the runner.
  plain: () => runDoctor({ env: SERVER_VARIABLES, state: ONLY_LLAMA }),
  strict: () => runDoctor({ args: ['--strict'], env: SERVER_VARIABLES, state: ONLY_LLAMA }),
  flag: () => runDoctor({ args: ['--ollama'], env: SERVER_VARIABLES, trapExit: true }),
  flagDown: () => runDoctor({ args: ['--ollama'], state: { down: true }, trapExit: true }),
  shell: () => runDoctor({ env: { OLLAMA_MODEL: GEMMA }, trapExit: true }),
  shellDown: () => runDoctor({ env: { OLLAMA_NUM_CTX: '65536' }, state: { down: true }, trapExit: true }),
  // --cli makes doctor skip its own load of the .env into the environment.
  dotenv: () => runDoctor({ args: ['--cli', 'claude'], dotenv: `OLLAMA_MODEL=${GEMMA}\n`, trapExit: true }),
  precedence: () => runDoctor({ dotenv: 'OLLAMA_MODEL=llama3.3\nOLLAMA_NUM_CTX=131072\n', env: { OLLAMA_MODEL: QWEN }, trapExit: true }),
  // The runner is configured, in the data root's .env, for a model the server
  // does not have: once as part of the general check, once asked for by name.
  missingModel: async () => {
    const at = dataRoot();
    writeFileSync(join(at.root, '.env'), 'OLLAMA_MODEL=llama3.3\n');
    const [general, byName] = await Promise.all([
      runDoctor({ at, trapExit: true }),
      runDoctor({ at, args: ['--ollama'], trapExit: true }),
    ]);
    return { general, byName };
  },
  // The data root is somewhere else than the folder the command is started
  // from, and the settings are in that folder's .env. This is the layout of
  // the first trial in docs/OLLAMA.md, and of anyone who keeps the data root
  // outside the checkout and the .env inside it.
  workingDirectory: async () => {
    const cwd = folderWithEnv('OLLAMA_MODEL=llama3.3\n');
    const target = dataRoot();
    const [fromCwd, both, withTarget] = await Promise.all([
      runDoctor({ cwd, trapExit: true }),
      // --cli, so that doctor has not already put the data root's .env into
      // the environment, where it would win as a shell value does.
      runDoctor({ cwd, dotenv: `OLLAMA_MODEL=${GEMMA}\n`, args: ['--ollama', '--cli', 'claude'], trapExit: true }),
      runDoctor({ cwd, at: target, args: ['--target', target.root] }),
    ]);
    return { fromCwd, both, withTarget };
  },
  // CAREER_OPS_PROFILE names a profile outside the data root, long enough to
  // need the next window size. The runner loads the .env files before it
  // looks for the profile, so the name counts from there as from the shell.
  profileElsewhere: async () => {
    const folder = folderWithEnv('');
    const profile = join(folder, 'work-profile.yml');
    writeFileSync(profile, `candidate:\n  full_name: "Jordan Rivera"\n${'# A long note about targeting.\n'.repeat(3000)}`);
    const short = join(folder, 'short-profile.yml');
    writeFileSync(short, 'candidate:\n  full_name: "Jordan Rivera"\n');
    const [fromEnvFile, fromShell, fromBoth] = await Promise.all([
      runDoctor({ args: ['--ollama'], dotenv: `CAREER_OPS_PROFILE=${profile}\n`, trapExit: true }),
      runDoctor({ args: ['--ollama'], env: { CAREER_OPS_PROFILE: profile }, trapExit: true }),
      runDoctor({ args: ['--ollama'], dotenv: `CAREER_OPS_PROFILE=${short}\n`, env: { CAREER_OPS_PROFILE: profile }, trapExit: true }),
    ]);
    return { profile, fromEnvFile, fromShell, fromBoth };
  },
  help: () => runDoctor({ args: ['--help'] }),
  json: async () => {
    const at = dataRoot();
    const before = await runDoctor({ args: ['--json'], at });
    const configured = await runDoctor({
      args: ['--json', '--ollama'],
      env: { OLLAMA_BASE_URL: DEFAULT_URL, OLLAMA_MODEL: QWEN, OLLAMA_THINK: 'on' },
      dotenv: 'OLLAMA_NUM_CTX=65536\nOLLAMA_MODEL_PREFERENCE=gemma4\n',
      at,
    });
    return { before, configured };
  },
};
for (const [name, start] of Object.entries(SCENARIOS)) scenario(name, start);
const ran = (name) => scenario(name, SCENARIOS[name]);

test('doctor: with nothing about Ollama set, no request is made and nothing is said about it', async () => {
  const bare = await ran('bare');
  assert.deepEqual(bare.requests, [], describeSpawn(bare));
  assert.deepEqual(lines(bare), [], describeSpawn(bare));
  assert.match(bare.stdout, /^Result: /m, describeSpawn(bare));
  // The data root is complete and the browser is stubbed, so this is a clean
  // run. The runs below are compared with it.
  assert.equal(bare.status, 0, describeSpawn(bare));
  // A run that skips the check still ends the way doctor always has. Only a
  // run that made requests has a reason to wait for the event loop.
  assert.match(bare.stderr, /^process\.exit called with 0$/m, describeSpawn(bare));
});

test('doctor: the variables of the Ollama server change nothing, with --strict or without it', async () => {
  // The server here has none of the models the runner would pick, so a check
  // that ran would have said so: counting any OLLAMA_* name as "the backend is
  // configured" sent four requests and ended this run with exit 1, for someone
  // who had only tuned their server.
  const [bare, plain, strict] = [await ran('bare'), await ran('plain'), await ran('strict')];
  for (const run of [plain, strict]) {
    assert.deepEqual(run.requests, [], describeSpawn(run));
    assert.deepEqual(lines(run), [], describeSpawn(run));
    assert.equal(run.status, bare.status, describeSpawn(run));
    // No check, so the run ends as upstream's doctor ends.
    assert.match(run.stderr, /^process\.exit called with 0$/m, describeSpawn(run));
  }
  assert.equal(plain.stdout, bare.stdout);
  // --strict is upstream's flag and keeps upstream's meaning: the portal
  // probe, which this data root gives nothing to probe.
  assert.match(strict.stdout, /^Result: /m, describeSpawn(strict));
});

test('doctor --ollama: runs the check, prints its four lines, and exits 0 without process.exit', async () => {
  const [bare, flag] = [await ran('bare'), await ran('flag')];
  assert.deepEqual(flag.requests, STUB_REQUESTS, describeSpawn(flag));
  assert.deepEqual(lines(flag), HEALTHY_LINES, describeSpawn(flag));
  assert.equal(flag.status, bare.status, describeSpawn(flag));
  assert.doesNotMatch(flag.stderr, /process\.exit called/, describeSpawn(flag));
  // The check adds lines to the report and changes nothing else in it.
  const withoutCheck = flag.stdout.split('\n').filter((line) => !OLLAMA_LINE.test(line)).join('\n');
  assert.equal(withoutCheck, bare.stdout);
});

test('doctor --ollama: a failed check is a failed run, exit 1, still without process.exit', async () => {
  const { byName } = await ran('missingModel');
  assert.equal(byName.status, 1, describeSpawn(byName));
  assert.match(byName.stdout, /^✗ Ollama model: llama3\.3 is not installed$/m);
  assert.match(byName.stdout, /^ {2}→ Model "llama3\.3" is not installed in Ollama\. Installed chat models: /m);
  assert.match(byName.stdout, /^Result: 1 issue found\./m);
  assert.doesNotMatch(byName.stderr, /process\.exit called/, describeSpawn(byName));

  const down = await ran('flagDown');
  assert.equal(down.status, 1, describeSpawn(down));
  assert.deepEqual(down.requests, [`GET ${DEFAULT_URL}/api/version`]);
  assert.match(down.stdout, new RegExp(`^✗ Ollama not reachable at ${DEFAULT_URL}$`, 'm'));
  assert.match(down.stdout, /^ {2}→ Start Ollama \(ollama serve\) or set OLLAMA_BASE_URL$/m);
  assert.doesNotMatch(down.stderr, /process\.exit called/, describeSpawn(down));
});

test('doctor: a setting of the runner in the shell switches the check on, and a server that is off is only a warning', async () => {
  const shell = await ran('shell');
  assert.deepEqual(shell.requests, STUB_REQUESTS, describeSpawn(shell));
  assert.deepEqual(lines(shell).slice(1), [`✓ Ollama model: ${GEMMA} (requested)`, fitsLine(65536), '✓ thinking: off (auto)']);
  assert.doesNotMatch(shell.stderr, /process\.exit called/, describeSpawn(shell));

  const [bare, down] = [await ran('bare'), await ran('shellDown')];
  assert.match(down.stdout, new RegExp(`^⚠ Ollama not reachable at ${DEFAULT_URL}$`, 'm'), describeSpawn(down));
  assert.equal(down.status, bare.status, describeSpawn(down));
  assert.doesNotMatch(down.stderr, /process\.exit called/, describeSpawn(down));
});

test('doctor: a model the server does not have is a warning in the general check, and a failure when asked for by name', async () => {
  const [bare, { general, byName }] = [await ran('bare'), await ran('missingModel')];
  // Not asked for by name: what a server holds must not turn the setup check
  // red. The same lines as the failure, under the other mark.
  assert.deepEqual(general.requests, STUB_REQUESTS.slice(0, 2), describeSpawn(general));
  assert.match(general.stdout, /^⚠ Ollama model: llama3\.3 is not installed$/m, describeSpawn(general));
  assert.match(general.stdout, /^ {2}→ Model "llama3\.3" is not installed in Ollama\. Installed chat models: /m);
  assert.doesNotMatch(general.stdout, /^✗/m, describeSpawn(general));
  assert.equal(general.status, bare.status, describeSpawn(general));
  // One warning more than the run without the check, and still no issue.
  const warnings = (run) => Number(/^Result: All checks passed \((\d+) warnings? — see above\)/m.exec(run.stdout)?.[1] ?? 0);
  assert.match(general.stdout, /^Result: All checks passed/m, describeSpawn(general));
  assert.equal(warnings(general), warnings(bare) + 1, describeSpawn(general));
  assert.doesNotMatch(general.stderr, /process\.exit called/, describeSpawn(general));

  assert.deepEqual(byName.requests, general.requests);
  assert.equal(byName.status, 1, describeSpawn(byName));
});

test('doctor: a setting of the runner in the data root .env switches the check on, even when doctor did not load that file', async () => {
  const dotenv = await ran('dotenv');
  assert.deepEqual(dotenv.requests, STUB_REQUESTS, describeSpawn(dotenv));
  assert.ok(lines(dotenv).includes(`✓ Ollama model: ${GEMMA} (requested)`), describeSpawn(dotenv));
  assert.doesNotMatch(dotenv.stderr, /process\.exit called/, describeSpawn(dotenv));
});

test('doctor: the shell wins over the .env, setting by setting', async () => {
  const precedence = await ran('precedence');
  // The model from the shell (the .env names one that is not installed), the
  // window from the .env.
  assert.deepEqual(lines(precedence).slice(1, 3), [`✓ Ollama model: ${QWEN} (requested)`, fitsLine(131072)], describeSpawn(precedence));
});

test('doctor: the .env of the working directory is read as the runner reads it, after the data root\'s', async () => {
  const { fromCwd, both, withTarget } = await ran('workingDirectory');
  // Only the folder the command was started from has a .env. The runner
  // would take its model from there; a check that read the data root's file
  // alone did not run at all, or with --ollama passed another model.
  assert.deepEqual(fromCwd.requests, STUB_REQUESTS.slice(0, 2), describeSpawn(fromCwd));
  assert.match(fromCwd.stdout, /^⚠ Ollama model: llama3\.3 is not installed$/m, describeSpawn(fromCwd));
  assert.doesNotMatch(fromCwd.stderr, /process\.exit called/, describeSpawn(fromCwd));

  // Both folders have one: the data root's is read first and wins.
  assert.deepEqual(both.requests, STUB_REQUESTS, describeSpawn(both));
  assert.deepEqual(lines(both).slice(1, 2), [`✓ Ollama model: ${GEMMA} (requested)`], describeSpawn(both));

  // --target diagnoses another checkout, and the folder the command happens
  // to be started from says nothing about that one.
  assert.deepEqual(withTarget.requests, [], describeSpawn(withTarget));
  assert.deepEqual(lines(withTarget), [], describeSpawn(withTarget));
  assert.match(withTarget.stdout, /^Result: /m, describeSpawn(withTarget));
});

test('doctor: the window is sized for the profile the runner reads, named in a .env or in the shell', async () => {
  const { profile, fromEnvFile, fromShell, fromBoth } = await ran('profileElsewhere');
  const size = measurePromptTokens({ codeRoot: ROOT, dataRoot: fromEnvFile.root, profilePath: profile });
  // The long profile takes the prompt past what the default window holds;
  // sized for the short one in the data root, the report said 65536 fits.
  assert.ok(size > ROOT_PROMPT + 20_000, `the long profile makes a prompt of ${size} tokens`);
  const raised = `⚠ num_ctx raised from 65536 to 98304 to fit ~${Math.round(size / 1000)}K prompt + output (model max 262144)`;
  for (const run of [fromEnvFile, fromShell, fromBoth]) {
    assert.ok(lines(run).includes(raised), describeSpawn(run));
    assert.doesNotMatch(run.stderr, /process\.exit called/, describeSpawn(run));
  }
});

test('doctor --json: the same bytes with the backend configured as without, and no request', async () => {
  const { before, configured } = await ran('json');
  assert.equal(before.status, 0, describeSpawn(before));
  assert.equal(configured.status, 0, describeSpawn(configured));
  assert.equal(JSON.parse(before.stdout).data_root, before.root);
  assert.equal(configured.stdout, before.stdout);
  assert.deepEqual(before.requests, []);
  assert.deepEqual(configured.requests, []);
});

test('doctor --help: lists --ollama', async () => {
  const help = await ran('help');
  assert.equal(help.status, 0, describeSpawn(help));
  assert.match(help.stdout, /^ {2}node doctor\.mjs --ollama +# check the local Ollama server, model and context size$/m);
  assert.deepEqual(help.requests, []);
});

test('doctor --ollama: a run whose event loop stays open still ends, through the backstop', async () => {
  // A timer nothing clears holds the loop open after the report. Without the
  // backstop this child would run until the test's own timeout kills it.
  const [bare, held] = [await ran('bare'), await ran('heldOpen')];
  assert.equal(held.error, null, describeSpawn(held));
  assert.equal(held.status, bare.status, describeSpawn(held));
  assert.deepEqual(lines(held), HEALTHY_LINES);
  assert.match(held.stderr, new RegExp(`^process\\.exit called with ${bare.status}$`, 'm'), describeSpawn(held));

  // The backstop carries the run's own verdict, not a fixed code.
  const failing = await ran('heldOpenFailing');
  assert.equal(failing.error, null, describeSpawn(failing));
  assert.equal(failing.status, 1, describeSpawn(failing));
  assert.match(failing.stderr, /^process\.exit called with 1$/m, describeSpawn(failing));
});

// ---------------------------------------------------------------------------
// doctor.mjs in a copy without the backend
// ---------------------------------------------------------------------------

// The files this fork adds for the backend (config/local-paths.txt). doctor.mjs
// is upstream's file with a few lines of this fork in it, so a copy can hold it
// without them: a commit or a merge that left one out, or a sandbox built from
// the tracked files before they were added. The second is what
// tests/doctor-without-node-modules.test.mjs and
// tests/claude-plugin-install.test.mjs build, and with the check imported at
// the top of doctor.mjs both died there on ERR_MODULE_NOT_FOUND.
const BACKEND_FILES = ['ollama-runner.mjs', 'lib/ollama-config.mjs', 'lib/ollama-client.mjs', 'lib/ollama-doctor.mjs'];

/**
 * A copy of what the checkout ships, without the backend, and with the
 * installed packages linked in. Null where git cannot list the files (a tree
 * that was copied, not cloned).
 */
function checkoutWithoutBackend() {
  const listed = spawnSync('git', ['ls-files', '-z', '--', '.', ':!tests', ':!web', ':!docs'], { cwd: ROOT, encoding: 'utf8' });
  if (listed.status !== 0 || !listed.stdout) return null;
  const copy = mkdtempSync(join(tmpdir(), 'ollama-doctor-test-'));
  scratch.push(copy);
  for (const file of listed.stdout.split('\0').filter(Boolean)) {
    if (BACKEND_FILES.includes(file)) continue;
    mkdirSync(dirname(join(copy, file)), { recursive: true });
    copyFileSync(join(ROOT, file), join(copy, file));
  }
  assert.equal(linkNodeModules(copy), null);
  return copy;
}

const withoutBackend = scenario('withoutBackend', async () => {
  const copy = checkoutWithoutBackend();
  if (!copy) return null;
  const doctor = join(copy, 'doctor.mjs');
  const [plain, configured, flag, json] = await Promise.all([
    runDoctor({ doctor }),
    runDoctor({ doctor, env: { OLLAMA_MODEL: GEMMA } }),
    runDoctor({ doctor, args: ['--ollama'] }),
    runDoctor({ doctor, args: ['--json'] }),
  ]);
  // Last, in the same copy: one of the files is there now, and does not parse.
  writeFileSync(join(copy, 'lib', 'ollama-doctor.mjs'), 'export const checkOllama = ;\n');
  const broken = await runDoctor({ doctor });
  return { plain, configured, flag, json, broken };
});

test('doctor in a copy without the backend: the report is whole, and --ollama says what is missing', async (t) => {
  const runs = await withoutBackend;
  if (!runs) {
    t.skip('git cannot list the files of this checkout');
    return;
  }
  const { plain, configured, flag, json, broken } = runs;
  for (const run of [plain, configured, flag, json]) {
    assert.doesNotMatch(run.stdout + run.stderr, /ERR_MODULE_NOT_FOUND|Cannot find module/, describeSpawn(run));
    assert.deepEqual(run.requests, [], describeSpawn(run));
  }
  // Every other check ran. Nothing is said about a backend that was not asked
  // about by name, configured or not: there is nothing here to check it with.
  for (const run of [plain, configured]) {
    assert.match(run.stdout, /^Result: /m, describeSpawn(run));
    assert.deepEqual(lines(run), [], describeSpawn(run));
  }
  assert.equal(configured.stdout, plain.stdout);
  assert.equal(configured.status, plain.status, describeSpawn(configured));

  // Asked for by name, a check that cannot run is a failed check.
  assert.deepEqual(lines(flag), ['✗ Ollama checks are not part of this checkout'], describeSpawn(flag));
  assert.match(flag.stdout, /^ {2}→ lib\/ollama-doctor\.mjs, .+ is missing: restore the files listed in config\/local-paths\.txt/m);
  assert.equal(flag.status, 1, describeSpawn(flag));
  // Every other check is there, with the verdict it has in the plain run.
  const verdicts = (run) => run.stdout.split('\n').filter((line) => /^[✓✗⚠] /.test(line) && !/Ollama checks/.test(line));
  assert.ok(verdicts(plain).length >= 10, describeSpawn(plain));
  assert.deepEqual(verdicts(flag), verdicts(plain));

  // --json is what an agent reads at the start of a session. It never loads the check.
  assert.equal(json.status, 0, describeSpawn(json));
  assert.equal(JSON.parse(json.stdout).data_root, json.root);

  // Only a file that is not there is skipped. One that is there and does not
  // load is a broken checkout, and doctor says so instead of passing over it.
  assert.equal(broken.status, 1, describeSpawn(broken));
  assert.match(broken.stderr, /^doctor\.mjs failed: /m, describeSpawn(broken));
  assert.doesNotMatch(broken.stdout, /^Result: /m, describeSpawn(broken));
});

// ---------------------------------------------------------------------------
// doctor.mjs over real HTTP
// ---------------------------------------------------------------------------

/** The stub as a real server on 127.0.0.1, for the runs that need real sockets. */
async function withStubServer(run) {
  const state = server();
  const requests = [];
  const http = createServer((req, res) => {
    let raw = '';
    req.setEncoding('utf8');
    req.on('data', (chunk) => { raw += chunk; });
    req.on('end', () => {
      requests.push(`${req.method} ${req.url}`);
      const reply = route(state, req.method, req.url, raw ? JSON.parse(raw) : null);
      res.writeHead(reply.status, { 'Content-Type': 'application/json' });
      res.end(reply.text ?? JSON.stringify(reply.json));
    });
  });
  await new Promise((done, fail) => {
    http.once('error', fail);
    http.listen(0, '127.0.0.1', done);
  });
  try {
    return await run({ address: `127.0.0.1:${http.address().port}`, requests });
  } finally {
    http.closeAllConnections();
    await new Promise((done) => http.close(done));
  }
}

const overHttp = scenario('overHttp', () => withStubServer(async ({ address, requests }) => {
  // The stub is named by OLLAMA_HOST, which is where a client finds a server
  // but is not a reason to go looking for one.
  const env = { OLLAMA_HOST: address, OLLAMA_FLASH_ATTENTION: '1' };
  const start = (args) => {
    const { root } = dataRoot();
    return runNode(['--import', PLAYWRIGHT_STUB, DOCTOR, ...args], { cwd: root, env: childEnv(root, env) });
  };
  const plain = await start([]);
  const seenByPlain = [...requests];
  const flag = await start(['--ollama']);
  return { address, plain, seenByPlain, flag, seenByFlag: requests.slice(seenByPlain.length) };
}));

test('doctor over real HTTP: no request without --ollama; with it, four requests and a clean exit', async () => {
  const { address, plain, seenByPlain, flag, seenByFlag } = await overHttp;
  assert.deepEqual(seenByPlain, [], describeSpawn(plain));
  assert.deepEqual(lines(plain), []);

  // The control: the same environment does reach the stub once the check is
  // asked for, so the empty list above is not a stub nobody could find.
  assert.deepEqual(seenByFlag, ['GET /api/version', 'GET /api/tags', 'POST /api/show', 'GET /api/ps'], describeSpawn(flag));
  assert.deepEqual(lines(flag), [`✓ Ollama 0.40.16 reachable at http://${address}`, ...HEALTHY_LINES.slice(1)]);
  // On Windows with an affected Node, ending this run through process.exit
  // prints a libuv assertion and exits with a crash code.
  assert.equal(flag.status, plain.status, describeSpawn(flag));
  assert.doesNotMatch(flag.stderr, /Assertion failed/, describeSpawn(flag));
  assert.equal(plain.status, 0, describeSpawn(plain));
});

// ---------------------------------------------------------------------------
// doctor.mjs and the runner, on the same files
// ---------------------------------------------------------------------------

/**
 * A data root whose prompt has a realistic size: the example CV and profile
 * this checkout ships, with both optional files, so that the prompt's own
 * wording is at its longest. `cvBytes` makes the CV that long.
 */
function exampleRoot({ cvBytes } = {}) {
  const { root } = dataRoot();
  const cv = readFileSync(join(ROOT, 'examples', 'cv-example.md'), 'utf8');
  writeFileSync(join(root, 'cv.md'), cvBytes ? cv.repeat(Math.ceil(cvBytes / cv.length)).slice(0, cvBytes) : cv);
  copyFileSync(join(ROOT, 'config', 'profile.example.yml'), join(root, 'config', 'profile.yml'));
  copyFileSync(join(ROOT, 'modes', '_profile.template.md'), join(root, 'modes', '_profile.md'));
  copyFileSync(join(ROOT, 'modes', '_custom.template.md'), join(root, 'modes', '_custom.md'));
  return root;
}

const sideBySide = scenario('sideBySide', () => withStubServer(async ({ address }) => {
  if (!existsSync(RUNNER)) return null;
  const start = (script, args, root, settings = {}) => runNode(
    [...(script === DOCTOR ? ['--import', PLAYWRIGHT_STUB] : []), script, ...args],
    { cwd: root, env: childEnv(root, { OLLAMA_BASE_URL: `http://${address}`, ...settings }) },
  );
  const both = async (root, settings) => {
    const [doctor, runner] = await Promise.all([
      start(DOCTOR, ['--ollama'], root, settings),
      start(RUNNER, ['doctor'], root, settings),
    ]);
    return { doctor, runner };
  };
  const example = exampleRoot();
  // A CV of 38 KB is past what the default window holds: the runner moves to
  // the next size for it.
  const longCv = exampleRoot({ cvBytes: 38_000 });
  // 56000 is between what a 40,000-token prompt needs (54,000, the number the
  // check used to assume and to recommend) and what this one needs.
  const [pinned, pinnedLongCv, unpinnedLongCv] = await Promise.all([
    both(example, { OLLAMA_NUM_CTX: '56000' }),
    both(longCv, { OLLAMA_NUM_CTX: '56000' }),
    both(longCv),
  ]);
  // The window doctor.mjs recommends, handed to the runner.
  const advised = /→ Raise OLLAMA_NUM_CTX to (\d+) or more/.exec(pinned.doctor.stdout)?.[1];
  const followed = advised ? await start(RUNNER, ['doctor'], example, { OLLAMA_NUM_CTX: advised }) : null;
  return { pinned, pinnedLongCv, unpinnedLongCv, advised, followed };
}));

/** The size and the need in "num_ctx too small: prompt ~N tokens + … needs M", exactly as the fit check counted them. */
function refused(run) {
  const [, prompt, need] = /^✗ num_ctx too small: prompt ~(\d+) tokens \+ \d+ output needs (\d+); num_ctx is 56000/m.exec(run.stdout) ?? [];
  assert.ok(prompt && need, `no refusal of the 56000 window in:\n${describeSpawn(run)}`);
  return { prompt: Number(prompt), need: Number(need) };
}

test('doctor.mjs and the runner size the same prompt alike, and what doctor.mjs recommends the runner accepts', async (t) => {
  const runs = await sideBySide;
  if (!runs) {
    t.skip('this copy has no ollama-runner.mjs');
    return;
  }
  const { pinned, pinnedLongCv, unpinnedLongCv, advised, followed } = runs;

  // The same verdict on a window that a fixed 40,000-token prompt passed and
  // the runner refused.
  for (const { doctor, runner } of [pinned, pinnedLongCv]) {
    assert.equal(doctor.status, 1, describeSpawn(doctor));
    assert.equal(runner.status, 1, describeSpawn(runner));
    // And the same numbers: both size the prompt with evaluationPromptTokens()
    // of lib/ollama-config.mjs, from the same files. While each counted for
    // itself they were about 400 tokens apart, and a window pinned at the
    // number the runner named was refused by doctor.mjs.
    assert.deepEqual(
      refused(doctor), refused(runner),
      `doctor.mjs and the runner size one prompt differently:\n${describeSpawn(doctor)}\n${describeSpawn(runner)}`,
    );
  }

  // What doctor.mjs says to set is enough for the runner, and it is the number
  // the runner's own check names.
  assert.equal(advised, String(refused(pinned.doctor).need));
  assert.equal(/→ Raise OLLAMA_NUM_CTX to (\d+) or more/.exec(pinned.runner.stdout)?.[1], advised, describeSpawn(pinned.runner));
  assert.match(followed.stdout, new RegExp(`^✓ num_ctx ${advised} fits `, 'm'), describeSpawn(followed));
  assert.equal(followed.status, 0, describeSpawn(followed));

  // Left to the runner, the window for the long CV is the next size up, and
  // both say so. A check that assumed a typical prompt reported 65536 here,
  // and then compared the loaded copy with a size the runner does not use.
  for (const run of [unpinnedLongCv.doctor, unpinnedLongCv.runner]) {
    assert.match(run.stdout, /^⚠ num_ctx raised from 65536 to 98304 to fit /m, describeSpawn(run));
  }
  assert.equal(unpinnedLongCv.runner.status, 0, describeSpawn(unpinnedLongCv.runner));
});

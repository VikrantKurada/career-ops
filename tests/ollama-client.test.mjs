// tests/ollama-client.test.mjs — the module that talks to Ollama
// (lib/ollama-client.mjs).
//
// Every way a local model fails is HTTP 200 with plausible text in it: a
// prompt the server cut to fit its window, a reply that stopped at the output
// cap, a model that only thought, a tool call, a connection that dropped
// halfway. Each one would be saved as a report if the client did not name it.
// These tests drive the client against a stub server that produces each shape,
// and check what goes out as well: where the settings sit in the request, that
// the privacy gate saw it first, that nothing is sent when the gate says no,
// and that nothing goes to an address a redirect names.
//
// No Ollama is needed, and none is contacted. The stub is a real HTTP server
// on 127.0.0.1 so that fragmentation, disconnects and timeouts are the
// transport's own; where the exact split matters, the reply is an injected
// stream instead.
//
// Run:  node --test tests/ollama-client.test.mjs
import { after, test } from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { isBuiltin } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';

import { OllamaError, chatWithRetry, createOllamaClient, stripThinkTags } from '../lib/ollama-client.mjs';
import { installPiiInterceptor, uninstallPiiInterceptor } from '../lib/pii-interceptor.mjs';
import { resetDefaultGuard } from '../lib/privacy-guard.mjs';

const exec = promisify(execFile);
const ROOT = fileURLToPath(new URL('../', import.meta.url));
const CLIENT_SOURCE = join(ROOT, 'lib', 'ollama-client.mjs');

// A data root of this suite's own, so the privacy guard reads the config a test
// writes and never the developer's. The guard's switches are cleared for the
// same reason: a bypass exported in the shell would make "blocked" untestable.
const DATA_ROOT = mkdtempSync(join(tmpdir(), 'ollama-client-test-'));
process.env.CAREER_OPS_ROOT = DATA_ROOT;
process.env.CAREER_OPS_DATA_DIR = '';
for (const name of ['CAREER_OPS_PRIVACY', 'PRIVACY_GUARD', 'PRIVACY_BYPASS_TOKEN', 'PRIVACY_REDACT']) delete process.env[name];
after(() => rmSync(DATA_ROOT, { recursive: true, force: true, maxRetries: 10 }));

// Importing the interceptor installs it over fetch. The client is tested on
// its own first; the one test that needs the interceptor puts it back.
uninstallPiiInterceptor();

// Fictional, like every value in this file.
const PROTECTED_NAME = 'Jordan A. Placeholder';
const REQUEST = Object.freeze({
  model: 'stub-model',
  system: 'You evaluate job offers.',
  user: 'JOB DESCRIPTION TO EVALUATE:\n\nBuild reliable software.',
});

// ---------------------------------------------------------------------------
// Stub server
// ---------------------------------------------------------------------------

/**
 * Run `run` against a stub server whose every request goes to `handler`.
 * `requests` holds what arrived: method, path, raw body and parsed body.
 */
async function withServer(handler, run, { timeoutMs = 10_000 } = {}) {
  const requests = [];
  const server = createServer((req, res) => {
    let raw = '';
    req.setEncoding('utf8');
    req.on('data', (chunk) => { raw += chunk; });
    req.on('end', () => {
      let body = null;
      try { body = raw ? JSON.parse(raw) : null; } catch { /* left null: not JSON */ }
      const seen = { method: req.method, url: req.url, headers: req.headers, raw, body, index: requests.length };
      requests.push(seen);
      handler(seen, res);
    });
  });
  await new Promise((done, fail) => {
    server.once('error', fail);
    server.listen(0, '127.0.0.1', done);
  });
  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  try {
    return await run({ baseUrl, requests, client: createOllamaClient({ baseUrl, timeoutMs }) });
  } finally {
    server.closeAllConnections();
    await new Promise((done) => server.close(done));
  }
}

/** A port nothing listens on: one that was just open, and is closed again. */
async function closedPort() {
  const server = createServer();
  await new Promise((done) => server.listen(0, '127.0.0.1', done));
  const { port } = server.address();
  await new Promise((done) => server.close(done));
  return port;
}

const record = (object) => `${JSON.stringify(object)}\n`;
const text = (content) => record({ model: 'stub-model', message: { role: 'assistant', content }, done: false });
const thinking = (thought) => record({ model: 'stub-model', message: { role: 'assistant', content: '', thinking: thought }, done: false });
/**
 * The final record. It reports the prompt as a quarter of the request's
 * characters, which is how callers estimate it, so the truncation check stays
 * quiet unless a test lowers the count.
 */
const done = (seen, extra = {}) => record({
  model: 'stub-model',
  message: { role: 'assistant', content: '' },
  done: true,
  done_reason: 'stop',
  prompt_eval_count: Math.ceil((seen?.raw.length ?? 0) / 4),
  eval_count: 7,
  ...extra,
});
const send = (res, body, status = 200) => {
  res.writeHead(status, { 'Content-Type': 'application/x-ndjson' });
  res.end(body);
};
const json = (res, object, status = 200) => {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(object));
};

/** A fetch stand-in that answers every call with `body`, recording the calls. */
function fakeFetch(body, { status = 200 } = {}) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url: String(url), init });
    return new Response(typeof body === 'function' ? body() : body, { status });
  };
  return { fetchImpl, calls };
}

/** A response body delivered in exactly these pieces. */
function pieces(parts) {
  return () => new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(typeof part === 'string' ? Buffer.from(part) : part);
      controller.close();
    },
  });
}

const stubClient = (body, options) => {
  const { fetchImpl, calls } = fakeFetch(body, options);
  return { client: createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs: 10_000, fetchImpl }), calls };
};

/** The OllamaError a promise rejects with, checked for its code. */
async function failsWith(promise, code) {
  let caught;
  try {
    await promise;
  } catch (err) {
    caught = err;
  }
  assert.ok(caught, `expected a rejection with code ${code}, but the call succeeded`);
  assert.ok(caught instanceof OllamaError, `expected an OllamaError, got ${caught.name}: ${caught.message}`);
  assert.equal(caught.code, code, caught.message);
  return caught;
}

/** Run `run` with a privacy config in this suite's data root. */
async function withPrivacyConfig(yaml, run, env = {}) {
  const file = join(DATA_ROOT, 'config', 'privacy.yml');
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, yaml);
  Object.assign(process.env, env);
  // The guard is loaded once per process; make it read the file just written.
  resetDefaultGuard();
  try {
    return await run();
  } finally {
    for (const name of Object.keys(env)) delete process.env[name];
    rmSync(file, { force: true });
    resetDefaultGuard();
  }
}

// ---------------------------------------------------------------------------
// The request
// ---------------------------------------------------------------------------

test('chat posts a streamed request to /api/chat with every setting where the server reads it', async () => {
  await withServer((seen, res) => send(res, text('Evaluation.') + done(seen)), async ({ client, requests }) => {
    await client.chat({ ...REQUEST, numCtx: 98304, numPredict: 20000, temperature: 0.3, think: false, keepAlive: '10m', seed: 7 });
    assert.equal(requests.length, 1);
    assert.equal(requests[0].method, 'POST');
    assert.equal(requests[0].url, '/api/chat');
    assert.equal(requests[0].headers['content-type'], 'application/json');
    assert.deepEqual(requests[0].body, {
      model: 'stub-model',
      // The system prompt is one plain string, not an array of parts.
      messages: [{ role: 'system', content: REQUEST.system }, { role: 'user', content: REQUEST.user }],
      // Without stream:true no response header arrives until the reply is
      // complete, and fetch gives up on a silent connection after 300 s.
      stream: true,
      keep_alive: '10m',
      think: false,
      // Generation settings are read from `options` and from nowhere else.
      options: { temperature: 0.3, num_ctx: 98304, num_predict: 20000, seed: 7 },
    });
    assert.deepEqual(Object.keys(requests[0].body), ['model', 'messages', 'stream', 'keep_alive', 'think', 'options']);
  });
});

test('think and keep_alive are left out when the caller has none, and the generation defaults are always sent', async () => {
  await withServer((seen, res) => send(res, text('Evaluation.') + done(seen)), async ({ client, requests }) => {
    await client.chat({ ...REQUEST });
    await client.chat({ ...REQUEST, think: undefined, seed: undefined, keepAlive: undefined });
    await client.chat({ ...REQUEST, think: true, keepAlive: 300 });
    for (const sent of requests.slice(0, 2)) {
      assert.equal('think' in sent.body, false, 'a model without the capability rejects the field');
      assert.equal('seed' in sent.body.options, false);
      // The model's own temperature is 1.0, so these go out even when the
      // caller names none.
      assert.deepEqual(sent.body.options, { temperature: 0.2, num_ctx: 65536, num_predict: 16384 });
      // How long the model stays in GPU memory is the server's setting. A
      // default sent from here ("30m", once) overrode it on every request and
      // held 18-24 GB for half an hour after each command.
      assert.equal('keep_alive' in sent.body, false);
      assert.deepEqual(Object.keys(sent.body), ['model', 'messages', 'stream', 'options']);
      assert.equal(sent.body.stream, true);
    }
    assert.equal(requests[2].body.think, true);
    assert.equal(requests[2].body.keep_alive, 300, 'a number of seconds stays a number');
    // No system prompt: no empty system message in its place.
    await client.chat({ model: 'stub-model', user: 'Just the user message.' });
    assert.deepEqual(requests[3].body.messages, [{ role: 'user', content: 'Just the user message.' }]);
    // Zero is a value, not "unset": OLLAMA_TEMPERATURE=0, OLLAMA_KEEP_ALIVE=0 and
    // OLLAMA_SEED=0 are all things a user can ask for.
    await client.chat({ ...REQUEST, temperature: 0, keepAlive: 0, seed: 0 });
    assert.equal(requests[4].body.keep_alive, 0);
    assert.deepEqual(requests[4].body.options, { temperature: 0, num_ctx: 65536, num_predict: 16384, seed: 0 });
    // -1 is "never unload", and a duration string goes out as it is.
    await client.chat({ ...REQUEST, keepAlive: -1 });
    await client.chat({ ...REQUEST, keepAlive: '1h30m' });
    assert.equal(requests[5].body.keep_alive, -1);
    assert.equal(requests[6].body.keep_alive, '1h30m');
    // What is not a keep-alive is not sent as one.
    for (const keepAlive of [null, '', '   ', NaN, {}, true]) {
      await client.chat({ ...REQUEST, keepAlive });
      assert.equal('keep_alive' in requests.at(-1).body, false, String(keepAlive));
    }
  });
});

test('chat refuses a request that names no model or no user message', async () => {
  const { client, calls } = stubClient(text('x') + done());
  await assert.rejects(client.chat({ user: 'hello' }), TypeError);
  await assert.rejects(client.chat({ model: 'stub-model', system: 'only a system prompt' }), TypeError);
  assert.equal(calls.length, 0);
});

// ---------------------------------------------------------------------------
// The reply
// ---------------------------------------------------------------------------

test('thinking is counted and reported, and never becomes part of the content', async () => {
  const body = (seen) => thinking('First I will ') + thinking('consider THINKING-MARKER.') + text('# Evaluation\n')
    + text('Body.\n') + done(seen, { prompt_eval_count: 1200, prompt_eval_cached_count: 1000, eval_count: 340 });
  await withServer((seen, res) => send(res, body(seen)), async ({ client }) => {
    const answer = [];
    const thought = [];
    const result = await client.chat({ ...REQUEST, think: true, onToken: (piece) => answer.push(piece), onThinking: (piece) => thought.push(piece) });
    assert.deepEqual(result, {
      content: '# Evaluation\nBody.',
      thinkingChars: 'First I will consider THINKING-MARKER.'.length,
      usage: { prompt_tokens: 1200, completion_tokens: 340, total_tokens: 1540, cached_tokens: 1000 },
      doneReason: 'stop',
    });
    assert.doesNotMatch(result.content, /THINKING-MARKER/);
    assert.deepEqual(answer, ['# Evaluation\n', 'Body.\n']);
    assert.deepEqual(thought, ['First I will ', 'consider THINKING-MARKER.']);
  });
});

test('a reply whose done record has no reason or no counts is still a reply', async () => {
  const { client } = stubClient(text('Evaluation.') + record({ done: true }));
  assert.deepEqual(await client.chat(REQUEST), {
    content: 'Evaluation.',
    thinkingChars: 0,
    // Nothing was counted, and nothing is invented.
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, cached_tokens: 0 },
    doneReason: 'stop',
  });
});

test('stripThinkTags removes inline reasoning in the three shapes it comes in', () => {
  assert.equal(stripThinkTags('<think>\nreasoning\n</think>\n\n# Evaluation'), '# Evaluation');
  assert.equal(stripThinkTags('Intro.\n<think>a</think>\nMiddle.<THINK>b</THINK> End.'), 'Intro.\n\nMiddle. End.');
  // Cut off while still reasoning: there is no answer in it.
  assert.equal(stripThinkTags('<think>reasoning that never ended'), '');
  assert.equal(stripThinkTags('  \n<think>reasoning that never ended'), '');
  // A template that opens the block itself leaves only the closing tag.
  assert.equal(stripThinkTags('reasoning without an opening tag</think>\n# Evaluation'), '# Evaluation');
  // Nothing to remove: only the surrounding whitespace goes.
  assert.equal(stripThinkTags('  # Evaluation\nBody.\n'), '# Evaluation\nBody.');
  assert.equal(stripThinkTags('A report may say "think" and <thinking> freely.'), 'A report may say "think" and <thinking> freely.');
  assert.equal(stripThinkTags(''), '');
  assert.equal(stripThinkTags(undefined), '');
  // Running it twice changes nothing more.
  const once = stripThinkTags('<think>x</think>Answer <think>y</think>here.');
  assert.equal(stripThinkTags(once), once);
});

test('inline think tags are stripped from the content chat returns', async () => {
  const { client } = stubClient(text('<think>Draft: SCORE 9.9</think>\n') + text('# Evaluation\nSCORE: 3.5') + done());
  const result = await client.chat(REQUEST);
  assert.equal(result.content, '# Evaluation\nSCORE: 3.5');
});

// ---------------------------------------------------------------------------
// Replies that must not be returned
// ---------------------------------------------------------------------------

test('a reply cut at the output cap is INCOMPLETE, with the two settings that fix it', async () => {
  await withServer((seen, res) => send(res, text('Partial report') + done(seen, { done_reason: 'length' })), async ({ client }) => {
    const err = await failsWith(client.chat({ ...REQUEST, numCtx: 65536, numPredict: 6000 }), 'INCOMPLETE');
    assert.equal(err.message, 'Reply stopped with done_reason "length" after 14 chars. Raise OLLAMA_NUM_PREDICT (now 6000) or OLLAMA_NUM_CTX (now 65536).');
    assert.equal(err.partialChars, 14);
  });
});

test('a reply cut while the model was still thinking says so, and does not ask for more room', async () => {
  // The measured case: gemma4 with thinking on spent 25,520 output tokens on
  // 78K-87K characters of thinking and never began the report. "Raise
  // OLLAMA_NUM_PREDICT" sends the user to buy more of the same.
  const thought = 'Considering the role. '.repeat(40);
  const cut = (seen) => thinking(thought) + done(seen, { done_reason: 'length' });
  await withServer((seen, res) => send(res, cut(seen)), async ({ client }) => {
    const on = await failsWith(client.chat({ ...REQUEST, think: true, numPredict: 28000 }), 'INCOMPLETE');
    assert.equal(
      on.message,
      `Reply stopped with done_reason "length" before the answer began: the model spent its output budget thinking (${thought.length} chars of it). `
      + 'Leave thinking off: drop --think on or OLLAMA_THINK=on. Off is the default.',
    );
    assert.doesNotMatch(on.message, /Raise OLLAMA_NUM_PREDICT|OLLAMA_NUM_CTX/);
    assert.equal(on.partialChars, 0);

    // Thinking was not asked for and came anyway: turning it off is not advice
    // this request can follow, so the message does not give it.
    for (const think of [false, undefined]) {
      const unasked = await failsWith(client.chat({ ...REQUEST, think, numPredict: 16384 }), 'INCOMPLETE');
      assert.equal(
        unasked.message,
        `Reply stopped with done_reason "length" before the answer began: the model spent its output budget thinking (${thought.length} chars of it). `
        + 'This request did not switch thinking on, so the model thinks whatever it is asked: use another one (--model), or raise OLLAMA_NUM_PREDICT (now 16384).',
      );
    }
  });
  // Reasoning written into the answer between think tags is the same case.
  const inline = await failsWith(stubClient(text('<think>still reasoning') + done(undefined, { done_reason: 'length' })).client.chat({ ...REQUEST, think: true }), 'INCOMPLETE');
  assert.match(inline.message, /before the answer began: the model spent its output budget thinking \(22 chars of it\)/);

  // Once the answer has begun, the cap really was too small for it, and the
  // usual message stands. So does it for a cut with nothing at all.
  const begun = await failsWith(stubClient(thinking(thought) + text('# Evaluation') + done(undefined, { done_reason: 'length' })).client.chat({ ...REQUEST, think: true }), 'INCOMPLETE');
  assert.match(begun.message, /^Reply stopped with done_reason "length" after 12 chars\. Raise OLLAMA_NUM_PREDICT/);
  const nothing = await failsWith(stubClient(done(undefined, { done_reason: 'length' })).client.chat(REQUEST), 'INCOMPLETE');
  assert.match(nothing.message, /^Reply stopped with done_reason "length" after 0 chars\. Raise OLLAMA_NUM_PREDICT/);
  // Another reason to stop is not the output budget running out.
  const other = await failsWith(stubClient(thinking(thought) + done(undefined, { done_reason: 'load' })).client.chat({ ...REQUEST, think: true }), 'INCOMPLETE');
  assert.match(other.message, /^Reply stopped with done_reason "load" after 0 chars\./);
});

test('a reply that is only thinking is EMPTY', async () => {
  await withServer((seen, res) => send(res, thinking('Thinking without an answer') + done(seen)), async ({ client }) => {
    const err = await failsWith(client.chat({ ...REQUEST, think: true }), 'EMPTY');
    assert.match(err.message, /only 26 chars of thinking/);
  });
  // The same with the reasoning written inline, and with nothing at all.
  await failsWith(stubClient(text('<think>still reasoning') + done()).client.chat(REQUEST), 'EMPTY');
  await failsWith(stubClient(text('<think>all of it</think>  \n') + done()).client.chat(REQUEST), 'EMPTY');
  const bare = await failsWith(stubClient(done()).client.chat(REQUEST), 'EMPTY');
  assert.equal(bare.message, 'Ollama returned an empty reply.');
});

test('a tool call is TOOL_CALL, alone or mixed with text', async () => {
  const call = { message: { role: 'assistant', content: '', tool_calls: [{ function: { name: 'read_file', arguments: {} } }] }, done: false };
  await withServer((seen, res) => send(res, record(call) + done(seen)), async ({ client }) => {
    await failsWith(client.chat(REQUEST), 'TOOL_CALL');
  });
  const mixed = await failsWith(stubClient(text('Let me look. ') + record(call) + done()).client.chat(REQUEST), 'TOOL_CALL');
  assert.equal(mixed.partialChars, 13);
});

test('a reply refused halfway stops being read, so the server stops generating it', async () => {
  const call = { message: { role: 'assistant', content: '', tool_calls: [{ function: { name: 'read_file', arguments: {} } }] }, done: false };
  // Ollama stops a generation when its client goes away. The stub notes that
  // moment; it never ends the reply itself.
  let closed = false;
  const neverEnds = (seen, res) => {
    res.on('close', () => { closed = true; });
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
    res.write(text('Let me look. ') + record(call));
  };
  await withServer(neverEnds, async ({ client }) => {
    await failsWith(client.chat(REQUEST), 'TOOL_CALL');
    for (let waited = 0; !closed && waited < 2_000; waited += 20) await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(closed, true, 'the connection was still open 2 s after the reply was refused');
  });
});

test('a prompt the server truncated is TRUNCATED_PROMPT', async () => {
  // The measured case: about 40K tokens sent at num_ctx 32768, 16.4K processed,
  // HTTP 200, done_reason "stop", and a complete-looking report.
  await withServer((seen, res) => send(res, text('# Evaluation\nLooks fine.') + done(seen, { prompt_eval_count: 16400 })), async ({ client }) => {
    const err = await failsWith(client.chat({ ...REQUEST, estimatedPromptTokens: 40000 }), 'TRUNCATED_PROMPT');
    assert.equal(err.message, 'Ollama processed 16400 prompt tokens but ~40000 were sent: the prompt was truncated. Raise OLLAMA_NUM_CTX.');
  });
});

test('the truncation check allows for the estimate, the cache and a server that reports no count', async () => {
  const reply = (final) => stubClient(text('# Evaluation') + record({ done: true, done_reason: 'stop', eval_count: 5, ...final })).client;
  // 85% of the estimate is the line: the estimate is characters / 4, not a tokenizer.
  await reply({ prompt_eval_count: 34000 }).chat({ ...REQUEST, estimatedPromptTokens: 40000 });
  await failsWith(reply({ prompt_eval_count: 33999 }).chat({ ...REQUEST, estimatedPromptTokens: 40000 }), 'TRUNCATED_PROMPT');
  // A repeated prompt is served from the server's cache. The count is still
  // the whole prompt; the cached count is the share of it that was reused.
  const cached = await reply({ prompt_eval_count: 39700, prompt_eval_cached_count: 39500 }).chat({ ...REQUEST, estimatedPromptTokens: 40000 });
  assert.deepEqual(cached.usage, { prompt_tokens: 39700, completion_tokens: 5, total_tokens: 39705, cached_tokens: 39500 });
  // No estimate, no check; no count, no check.
  await reply({ prompt_eval_count: 10 }).chat(REQUEST);
  await reply({}).chat({ ...REQUEST, estimatedPromptTokens: 40000 });
  // A count that is not a count is no count: it neither trips the check nor is reported.
  const invalid = await reply({ prompt_eval_count: -1 }).chat({ ...REQUEST, estimatedPromptTokens: 40000 });
  assert.deepEqual(invalid.usage, { prompt_tokens: 0, completion_tokens: 5, total_tokens: 5, cached_tokens: 0 });
  // Truncated AND cut at the cap: the truncation is the cause, so it is the one named.
  await failsWith(reply({ prompt_eval_count: 16400, done_reason: 'length' }).chat({ ...REQUEST, estimatedPromptTokens: 40000 }), 'TRUNCATED_PROMPT');
});

test('a truncated prompt is refused when it was served from the cache too', async () => {
  // Ollama 0.40.16 reports a warm prompt as 39118 with 39114 of them cached:
  // the cached count is part of the count. Read as an amount on top of it, a
  // truncated prompt doubled and passed. That is every rerun of a command that
  // was just refused, every retry, and every pipeline row after the first.
  const reply = (final) => stubClient(text('# Evaluation') + record({ done: true, done_reason: 'stop', eval_count: 5, ...final })).client;
  const warm = await failsWith(
    reply({ prompt_eval_count: 16400, prompt_eval_cached_count: 16396 }).chat({ ...REQUEST, estimatedPromptTokens: 40000 }),
    'TRUNCATED_PROMPT',
  );
  assert.equal(warm.message, 'Ollama processed 16400 prompt tokens but ~40000 were sent: the prompt was truncated. Raise OLLAMA_NUM_CTX.');
  // The window-sized case: 32768 kept of ~39.6K, all of it cached.
  await failsWith(
    reply({ prompt_eval_count: 32768, prompt_eval_cached_count: 32764 }).chat({ ...REQUEST, estimatedPromptTokens: 39577 }),
    'TRUNCATED_PROMPT',
  );
  // The same counts cold and warm give the same answer, either way.
  for (const cachedCount of [0, 39114]) {
    const whole = await reply({ prompt_eval_count: 39118, prompt_eval_cached_count: cachedCount }).chat({ ...REQUEST, estimatedPromptTokens: 40500 });
    assert.equal(whole.usage.prompt_tokens, 39118);
    assert.equal(whole.usage.cached_tokens, cachedCount);
  }
});

// ---------------------------------------------------------------------------
// HTTP errors
// ---------------------------------------------------------------------------

test('a 404 for a missing model is MODEL_NOT_FOUND; any other error status is HTTP', async () => {
  await withServer((seen, res) => json(res, { error: "model 'stub-model' not found" }, 404), async ({ client, baseUrl, requests }) => {
    const err = await failsWith(client.chat(REQUEST), 'MODEL_NOT_FOUND');
    assert.equal(err.status, 404);
    assert.equal(err.message, `Model "stub-model" was not found by Ollama at ${baseUrl}. Run: ollama pull stub-model`);
    assert.equal(requests.length, 1);
  });
  await withServer((seen, res) => json(res, { error: 'page not found' }, 404), async ({ client }) => {
    const err = await failsWith(client.chat(REQUEST), 'HTTP');
    assert.equal(err.status, 404);
  });
  // Only the first 300 characters of a server's error text are repeated.
  const long = `server-error:${'x'.repeat(300)}TAIL-THAT-MUST-NOT-APPEAR`;
  await withServer((seen, res) => send(res, long, 500), async ({ client, requests }) => {
    const err = await failsWith(client.chat(REQUEST), 'HTTP');
    assert.equal(err.status, 500);
    assert.equal(err.message, `Ollama API error: HTTP 500: ${long.slice(0, 300)}`);
    assert.doesNotMatch(err.message, /TAIL-THAT-MUST-NOT-APPEAR/);
    assert.equal(requests.length, 1, 'a server error is not retried here');
  });
});

test('a 400 "does not support thinking" is retried once, in place, without think', async () => {
  const refusal = { error: '"stub-model" does not support thinking' };
  await withServer((seen, res) => (seen.index === 0 ? json(res, refusal, 400) : send(res, text('Evaluation.') + done(seen))), async ({ client, requests }) => {
    const result = await client.chat({ ...REQUEST, think: false, seed: 3 });
    assert.equal(result.content, 'Evaluation.');
    assert.equal(requests.length, 2);
    assert.equal(requests[0].body.think, false);
    assert.equal('think' in requests[1].body, false);
    const { think, ...first } = requests[0].body;
    assert.deepEqual(requests[1].body, first, 'the retry changes only the rejected field');
    assert.deepEqual(Object.keys(requests[1].body), ['model', 'messages', 'stream', 'options']);
  });
  // A keep-alive the caller did set stays on the second request as well.
  await withServer((seen, res) => (seen.index === 0 ? json(res, refusal, 400) : send(res, text('Evaluation.') + done(seen))), async ({ client, requests }) => {
    await client.chat({ ...REQUEST, think: true, keepAlive: '10m' });
    assert.deepEqual(Object.keys(requests[1].body), ['model', 'messages', 'stream', 'keep_alive', 'options']);
    assert.equal(requests[1].body.keep_alive, '10m');
  });
});

test('the think retry happens once, and only for that refusal', async () => {
  const refusal = { error: '"stub-model" does not support thinking' };
  // Refused again without the field: an HTTP error, not a loop.
  await withServer((seen, res) => json(res, refusal, 400), async ({ client, requests }) => {
    const err = await failsWith(client.chat({ ...REQUEST, think: true }), 'HTTP');
    assert.equal(err.status, 400);
    assert.equal(requests.length, 2);
  });
  // The field was never sent, so there is nothing to take out.
  await withServer((seen, res) => json(res, refusal, 400), async ({ client, requests }) => {
    await failsWith(client.chat(REQUEST), 'HTTP');
    assert.equal(requests.length, 1);
  });
  // A different 400.
  await withServer((seen, res) => json(res, { error: 'invalid options' }, 400), async ({ client, requests }) => {
    await failsWith(client.chat({ ...REQUEST, think: false }), 'HTTP');
    assert.equal(requests.length, 1);
  });
});

test('a redirect is an HTTP error that names where it points, and the other host sees nothing', async () => {
  // The loopback rule and the privacy gate look at the configured address. On
  // a 307 or 308 fetch would repeat the POST, CV included, at whatever host
  // the answer names; on the others it would fetch that host's answer and
  // hand it back as the model's reply.
  const prompt = { ...REQUEST, user: 'CV-MARKER and the job description.' };
  await withServer((seen, res) => send(res, text('from the other host') + done(seen)), async ({ baseUrl: other, requests: atOther }) => {
    for (const status of [301, 302, 303, 307, 308]) {
      const redirect = (seen, res) => {
        // What a proxy in front of an SSO page sends: credentials and a token on the address.
        res.writeHead(status, { Location: `${other.replace('://', '://someone:hunter2@')}/api/chat?token=s3cr3t#frag` });
        res.end('moved');
      };
      await withServer(redirect, async ({ client, baseUrl, requests }) => {
        const err = await failsWith(client.chat(prompt), 'HTTP');
        assert.equal(err.status, status);
        assert.equal(err.location, `${other}/api/chat`);
        assert.equal(
          err.message,
          `Ollama API error: HTTP ${status}: ${baseUrl} answered with a redirect to ${other}/api/chat. `
          + 'Redirects are not followed: only the configured address was checked, and requests to it carry your CV. '
          + 'Set OLLAMA_BASE_URL (or --url) to the Ollama server itself.',
        );
        assert.doesNotMatch(err.message, /hunter2|someone|s3cr3t|frag/, 'the address is shown without what may be a secret');
        assert.equal(requests.length, 1, 'asked once, and not again');

        // The metadata requests carry no prompt, and are not followed either:
        // their answer decides which model the prompt is sent to.
        for (const call of [() => client.version(), () => client.listModels(), () => client.ps(), () => client.showModel('stub-model')]) {
          const meta = await failsWith(call(), 'HTTP');
          assert.equal(meta.status, status);
          assert.equal(meta.location, `${other}/api/chat`);
        }
        // The preload goes to the chat endpoint, and reports only that it did not work.
        assert.equal(await client.preload('stub-model'), false);
      });
    }
    assert.equal(atOther.length, 0, `the other host received ${atOther.length} request(s): ${atOther.map((seen) => `${seen.method} ${seen.url}`).join(', ')}`);
  });
});

test('a redirect that names no usable address is still an HTTP error', async () => {
  // A path alone points back at the same server: shown resolved, and not followed.
  await withServer((seen, res) => { res.writeHead(302, { Location: '/login?next=%2Fapi%2Fchat' }); res.end(); }, async ({ client, baseUrl, requests }) => {
    const err = await failsWith(client.chat(REQUEST), 'HTTP');
    assert.equal(err.location, `${baseUrl}/login`);
    assert.match(err.message, new RegExp(`^Ollama API error: HTTP 302: ${baseUrl} answered with a redirect to ${baseUrl}/login\\. `));
    assert.equal(requests.length, 1);
  });
  // No Location at all, and one that is not an address.
  for (const headers of [{}, { Location: 'http://[' }]) {
    await withServer((seen, res) => { res.writeHead(307, headers); res.end(); }, async ({ client, baseUrl }) => {
      const err = await failsWith(client.chat(REQUEST), 'HTTP');
      assert.equal(err.status, 307);
      assert.equal('location' in err, false);
      assert.match(err.message, new RegExp(`^Ollama API error: HTTP 307: ${baseUrl} answered with a redirect\\. Redirects are not followed`));
    });
  }
  // A stand-in for fetch that has no headers to read.
  const bare = createOllamaClient({ fetchImpl: async () => ({ ok: false, status: 308, text: async () => '' }) });
  const err = await failsWith(bare.version(), 'HTTP');
  assert.equal(err.status, 308);
  // Every request asks fetch not to follow one.
  const { client, calls } = stubClient(text('Evaluation.') + done());
  await client.chat(REQUEST);
  await client.preload('stub-model');
  const meta = fakeFetch(JSON.stringify({ version: '1.2.3', models: [] }));
  const metaClient = createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', fetchImpl: meta.fetchImpl });
  await metaClient.version();
  await metaClient.listModels();
  await metaClient.ps();
  await metaClient.showModel('stub-model');
  for (const call of [...calls, ...meta.calls]) assert.equal(call.init.redirect, 'manual', call.url);
  assert.equal(calls.length + meta.calls.length, 6);
});

// ---------------------------------------------------------------------------
// The stream
// ---------------------------------------------------------------------------

test('a reply split anywhere, even inside a character, is reassembled', async () => {
  const reply = text('Synthetic evaluation: caf\u00e9 \u2615.\n') + text('Second line.') + done();
  const bytes = Buffer.from(reply);
  // One byte at a time: every record and every multi-byte character is split.
  const oneByOne = stubClient(pieces([...bytes].map((byte) => Buffer.from([byte])))).client;
  assert.equal((await oneByOne.chat(REQUEST)).content, 'Synthetic evaluation: caf\u00e9 \u2615.\nSecond line.');
  // Split inside the two-byte e-acute.
  const split = bytes.indexOf(Buffer.from('\u00e9')) + 1;
  const inCharacter = stubClient(pieces([bytes.subarray(0, split), bytes.subarray(split)])).client;
  assert.equal((await inCharacter.chat(REQUEST)).content, 'Synthetic evaluation: caf\u00e9 \u2615.\nSecond line.');
  // CRLF line ends, and a last record with no newline after it.
  const crlf = stubClient(reply.replace(/\n/g, '\r\n').trimEnd()).client;
  assert.equal((await crlf.chat(REQUEST)).content, 'Synthetic evaluation: caf\u00e9 \u2615.\nSecond line.');
  // The whole reply as one JSON object, which is what a server that ignores stream:true sends.
  const single = stubClient(JSON.stringify({ message: { content: 'All at once.' }, done: true, done_reason: 'stop' })).client;
  assert.equal((await single.chat(REQUEST)).content, 'All at once.');
});

test('a reply dripped over a real socket is reassembled', async () => {
  const drip = (seen, res) => {
    const bytes = Buffer.from(text('Synthetic evaluation: caf\u00e9.\n') + text('Second line.') + done(seen));
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
    let offset = 0;
    const next = () => {
      if (offset >= bytes.length) { res.end(); return; }
      res.write(bytes.subarray(offset, offset + 3));
      offset += 3;
      setImmediate(next);
    };
    next();
  };
  await withServer(drip, async ({ client }) => {
    assert.equal((await client.chat(REQUEST)).content, 'Synthetic evaluation: caf\u00e9.\nSecond line.');
  });
});

test('data after the done record is a STREAM error, and so is a stream that never finishes', async () => {
  await withServer((seen, res) => send(res, text('Evaluation.') + done(seen) + text('late')), async ({ client }) => {
    const err = await failsWith(client.chat(REQUEST), 'STREAM');
    assert.match(err.message, /after the completed reply/);
  });
  await failsWith(stubClient(text('Evaluation.') + done() + '{broken}\n').client.chat(REQUEST), 'STREAM');
  // Blank lines after the end are not data.
  assert.equal((await stubClient(text('Evaluation.') + done() + '\n\r\n').client.chat(REQUEST)).content, 'Evaluation.');

  const early = await failsWith(stubClient(text('Evaluation.')).client.chat(REQUEST), 'STREAM');
  assert.match(early.message, /before done:true/);
  assert.equal(early.partialChars, 11);
  await failsWith(stubClient('').client.chat(REQUEST), 'STREAM');
  await failsWith(stubClient(text('Evaluation.') + '{broken}\n' + done()).client.chat(REQUEST), 'STREAM');
  await failsWith(stubClient(text('Evaluation.') + '{"done":').client.chat(REQUEST), 'STREAM');
  await failsWith(stubClient(text('Evaluation.') + 'null\n' + done()).client.chat(REQUEST), 'STREAM');
});

test('an error record in the stream is a STREAM error, with its text capped', async () => {
  const message = `${'provider-error:'.padEnd(300, 'x')}TAIL-THAT-MUST-NOT-APPEAR`;
  await withServer((seen, res) => send(res, text('Partial. ') + record({ error: message })), async ({ client }) => {
    const err = await failsWith(client.chat(REQUEST), 'STREAM');
    assert.ok(err.message.includes(message.slice(0, 300)));
    assert.doesNotMatch(err.message, /TAIL-THAT-MUST-NOT-APPEAR/);
    assert.equal(err.partialChars, 9);
  });
  // An error after the done record still fails the reply.
  await failsWith(stubClient(text('Evaluation.') + done() + record({ error: 'model crashed' })).client.chat(REQUEST), 'STREAM');
});

test('bytes that are not UTF-8 are a STREAM error', async () => {
  const body = pieces([Buffer.from(text('Evaluation.')), Buffer.from([0xc3]), Buffer.from(done())]);
  const err = await failsWith(stubClient(body).client.chat(REQUEST), 'STREAM');
  assert.match(err.message, /not valid UTF-8/);
});

test('one record may not exceed 1 Mi code units; the reply as a whole may', async () => {
  const limit = 1024 * 1024;
  const oversized = JSON.stringify({ padding: 'x'.repeat(limit) });
  const terminated = await failsWith(stubClient(text('Evaluation.') + oversized + '\n' + done()).client.chat(REQUEST), 'STREAM');
  assert.match(terminated.message, /exceeds the 1 Mi code-unit limit/);
  const unterminated = await failsWith(stubClient(text('Evaluation.') + oversized).client.chat(REQUEST), 'STREAM');
  assert.match(unterminated.message, /exceeds the 1 Mi code-unit limit/);

  // Many small records in one transport chunk add up to more than the cap.
  const small = record({ padding: 'x'.repeat(2048) });
  assert.equal((await stubClient(text('Evaluation.') + small.repeat(600) + done()).client.chat(REQUEST)).content, 'Evaluation.');
  // The cap counts decoded characters: this record is over 1 MiB of UTF-8 bytes and under the cap.
  const unicode = record({ padding: '\u00e9'.repeat(limit / 2) });
  assert.equal((await stubClient(text('Evaluation.') + unicode + done()).client.chat(REQUEST)).content, 'Evaluation.');
});

test('a record that never ends is refused when it passes the limit, not when the stream does', async () => {
  // 16 MiB with no newline in it, then silence. Waiting for the end of the
  // stream to measure the record would mean holding all of it until the timeout.
  let pulls = 0;
  const endless = () => new ReadableStream({
    pull(controller) {
      pulls += 1;
      if (pulls > 64) return new Promise(() => {});
      controller.enqueue(Buffer.from('x'.repeat(256 * 1024)));
      return undefined;
    },
  });
  const { fetchImpl } = fakeFetch(endless);
  const client = createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs: 2_000, fetchImpl });
  const err = await failsWith(client.chat(REQUEST), 'STREAM');
  assert.match(err.message, /exceeds the 1 Mi code-unit limit/);
  assert.ok(pulls <= 8, `read ${pulls} chunks of 256 KiB before refusing a record capped at 1 Mi`);
});

test('an exception from the caller\u2019s own callback is not relabelled as a server failure', async () => {
  const { client } = stubClient(text('Evaluation.') + done());
  await assert.rejects(
    client.chat({ ...REQUEST, onToken: () => { throw new RangeError('a bug in the caller'); } }),
    (err) => err instanceof RangeError && err.message === 'a bug in the caller',
  );
});

// ---------------------------------------------------------------------------
// Transport failures
// ---------------------------------------------------------------------------

test('a server that is not there is UNREACHABLE, with the two ways to fix it', async () => {
  const baseUrl = `http://127.0.0.1:${await closedPort()}`;
  const client = createOllamaClient({ baseUrl, timeoutMs: 10_000 });
  const expected = `Ollama not reachable at ${baseUrl}. Start it (ollama serve) or set OLLAMA_BASE_URL.`;
  for (const call of [() => client.chat(REQUEST), () => client.version(), () => client.listModels(), () => client.showModel('stub-model'), () => client.ps()]) {
    const err = await failsWith(call(), 'UNREACHABLE');
    assert.equal(err.message, expected);
    assert.equal(err.midStream, undefined);
    assert.ok(err.cause, 'the transport error is kept as the cause');
  }
});

test('a chat request whose connection fails before the reply starts is sent once more', async () => {
  // What fetch throws when it reuses a kept-alive connection the far end has
  // closed. The runner renders PDFs and merges the tracker with spawnSync, so
  // its event loop cannot see a close that happens meanwhile, and the next
  // request went out on the dead connection: "Ollama not reachable", from a
  // server that was up. Nothing was generated, so asking again costs nothing.
  const lost = () => Object.assign(new TypeError('fetch failed'), {
    cause: Object.assign(new Error('other side closed'), { name: 'SocketError', code: 'UND_ERR_SOCKET' }),
  });
  let sent = 0;
  const bodies = [];
  const lostOnce = async (url, init) => {
    sent += 1;
    bodies.push(init.body);
    if (sent === 1) throw lost();
    return new Response(text('Evaluation.') + done());
  };
  const client = createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs: 10_000, fetchImpl: lostOnce });
  const tokens = [];
  assert.equal((await client.chat({ ...REQUEST, onToken: (piece) => tokens.push(piece) })).content, 'Evaluation.');
  assert.equal(sent, 2);
  assert.equal(bodies[0], bodies[1], 'the same request, byte for byte');
  assert.deepEqual(tokens, ['Evaluation.']);
  // It is not one of chatWithRetry's attempts: no reply was rejected.
  sent = 0;
  const retried = [];
  const result = await chatWithRetry(client, REQUEST, { retries: 0, onRetry: (info) => retried.push(info) });
  assert.equal(result.attempts, 1);
  assert.equal(sent, 2);
  assert.deepEqual(retried, []);

  // Lost twice, the server is not there: two sends, no more, and the usual words.
  let attempts = 0;
  const causes = [];
  const alwaysLost = async () => {
    attempts += 1;
    causes.push(lost());
    throw causes.at(-1);
  };
  const down = createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs: 10_000, fetchImpl: alwaysLost });
  const err = await failsWith(down.chat(REQUEST), 'UNREACHABLE');
  assert.equal(err.message, 'Ollama not reachable at http://127.0.0.1:11434. Start it (ollama serve) or set OLLAMA_BASE_URL.');
  assert.equal(err.midStream, undefined);
  assert.equal(attempts, 2);
  assert.equal(err.cause, causes[1], 'the cause is the failure of the last send');
  // The preload goes to the same endpoint, the same way.
  attempts = 0;
  assert.equal(await down.preload('stub-model'), false);
  assert.equal(attempts, 2);

  // Over a real socket: the server reads the request and closes the
  // connection without a byte of reply.
  const hangUpOnce = (seen, res) => (seen.index === 0 ? res.socket.destroy() : send(res, text('Evaluation.') + done(seen)));
  await withServer(hangUpOnce, async ({ client: real, requests }) => {
    assert.equal((await real.chat(REQUEST)).content, 'Evaluation.');
    assert.equal(requests.length, 2);
    assert.equal(requests[1].raw, requests[0].raw);
  });
  await withServer((seen, res) => res.socket.destroy(), async ({ client: real, baseUrl, requests }) => {
    const refused = await failsWith(real.chat(REQUEST), 'UNREACHABLE');
    assert.equal(refused.message, `Ollama not reachable at ${baseUrl}. Start it (ollama serve) or set OLLAMA_BASE_URL.`);
    assert.equal(requests.length, 2);
  });
});

test('only a lost connection is sent again: a time limit, a refusal and an answer are final', async () => {
  const counting = (failure) => {
    let calls = 0;
    const fetchImpl = async () => {
      calls += 1;
      throw failure();
    };
    return { client: createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs: 10_000, fetchImpl }), calls: () => calls };
  };
  // The request's own time limit: a second send would be over before it began.
  const timedOut = counting(() => new DOMException('The operation was aborted due to timeout', 'TimeoutError'));
  await failsWith(timedOut.client.chat(REQUEST), 'TIMEOUT');
  assert.equal(timedOut.calls(), 1);
  // Five minutes of silence: the server is busy, and a second send waits five more.
  const silent = counting(() => Object.assign(new TypeError('fetch failed'), {
    cause: Object.assign(new Error('Headers Timeout Error'), { code: 'UND_ERR_HEADERS_TIMEOUT' }),
  }));
  await failsWith(silent.client.chat(REQUEST), 'TIMEOUT');
  assert.equal(silent.calls(), 1);
  // The privacy guard, raised from inside fetch by the interceptor.
  const refusal = Object.assign(new Error('Privacy guard blocked an outbound LLM request'), { name: 'PrivacyGuardError', code: 'PRIVACY_GUARD_BLOCKED' });
  const refused = counting(() => refusal);
  await assert.rejects(refused.client.chat(REQUEST), (err) => err === refusal);
  assert.equal(refused.calls(), 1);
  // An answer, whatever it says, is an answer.
  await withServer((seen, res) => send(res, 'bad gateway', 502), async ({ client, requests }) => {
    await failsWith(client.chat(REQUEST), 'HTTP');
    assert.equal(requests.length, 1);
  });
  // A connection that drops after the reply began is not this case: text was
  // generated, and whether to pay for it again is chatWithRetry's decision.
  let streams = 0;
  const dropsMidReply = async () => {
    streams += 1;
    let reads = 0;
    return new Response(new ReadableStream({
      pull(controller) {
        if (reads++ === 0) controller.enqueue(Buffer.from(text('Partial ')));
        else controller.error(new TypeError('terminated'));
      },
    }));
  };
  const dropped = await failsWith(createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs: 10_000, fetchImpl: dropsMidReply }).chat(REQUEST), 'UNREACHABLE');
  assert.equal(dropped.midStream, true);
  assert.equal(streams, 1);
  // The metadata requests are asked once: the runner reads them before any
  // blocking step, and a server that is not there should be reported at once.
  const lost = counting(() => Object.assign(new TypeError('fetch failed'), { cause: Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED' }) }));
  await failsWith(lost.client.version(), 'UNREACHABLE');
  await failsWith(lost.client.listModels(), 'UNREACHABLE');
  assert.equal(lost.calls(), 2);
});

test('a metadata request that gets no answer in time is UNREACHABLE: such a server cannot run a job either', async () => {
  // What fetch rejects with when the request's own AbortSignal.timeout fires.
  const signals = [];
  const fetchImpl = async (url, init) => {
    signals.push(init.signal);
    throw new DOMException('The operation was aborted due to timeout', 'TimeoutError');
  };
  const client = createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs: 900_000, fetchImpl });
  for (const call of [() => client.version(), () => client.listModels(), () => client.ps(), () => client.showModel('stub-model')]) {
    const err = await failsWith(call(), 'UNREACHABLE');
    assert.equal(err.message, 'Ollama not reachable at http://127.0.0.1:11434. Start it (ollama serve) or set OLLAMA_BASE_URL.');
  }
  // Each one carries a limit of its own: the 15-minute chat timeout is not it.
  assert.equal(signals.length, 4);
  for (const signal of signals) assert.ok(signal instanceof AbortSignal);
});

test('a connection that drops mid-reply is UNREACHABLE with midStream, which is what makes it worth a retry', async () => {
  const dropped = (seen, res) => {
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
    res.write(text('Partial '));
    setTimeout(() => res.destroy(), 30);
  };
  await withServer(dropped, async ({ client, baseUrl }) => {
    const err = await failsWith(client.chat(REQUEST), 'UNREACHABLE');
    assert.equal(err.midStream, true);
    assert.equal(err.message, `Connection to Ollama at ${baseUrl} was lost after ${err.partialChars} chars of the reply.`);
  });
  // The same from a stream that fails after its first chunk, where the count is exact.
  let reads = 0;
  const failing = () => new ReadableStream({
    pull(controller) {
      if (reads++ === 0) controller.enqueue(Buffer.from(text('Partial ')));
      else controller.error(new TypeError('terminated'));
    },
  });
  const err = await failsWith(stubClient(failing).client.chat(REQUEST), 'UNREACHABLE');
  assert.equal(err.midStream, true);
  assert.equal(err.partialChars, 8);
});

test('a reply that does not arrive in time is TIMEOUT, before the first byte or after it', async () => {
  // 600 ms rounds to the "1s" in the message.
  await withServer(() => { /* never answers */ }, async ({ client }) => {
    const err = await failsWith(client.chat(REQUEST), 'TIMEOUT');
    assert.equal(err.message, 'Request timed out after 1s. Raise OLLAMA_TIMEOUT_MS.');
  }, { timeoutMs: 600 });
  // The limit covers the stream as well: a reply that starts and then stalls.
  // A longer limit here, so the first chunk has arrived before it runs out
  // even on a machine that is busy with the rest of the suite.
  const stalled = (seen, res) => {
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
    res.write(text('Partial '));
  };
  await withServer(stalled, async ({ client }) => {
    const err = await failsWith(client.chat(REQUEST), 'TIMEOUT');
    assert.equal(err.message, 'Request timed out after 2s. Raise OLLAMA_TIMEOUT_MS.');
    assert.equal(err.partialChars, 8);
  }, { timeoutMs: 1500 });
});

test('fetch giving up on a silent connection is TIMEOUT, without sending the user to a setting that cannot help', async () => {
  // What Node's fetch throws 300 s into a connection that has sent nothing: the
  // request is queued behind another client, or the model is still loading.
  const silent = Object.assign(new TypeError('fetch failed'), {
    cause: Object.assign(new Error('Headers Timeout Error'), { code: 'UND_ERR_HEADERS_TIMEOUT' }),
  });
  const client = createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs: 900_000, fetchImpl: async () => { throw silent; } });
  const err = await failsWith(client.chat(REQUEST), 'TIMEOUT');
  assert.match(err.message, /sent nothing for 5 minutes/);
  assert.doesNotMatch(err.message, /Raise OLLAMA_TIMEOUT_MS/);
  assert.equal(err.cause, silent);
});

// ---------------------------------------------------------------------------
// chatWithRetry
// ---------------------------------------------------------------------------

const hasSummary = (content) => (content.includes('---SCORE_SUMMARY---') ? null : 'no SCORE_SUMMARY block');
const GOOD = '# Evaluation\n---SCORE_SUMMARY---\nSCORE: 3.5\n---END_SUMMARY---';
const PREAMBLE_ONLY = 'I will start by reading the candidate profile and configuration files.';

test('chatWithRetry tries once more when validate gives a reason, and a request with no seed stays without one', async () => {
  const handler = (seen, res) => send(res, text(seen.index === 0 ? PREAMBLE_ONLY : GOOD) + done(seen));
  await withServer(handler, async ({ client, requests }) => {
    const retried = [];
    const result = await chatWithRetry(client, REQUEST, { retries: 1, validate: hasSummary, onRetry: (info) => retried.push(info) });
    assert.equal(result.content, GOOD);
    assert.equal(result.attempts, 2);
    assert.deepEqual(Object.keys(result), ['content', 'thinkingChars', 'usage', 'doneReason', 'attempts']);
    assert.equal(requests.length, 2);
    // No seed asked for: the server draws one for each request, so the retry
    // is a new sample as it is. It used to be given seed 1, which made it the
    // same reply on every run of a command, a bad one included.
    assert.equal('seed' in requests[0].body.options, false);
    assert.equal('seed' in requests[1].body.options, false);
    assert.equal(requests[1].raw, requests[0].raw, 'the retry is the same request');
    assert.deepEqual(retried, [{ attempt: 1, reason: 'no SCORE_SUMMARY block', error: undefined, nextSeed: undefined }]);
    assert.equal('nextSeed' in retried[0], true, 'onRetry keeps its shape');
  });
});

test('chatWithRetry gives up after the last attempt with EMPTY and the reason', async () => {
  await withServer((seen, res) => send(res, text(PREAMBLE_ONLY) + done(seen)), async ({ client, requests }) => {
    const err = await failsWith(chatWithRetry(client, REQUEST, { retries: 1, validate: hasSummary }), 'EMPTY');
    assert.equal(err.message, 'Reply rejected after 2 attempts: no SCORE_SUMMARY block.');
    assert.equal(err.reason, 'no SCORE_SUMMARY block');
    assert.equal(err.attempts, 2);
    assert.equal(err.content, PREAMBLE_ONLY, 'the rejected reply is there for a caller that wants to show it');
    assert.equal(Object.keys(err).includes('content'), false, 'but a logged error does not print a whole report');
    assert.equal(requests.length, 2);
  });
  // retries: 0 means one attempt.
  await withServer((seen, res) => send(res, text(PREAMBLE_ONLY) + done(seen)), async ({ client, requests }) => {
    const err = await failsWith(chatWithRetry(client, REQUEST, { retries: 0, validate: hasSummary }), 'EMPTY');
    assert.equal(err.message, 'Reply rejected after 1 attempt: no SCORE_SUMMARY block.');
    assert.equal(requests.length, 1);
  });
  // One retry is the default.
  await withServer((seen, res) => send(res, text(PREAMBLE_ONLY) + done(seen)), async ({ client, requests }) => {
    await failsWith(chatWithRetry(client, REQUEST, { validate: hasSummary }), 'EMPTY');
    assert.equal(requests.length, 2);
  });
});

test('chatWithRetry moves a seed the caller set on by one per attempt, and sets none of its own', async () => {
  const seeds = (requests) => requests.map((sent) => sent.body.options.seed);
  await withServer((seen, res) => send(res, text(seen.index < 2 ? PREAMBLE_ONLY : GOOD) + done(seen)), async ({ client, requests }) => {
    const told = [];
    const result = await chatWithRetry(client, REQUEST, { retries: 2, validate: hasSummary, onRetry: ({ nextSeed }) => told.push(nextSeed) });
    assert.equal(result.attempts, 3);
    assert.deepEqual(seeds(requests), [undefined, undefined, undefined]);
    assert.deepEqual(told, [undefined, undefined]);
  });
  // A fixed seed gives the same sample every time, so that one has to move.
  await withServer((seen, res) => send(res, text(seen.index < 2 ? PREAMBLE_ONLY : GOOD) + done(seen)), async ({ client, requests }) => {
    const told = [];
    await chatWithRetry(client, { ...REQUEST, seed: 41 }, { retries: 2, validate: hasSummary, onRetry: ({ nextSeed }) => told.push(nextSeed) });
    assert.deepEqual(seeds(requests), [41, 42, 43]);
    assert.deepEqual(told, [42, 43], 'onRetry is told the seed the next request carries');
  });
  await withServer((seen, res) => send(res, text(seen.index === 0 ? PREAMBLE_ONLY : GOOD) + done(seen)), async ({ client, requests }) => {
    await chatWithRetry(client, { ...REQUEST, seed: 0 }, { retries: 1, validate: hasSummary });
    assert.deepEqual(seeds(requests), [0, 1]);
  });
  // A seed that is not one is not a seed the caller set.
  await withServer((seen, res) => send(res, text(seen.index === 0 ? PREAMBLE_ONLY : GOOD) + done(seen)), async ({ client, requests }) => {
    await chatWithRetry(client, { ...REQUEST, seed: '41' }, { retries: 1, validate: hasSummary });
    assert.deepEqual(seeds(requests), [undefined, undefined]);
  });
  // The caller's request object is not the one that gets the new seed.
  const original = { ...REQUEST };
  await withServer((seen, res) => send(res, text(seen.index === 0 ? PREAMBLE_ONLY : GOOD) + done(seen)), async ({ client }) => {
    await chatWithRetry(client, original, { retries: 1, validate: hasSummary });
    assert.deepEqual(original, { ...REQUEST });
  });
  // A reply that passes is returned from the first attempt, untouched.
  await withServer((seen, res) => send(res, text(GOOD) + done(seen)), async ({ client, requests }) => {
    const result = await chatWithRetry(client, REQUEST, { retries: 3, validate: async (content) => hasSummary(content) });
    assert.equal(result.attempts, 1);
    assert.equal(requests.length, 1);
  });
});

test('chatWithRetry retries an empty reply and a connection dropped mid-reply', async () => {
  await withServer((seen, res) => send(res, (seen.index === 0 ? thinking('Only thinking.') : text(GOOD)) + done(seen)), async ({ client, requests }) => {
    const retried = [];
    const result = await chatWithRetry(client, REQUEST, { retries: 1, onRetry: (info) => retried.push(info) });
    assert.equal(result.attempts, 2);
    assert.equal(requests.length, 2);
    assert.equal(retried.length, 1);
    assert.equal(retried[0].error.code, 'EMPTY');
    assert.equal(retried[0].reason, retried[0].error.message);
  });
  // A reply that breaks off after its first chunk, then a complete one.
  let sent = 0;
  const dropThenAnswer = async () => {
    sent += 1;
    if (sent > 1) return new Response(text(GOOD) + done());
    let reads = 0;
    return new Response(new ReadableStream({
      pull(controller) {
        if (reads++ === 0) controller.enqueue(Buffer.from(text('Partial ')));
        else controller.error(new TypeError('terminated'));
      },
    }));
  };
  const flaky = createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs: 10_000, fetchImpl: dropThenAnswer });
  const recovered = await chatWithRetry(flaky, REQUEST, { retries: 1 });
  assert.equal(recovered.content, GOOD);
  assert.equal(recovered.attempts, 2);
  assert.equal(sent, 2);
  // Still empty on the last attempt: the server's own error, with the count.
  await withServer((seen, res) => send(res, thinking('Only thinking.') + done(seen)), async ({ client, requests }) => {
    const err = await failsWith(chatWithRetry(client, REQUEST, { retries: 1 }), 'EMPTY');
    assert.equal(err.attempts, 2);
    assert.equal(requests.length, 2);
  });
});

test('chatWithRetry retries a tool call', async () => {
  // Live, on the default model: 13 tokens of "I'll start by reading the
  // required source files before tailoring." and then a Read call, 0.3 s into
  // a request that offers no tools. As plain text that reply fails validate()
  // and is asked again; as a tool call it used to end the command.
  const call = { message: { role: 'assistant', content: '', tool_calls: [{ function: { name: 'Read', arguments: { file_path: '/data/cv.md' } } }] }, done: false };
  const misfire = (seen) => text('I’ll start by reading the required source files before tailoring.') + record(call) + done(seen);
  await withServer((seen, res) => send(res, seen.index === 0 ? misfire(seen) : text(GOOD) + done(seen)), async ({ client, requests }) => {
    const retried = [];
    const result = await chatWithRetry(client, REQUEST, { retries: 1, validate: hasSummary, onRetry: (info) => retried.push(info) });
    assert.equal(result.content, GOOD);
    assert.equal(result.attempts, 2);
    assert.equal(requests.length, 2);
    assert.equal('tools' in requests[0].body, false, 'the premise: no tools were offered');
    assert.equal(retried.length, 1);
    assert.equal(retried[0].attempt, 1);
    assert.equal(retried[0].error.code, 'TOOL_CALL');
    assert.equal(retried[0].reason, 'Ollama returned a tool call instead of a text reply.');
  });
  // A seed the caller set moves on for it, as for any other retry.
  await withServer((seen, res) => send(res, seen.index === 0 ? misfire(seen) : text(GOOD) + done(seen)), async ({ client, requests }) => {
    await chatWithRetry(client, { ...REQUEST, seed: 7 }, { retries: 1, validate: hasSummary });
    assert.deepEqual(requests.map((sent) => sent.body.options.seed), [7, 8]);
  });
  // Every reply a tool call: the last one is reported, after retries + 1 requests.
  await withServer((seen, res) => send(res, misfire(seen)), async ({ client, requests }) => {
    const err = await failsWith(chatWithRetry(client, REQUEST, { retries: 3, validate: hasSummary }), 'TOOL_CALL');
    assert.equal(err.message, 'Ollama returned a tool call instead of a text reply.');
    assert.equal(err.attempts, 4);
    assert.equal(requests.length, 4);
  });
  // With no retries allowed it is one request, as before.
  await withServer((seen, res) => send(res, misfire(seen)), async ({ client, requests }) => {
    const err = await failsWith(chatWithRetry(client, REQUEST, { retries: 0 }), 'TOOL_CALL');
    assert.equal(err.attempts, 1);
    assert.equal(requests.length, 1);
  });
});

test('chatWithRetry does not retry a reply cut at the cap', async () => {
  await withServer((seen, res) => send(res, text('Partial report') + done(seen, { done_reason: 'length' })), async ({ client, requests }) => {
    await failsWith(chatWithRetry(client, REQUEST, { retries: 3, validate: hasSummary }), 'INCOMPLETE');
    assert.equal(requests.length, 1, 'the same cap would cut the same reply again');
  });
});

test('chatWithRetry does not retry what would fail the same way, or a refusal', async () => {
  // The privacy guard's refusal, as lib/privacy-guard.mjs raises it.
  const refusal = Object.assign(new Error('Privacy guard blocked an outbound LLM request'), { name: 'PrivacyGuardError', code: 'PRIVACY_GUARD_BLOCKED' });
  const failures = [
    ...['INCOMPLETE', 'TRUNCATED_PROMPT', 'MODEL_NOT_FOUND', 'TIMEOUT', 'HTTP', 'STREAM', 'UNREACHABLE']
      .map((code) => new OllamaError(code, `synthetic ${code}`)),
    refusal,
    new TypeError('a bug'),
  ];
  for (const failure of failures) {
    let calls = 0;
    let retried = 0;
    const client = { chat: async () => { calls += 1; throw failure; } };
    await assert.rejects(chatWithRetry(client, REQUEST, { retries: 3, validate: hasSummary, onRetry: () => { retried += 1; } }), (err) => err === failure);
    assert.equal(calls, 1, `${failure.code ?? failure.name} must not be retried`);
    assert.equal(retried, 0);
  }
});

// ---------------------------------------------------------------------------
// The privacy gate
// ---------------------------------------------------------------------------

test('a privacy.yml in the data root blocks the request before anything reaches the server', async () => {
  // A fresh process, so the guard is loaded the way a real run loads it: from
  // the data root, on the first request, with no test hook involved.
  const root = mkdtempSync(join(tmpdir(), 'ollama-client-privacy-'));
  try {
    mkdirSync(join(root, 'config'), { recursive: true });
    writeFileSync(join(root, 'config', 'privacy.yml'), `protected:\n  full_name: "${PROTECTED_NAME}"\n`);
    const driver = join(root, 'driver.mjs');
    writeFileSync(driver, `import { chatWithRetry, createOllamaClient } from ${JSON.stringify(pathToFileURL(CLIENT_SOURCE).href)};
const [baseUrl, user] = process.argv.slice(2);
const client = createOllamaClient({ baseUrl, timeoutMs: 10000 });
let retried = 0;
try {
  const result = await chatWithRetry(client, { model: 'stub-model', system: 'You evaluate job offers.', user }, { retries: 2, onRetry: () => { retried += 1; } });
  console.log(JSON.stringify({ sent: true, content: result.content }));
} catch (err) {
  console.log(JSON.stringify({ sent: false, name: err.name, code: err.code, message: err.message, retried }));
}
`);
    // Only the fixture: no override, proxy, Ollama or privacy setting from the
    // developer's shell reaches the child.
    const env = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
      !/^(CAREER_OPS_|OPENAI_|OLLAMA_|DOTENV_|PRIVACY_|NODE_OPTIONS$|NODE_PATH$|HTTPS?_PROXY$|ALL_PROXY$)/i.test(key)));
    Object.assign(env, { CAREER_OPS_ROOT: root, DOTENV_CONFIG_QUIET: 'true' });
    const run = async (baseUrl, user) => {
      const { stdout } = await exec(process.execPath, [driver, baseUrl, user], { cwd: root, env, timeout: 30_000 });
      return JSON.parse(stdout);
    };

    await withServer((seen, res) => send(res, text('Evaluation.') + done(seen)), async ({ baseUrl, requests }) => {
      const blocked = await run(baseUrl, `CV of ${PROTECTED_NAME}, engineer.`);
      assert.equal(blocked.sent, false);
      assert.equal(blocked.name, 'PrivacyGuardError');
      assert.equal(blocked.code, 'PRIVACY_GUARD_BLOCKED');
      assert.equal(blocked.retried, 0, 'a refusal is not retried');
      assert.equal(blocked.message.includes(PROTECTED_NAME), false, 'the refusal does not repeat the value');
      assert.equal(requests.length, 0, 'nothing reached the server');

      // The control: the same child does reach the stub when there is nothing to block.
      const allowed = await run(baseUrl, 'CV of an engineer.');
      assert.deepEqual(allowed, { sent: true, content: 'Evaluation.' });
      assert.equal(requests.length, 1);
    });
  } finally {
    rmSync(root, { recursive: true, force: true, maxRetries: 10 });
  }
});

test('the gate inspects every chat request, including the retry without think and the preload', async () => {
  const yaml = `protected:\n  full_name: "${PROTECTED_NAME}"\n`;
  await withPrivacyConfig(yaml, async () => {
    await withServer((seen, res) => send(res, text('Evaluation.') + done(seen)), async ({ client, requests }) => {
      const blocked = (promise) => assert.rejects(promise, (err) => err.name === 'PrivacyGuardError' && err.code === 'PRIVACY_GUARD_BLOCKED');
      await blocked(client.chat({ ...REQUEST, user: `CV of ${PROTECTED_NAME}.` }));
      await blocked(client.chat({ ...REQUEST, system: `The candidate is ${PROTECTED_NAME.toUpperCase()}.` }));
      assert.equal(requests.length, 0);
      // The preload swallows its failures, and still sends nothing.
      assert.equal(await client.preload(PROTECTED_NAME), false);
      assert.equal(requests.length, 0);
      // A request with nothing protected in it goes through the same gate.
      assert.equal((await client.chat(REQUEST)).content, 'Evaluation.');
      assert.equal(requests.length, 1);
    });
  });
});

test('a privacy config that cannot be read stops the request too', async () => {
  await withPrivacyConfig('protected: [unclosed\n', async () => {
    await withServer((seen, res) => send(res, text('Evaluation.') + done(seen)), async ({ client, requests }) => {
      await assert.rejects(client.chat(REQUEST), (err) => err.name === 'PrivacyConfigError' && err.code === 'PRIVACY_GUARD_CONFIG');
      assert.equal(requests.length, 0);
    });
  });
});

test('with PRIVACY_REDACT=1 a protected value is replaced by the interceptor, or the request is refused', async () => {
  const yaml = `protected:\n  full_name: "${PROTECTED_NAME}"\n`;
  const withName = { ...REQUEST, user: `CV of ${PROTECTED_NAME}, engineer.` };
  await withPrivacyConfig(yaml, async () => {
    // Redact mode makes the gate let configured text through, on the strength of
    // the interceptor replacing it on the way out. Here nothing would: this
    // request does not go through the interceptor. So it is refused.
    await withServer((seen, res) => send(res, text('Evaluation.') + done(seen)), async ({ client, requests }) => {
      await assert.rejects(client.chat(withName), (err) => err.code === 'PRIVACY_GUARD_BLOCKED');
      assert.equal(requests.length, 0);
    });
    // With the interceptor over fetch, as the runner has it, the request goes
    // out with the value replaced.
    installPiiInterceptor();
    try {
      await withServer((seen, res) => send(res, text('Evaluation.') + done(seen)), async ({ client, requests }) => {
        assert.equal((await client.chat(withName)).content, 'Evaluation.');
        assert.equal(requests.length, 1);
        assert.equal(requests[0].raw.includes(PROTECTED_NAME), false, 'the value did not leave the process');
        assert.match(requests[0].body.messages[1].content, /^CV of \{\{PRIVACY_REDACTED_\d+\}\}, engineer\.$/);
      });
    } finally {
      uninstallPiiInterceptor();
    }
  }, { PRIVACY_REDACT: '1' });
});

test('a refusal raised inside fetch comes out as it went in', async () => {
  // The interceptor raises from inside fetch. Relabelled "not reachable", it
  // would send the user looking for a server problem.
  const refusal = Object.assign(new Error('Privacy guard blocked an outbound LLM request'), { name: 'PrivacyGuardError', code: 'PRIVACY_GUARD_BLOCKED' });
  const client = createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs: 10_000, fetchImpl: async () => { throw refusal; } });
  await assert.rejects(client.chat(REQUEST), (err) => err === refusal);
  await assert.rejects(client.version(), (err) => err === refusal);
  await assert.rejects(client.showModel('stub-model'), (err) => err === refusal);
});

// ---------------------------------------------------------------------------
// Model metadata
// ---------------------------------------------------------------------------

test('listModels returns an empty list when the server names no models', async () => {
  for (const answer of [{}, { models: [] }, { models: null }]) {
    await withServer((seen, res) => json(res, answer), async ({ client, requests }) => {
      assert.deepEqual(await client.listModels(), []);
      assert.equal(requests[0].method, 'GET');
      assert.equal(requests[0].url, '/api/tags');
    });
  }
});

test('listModels maps what the server lists, and marks the embedding models', async () => {
  const tags = { models: [
    { name: 'qwen3.8:27b', model: 'qwen3.8:27b', size: 17741872154, details: { family: 'qwen35', parameter_size: '27.3B', quantization_level: 'Q4_K_M' }, capabilities: ['completion', 'vision', 'tools', 'thinking'] },
    { name: 'bge-m3:latest', size: 1157672605, details: { family: 'bert', parameter_size: '566.70M', quantization_level: 'F16' }, capabilities: ['embedding'] },
    // An older server: no capabilities, so the name and the family decide.
    { name: 'nomic-embed-text:latest', size: 274302450, details: { family: 'nomic-bert', parameter_size: '137M', quantization_level: 'F16' } },
    { name: 'mistral-small:latest', size: 14333921662, details: { family: 'llama', parameter_size: '23.6B', quantization_level: 'Q4_K_M' } },
    // Where the server says what a model can do, that outranks its name.
    { name: 'vector-store:1', size: 10, details: { family: 'llama' }, capabilities: ['embedding'] },
    { name: 'team/embed-chat:1', size: 11, details: { family: 'llama' }, capabilities: ['completion', 'embedding'] },
    // Only `model`, no details at all.
    { model: 'bare:1' },
    { size: 5 },
    null,
  ] };
  await withServer((seen, res) => json(res, tags), async ({ client }) => {
    assert.deepEqual(await client.listModels(), [
      { name: 'qwen3.8:27b', size: 17741872154, family: 'qwen35', parameterSize: '27.3B', quant: 'Q4_K_M', isEmbedding: false },
      { name: 'bge-m3:latest', size: 1157672605, family: 'bert', parameterSize: '566.70M', quant: 'F16', isEmbedding: true },
      { name: 'nomic-embed-text:latest', size: 274302450, family: 'nomic-bert', parameterSize: '137M', quant: 'F16', isEmbedding: true },
      { name: 'mistral-small:latest', size: 14333921662, family: 'llama', parameterSize: '23.6B', quant: 'Q4_K_M', isEmbedding: false },
      { name: 'vector-store:1', size: 10, family: 'llama', parameterSize: '', quant: '', isEmbedding: true },
      { name: 'team/embed-chat:1', size: 11, family: 'llama', parameterSize: '', quant: '', isEmbedding: false },
      { name: 'bare:1', size: 0, family: '', parameterSize: '', quant: '', isEmbedding: false },
    ]);
  });
});

test('showModel reads the capabilities and the model\u2019s own context limit, once per model', async () => {
  const shown = {
    capabilities: ['completion', 'vision', 'tools', 'thinking'],
    details: { family: 'qwen35', parameter_size: '27.3B', quantization_level: 'Q4_K_M' },
    // The limit sits under the architecture's own key.
    model_info: { 'general.architecture': 'qwen35', 'general.parameter_count': 27320697856, 'qwen35.context_length': 262144 },
    parameters: 'temperature 1',
  };
  await withServer((seen, res) => json(res, shown), async ({ client, requests }) => {
    const first = await client.showModel('qwen3.8:27b');
    assert.deepEqual(first, { capabilities: ['completion', 'vision', 'tools', 'thinking'], contextLength: 262144, parameterSize: '27.3B', quant: 'Q4_K_M' });
    assert.equal(requests[0].method, 'POST');
    assert.equal(requests[0].url, '/api/show');
    assert.deepEqual(requests[0].body, { model: 'qwen3.8:27b' });

    // Asked again, and at the same time: still one request.
    first.capabilities.push('edited by a caller');
    const [second, third] = await Promise.all([client.showModel('qwen3.8:27b'), client.showModel('qwen3.8:27b')]);
    assert.deepEqual(second, { capabilities: ['completion', 'vision', 'tools', 'thinking'], contextLength: 262144, parameterSize: '27.3B', quant: 'Q4_K_M' });
    assert.deepEqual(third, second);
    assert.equal(requests.length, 1);
    await client.showModel('another:1');
    assert.equal(requests.length, 2);
  });
});

test('showModel copes with what a server may leave out, and does not remember a failure', async () => {
  // No architecture key: any `*.context_length` will do. No capabilities: none.
  await withServer((seen, res) => json(res, { model_info: { 'llama.context_length': 32768 } }), async ({ client }) => {
    assert.deepEqual(await client.showModel('mistral-small'), { capabilities: [], contextLength: 32768, parameterSize: '', quant: '' });
  });
  // Two such keys: the one the architecture names is the model's.
  const twoLimits = { model_info: { 'encoder.context_length': 512, 'general.architecture': 'gemma4', 'gemma4.context_length': 262144 } };
  await withServer((seen, res) => json(res, twoLimits), async ({ client }) => {
    assert.equal((await client.showModel('gemma4:26b')).contextLength, 262144);
  });
  // The limit under `details`, where the model list has it.
  await withServer((seen, res) => json(res, { details: { context_length: 131072, parameter_size: '32.8B' } }), async ({ client }) => {
    assert.deepEqual(await client.showModel('deepseek-r1:32b'), { capabilities: [], contextLength: 131072, parameterSize: '32.8B', quant: '' });
  });
  for (const answer of [{}, { model_info: { 'llama.context_length': 0 } }, { model_info: null, details: null }]) {
    await withServer((seen, res) => json(res, answer), async ({ client }) => {
      assert.deepEqual(await client.showModel('unknown-shape'), { capabilities: [], contextLength: null, parameterSize: '', quant: '' });
    });
  }
  // Missing, then pulled: the second call asks again.
  const handler = (seen, res) => (seen.index === 0
    ? json(res, { error: "model 'late:1' not found" }, 404)
    : json(res, { capabilities: ['completion'], model_info: { 'general.architecture': 'llama', 'llama.context_length': 8192 } }));
  await withServer(handler, async ({ client, requests }) => {
    const err = await failsWith(client.showModel('late:1'), 'MODEL_NOT_FOUND');
    assert.equal(err.status, 404);
    assert.equal((await client.showModel('late:1')).contextLength, 8192);
    assert.equal(requests.length, 2);
  });
});

test('ps and version map the server\u2019s answers', async () => {
  const handler = (seen, res) => {
    if (seen.url === '/api/version') { json(res, { version: '0.40.16' }); return; }
    json(res, { models: [
      { name: 'qwen3.8:27b', model: 'qwen3.8:27b', size: 17613654587, size_vram: 17613654587, context_length: 65536 },
      { name: 'deepseek-r1:32b', size: 37522883214, size_vram: 31244744130, context_length: 65536 },
      { model: 'cpu-only:1', size: 1000 },
    ] });
  };
  await withServer(handler, async ({ client, requests }) => {
    assert.equal(await client.version(), '0.40.16');
    assert.deepEqual(await client.ps(), [
      { name: 'qwen3.8:27b', size: 17613654587, sizeVram: 17613654587, contextLength: 65536 },
      { name: 'deepseek-r1:32b', size: 37522883214, sizeVram: 31244744130, contextLength: 65536 },
      { name: 'cpu-only:1', size: 1000, sizeVram: 0, contextLength: null },
    ]);
    assert.deepEqual(requests.map((seen) => `${seen.method} ${seen.url}`), ['GET /api/version', 'GET /api/ps']);
  });
  await withServer((seen, res) => json(res, {}), async ({ client }) => {
    assert.equal(await client.version(), 'unknown');
    assert.deepEqual(await client.ps(), []);
  });
});

test('an answer that is not Ollama\u2019s is an HTTP error, not a crash', async () => {
  await withServer((seen, res) => { res.writeHead(200, { 'Content-Type': 'text/html' }); res.end('<html>some other server</html>'); }, async ({ client }) => {
    const err = await failsWith(client.version(), 'HTTP');
    assert.match(err.message, /not JSON/);
    await failsWith(client.listModels(), 'HTTP');
  });
  await withServer((seen, res) => send(res, 'bad gateway', 502), async ({ client }) => {
    const err = await failsWith(client.listModels(), 'HTTP');
    assert.equal(err.status, 502);
    assert.equal(err.message, 'Ollama API error: HTTP 502: bad gateway');
  });
});

test('preload asks the server to load the model at the window the jobs will use, and never throws', async () => {
  await withServer((seen, res) => json(res, { model: 'stub-model', done: true, done_reason: 'load' }), async ({ client, requests }) => {
    assert.equal(await client.preload('stub-model', 98304, '10m'), true);
    assert.equal(requests[0].method, 'POST');
    assert.equal(requests[0].url, '/api/chat');
    assert.deepEqual(requests[0].body, { model: 'stub-model', messages: [], keep_alive: '10m', options: { num_ctx: 98304 } });
    // When the caller names neither: the default window, and no keep-alive.
    // The server's own setting then decides how long the model stays loaded.
    assert.equal(await client.preload('stub-model'), true);
    assert.deepEqual(requests[1].body, { model: 'stub-model', messages: [], options: { num_ctx: 65536 } });
    // What the runner passes when OLLAMA_KEEP_ALIVE is unset, and when it is 0.
    assert.equal(await client.preload('stub-model', 98304, undefined), true);
    assert.deepEqual(requests[2].body, { model: 'stub-model', messages: [], options: { num_ctx: 98304 } });
    assert.equal(await client.preload('stub-model', 98304, 0), true);
    assert.deepEqual(requests[3].body, { model: 'stub-model', messages: [], keep_alive: 0, options: { num_ctx: 98304 } });
  });
  await withServer((seen, res) => json(res, { error: "model 'stub-model' not found" }, 404), async ({ client }) => {
    assert.equal(await client.preload('stub-model'), false);
  });
  const down = createOllamaClient({ baseUrl: `http://127.0.0.1:${await closedPort()}`, timeoutMs: 10_000 });
  assert.equal(await down.preload('stub-model'), false);
});

// ---------------------------------------------------------------------------
// The client itself
// ---------------------------------------------------------------------------

test('the client uses the fetch that is current at each request', async () => {
  // A script that installs the privacy interceptor after creating the client
  // must still send through it.
  const client = createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs: 10_000 });
  const original = globalThis.fetch;
  const seen = [];
  globalThis.fetch = async (url) => {
    seen.push(String(url));
    return new Response(JSON.stringify({ version: 'replaced-later' }), { status: 200 });
  };
  try {
    assert.equal(await client.version(), 'replaced-later');
  } finally {
    globalThis.fetch = original;
  }
  assert.deepEqual(seen, ['http://127.0.0.1:11434/api/version']);
});

test('the client accepts a base URL with a trailing slash and refuses an unusable timeout', async () => {
  const { fetchImpl, calls } = fakeFetch(JSON.stringify({ version: '1.2.3' }));
  await createOllamaClient({ baseUrl: 'http://127.0.0.1:11434/', timeoutMs: 1000, fetchImpl }).version();
  assert.equal(calls[0].url, 'http://127.0.0.1:11434/api/version');
  for (const timeoutMs of [0, -5, 1.5, NaN, 2147483648, '900000']) {
    assert.throws(() => createOllamaClient({ baseUrl: 'http://127.0.0.1:11434', timeoutMs }), (err) => err.code === 'OLLAMA_CONFIG', String(timeoutMs));
  }
  // Nothing given: the defaults.
  assert.equal(typeof createOllamaClient().chat, 'function');
  assert.deepEqual(Object.keys(createOllamaClient()), ['version', 'listModels', 'showModel', 'ps', 'chat', 'preload']);
});

test('a stand-in response with only json() or only text() is read', async () => {
  // What a test double for fetch tends to return in place of a Response.
  const jsonOnly = createOllamaClient({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({ version: '9.9.9' }) }) });
  assert.equal(await jsonOnly.version(), '9.9.9');
  const textOnly = createOllamaClient({ fetchImpl: async () => ({ ok: true, status: 200, text: async () => text('Evaluation.') + done() }) });
  assert.equal((await textOnly.chat(REQUEST)).content, 'Evaluation.');
});

test('OllamaError carries its code however it is constructed', () => {
  const canonical = new OllamaError('HTTP', 'Ollama API error: HTTP 500', { status: 500, partialChars: 3 });
  assert.equal(canonical.name, 'OllamaError');
  assert.equal(canonical.code, 'HTTP');
  assert.equal(canonical.message, 'Ollama API error: HTTP 500');
  assert.equal(canonical.status, 500);
  assert.equal(canonical.partialChars, 3);
  assert.ok(canonical instanceof Error);
  // The forms a caller's test double may use.
  const viaProps = new OllamaError('down', { code: 'UNREACHABLE' });
  assert.equal(viaProps.code, 'UNREACHABLE');
  assert.equal(viaProps.message, 'down');
  const assigned = Object.assign(new OllamaError('down'), { code: 'UNREACHABLE' });
  assert.equal(assigned.code, 'UNREACHABLE');
  assert.equal(assigned.message, 'down');
  assert.equal(new OllamaError('EMPTY').code, 'EMPTY');
  const cause = new Error('socket');
  assert.equal(new OllamaError('UNREACHABLE', 'down', { cause }).cause, cause);
});

// ---------------------------------------------------------------------------
// What the module must not contain
// ---------------------------------------------------------------------------

test('the client never ends the process and has no second way out to a model', () => {
  const source = readFileSync(CLIENT_SOURCE, 'utf8');
  // On Windows, Node 24.x can abort when told to exit after a request
  // (nodejs/node#56645); a library throws and lets its caller set the exit code.
  assert.doesNotMatch(source, /process\.exit\s*\(/);
  // The ambient guard does not recognise these two paths on a host or port
  // other than the default, so nothing would inspect a payload sent to them.
  assert.equal(source.includes('/api/generate'), false);
  assert.equal(source.includes('/api/embed'), false);
  // The OpenAI-compatible route ignores num_ctx and think.
  assert.equal(source.includes('/v1/chat'), false);
});

test('importing the client reaches no npm package, so doctor can load it without node_modules', () => {
  // doctor.mjs loads lib/ollama-doctor.mjs at its Ollama step, and that imports
  // this module. doctor is the script that says "run npm install"
  // (tests/doctor-missing-deps.test.mjs), and it reads a module that cannot be
  // found at that step as a copy without the backend: a package reached from
  // here would turn an incomplete install into that answer. So the privacy
  // gate, which does reach js-yaml, is imported when the first chat needs it.
  const staticImports = (file) => [...readFileSync(file, 'utf8')
    .matchAll(/^\s*(?:import|export)\s[^'"]*?from\s*['"]([^'"]+)['"]|^\s*import\s*['"]([^'"]+)['"]/gm)]
    .map((match) => match[1] ?? match[2]);
  const reached = new Set();
  const packages = [];
  const visit = (file) => {
    if (reached.has(file)) return;
    reached.add(file);
    for (const specifier of staticImports(file)) {
      if (specifier.startsWith('.')) visit(resolve(dirname(file), specifier));
      else if (!isBuiltin(specifier)) packages.push(`${specifier} (from ${file})`);
    }
  };
  visit(CLIENT_SOURCE);
  assert.deepEqual(packages, []);
  assert.deepEqual([...reached].map((file) => file.slice(ROOT.length).replace(/\\/g, '/')).sort(), ['lib/ollama-client.mjs', 'lib/ollama-config.mjs']);
  assert.match(readFileSync(CLIENT_SOURCE, 'utf8'), /import\('\.\/llm-outbound\.mjs'\)/, 'the gate is still imported, on demand');
});

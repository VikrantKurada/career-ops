// tests/helpers/ollama-runner-eval-harness.mjs — what the two suites of the
// local Ollama runner's first steps share: tests/ollama-runner.test.mjs
// (`evaluate` and the functions an evaluation is made of) and
// tests/ollama-runner-pipeline.test.mjs (the command line, `models`, `doctor`,
// `scan` and `pipeline`).
//
// They were one file. It started 136 runner processes and needed 27 to 28
// seconds on one core, where test-all.mjs gives a suite it runs by itself 30.
// Split in two, each half needs its own copy of everything below, so it lives
// here: the fixtures, a stub that answers like Ollama, and `cli()`, which
// starts the runner as a child process.
//
// tests/helpers/ollama-runner-harness.mjs is the same for the two suites of
// the later steps, and stays a file of its own, because its fixtures differ: a
// data root there always has a CV and a tracker and every child gets a
// stand-in for Playwright, where the tests here need a data root without a CV,
// a child that reads standard input and the checks of `doctor` replaced.
//
// Importing this module prepares the process the way a suite needs it: it
// creates the sandbox every data root lives in (removed when the suite ends),
// and it clears the environment of anything that could point a call made in
// this process at the developer's own data.
import { after } from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { NO_TOOLS_NOTE } from '../../ollama-runner.mjs';
import { localToday } from '../../lib/local-today.mjs';
import { describeSpawn } from './describe-spawn.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const RUNNER = join(ROOT, 'ollama-runner.mjs');
// As in tests/helpers/ollama-runner-harness.mjs: the children read the day
// again when they save, so a midnight closer than the suite is long is waited
// out, the sandbox has a space in its name, and it is removed with rm().
const untilMidnight = new Date().setHours(24, 0, 0, 0) - Date.now();
if (untilMidnight < 15_000) await new Promise((done) => { setTimeout(done, untilMidnight + 50); });
const TODAY = localToday();

// Every data root of this suite lives under one folder, removed at the end.
const SANDBOX = mkdtempSync(join(tmpdir(), 'ollama runner test-'));
after(() => rm(SANDBOX, { recursive: true, force: true, maxRetries: 10 }));

// The few calls made in this process (evaluateJob with a stand-in client)
// resolve paths like any script does. Nothing here may follow a variable the
// developer's shell happens to export to a real tracker or a real profile.
// Every inherited CAREER_OPS_* variable goes, not a list of the ones known
// today: the next one upstream adds would otherwise point this process at the
// developer's real files.
for (const name of Object.keys(process.env)) {
  if (/^(CAREER_OPS_|PRIVACY_)/i.test(name)) delete process.env[name];
}
process.env.CAREER_OPS_ROOT = mkdtempSync(join(SANDBOX, 'process-root-'));
process.env.CAREER_OPS_DATA_DIR = '';
// The report-number allocator also reads batch/batch-state.tsv of the
// CHECKOUT, wherever the data root is. One failed batch row there (the file is
// what batch-runner.sh leaves behind) moved every report number of this suite:
// the first report was 089 where 001 is expected. Pinned to a file that does
// not exist, here and for every child.
const NO_BATCH_STATE = join(SANDBOX, 'no-batch-state.tsv');
process.env.CAREER_OPS_BATCH_STATE = NO_BATCH_STATE;

// ---------------------------------------------------------------------------
// Fixtures. Everything is fictional.
// ---------------------------------------------------------------------------

// A posting that states its pay, so the evidence pass has nothing to add, and
// that holds a "## " line of its own, as pasted postings do.
const JD = [
  'Senior AI Engineer at Northwind Robotics',
  '',
  'Build the retrieval and agent layer of a warehouse assistant.',
  '## Requirements',
  '- 5+ years shipping production software',
  'Salary: USD 150,000 to 180,000 per year.',
  '',
].join('\n');

// What the default model wrote ahead of its report in the probe: working notes
// with bold labels, then a rule, then the title.
const PREAMBLE = [
  "I'll evaluate this posting against the candidate's profile. Let me work through the gates.",
  '',
  '**Gates check:**',
  '- **Blacklist:** no entry, no gate.',
  '',
  'Let me write the full evaluation.',
  '',
  '---',
  '',
].join('\n');

function reportBody({ company = 'Northwind Robotics', role = 'Senior AI Engineer', score = '4.2' } = {}) {
  return [
    `# Evaluation: ${company} — ${role}`,
    '',
    '**Date:** 2026-07-08',
    `**Score:** ${score}/5`,
    '**Decision:** Apply',
    '',
    '---',
    '',
    '## A) Role Summary',
    '',
    `EVALUATION_BODY_MARKER. ${'The candidate ships production ML systems and meets the core requirements. '.repeat(6)}`,
    '',
    '## G) Posting Legitimacy',
    '',
    'High Confidence: a specific, well written posting.',
    '',
  ].join('\n');
}

function scoreSummary({ company = 'Northwind Robotics', role = 'Senior AI Engineer', score = '4.2' } = {}) {
  return [
    '---SCORE_SUMMARY---',
    `COMPANY: ${company}`,
    `ROLE: ${role}`,
    `SCORE: ${score}`,
    'ARCHETYPE: AI Platform',
    'LEGITIMACY: High Confidence',
    '---END_SUMMARY---',
  ].join('\n');
}

const goodReply = (fields) => `${reportBody(fields)}\n${scoreSummary(fields)}`;

// What follows the posting in every evaluation request: the no-tools sentence,
// then the grounding rule, then the rule against copying the posting. All
// three were measured on the real model in exactly these words, so the last
// two are spelled out here and not imported.
const USER_TAIL = `\n\n${NO_TOOLS_NOTE}\n\n`
  + 'Every fact, number and result you state about the candidate must come from the CV, the profile or the article digest '
  + 'in this prompt. Do not invent one, not even as an example. In Block F, build each story only from what those files '
  + 'say; where they give no result or detail, write "(candidate to add)".'
  + '\n\nDo not write the "## Job Description (archived verbatim)" section: the script adds the posting to the saved '
  + 'report itself. After the last block of the report, go straight to "## Machine Summary" and then the SCORE_SUMMARY block.';
const userMessage = (posting, url) => `JOB DESCRIPTION TO EVALUATE:\n\n${url ? `URL: ${url}\n\n` : ''}${posting}${USER_TAIL}`;

// A reply no source of a verdict can be read from: no title, no header
// fields, no Machine Summary, no summary block. Long enough to be a report.
const NO_VERDICT_REPLY = `## A) Role Summary\n\n${'The role is a good match for the candidate in most respects. '.repeat(8)}`;

// Context sizes and capabilities as the installed models report them. The
// embedding model is here because `ollama list` shows it beside the chat models.
const MODELS = {
  'qwen3.8:27b': { caps: ['completion', 'vision', 'tools', 'thinking'], ctx: 262144, params: '27.3B', family: 'qwen35' },
  'gemma4:26b-a4b-it-q4_K_M': { caps: ['completion', 'vision', 'tools', 'thinking'], ctx: 262144, params: '25.8B', family: 'gemma4' },
  'qwen3-coder:30b': { caps: ['completion', 'tools'], ctx: 262144, params: '30.5B', family: 'qwen3' },
  'mistral-small:latest': { caps: ['completion', 'tools'], ctx: 32768, params: '23.6B', family: 'mistral' },
  'nomic-embed-text:latest': { caps: ['embedding'], ctx: 2048, params: '137M', family: 'nomic-bert' },
};
const CHAT_MODEL_NAMES = Object.keys(MODELS).filter((name) => !MODELS[name].caps.includes('embedding'));

const TRACKER = [
  '# Applications Tracker',
  '',
  '| # | Date | Company | Role | Score | Status | PDF | Report | Notes |',
  '|---|------|---------|------|-------|--------|-----|--------|-------|',
  '',
].join('\n');

/**
 * A data root with the example CV and profile, the way the smoke test in
 * docs/OLLAMA.md builds one. modes/_profile.md is the template with one line
 * of the candidate's own, so it counts as edited; `profileMode: 'template'`
 * leaves it as shipped.
 */
function makeRoot({ cv = true, profile = true, profileMode = true } = {}) {
  const root = mkdtempSync(join(SANDBOX, 'root-'));
  if (cv === 'empty') writeFileSync(join(root, 'cv.md'), '\n  \n');
  else if (cv) cpSync(join(ROOT, 'examples', 'cv-example.md'), join(root, 'cv.md'));
  if (profile) {
    mkdirSync(join(root, 'config'), { recursive: true });
    cpSync(join(ROOT, 'config', 'profile.example.yml'), join(root, 'config', 'profile.yml'));
  }
  if (profileMode) {
    mkdirSync(join(root, 'modes'), { recursive: true });
    const template = readFileSync(join(ROOT, 'modes', '_profile.template.md'), 'utf-8');
    writeFileSync(join(root, 'modes', '_profile.md'), profileMode === 'template' ? template : `${template}\nTarget: staff ML platform roles.\n`);
  }
  writeFileSync(join(root, 'jd.txt'), JD);
  return root;
}

const listing = (dir) => (existsSync(dir) ? readdirSync(dir).sort() : []);
const read = (...parts) => readFileSync(join(...parts), 'utf-8');

// ---------------------------------------------------------------------------
// Stub server
// ---------------------------------------------------------------------------

const isGeneration = (seen) => seen.url === '/api/chat' && Array.isArray(seen.body?.messages) && seen.body.messages.length > 0;

/**
 * Run `run` against a server that answers like Ollama.
 *
 * `reply(seen, n)` decides what the n-th generation request gets: the reply
 * text, or `{ content, thinking, doneReason, promptEvalCount, status, error }`.
 * The final record reports the prompt as a quarter of the request's characters,
 * which is how the runner estimates it, so the truncation check stays quiet
 * unless a test lowers the count.
 */
async function withStub({ models = MODELS, reply = () => goodReply() } = {}, run) {
  const requests = [];
  let generations = 0;
  const json = (res, object, status = 200) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(object));
  };
  const server = createServer((req, res) => {
    let raw = '';
    req.setEncoding('utf8');
    req.on('data', (chunk) => { raw += chunk; });
    req.on('end', () => {
      let body = null;
      try { body = raw ? JSON.parse(raw) : null; } catch { /* left null: not JSON */ }
      const seen = { method: req.method, url: req.url, raw, body };
      requests.push(seen);

      if (req.url === '/api/version') return json(res, { version: '0.0.0-stub' });
      if (req.url === '/api/ps') return json(res, { models: [] });
      if (req.url === '/api/tags') {
        return json(res, {
          models: Object.entries(models).map(([name, model]) => ({
            name, model: name, size: 17_500_000_000,
            details: { family: model.family, parameter_size: model.params, quantization_level: 'Q4_K_M' },
          })),
        });
      }
      if (req.url === '/api/show') {
        const model = models[body?.model];
        if (!model) return json(res, { error: `model '${body?.model}' not found` }, 404);
        return json(res, {
          capabilities: model.caps,
          model_info: { 'general.architecture': 'stub', 'stub.context_length': model.ctx },
          details: { parameter_size: model.params, quantization_level: 'Q4_K_M' },
        });
      }
      if (req.url !== '/api/chat') return json(res, { error: 'not found' }, 404);
      // An empty message list is the request that loads a model.
      if (!isGeneration(seen)) return json(res, { model: body?.model, done: true, done_reason: 'load' });

      const spec = reply(seen, generations);
      generations += 1;
      const answer = typeof spec === 'string' ? { content: spec } : spec;
      if (answer.status) return json(res, { error: answer.error ?? 'stub failure' }, answer.status);

      res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
      const record = (object) => `${JSON.stringify({ model: body.model, ...object })}\n`;
      const text = answer.content ?? '';
      let stream = '';
      if (answer.thinking) stream += record({ message: { role: 'assistant', content: '', thinking: answer.thinking }, done: false });
      // In pieces, the way a stream arrives.
      const piece = Math.max(1, Math.ceil(text.length / 3));
      for (let at = 0; at < text.length; at += piece) {
        stream += record({ message: { role: 'assistant', content: text.slice(at, at + piece) }, done: false });
      }
      stream += record({
        message: { role: 'assistant', content: '' },
        done: true,
        done_reason: answer.doneReason ?? 'stop',
        prompt_eval_count: answer.promptEvalCount ?? Math.ceil(raw.length / 4),
        eval_count: Math.ceil(text.length / 4),
      });
      return res.end(stream);
    });
  });
  await new Promise((done, fail) => {
    server.once('error', fail);
    server.listen(0, '127.0.0.1', done);
  });
  const stub = {
    url: `http://127.0.0.1:${server.address().port}`,
    requests,
    /** Requests that asked the model to write something. */
    generations: () => requests.filter(isGeneration),
    paths: () => requests.map((seen) => `${seen.method} ${seen.url}`),
  };
  try {
    return await run(stub);
  } finally {
    server.closeAllConnections();
    await new Promise((done) => server.close(done));
  }
}

// ---------------------------------------------------------------------------
// Starting the runner
// ---------------------------------------------------------------------------

/** The environment without anything that could point a child at real data or a real server. */
function cleanEnv() {
  return Object.fromEntries(Object.entries(process.env).filter(([name]) => (
    // npm_*: a suite started through `npm test` must not look like a runner
    // that npm started with options of its own.
    !/^(CAREER_OPS_|OPENAI_|OLLAMA_|PRIVACY_|DOTENV_|npm_config_|npm_lifecycle_|NODE_OPTIONS$|NODE_PATH$|HTTPS?_PROXY$|ALL_PROXY$)/i.test(name)
  )));
}

// main() with one of its two seams set, for what a test cannot reach otherwise.
//   pages   maps a URL to its text and replaces the page reader; a URL that
//           is not in it reads like a posting that was taken down.
//   checks  is what the doctor checks return. What they were called with is
//           written to checksSeenAt, so a test can read it back.
const SEAM_DRIVER = `
import { writeFileSync } from 'node:fs';
import { main } from ${JSON.stringify(pathToFileURL(RUNNER).href)};
const { pages, checks, checksSeenAt } = JSON.parse(process.env.RUNNER_TEST_SEAMS);
const seams = {};
if (pages) {
  seams.fetchPage = async (url) => {
    if (!Object.hasOwn(pages, url)) throw new Error('No job description to evaluate at ' + url + ' (http_404)');
    return pages[url];
  };
}
if (checks) {
  seams.checkOllama = async (options) => {
    const settings = Object.fromEntries(Object.entries(options.env).filter(([name]) => name.startsWith('OLLAMA_')));
    writeFileSync(checksSeenAt, JSON.stringify({ ...options, env: settings }));
    return checks;
  };
}
process.exitCode = await main(JSON.parse(process.env.RUNNER_TEST_ARGV), seams);
`;

/**
 * Run the runner in a child process, in `root`, and wait for it.
 *
 * `status` is a number only when the child exited on its own, so an assertion
 * on it can never be satisfied by a child that was killed at the timeout.
 *
 * @param {string[]} args
 * @param {{root: string, cwd?: string, env?: object, input?: string, stdinOpen?: boolean, pages?: Record<string, string>,
 *   checks?: object[]}} options
 *   `cwd` defaults to the data root, so no test reads a .env from wherever the suite was started.
 *   `stdinOpen` leaves standard input open for as long as the child runs, like a terminal nobody types into.
 */
function cli(args, { root, cwd = root, env = {}, input = '', stdinOpen = false, pages, checks } = {}) {
  // DOTENV_CONFIG_QUIET is deliberately not set, as other suites set it for
  // their children: the runner has to keep dotenv quiet by itself.
  const childEnv = {
    ...cleanEnv(),
    CAREER_OPS_ROOT: root,
    CAREER_OPS_BATCH_STATE: NO_BATCH_STATE,
    // Nothing in this suite may start a browser.
    CAREER_OPS_NO_PLAYWRIGHT: '1',
    ...env,
  };
  const seamed = Boolean(pages || checks);
  const argv = seamed ? ['--input-type=module', '-e', SEAM_DRIVER] : [RUNNER, ...args];
  if (seamed) {
    childEnv.RUNNER_TEST_ARGV = JSON.stringify(args);
    childEnv.RUNNER_TEST_SEAMS = JSON.stringify({ pages, checks, checksSeenAt: join(root, 'checks-seen.json') });
  }
  return new Promise((resolve) => {
    const child = execFile(process.execPath, argv, {
      cwd, env: childEnv, timeout: 25_000, maxBuffer: 32 * 1024 * 1024, windowsHide: true,
    }, (error, stdout, stderr) => {
      const exited = !error || typeof error.code === 'number';
      child.stdin.destroy();
      resolve({
        status: !error ? 0 : exited ? error.code : null,
        signal: error?.signal ?? null,
        error: exited ? undefined : error,
        stdout,
        stderr,
      });
    });
    if (!stdinOpen) child.stdin.end(input);
  });
}

/** Assert how a run ended, with both streams in the message when it ended another way. */
function assertExit(result, status, label) {
  assert.equal(result.status, status, describeSpawn(result, label));
}

export {
  ROOT, RUNNER, TODAY, SANDBOX, NO_BATCH_STATE, JD, PREAMBLE, reportBody, scoreSummary, goodReply, userMessage,
  NO_VERDICT_REPLY, MODELS, CHAT_MODEL_NAMES, TRACKER, makeRoot, listing, read, withStub, cleanEnv, cli, assertExit,
};

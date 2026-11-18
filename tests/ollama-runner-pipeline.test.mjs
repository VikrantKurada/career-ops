// tests/ollama-runner-pipeline.test.mjs — the local Ollama runner
// (ollama-runner.mjs): its command line, `models`, `doctor`, `scan`, and
// `pipeline`, which evaluates the postings of an inbox one after another.
//
// tests/ollama-runner.test.mjs covers `evaluate` and the functions an
// evaluation is made of. The two were one suite, use
// tests/helpers/ollama-runner-eval-harness.mjs and work the same way: the
// runner is started as a child process, `node ollama-runner.mjs …`, against a
// stub that answers like Ollama, and what is checked is what the run leaves
// behind. That is the requests that reached the server (and the ones that must
// not), the reports, the tracker, the pipeline file and the exit code.
// runPipeline, parseArgs and parsePipeline are also called in this process.
//
// The failures that matter here are the quiet ones: an option that is dropped
// without a word, so that `--limit 5` runs the whole inbox and `--no-save`
// saves; a pipeline row ticked for a posting nobody evaluated; a closed posting
// tried again on every run; a run that exits 0 and fills no tracker; a report
// saved as Skip behind a summary row that shows only its score.
//
// No Ollama is needed and none is contacted. A child that asks a server for
// anything gets `--url` pointing at the stub. Every child is given a data root
// made for its test under the OS temp dir, and an environment without the
// CAREER_OPS_*, OLLAMA_* and PRIVACY_* variables of the shell that started the
// suite. Job pages are never fetched either: the page reader refuses the
// loopback addresses a test could serve from, so the tests that need a posting
// hand main() a reader of their own. The `doctor` tests replace the checks the
// same way: those live in lib/ollama-doctor.mjs and have their own suite, and
// this one must pass whatever state that file is in.
//
// Run:  node --test tests/ollama-runner-pipeline.test.mjs
import { describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseArgs, parsePipeline, runPipeline } from '../ollama-runner.mjs';
import { TYPICAL_PROMPT_TOKENS } from '../lib/ollama-config.mjs';
import { TSV_ADDITION_HEADER } from '../tracker-parse.mjs';
import {
  RUNNER, TODAY, SANDBOX, NO_BATCH_STATE, goodReply, CHAT_MODEL_NAMES, TRACKER, makeRoot, listing, read, withStub,
  cleanEnv, cli, assertExit,
} from './helpers/ollama-runner-eval-harness.mjs';

// ---------------------------------------------------------------------------
// The command line
// ---------------------------------------------------------------------------

describe('command line', { concurrency: 6 }, () => {
  test('--help, -h, help and no arguments print the usage and do nothing else', async () => {
    const root = makeRoot();
    await withStub({}, async (stub) => {
      const runs = await Promise.all([['--help'], ['-h'], ['help'], [], ['evaluate', '--url', stub.url, '--file', join(root, 'jd.txt'), '--help']]
        .map((args) => cli(args, { root })));
      for (const run of runs) {
        assertExit(run, 0);
        assert.match(run.stdout, /^Usage: node ollama-runner\.mjs <command> \[options\]/);
        assert.equal(run.stdout, runs[0].stdout);
      }
      assert.deepEqual(stub.requests, []);
      assert.deepEqual(listing(join(root, 'reports')), []);
    });
  });

  test('the usage lists only options the parser accepts', async () => {
    const run = await cli(['--help'], { root: makeRoot() });
    const documented = [...run.stdout.matchAll(/^\s+(--[a-z-]+)/gm)].map((match) => match[1]);
    assert.ok(documented.length >= 14, `only ${documented.length} options found in the usage`);
    for (const flag of documented) {
      // An option that takes no value reports that; neither answer is "unknown".
      const known = ['evaluate', 'pipeline', 'tailor', 'pdf', 'apply', 'run']
        .some((command) => !/Unknown option/.test(parseArgs([command, flag, 'x']).error ?? ''));
      assert.ok(known, `${flag} is in the usage and the parser does not know it`);
    }
  });

  test('a misspelled option stops the run instead of being ignored', async () => {
    const root = makeRoot();
    await withStub({}, async (stub) => {
      // The old script skipped unknown flags, so this ran the default model without a word.
      const run = await cli(['evaluate', '--url', stub.url, '--modle', 'qwen3.8:27b', '--file', join(root, 'jd.txt')], { root });
      assertExit(run, 1);
      assert.match(run.stderr, /Unknown option: --modle\. Run with --help\./);
      const noValue = await cli(['evaluate', '--url', stub.url, '--file', join(root, 'jd.txt'), '--model'], { root });
      assertExit(noValue, 1);
      assert.match(noValue.stderr, /--model needs a value/);
      assert.deepEqual(stub.requests, []);
    });
  });

  test('a command that does not exist stops the run, and is not listed in the usage', async () => {
    const root = makeRoot();
    const help = (await cli(['--help'], { root })).stdout;
    // `rank` and `cover` are parts of career-ops this runner does not cover.
    for (const name of ['rank', 'cover', 'evalute']) {
      const run = await cli([name, '001'], { root });
      assertExit(run, 1, name);
      assert.match(run.stderr, new RegExp(`Unknown command: ${name}\\. Run with --help\\.`));
      assert.doesNotMatch(help, new RegExp(`^  ${name}\\b`, 'm'), `${name} is in the usage but is not a command`);
    }
  });

  // `pipeline 5`, meant as `--limit 5`, used to evaluate the whole inbox: the
  // 5 was dropped without a word. It is what `npm run ollama:pipeline --limit 5`
  // becomes when the "--" before the options is forgotten.
  test('a command that takes no argument refuses one instead of dropping it', async () => {
    const root = makePipelineRoot();
    await withStub({}, async (stub) => {
      const five = await cli(['pipeline', '5', '--url', stub.url, '--json'], { root, pages: PAGES });
      assertExit(five, 1);
      assert.deepEqual(JSON.parse(five.stdout), {
        command: 'pipeline', success: false, reason: 'pipeline takes no argument, and "5" was given. Did you mean --limit 5?', code: 'USAGE',
      });
      const lane = await cli(['pipeline', 'data/pipeline-lane-b.md', '--url', stub.url], { root, pages: PAGES });
      assertExit(lane, 1);
      assert.match(lane.stderr, /pipeline takes no argument, and "data\/pipeline-lane-b\.md" was given\.$/m);
      for (const command of ['models', 'doctor']) {
        const run = await cli([command, 'stray', 'words', '--url', stub.url], { root, checks: [{ pass: true, label: 'ok' }] });
        assertExit(run, 1, command);
        assert.match(run.stderr, new RegExp(`${command} takes no argument, and "stray", "words" were given\\.`));
      }
      // Started by npm, the usual cause is named.
      const viaNpm = await cli(['pipeline', '5', '--url', stub.url], { root, pages: PAGES, env: { npm_lifecycle_event: 'ollama:pipeline' } });
      assertExit(viaNpm, 1);
      assert.match(viaNpm.stderr, /Did you mean --limit 5\? With npm run, options go after "--": npm run ollama:pipeline -- <options>/);
      assert.deepEqual(stub.requests, []);
      assert.deepEqual(pendingUrls(root), [ACME, GLOBEX, INITECH]);
    });
  });

  // Without the "--", npm keeps the runner's options. In PowerShell the same
  // happens with the "--" in place whenever something is piped into npm run.
  // `--no-save` then saved, and `--limit 5` ran every row. npm leaves a trace
  // of each option it kept, and a run that npm started looks for it.
  test('refuses to start when npm kept an option that was meant for the runner', async () => {
    const root = makePipelineRoot();
    await withStub({}, async (stub) => {
      // What `npm run ollama:pipeline --limit 5 --no-merge` leaves in the child's environment.
      const kept = await cli(['pipeline', '--url', stub.url, '--json'], {
        root, pages: PAGES, env: { npm_lifecycle_event: 'ollama:pipeline', npm_config_limit: 'true', npm_config_merge: '' },
      });
      assertExit(kept, 1);
      const receipt = JSON.parse(kept.stdout);
      assert.deepEqual([receipt.command, receipt.success, receipt.code], ['pipeline', false, 'USAGE']);
      assert.equal(receipt.reason, 'npm kept --limit, --no-merge for itself, so they never reached the runner. '
        + 'With npm run, the runner\'s options go after "--": npm run ollama:pipeline -- --limit <n> --no-merge. '
        // Seen on this machine: `Get-Content jd.txt | npm run ollama:eval -- --no-save` arrives without the "--".
        + 'If the "--" was there, PowerShell dropped it, as it does when input is piped into npm run: '
        + 'use --file instead of the pipe, or start the runner directly (node ollama-runner.mjs …).');

      // `npm run ollama:eval --no-save <url>`: npm takes --no-save as its own and says nothing.
      const noSave = await cli(['evaluate', '--url', stub.url, '--file', join(root, 'jd.txt')], {
        root, env: { npm_lifecycle_event: 'ollama:eval', npm_config_save: '' },
      });
      assertExit(noSave, 1);
      assert.match(noSave.stderr, /npm kept --no-save for itself, so it never reached the runner\./);
      assert.match(noSave.stderr, /An npm configuration with save=false gives this message too: start the runner directly then as well\./);

      assert.deepEqual(stub.requests, []);
      assert.deepEqual(listing(join(root, 'reports')), []);

      // With the "--" in place npm leaves no trace, and neither does a run npm did not start.
      const fine = await cli(['pipeline', '--url', stub.url, '--quiet', '--no-merge', '--limit', '1'], {
        root, pages: PAGES, env: { npm_lifecycle_event: 'ollama:pipeline' },
      });
      assertExit(fine, 0);
      const direct = await cli(['evaluate', '--url', stub.url, '--no-save', '--quiet', '--file', join(root, 'jd.txt')], {
        root, env: { npm_config_limit: 'true', npm_config_save: '' },
      });
      assertExit(direct, 0);
      // npm's own options say nothing about what was typed for the runner.
      const own = await cli(['evaluate', '--url', stub.url, '--no-save', '--quiet', '--file', join(root, 'jd.txt')], {
        root, env: { npm_lifecycle_event: 'ollama:eval', npm_config_json: 'true', npm_config_force: 'true' },
      });
      assertExit(own, 0);
    });
  });

  test('--json leaves a receipt for every refusal, the ones made before a command runs included', async () => {
    const root = makePipelineRoot();
    await withStub({}, async (stub) => {
      for (const [args, code, pattern] of [
        [['pipeline', '--no-save'], 'USAGE', /--no-save cannot be used with pipeline/],
        [['pipeline', '--min-score', '9'], 'USAGE', /Invalid --min-score: "9"/],
        [['run', '--min-score', '9', ACME], 'USAGE', /Invalid --min-score: "9"/],
        [['evaluate', '--bogus'], 'USAGE', /Unknown option: --bogus/],
        [['evaluate', '--file'], 'USAGE', /--file needs a value/],
        [['bogus-command'], 'USAGE', /Unknown command: bogus-command/],
      ]) {
        const run = await cli([...args, '--url', stub.url, '--json'], { root, pages: PAGES });
        assertExit(run, 1, args.join(' '));
        // Exactly one line, and it parses.
        assert.equal(run.stdout.trim().split('\n').length, 1, `${args.join(' ')}: ${run.stdout}`);
        const receipt = JSON.parse(run.stdout);
        assert.deepEqual([receipt.success, receipt.code], [false, code], args.join(' '));
        assert.match(receipt.reason, pattern);
        assert.match(run.stderr, pattern);
      }
      // Without --json a command-line error prints nothing on stdout.
      const plain = await cli(['evaluate', '--bogus'], { root });
      assert.equal(plain.stdout, '');
      assert.deepEqual(stub.requests, []);
    });
  });

  // PowerShell 5.1 decodes a program's output with the console code page:
  // "Zürich" in a UTF-8 receipt reached ConvertFrom-Json as "Z├╝rich".
  test('--json receipt is pure ASCII and parses back to the same strings', async () => {
    const root = makeRoot();
    const fields = { company: 'Société Générale — Zürich', role: 'Ingénieur IA · 東京' };
    await withStub({ reply: () => goodReply(fields) }, async (stub) => {
      const run = await cli(['evaluate', '--url', stub.url, '--json', '--no-save', '--file', join(root, 'jd.txt')], { root });
      assertExit(run, 0);
      assert.ok(/^[\x00-\x7e]*$/.test(run.stdout), 'a character outside ASCII is in the receipt');
      const receipt = JSON.parse(run.stdout);
      assert.equal(receipt.company, fields.company);
      assert.equal(receipt.role, fields.role);
      assert.ok(receipt.text.includes(`# Evaluation: ${fields.company} — ${fields.role}`));
    });
  });

  test('parseArgs reads both spellings of a value, and text that only looks like an option', () => {
    const spaced = parseArgs(['evaluate', '--model', 'qwen3.8:27b', '--num-ctx', '98304', '--file', 'jd.txt', '--no-save']);
    const equals = parseArgs(['eval', '--model=qwen3.8:27b', '--num-ctx=98304', '--file=jd.txt', '--no-save']);
    for (const parsed of [spaced, equals]) {
      assert.equal(parsed.error, undefined);
      assert.equal(parsed.command.name, 'evaluate');
      assert.deepEqual(parsed.flags, { model: 'qwen3.8:27b', numCtx: '98304', file: 'jd.txt', noSave: true });
    }
    // Global options may come before the command.
    assert.deepEqual(parseArgs(['--think=off', 'models']).flags, { think: 'off' });
    // A pasted posting can start with a dash; "--" covers text that is shaped like an option.
    assert.deepEqual(parseArgs(['evaluate', '- Build the retrieval layer', '--', '--not-an-option']).positionals,
      ['- Build the retrieval layer', '--not-an-option']);
    assert.deepEqual(parseArgs(['pipeline', '-1', '--limit', '3', '--min-score=4.5', '--tailor', '--pdf', '--no-merge']).flags,
      { one: true, limit: '3', minScore: '4.5', tailor: true, pdf: true, noMerge: true });
    // An option of another command is unknown here.
    assert.match(parseArgs(['evaluate', '--limit', '3']).error, /Unknown option: --limit/);
    assert.match(parseArgs(['evaluate', '--no-save=yes']).error, /--no-save does not take a value/);
    assert.match(parseArgs(['evaluate', '--file', '--quiet']).error, /--file needs a value/);
    assert.match(parseArgs(['evaluate', '--model=']).error, /--model needs a value/);
    assert.match(parseArgs(['evalute', 'x']).error, /Unknown command: evalute/);
    // scan's arguments belong to scan.mjs, options included.
    assert.deepEqual(parseArgs(['scan', '--dry-run', '--company', 'Acme', '--help']).positionals, ['--dry-run', '--company', 'Acme', '--help']);
    assert.deepEqual(parseArgs(['scan', '--dry-run']).flags, {});
  });
});

// ---------------------------------------------------------------------------
// models, doctor, scan
// ---------------------------------------------------------------------------

describe('models, doctor and scan', { concurrency: 6 }, () => {
  test('models lists the chat models, marks the pick and says which ones are too small', async () => {
    const root = makeRoot();
    await withStub({}, async (stub) => {
      const run = await cli(['models', '--url', stub.url], { root });
      assertExit(run, 0);
      assert.match(run.stdout, /Ollama 0\.0\.0-stub at http:\/\/127\.0\.0\.1:\d+: 4 chat model\(s\) \(1 embedding model\(s\) not shown\)/);
      const row = (name) => run.stdout.split('\n').find((line) => line.includes(` ${name} `));
      assert.match(row('qwen3.8:27b'), /^\s+\*\s+qwen3\.8:27b\s+27\.3B\s+17\.5 GB\s+262144\s+completion, vision, tools, thinking\s+fits$/);
      assert.match(row('gemma4:26b-a4b-it-q4_K_M'), /^\s+gemma4\S+\s+25\.8B\s.*\sfits$/, 'only one model is marked');
      assert.match(row('mistral-small:latest'), /\s32768\s+completion, tools\s+too small$/);
      assert.ok(!run.stdout.includes('nomic-embed-text'));
      // Measured from this data root's own CV and profile, not a guess.
      assert.doesNotMatch(run.stdout, /a typical prompt/);
      assert.deepEqual(stub.generations(), []);
      assert.ok(!stub.paths().includes('POST /api/chat'), 'listing models must not load one');
    });
  });

  test('models --json carries the same list, and a named model that is missing is said, not guessed around', async () => {
    const root = makeRoot({ cv: false });
    await withStub({}, async (stub) => {
      const run = await cli(['models', '--url', stub.url, '--json', '--model', 'llama3.3'], { root });
      assertExit(run, 0);
      const receipt = JSON.parse(run.stdout);
      assert.deepEqual(receipt.models.map((model) => model.name), CHAT_MODEL_NAMES);
      assert.equal(receipt.pick, null);
      assert.match(receipt.pickError, /Model "llama3\.3" is not installed/);
      assert.deepEqual(receipt.models.map((model) => model.picked), [false, false, false, false]);
      assert.deepEqual(receipt.models.map((model) => model.fits), [true, true, true, false]);
      // No CV in this data root: the typical prompt, plus 5% and room for the
      // reply. The size is the one both setup checks assume in the same case
      // (TYPICAL_PROMPT_TOKENS): `models` used to say 40000 and 54000 here
      // while the checks said 44000 and 58200.
      assert.equal(receipt.promptTokens, TYPICAL_PROMPT_TOKENS);
      assert.equal(receipt.promptTokens, 44000);
      assert.equal(receipt.need, 58200);
      assert.deepEqual(receipt.models.map((model) => model.need), [58200, 58200, 58200, 58200]);
    });
  });

  test('models with thinking forced on: a model that can think needs room for its whole output cap', async () => {
    const root = makeRoot({ cv: false });
    await withStub({}, async (stub) => {
      const asJson = await cli(['models', '--url', stub.url, '--json', '--think', 'on'], { root });
      assertExit(asJson, 0);
      const receipt = JSON.parse(asJson.stdout);
      // 44000 tokens of prompt + 5% + the cap of 28000 a thinking request gets,
      // for the two models that can think; 12000 of reply for the two that cannot.
      assert.deepEqual(receipt.models.map((model) => [model.name, model.need, model.numCtx, model.fits]), [
        ['qwen3.8:27b', 74200, 98304, true],
        ['gemma4:26b-a4b-it-q4_K_M', 74200, 98304, true],
        ['qwen3-coder:30b', 58200, 65536, true],
        ['mistral-small:latest', 58200, 32768, false],
      ]);
      assert.equal(receipt.need, 58200);

      const asText = await cli(['models', '--url', stub.url, '--think=on'], { root });
      assertExit(asText, 0);
      assert.match(asText.stdout, /plus room for the reply, 58\.2K in all\.\n {2}Thinking is forced on, so a model that can think needs room for its whole output cap: 74\.2K in all\.\n$/);
      // Left to the default, no model thinks, and nothing is said about it.
      const automatic = await cli(['models', '--url', stub.url], { root });
      assert.doesNotMatch(automatic.stdout, /Thinking is forced on/);
    });
  });

  test('models says so when the server is not there', async () => {
    const root = makeRoot();
    let url;
    await withStub({}, async (stub) => { url = stub.url; });
    // The stub is closed now, so its port refuses connections.
    const run = await cli(['models', '--url', url], { root, env: { OLLAMA_HOST: '0.0.0.0' } });
    assertExit(run, 1);
    assert.match(run.stderr, /Ollama not reachable at http:\/\/127\.0\.0\.1:\d+\. Start it \(ollama serve\) or set OLLAMA_BASE_URL\./);
    // The address came from --url, so OLLAMA_HOST is not what went wrong.
    assert.doesNotMatch(run.stderr, /OLLAMA_HOST/);
  });

  // The checks themselves live in lib/ollama-doctor.mjs and have a suite of
  // their own. What is tested here is the command around them, with the checks
  // replaced: what it hands them, how it prints what comes back, how it exits.
  test('doctor passes the run\'s settings to the checks and prints each result with its fix', async () => {
    const root = makeRoot();
    const checks = [
      { pass: true, label: 'Ollama 0.0.0-stub reachable at http://localhost:11434' },
      { warn: true, label: 'loaded at 131072, runner uses 65536: the next call reloads the model', fix: 'Set OLLAMA_NUM_CTX=131072 to match' },
      { pass: true, label: 'thinking: off (auto)' },
    ];
    const run = await cli(['doctor', '--url', 'http://127.0.0.1:9', '--model', 'gemma4:26b-a4b-it-q4_K_M', '--num-ctx=98304', '--think', 'off'], { root, checks });
    // A warning is not a failure.
    assertExit(run, 0);
    assert.equal(run.stdout, [
      '✓ Ollama 0.0.0-stub reachable at http://localhost:11434',
      '⚠ loaded at 131072, runner uses 65536: the next call reloads the model',
      '  → Set OLLAMA_NUM_CTX=131072 to match',
      '✓ thinking: off (auto)',
      '',
      'Result: the local Ollama setup is ready (1 warning — see above).',
      '',
    ].join('\n'));

    const seen = JSON.parse(read(root, 'checks-seen.json'));
    // The options of this run reach the checks as the variables they read.
    assert.deepEqual(seen.env, {
      OLLAMA_BASE_URL: 'http://127.0.0.1:9', OLLAMA_MODEL: 'gemma4:26b-a4b-it-q4_K_M', OLLAMA_NUM_CTX: '98304', OLLAMA_THINK: 'off',
    });
    // The prompt size is this data root's own, not a typical one.
    assert.ok(seen.promptTokens > 40000 && seen.promptTokens < 50000, `promptTokens ${seen.promptTokens}`);
    // Run on purpose, so an unreachable server is a failure and not a warning.
    assert.equal('soft' in seen, false);
  });

  test('doctor exits 1 when a check fails, and --json carries the checks', async () => {
    const root = makeRoot({ cv: false });
    const checks = [
      { pass: true, label: 'Ollama 0.0.0-stub reachable at http://localhost:11434' },
      { pass: false, label: 'num_ctx too small', fix: ['Pick a model with a larger context', 'or raise OLLAMA_NUM_CTX'] },
    ];
    const run = await cli(['doctor'], { root, checks });
    assertExit(run, 1);
    assert.match(run.stdout, /^✗ num_ctx too small\n {2}→ Pick a model with a larger context\n {2}→ or raise OLLAMA_NUM_CTX\n\nResult: 1 issue found\.\n$/m);
    const seen = JSON.parse(read(root, 'checks-seen.json'));
    // No CV to measure: the checks are left to use their own typical size.
    assert.equal('promptTokens' in seen, false);
    assert.deepEqual(seen.env, {});

    const asJson = await cli(['doctor', '--json'], { root, checks });
    assertExit(asJson, 1);
    assert.deepEqual(JSON.parse(asJson.stdout), { command: 'doctor', success: false, failures: 1, warnings: 0, checks });
  });

  test('scan hands its arguments to scan.mjs and returns its exit code', async () => {
    const root = makeRoot();
    const help = await cli(['scan', '--help'], { root });
    assertExit(help, 0);
    assert.match(help.stdout, /node scan\.mjs --dry-run/, 'this is scan.mjs\'s own usage, not the runner\'s');
    const refused = await cli(['scan', '--definitely-not-a-flag'], { root });
    assertExit(refused, 1);
    assert.match(refused.stdout + refused.stderr, /unrecognized flag\(s\): --definitely-not-a-flag/);
  });
});

// ---------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------

const ACME = 'https://jobs.example.com/acme/1';
const GLOBEX = 'https://jobs.example.com/globex/2';
const INITECH = 'https://jobs.example.com/initech/3';
const PIPELINE = [
  '# Pipeline',
  '',
  '## Pending',
  '',
  `- [ ] ${ACME} | Acme | Senior Engineer`,
  `- [ ] ${GLOBEX} | Globex | Staff Engineer`,
  `- [ ] ${INITECH} | Initech | Principal Engineer`,
  '- [x] https://jobs.example.com/old/9 | Old | Done long ago',
  '',
].join('\n');
const page = (company) => `${company} is hiring a Senior Engineer.\nSalary: USD 150,000 per year.\nBuild reliable software.`;
const PAGES = { [ACME]: page('Acme'), [GLOBEX]: page('Globex'), [INITECH]: page('Initech') };

/** The stub writes a report about whichever company the posting names. */
function replyForPosting(seen, scores = {}) {
  const company = ['Acme', 'Globex', 'Initech'].find((name) => seen.body.messages[1].content.includes(`${name} is hiring`));
  return goodReply({ company, role: 'Senior Engineer', score: scores[company] ?? '4.2' });
}

/**
 * The same, with Acme's report as the default model wrote one in the third
 * live round: in its Machine Summary the work-authorisation tier that is a
 * hard blocker, no hard stop, and "Apply". The runner saves such a report as
 * Skip, with the hard stop that tier stands for.
 */
function replyWithBlockingTier(seen, scores = {}) {
  const reply = replyForPosting(seen, scores);
  if (!seen.body.messages[1].content.includes('Acme is hiring')) return reply;
  return reply.replace('---SCORE_SUMMARY---',
    `## Machine Summary\n\n\`\`\`yaml\nscore: ${scores.Acme ?? '4.2'}\nfinal_decision: "Apply"\nwork_auth: "no_sponsorship"\n\`\`\`\n\n---SCORE_SUMMARY---`);
}
/** That hard stop, as a run names it: under the posting and in its receipt. */
const TIER_STOP = 'hard stop: No visa sponsorship (listed by the runner from the tier work_auth: no_sponsorship): the posting refuses it, '
  + 'and the role is outside the countries the profile lists as authorized';

function makePipelineRoot({ tracker = true } = {}) {
  const root = makeRoot();
  mkdirSync(join(root, 'data'), { recursive: true });
  writeFileSync(join(root, 'data', 'pipeline.md'), PIPELINE);
  if (tracker) writeFileSync(join(root, 'data', 'applications.md'), TRACKER);
  return root;
}

const pendingUrls = (root) => parsePipeline(read(root, 'data', 'pipeline.md')).map((item) => item.url);

describe('pipeline', { concurrency: 6 }, () => {
  test('each posting is evaluated, ticked and merged into the tracker', async () => {
    const root = makePipelineRoot();
    await withStub({ reply: (seen) => replyForPosting(seen) }, async (stub) => {
      const run = await cli(['pipeline', '--url', stub.url, '--json'], { root, pages: PAGES });
      assertExit(run, 0);
      // One object, although merge-tracker.mjs prints to its own stdout while it runs.
      assert.equal(run.stdout.trim().split('\n').length, 1, run.stdout);
      const receipt = JSON.parse(run.stdout);
      assert.equal(receipt.success, true);
      assert.equal(receipt.merged, true);
      assert.deepEqual(receipt.done.map((entry) => [entry.url, entry.company, entry.score, entry.report]), [
        [ACME, 'Acme', 4.2, `reports/001-acme-${TODAY}.md`],
        [GLOBEX, 'Globex', 4.2, `reports/002-globex-${TODAY}.md`],
        [INITECH, 'Initech', 4.2, `reports/003-initech-${TODAY}.md`],
      ]);
      assert.deepEqual([receipt.failed, receipt.stoppedEarly, receipt.pending], [[], false, 0]);

      // One model lookup for the run, the model loaded once, then one request per posting.
      assert.deepEqual(stub.paths(), ['GET /api/tags', 'POST /api/show', 'POST /api/chat', 'POST /api/chat', 'POST /api/chat', 'POST /api/chat']);
      const load = stub.requests[2].body;
      assert.deepEqual(load.messages, []);
      assert.equal(load.options.num_ctx, 65536);
      // Every posting at the size the model was loaded at: another size reloads it.
      assert.deepEqual(stub.generations().map((seen) => seen.body.options.num_ctx), [65536, 65536, 65536]);
      assert.ok(stub.generations()[0].body.messages[1].content.includes(`URL: ${ACME}`));

      assert.deepEqual(pendingUrls(root), []);
      assert.ok(read(root, 'data', 'pipeline.md').includes(`- [x] ${GLOBEX} | Globex | Staff Engineer`));
      assert.deepEqual(listing(join(root, 'reports')), [
        `001-acme-${TODAY}.md`, `002-globex-${TODAY}.md`, `003-initech-${TODAY}.md`,
      ]);
      // merge-tracker.mjs ran: the rows are in the tracker.
      const tracker = read(root, 'data', 'applications.md');
      for (const company of ['Acme', 'Globex', 'Initech']) assert.match(tracker, new RegExp(`\\| ${company} \\|`));
    });
  });

  // `pipeline` is the one command that saves an evaluation without printing
  // its verdict. A report the runner had just saved as Skip, with a hard stop,
  // stood in the summary as "evaluated  4.2/5", and with the reply streaming,
  // the model's own "Apply" was the last word the terminal had on it. The
  // report's advice was named only in a run that was asked for CVs.
  test('a run that only evaluates says which report advises against applying: under the posting, in its row of the summary and in the receipt', async () => {
    const root = makePipelineRoot();
    await withStub({ reply: (seen) => replyWithBlockingTier(seen) }, async (stub) => {
      // Not --quiet: the replies stream, as they do for whoever types the command.
      const run = await cli(['pipeline', '--url', stub.url, '--limit', '2', '--no-merge'], { root, pages: PAGES });
      assertExit(run, 0);
      // What the run said about each posting, under the line that names it.
      const said = run.stderr.split(/^\[\d\/3\] /m);
      assert.equal(said.length, 3, run.stderr);
      const [, acme, globex] = said;
      assert.ok(acme.startsWith('Acme — Senior Engineer\n') && globex.startsWith('Globex — Staff Engineer\n'), run.stderr);
      assert.ok(acme.includes(`    Ticked in the pipeline: ${ACME}\n    The report advises against applying: ${TIER_STOP}\n`), acme);
      assert.ok(!globex.includes('advises against applying'), globex);
      // No CV was asked for, so none is said to be withheld, and no way to make one anyway is named.
      assert.doesNotMatch(run.stderr, /No CV for this one|To make the CV and the answers anyway/);

      // The reply as it streamed says Apply. The report that was saved says Skip and lists the hard stop.
      const [streamed, afterSaving] = run.stdout.split(`Report saved: reports/001-acme-${TODAY}.md`);
      assert.ok(afterSaving !== undefined && streamed.includes('final_decision: "Apply"'), run.stdout.slice(0, 1500));
      const saved = read(root, 'reports', `001-acme-${TODAY}.md`);
      assert.match(saved, /^\*\*Decision:\*\* Skip$/m);
      assert.match(saved, /^final_decision: "Skip"$/m);
      assert.ok(saved.includes(`hard_stops:\n  - "${TIER_STOP.slice('hard stop: '.length)}"\n`), saved);
      assert.match(read(root, 'reports', `002-globex-${TODAY}.md`), /^\*\*Decision:\*\* Apply$/m);

      const summary = run.stdout.slice(run.stdout.indexOf('Pipeline summary'));
      assert.match(summary, /^\s+1\s+evaluated\s+4\.2\/5\s+Acme — Senior Engineer\s+reports\/001-acme-\S+\.md · the report advises against applying$/m);
      // The row of a report that gives no such advice ends with the report.
      assert.match(summary, /^\s+2\s+evaluated\s+4\.2\/5\s+Globex — Staff Engineer\s+reports\/002-globex-\S+\.md$/m);
      // Evaluated and ticked like the other: the advice is about applying, not about the evaluation.
      assert.deepEqual(pendingUrls(root), [INITECH]);
    });

    const second = makePipelineRoot();
    await withStub({ reply: (seen) => replyWithBlockingTier(seen) }, async (stub) => {
      const run = await cli(['pipeline', '--url', stub.url, '--json', '--limit', '2', '--no-merge'], { root: second, pages: PAGES });
      assertExit(run, 0);
      const receipt = JSON.parse(run.stdout);
      assert.deepEqual(receipt.done, [
        { url: ACME, company: 'Acme', role: 'Senior Engineer', score: 4.2, report: `reports/001-acme-${TODAY}.md`, stops: [TIER_STOP] },
        // No `stops` key for a report that gives none.
        { url: GLOBEX, company: 'Globex', role: 'Senior Engineer', score: 4.2, report: `reports/002-globex-${TODAY}.md` },
      ]);
      // --min-score is the line for --tailor and --pdf. A run that was asked for neither does not name it.
      assert.equal('minScore' in receipt, false);
      // The line under the posting is on stderr, where --json leaves it.
      assert.ok(run.stderr.includes(`    The report advises against applying: ${TIER_STOP}\n`), run.stderr);
    });
  });

  test('a posting that fails stays pending and the run exits 1', async () => {
    const root = makePipelineRoot();
    // Globex's page is gone. The other two are fine.
    const pages = { [ACME]: PAGES[ACME], [INITECH]: PAGES[INITECH] };
    await withStub({ reply: (seen) => replyForPosting(seen) }, async (stub) => {
      const run = await cli(['pipeline', '--url', stub.url, '--quiet', '--no-merge'], { root, pages });
      assertExit(run, 1);
      assert.deepEqual(pendingUrls(root), [GLOBEX], 'only the posting that was not evaluated is still pending');
      assert.equal(stub.generations().length, 2);
      assert.match(run.stderr, /Could not read the job page: No job description to evaluate at .*globex/);
      assert.match(run.stdout, /2 evaluated, 1 failed, 1 still pending\./);
      // --no-merge: the additions wait where they were written.
      assert.deepEqual(listing(join(root, 'batch', 'tracker-additions')), ['001-acme.tsv', '002-initech.tsv']);
      assert.equal(read(root, 'data', 'applications.md'), TRACKER);
    });
  });

  // merge-tracker.mjs exits 0 over a tracker that is not there and leaves the
  // rows where they are. The runner used to pass that on as a run that went
  // well: a scheduled pipeline could succeed every night and fill no tracker.
  test('without a tracker one is created with a URL column, the run says so, and the rows are merged into it', async () => {
    const root = makePipelineRoot({ tracker: false });
    await withStub({ reply: (seen) => replyForPosting(seen) }, async (stub) => {
      const run = await cli(['pipeline', '--url', stub.url, '--json', '--limit', '2'], { root, pages: PAGES });
      assertExit(run, 0);
      const receipt = JSON.parse(run.stdout);
      assert.deepEqual([receipt.success, receipt.merged, receipt.trackerCreated], [true, true, 'data/applications.md']);
      assert.equal('trackerMissing' in receipt, false);
      assert.match(run.stderr, /No tracker yet: created data\/applications\.md \(ten columns, the last one the posting URL\)\./);
      const tracker = read(root, 'data', 'applications.md').split('\n');
      assert.deepEqual(tracker.slice(0, 4), [
        '# Applications Tracker',
        '',
        '| # | Date | Company | Role | Score | Status | PDF | Report | Notes | URL |',
        '|---|------|---------|------|-------|--------|-----|--------|-------|-----|',
      ]);
      // Both rows are in it, each with the address it was read from in the tenth column.
      const rows = tracker.filter((line) => /^\| \d+ \|/.test(line));
      assert.equal(rows.length, 2, tracker.join('\n'));
      assert.ok(rows[0].includes('| Acme |') && rows[0].includes(`(${ACME})`), rows[0]);
      assert.ok(rows[1].includes('| Globex |') && rows[1].includes(`(${GLOBEX})`), rows[1]);
      // With the column there, the merge has nothing to warn about.
      assert.doesNotMatch(run.stderr, /URL dedup is INACTIVE|Nothing to merge into/);
      assert.deepEqual(listing(join(root, 'batch', 'tracker-additions')), ['merged']);
      assert.deepEqual(pendingUrls(root), [INITECH]);
    });
  });

  test('an existing tracker is merged into as it is, never restructured', async () => {
    const root = makePipelineRoot();
    await withStub({ reply: (seen) => replyForPosting(seen) }, async (stub) => {
      const run = await cli(['pipeline', '--url', stub.url, '--quiet', '--one'], { root, pages: PAGES });
      assertExit(run, 0);
      const tracker = read(root, 'data', 'applications.md');
      // The nine columns the user's tracker has: no tenth was added behind their back.
      assert.ok(tracker.includes('| # | Date | Company | Role | Score | Status | PDF | Report | Notes |\n'), tracker);
      assert.match(tracker, /\| Acme \|/);
      assert.doesNotMatch(run.stderr, /No tracker yet/);
    });
  });

  // A run that is stopped after a report was saved leaves its tracker row
  // waiting, with the pipeline row already ticked: no later run writes it
  // again, and one that found nothing to evaluate ended without a merge.
  test('tracker rows an earlier run left are merged by a run with nothing pending, and by one in which every row fails', async () => {
    const waitingRow = (root, num, company) => {
      mkdirSync(join(root, 'batch', 'tracker-additions'), { recursive: true });
      writeFileSync(join(root, 'batch', 'tracker-additions', `${num}-${company.toLowerCase()}.tsv`), [
        TSV_ADDITION_HEADER,
        [String(Number(num)), TODAY, company, 'Senior Engineer', 'Evaluated', '4.2/5', '❌', `[${num}](reports/${num}-x-${TODAY}.md)`, 'left by an interrupted run'].join('\t'),
        '',
      ].join('\n'));
    };
    await withStub({}, async (stub) => {
      const nothingPending = makePipelineRoot();
      writeFileSync(join(nothingPending, 'data', 'pipeline.md'), '## Pending\n\n- [x] https://jobs.example.com/old/9 | Old | Done\n');
      waitingRow(nothingPending, '007', 'Hooli');
      const idle = await cli(['pipeline', '--url', stub.url, '--json'], { root: nothingPending, pages: {} });
      assertExit(idle, 0);
      assert.equal(JSON.parse(idle.stdout).merged, true);
      assert.match(idle.stderr, /1 tracker row of an earlier run was still waiting in batch\/tracker-additions\/, and will be merged now\./);
      assert.match(read(nothingPending, 'data', 'applications.md'), /\| Hooli \|/);

      const allFail = makePipelineRoot();
      waitingRow(allFail, '007', 'Hooli');
      // No page can be read: every row fails before the model.
      const failing = await cli(['pipeline', '--url', stub.url, '--quiet'], { root: allFail, pages: {} });
      assertExit(failing, 1);
      assert.match(failing.stdout, /0 evaluated, 3 failed, 3 still pending\./);
      assert.match(read(allFail, 'data', 'applications.md'), /\| Hooli \|/);

      // --no-merge is still honoured, and says what is waiting.
      const kept = makePipelineRoot();
      waitingRow(kept, '007', 'Hooli');
      const unmerged = await cli(['pipeline', '--url', stub.url, '--quiet', '--no-merge'], { root: kept, pages: {} });
      assert.match(unmerged.stderr, /Tracker rows were left in batch\/tracker-additions\/ \(--no-merge\)/);
      assert.equal(read(kept, 'data', 'applications.md'), TRACKER);
      // No model was loaded for any of this: no page was read.
      assert.ok(!stub.paths().includes('POST /api/chat'), stub.paths().join(', '));
    });
  });

  // Closed postings are routine in a scanned inbox, and the scanner appends
  // below the old rows, so they collect at the head. Counted toward --limit
  // and left pending, two of them made `--limit 2` evaluate nothing, today
  // and on every later day.
  test('closed postings at the head of the inbox are marked, do not use up --limit, and are not tried again', async () => {
    const root = makePipelineRoot();
    const DEAD1 = 'https://jobs.example.com/dead/1';
    const DEAD2 = 'https://jobs.example.com/dead/2';
    writeFileSync(join(root, 'data', 'pipeline.md'), PIPELINE.replace('## Pending\n\n', `## Pending\n\n- [ ] ${DEAD1} | Dead | Gone\r\n- [ ] ${DEAD2} | Dead | Also gone\n- [ ] local:jds/saved.md | Saved | Posting\n- [ ] not a url | Odd | Row\n`));
    const driver = `
      import { main } from ${JSON.stringify(pathToFileURL(RUNNER).href)};
      const pages = JSON.parse(process.env.PAGES);
      process.exitCode = await main(JSON.parse(process.env.ARGV), { fetchPage: async (url) => {
        // The page reader's own words for a posting that is closed, and for one that is gone.
        if (url.endsWith('/dead/1')) throw new Error('No job description to evaluate at ' + url + ' (expired_body: pattern matched: no longer accepting)');
        if (url.endsWith('/dead/2')) throw new Error('Could not fetch job page: HTTP 404 Not Found');
        return pages[url];
      } });`;
    const pipeline = (args) => new Promise((resolve) => {
      execFile(process.execPath, ['--input-type=module', '-e', driver], {
        cwd: root, timeout: 25_000, windowsHide: true,
        env: { ...cleanEnv(), CAREER_OPS_ROOT: root, CAREER_OPS_BATCH_STATE: NO_BATCH_STATE, CAREER_OPS_NO_PLAYWRIGHT: '1', PAGES: JSON.stringify(PAGES), ARGV: JSON.stringify(args) },
      }, (error, stdout, stderr) => resolve({ status: !error ? 0 : typeof error.code === 'number' ? error.code : null, stdout, stderr }));
    });
    await withStub({ reply: (seen) => replyForPosting(seen) }, async (stub) => {
      const first = await pipeline(['pipeline', '--url', stub.url, '--json', '--limit', '2', '--no-merge']);
      assertExit(first, 1);
      const receipt = JSON.parse(first.stdout);
      // Two postings were evaluated although four rows stood before them.
      assert.deepEqual(receipt.done.map((entry) => entry.url), [ACME, GLOBEX]);
      assert.deepEqual(receipt.failed.map((entry) => [entry.url, entry.code, entry.marked === true]), [
        [DEAD1, 'JOB_PAGE', true], [DEAD2, 'JOB_PAGE', true], ['local:jds/saved.md', 'BAD_INPUT', false], ['not a url', 'BAD_INPUT', true],
      ]);
      const file = read(root, 'data', 'pipeline.md');
      // The CR of a line saved on Windows stays at the end of the line.
      assert.ok(file.includes(`- [!] ${DEAD1} | Dead | Gone — posting closed (expired_body), checked ${TODAY}\r\n`), file);
      assert.ok(file.includes(`- [!] ${DEAD2} | Dead | Also gone — posting closed (HTTP 404), checked ${TODAY}\n`), file);
      assert.ok(file.includes('- [!] not a url | Odd | Row — not a posting URL\n'), file);
      // A row an agent's pipeline mode can read stays as it is.
      assert.ok(file.includes('- [ ] local:jds/saved.md | Saved | Posting\n'), file);
      assert.deepEqual(pendingUrls(root), ['local:jds/saved.md', INITECH]);
      assert.equal(receipt.pending, 2);
      // The model was loaded after the first page that could be read, at the size the postings run at.
      assert.deepEqual(stub.paths(), ['GET /api/tags', 'POST /api/show', 'POST /api/chat', 'POST /api/chat', 'POST /api/chat']);
      assert.match(first.stderr, /Marked `- \[!\]` in the pipeline with the reason: later runs will not try it again\./);

      const second = await pipeline(['pipeline', '--url', stub.url, '--json', '--limit', '2', '--no-merge']);
      const again = JSON.parse(second.stdout);
      assert.deepEqual(again.done.map((entry) => entry.url), [INITECH]);
      assert.deepEqual(again.failed.map((entry) => entry.url), ['local:jds/saved.md'], 'a marked row was tried again');
    });
  });

  test('a run in which no page can be read loads no model', async () => {
    const root = makePipelineRoot();
    await withStub({}, async (stub) => {
      const run = await cli(['pipeline', '--url', stub.url, '--quiet', '--no-merge'], { root, pages: {} });
      assertExit(run, 1);
      // The model list and the model's details, and nothing after them.
      assert.deepEqual(stub.paths(), ['GET /api/tags', 'POST /api/show']);
      assert.doesNotMatch(run.stderr, /Loading qwen3\.8:27b/);
      // The seam's "http_404" is not closure evidence: the rows stay pending.
      assert.deepEqual(pendingUrls(root), [ACME, GLOBEX, INITECH]);
    });
  });

  // The scanner read the company and the role off the board. A reply that
  // names neither used to leave an orphan report on every run.
  test('a reply that names no company and role is tracked under the pipeline row\'s own', async () => {
    const root = makePipelineRoot();
    const nameless = [
      '## A) Role Summary',
      '',
      `EVALUATION_BODY_MARKER. ${'The candidate meets the core requirements of this role. '.repeat(8)}`,
      '',
      '---SCORE_SUMMARY---',
      'COMPANY: Unknown',
      'ROLE: Unknown',
      'SCORE: 3.9',
      'ARCHETYPE: AI Platform',
      'LEGITIMACY: High Confidence',
      '---END_SUMMARY---',
    ].join('\n');
    await withStub({ reply: () => nameless }, async (stub) => {
      const run = await cli(['pipeline', '--url', stub.url, '--json', '--one'], { root, pages: PAGES });
      assertExit(run, 0);
      const receipt = JSON.parse(run.stdout);
      assert.deepEqual(receipt.done.map((entry) => [entry.url, entry.company, entry.role, entry.score, entry.report]), [
        [ACME, 'Acme', 'Senior Engineer', 3.9, `reports/001-acme-${TODAY}.md`],
      ]);
      assert.equal(stub.generations().length, 1);
      assert.ok(read(root, 'reports', `001-acme-${TODAY}.md`).startsWith('# Evaluation: Acme — Senior Engineer\n'));
      assert.match(read(root, 'data', 'applications.md'), /\| Acme \| Senior Engineer \|/);
      assert.deepEqual(pendingUrls(root), [GLOBEX, INITECH]);
    });
  });

  // The report is written before its tracker row. When the row cannot be
  // written, the posting used to stay pending, and every later run evaluated
  // it again and saved one more report.
  test('a posting whose report is saved but whose tracker row is not is marked, not evaluated again', async () => {
    const root = makePipelineRoot();
    // The additions folder cannot be created: a file is in its way.
    mkdirSync(join(root, 'batch'), { recursive: true });
    writeFileSync(join(root, 'batch', 'tracker-additions'), 'not a directory');
    await withStub({ reply: (seen) => replyForPosting(seen) }, async (stub) => {
      const first = await cli(['pipeline', '--url', stub.url, '--quiet', '--one'], { root, pages: PAGES });
      assertExit(first, 1);
      assert.ok(read(root, 'data', 'pipeline.md').includes(`- [!] ${ACME} | Acme | Senior Engineer — report 001 is saved, but its tracker row could not be written\n`));
      const second = await cli(['pipeline', '--url', stub.url, '--quiet', '--one'], { root, pages: PAGES });
      assertExit(second, 1);
      // One report for each posting: Acme was not evaluated a second time.
      assert.deepEqual(listing(join(root, 'reports')), [`001-acme-${TODAY}.md`, `002-globex-${TODAY}.md`]);
      assert.deepEqual(pendingUrls(root), [INITECH]);
    });
  });

  test('a tracker merge does not run code a .env in the working directory names in NODE_OPTIONS', async () => {
    const root = makePipelineRoot();
    const elsewhere = mkdtempSync(join(SANDBOX, 'cwd-'));
    const loaded = join(elsewhere, 'loaded.txt');
    const marker = join(elsewhere, 'marker.cjs');
    // Every process that loads the marker leaves its own id and its parent's.
    // Not the script it runs: tests/main-guard-convention.test.mjs allows no
    // tracked file to read the process entry path, in a string or out of one.
    writeFileSync(marker, `require('node:fs').appendFileSync(${JSON.stringify(loaded)}, process.pid + ' ' + process.ppid + '\\n');\n`);
    await withStub({ reply: (seen) => replyForPosting(seen) }, async (stub) => {
      // The path in quotes: Node splits NODE_OPTIONS at a space, and a temp folder can have one in its path.
      const option = `--require "${marker.replace(/\\/g, '/')}"`;
      // From a .env in the folder the runner happens to be started in: not passed on.
      writeFileSync(join(elsewhere, '.env'), `NODE_OPTIONS=${option}\n`);
      const fromFile = await cli(['pipeline', '--url', stub.url, '--quiet', '--one'], { root, cwd: elsewhere, pages: PAGES });
      assertExit(fromFile, 0);
      assert.match(read(root, 'data', 'applications.md'), /\| Acme \|/, 'the merge did not run');
      assert.equal(existsSync(loaded), false, existsSync(loaded) ? `loaded by: ${readFileSync(loaded, 'utf-8')}` : '');

      // Exported in the shell, it is the user's own choice and reaches the child.
      rmSync(join(elsewhere, '.env'));
      const fromShell = await cli(['pipeline', '--url', stub.url, '--quiet', '--one'], {
        root, cwd: elsewhere, pages: PAGES, env: { NODE_OPTIONS: option },
      });
      assertExit(fromShell, 0);
      // The runner loaded it, and so did a process the runner started: the
      // only one this run starts is merge-tracker.mjs.
      const loaders = readFileSync(loaded, 'utf-8').trim().split('\n').map((line) => line.split(' '));
      assert.ok(
        loaders.some(([, parent]) => loaders.some(([pid]) => pid === parent)),
        `no process the runner started loaded the module: ${JSON.stringify(loaders)}`,
      );
    });
  });

  test('--one and --limit stop early and leave the rest pending', async () => {
    for (const [option, left] of [[['--one'], [GLOBEX, INITECH]], [['-1'], [GLOBEX, INITECH]], [['--limit', '2'], [INITECH]]]) {
      const root = makePipelineRoot();
      await withStub({ reply: (seen) => replyForPosting(seen) }, async (stub) => {
        const run = await cli(['pipeline', '--url', stub.url, '--quiet', '--no-merge', ...option], { root, pages: PAGES });
        assertExit(run, 0, option.join(' '));
        assert.deepEqual(pendingUrls(root), left, option.join(' '));
        assert.equal(stub.generations().length, 3 - left.length);
      });
    }
  });

  // What --tailor and --pdf do with a posting once it is evaluated is covered
  // in tests/ollama-runner-steps.test.mjs, beside `run`, which strings the
  // same steps together. What is here is the receipt of a run that was given
  // a score of its own and makes no CV, so that the stub is never asked for
  // one.
  test('the receipt of a run that was asked for CVs names the --min-score it was given: beside the stops, it says why a posting has none', async () => {
    // Acme reaches the score, and its report advises against applying. Globex does not reach it.
    const scores = { Acme: '4.6', Globex: '3.1' };
    const root = makePipelineRoot();
    await withStub({ reply: (seen) => replyWithBlockingTier(seen, scores) }, async (stub) => {
      const run = await cli(['pipeline', '--pdf', '--min-score', '4.5', '--url', stub.url, '--json', '--limit', '2', '--no-merge'], { root, pages: PAGES });
      assertExit(run, 0);
      const receipt = JSON.parse(run.stdout);
      assert.equal(receipt.minScore, 4.5);
      assert.deepEqual(receipt.done, [
        { url: ACME, company: 'Acme', role: 'Senior Engineer', score: 4.6, report: `reports/001-acme-${TODAY}.md`, stops: [TIER_STOP] },
        { url: GLOBEX, company: 'Globex', role: 'Senior Engineer', score: 3.1, report: `reports/002-globex-${TODAY}.md` },
      ]);
      // Two evaluations, and no request for a CV.
      assert.equal(stub.generations().length, 2);
      assert.ok(stub.generations().every((seen) => seen.body.messages[1].content.startsWith('JOB DESCRIPTION TO EVALUATE:')));
    });
  });

  // A posting can open a code fence it never closes: one line of tildes is
  // enough. Upstream's reader refuses a Machine Summary behind such a fence,
  // and a reply that copies the posting ahead of its summary has its summary
  // there. The run read no hard stop from that reply, although the report it
  // saved from it lists the stop and says Skip: no advice was printed, the CV
  // was asked for, and the receipt carried the CV step and no `stops`.
  test('a reply that copies a posting with an open code fence ahead of its hard stop gets no CV: one request, the advice under the posting, in its row of the summary and in the receipt', async () => {
    const fenced = PAGES[ACME].replace('\nBuild', '\n~~~~~~~~~~~~\nBuild');
    const pages = { ...PAGES, [ACME]: fenced };
    const stop = 'hard stop: Onsite in Lagos five days a week';
    // The copy of the posting, which the request asks the model not to make, then the summary with the hard stop.
    const copying = (seen) => replyForPosting(seen).replace('---SCORE_SUMMARY---', [
      '## Job Description (archived verbatim)', '', fenced, '',
      '## Machine Summary', '', '```yaml', 'score: 4.2', 'final_decision: "Apply"', 'hard_stops:', '  - "Onsite in Lagos five days a week"', '```', '',
      '---SCORE_SUMMARY---',
    ].join('\n'));
    const evaluations = (stub) => stub.generations().map((seen) => seen.body.messages[1].content.startsWith('JOB DESCRIPTION TO EVALUATE:'));

    const root = makePipelineRoot();
    await withStub({ reply: copying }, async (stub) => {
      const run = await cli(['pipeline', '--tailor', '--url', stub.url, '--quiet', '--limit', '1', '--no-merge'], { root, pages });
      assertExit(run, 0);
      // The posting reached the score for a CV. One request, the evaluation: no CV was asked for.
      assert.deepEqual(evaluations(stub), [true]);
      assert.deepEqual(listing(join(root, 'output')), []);
      const [, acme] = run.stderr.split(/^\[1\/3\] /m);
      assert.ok(acme?.startsWith('Acme — Senior Engineer\n'), run.stderr);
      assert.ok(acme.includes(`    Ticked in the pipeline: ${ACME}\n    The report advises against applying: ${stop}\n`
        + '    No CV for this one, for that reason.\n'), acme);
      assert.match(run.stdout, /^\s+1\s+evaluated\s+4\.2\/5\s+Acme — Senior Engineer\s+reports\/001-acme-\S+\.md · no CV: the report advises against applying$/m);
    });

    const second = makePipelineRoot();
    await withStub({ reply: copying }, async (stub) => {
      const run = await cli(['pipeline', '--tailor', '--url', stub.url, '--json', '--limit', '1', '--no-merge'], { root: second, pages });
      assertExit(run, 0);
      const receipt = JSON.parse(run.stdout);
      // The stop, and no `steps` key: the CV was not tried.
      assert.deepEqual(receipt.done, [
        { url: ACME, company: 'Acme', role: 'Senior Engineer', score: 4.2, report: `reports/001-acme-${TODAY}.md`, stops: [stop] },
      ]);
      // The score a CV needs, which 4.2 reached: the report's advice is what withheld it.
      assert.equal(receipt.minScore, 4);
      assert.deepEqual(evaluations(stub), [true]);
    });
  });

  test('the pipeline file, its options and the CV are checked before any model is loaded', async () => {
    await withStub({}, async (stub) => {
      const empty = makeRoot();
      const none = await cli(['pipeline', '--url', stub.url], { root: empty, pages: {} });
      assertExit(none, 0);
      assert.match(none.stdout, /No pipeline at .*pipeline\.md\. Run: npm run ollama:scan/);

      const done = makeRoot();
      mkdirSync(join(done, 'data'));
      writeFileSync(join(done, 'data', 'pipeline.md'), '## Pending\n\n- [x] https://jobs.example.com/old/9 | Old | Done\n');
      const nothing = await cli(['pipeline', '--url', stub.url], { root: done, pages: {} });
      assertExit(nothing, 0);
      assert.match(nothing.stdout, /No pending postings in /);

      const root = makePipelineRoot();
      for (const [args, pattern] of [
        [['--limit', 'two'], /Invalid --limit: "two" — must be a positive integer\./],
        [['--min-score', '7'], /Invalid --min-score: "7" — must be a number from 0 to 5\./],
        [['--no-save'], /--no-save cannot be used with pipeline/],
      ]) {
        const run = await cli(['pipeline', '--url', stub.url, ...args], { root, pages: PAGES });
        assertExit(run, 1, args.join(' '));
        assert.match(run.stderr, pattern);
      }

      const noCv = makePipelineRoot();
      rmSync(join(noCv, 'cv.md'));
      const run = await cli(['pipeline', '--url', stub.url], { root: noCv, pages: PAGES });
      assertExit(run, 1);
      assert.match(run.stderr, /cv\.md not found/);

      assert.deepEqual(stub.requests, []);
      assert.deepEqual(pendingUrls(root), [ACME, GLOBEX, INITECH]);
    });
  });

  test('CAREER_OPS_PIPELINE names another inbox', async () => {
    const root = makePipelineRoot();
    const lane = join(root, 'lane-b.md');
    writeFileSync(lane, `- [ ] ${INITECH} | Initech | Principal Engineer\n`);
    await withStub({ reply: (seen) => replyForPosting(seen) }, async (stub) => {
      const run = await cli(['pipeline', '--url', stub.url, '--quiet', '--no-merge'], { root, pages: PAGES, env: { CAREER_OPS_PIPELINE: lane } });
      assertExit(run, 0);
      assert.equal(read(lane), `- [x] ${INITECH} | Initech | Principal Engineer\n`);
      assert.deepEqual(pendingUrls(root), [ACME, GLOBEX, INITECH], 'the default inbox was touched');
    });
  });

  test('parsePipeline reads pending rows the way the tick finds them again', () => {
    const rows = parsePipeline([
      '## Pending',
      '- [ ] https://jobs.example.com/a/1 | Acme | Engineer | Remote (US)',
      '- [ ] https://jobs.example.com/b/2\r',
      '- [x] https://jobs.example.com/c/3 | Done | Done',
      '- [!] https://jobs.example.com/d/4 — Error: login required',
      '  - [ ] https://jobs.example.com/e/5 | Indented | Not a row',
      '- [ ] ',
    ].join('\n'));
    assert.deepEqual(rows, [
      { url: 'https://jobs.example.com/a/1', company: 'Acme', role: 'Engineer' },
      // A file saved with Windows line endings must not carry the CR into the URL.
      { url: 'https://jobs.example.com/b/2', company: 'Unknown', role: 'Unknown' },
    ]);
  });
});

describe('runPipeline', () => {
  const items = ['a', 'b', 'c', 'd', 'e'].map((name) => ({ url: `https://jobs.example.com/${name}`, company: name, role: 'Engineer' }));
  const unreachable = { success: false, reason: 'Ollama not reachable at http://localhost:11434.', code: 'UNREACHABLE' };
  const ok = { success: true, relPath: 'reports/001-x.md' };

  /** Run with `outcomes[i]` as the result of the i-th evaluation; a function is called and may throw. */
  async function run(outcomes, options = {}) {
    const evaluated = [];
    const ticked = [];
    const marked = [];
    const events = [];
    const result = await runPipeline({
      items,
      evaluateFn: async (item) => {
        evaluated.push(item.url);
        const outcome = outcomes[evaluated.length - 1];
        return typeof outcome === 'function' ? outcome() : outcome;
      },
      markDoneFn: async (url) => {
        if (options.tickFails === url) throw new Error('lock timeout');
        ticked.push(url);
      },
      markFailedFn: async (item, why) => {
        if (options.markFails === item.url) throw new Error('lock timeout');
        marked.push([item.url, why]);
      },
      // `gone` holds the rows another program took out of the queue meanwhile.
      ...(options.gone ? { stillPendingFn: (item) => !options.gone.includes(item.url) } : {}),
      onItem: ({ phase, index }) => { events.push(`${phase} ${index}`); },
      limit: options.limit,
      ...(options.maxUnreadable ? { maxUnreadable: options.maxUnreadable } : {}),
    });
    return { ...result, evaluated, ticked, marked, events };
  }

  test('a row is ticked only when its evaluation succeeded', async () => {
    const failure = { success: false, reason: 'Reply rejected after 2 attempts: no SCORE_SUMMARY block.', code: 'EMPTY' };
    const result = await run([ok, failure, ok, () => { throw new Error('disk full'); }, ok]);
    assert.deepEqual(result.ticked, [items[0].url, items[2].url, items[4].url]);
    assert.deepEqual(result.done.map(({ item }) => item.url), result.ticked);
    assert.deepEqual(result.failed.map(({ item, reason }) => [item.url, reason]), [
      [items[1].url, failure.reason],
      // An evaluation that throws is one failed posting, not the end of the run.
      [items[3].url, 'disk full'],
    ]);
    assert.equal(result.stoppedEarly, false);
    assert.deepEqual(result.events, ['start 0', 'done 0', 'start 1', 'failed 1', 'start 2', 'done 2', 'start 3', 'failed 3', 'start 4', 'done 4']);
  });

  test('two unreachable-server failures in a row end the run', async () => {
    const result = await run([ok, unreachable, unreachable, ok, ok]);
    // The fourth and fifth were never tried: each would have fetched its page first.
    assert.deepEqual(result.evaluated, items.slice(0, 3).map((item) => item.url));
    assert.deepEqual(result.ticked, [items[0].url]);
    assert.equal(result.failed.length, 2);
    assert.equal(result.stoppedEarly, true);
  });

  test('a missing model counts the same, and any other outcome in between resets the count', async () => {
    const missing = { success: false, reason: 'Model "x" was not found by Ollama.', code: 'MODEL_NOT_FOUND' };
    const stopped = await run([unreachable, missing, ok]);
    assert.equal(stopped.evaluated.length, 2);
    assert.equal(stopped.stoppedEarly, true);

    const page = { success: false, reason: 'Could not read the job page: http_404', code: 'JOB_PAGE' };
    const carried = await run([unreachable, page, unreachable, ok, unreachable]);
    assert.equal(carried.evaluated.length, 5, 'failures that were not in a row ended the run');
    assert.equal(carried.stoppedEarly, false);
    assert.deepEqual(carried.ticked, [items[3].url]);
  });

  test('--limit counts postings that reached the model, and a tick that fails is reported as a failure', async () => {
    const limited = await run([ok, ok, ok, ok, ok], { limit: 2 });
    assert.deepEqual(limited.ticked, [items[0].url, items[1].url]);
    assert.equal(limited.stoppedEarly, false);

    const stuck = await run([ok, ok], { limit: 2, tickFails: items[1].url });
    assert.deepEqual(stuck.done.map(({ item }) => item.url), [items[0].url]);
    assert.equal(stuck.failed[0].code, 'NOT_TICKED');
    assert.match(stuck.failed[0].reason, /could not be ticked: lock timeout/);

    // Two rows whose page cannot be read stand before the two that can. They
    // are tried and do not count: the limit is for evaluations.
    const page = { success: false, reason: 'Could not read the job page: HTTP 503', code: 'JOB_PAGE' };
    const past = await run([page, page, ok, ok, ok], { limit: 2 });
    assert.deepEqual(past.ticked, [items[2].url, items[3].url]);
    assert.equal(past.evaluated.length, 4);
    // A reply that was rejected did reach the model, and uses a place.
    const rejected = { success: false, reason: 'Reply rejected after 2 attempts.', code: 'EMPTY' };
    assert.equal((await run([rejected, ok, ok], { limit: 2 })).evaluated.length, 2);
  });

  test('a row that can never succeed is marked once, and a failed mark leaves it pending', async () => {
    const closed = { success: false, reason: 'Could not read the job page: HTTP 404', code: 'JOB_PAGE', retire: 'posting closed (HTTP 404)' };
    const result = await run([closed, ok, closed], { limit: 1, markFails: items[2].url });
    assert.deepEqual(result.marked, [[items[0].url, 'posting closed (HTTP 404)']]);
    assert.deepEqual(result.failed.map((failure) => failure.marked === true), [true]);
    assert.deepEqual(result.events, ['start 0', 'failed 0', 'start 1', 'done 1']);
    // The mark could not be written (a lock, a read-only file): the row is reported failed, not marked.
    const unmarked = await run([closed, closed, closed, ok, ok], { markFails: items[1].url });
    assert.deepEqual(unmarked.failed.map((failure) => failure.marked === true), [true, false, true]);
    assert.deepEqual(unmarked.ticked, [items[3].url, items[4].url]);
  });

  test('a row that is no longer pending when its turn comes is skipped', async () => {
    // Another program ticked b and d while a was being evaluated.
    const result = await run([ok, ok, ok], { gone: [items[1].url, items[3].url] });
    assert.deepEqual(result.evaluated, [items[0].url, items[2].url, items[4].url]);
    assert.deepEqual(result.skipped.map((item) => item.url), [items[1].url, items[3].url]);
    assert.deepEqual(result.events, ['start 0', 'done 0', 'skipped 1', 'start 2', 'done 2', 'skipped 3', 'start 4', 'done 4']);
  });

  test('two timeouts in a row, or two refusals by the privacy guard, end the run and say why', async () => {
    const timeout = { success: false, reason: 'Request timed out after 900s. Raise OLLAMA_TIMEOUT_MS.', code: 'TIMEOUT' };
    const timedOut = await run([timeout, timeout, ok]);
    // The third row was never tried: it would have cost its page and another fifteen minutes.
    assert.equal(timedOut.evaluated.length, 2);
    assert.equal(timedOut.stoppedEarly, true);
    assert.equal(timedOut.stopReason, 'two postings in a row failed the same way, and the rest would too: Request timed out after 900s. Raise OLLAMA_TIMEOUT_MS.');

    for (const code of ['PRIVACY_GUARD_BLOCKED', 'PRIVACY_GUARD_CONFIG']) {
      const refusal = { success: false, reason: 'Privacy guard blocked an outbound LLM request', code };
      const refused = await run([refusal, refusal, ok]);
      assert.equal(refused.evaluated.length, 2, code);
      assert.equal(refused.stoppedEarly, true, code);
    }
    // One of them and then a success: the run goes on, and says nothing about stopping.
    const carried = await run([timeout, ok, timeout, ok, ok]);
    assert.equal(carried.evaluated.length, 5);
    assert.deepEqual([carried.stoppedEarly, carried.stopReason], [false, null]);
  });

  test('a run in which page after page cannot be read stops instead of walking the whole inbox', async () => {
    const page = { success: false, reason: 'Could not read the job page: fetch failed', code: 'JOB_PAGE' };
    const result = await run([page, page, page, page, page], { maxUnreadable: 3 });
    assert.equal(result.evaluated.length, 3);
    assert.equal(result.stoppedEarly, true);
    assert.match(result.stopReason, /3 postings could not be read\. Is this machine online\?/);
    // The cap reached on the last row cut nothing short.
    assert.equal((await run([page, page, page, page, page], { maxUnreadable: 5 })).stoppedEarly, false);
    // A posting found closed was read, and a row that is no URL cost nothing: neither says the machine is offline.
    const closed = { success: false, reason: 'Could not read the job page: HTTP 404', code: 'JOB_PAGE', retire: 'posting closed (HTTP 404)' };
    const notUrl = { success: false, reason: 'Not a complete http(s) URL: "local:jds/x.md"', code: 'BAD_INPUT' };
    const walked = await run([closed, notUrl, closed, notUrl, closed], { maxUnreadable: 2 });
    assert.equal(walked.evaluated.length, 5);
    assert.equal(walked.stoppedEarly, false);
  });

  test('rows that fail before the model is asked leave --limit untouched, whatever their number', async () => {
    const notUrl = { success: false, reason: 'Not a complete http(s) URL: "local:jds/x.md"', code: 'BAD_INPUT' };
    const result = await run([notUrl, notUrl, notUrl, ok, ok], { limit: 1 });
    assert.deepEqual(result.ticked, [items[3].url]);
    assert.equal(result.evaluated.length, 4);
  });
});

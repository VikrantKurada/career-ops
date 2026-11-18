// tests/ollama-fork-guard.test.mjs — the wiring of this fork's local Ollama
// backend is still in place, and its guide still describes the code.
//
// The backend itself (ollama-runner.mjs and lib/ollama-*.mjs) has suites of its
// own. This one guards the few lines the fork had to add to files upstream
// owns, because nothing else notices when they go missing:
//
//   - `node update-system.mjs apply --force` overwrites every locally edited
//     system file with upstream's copy. package.json then has one ollama
//     script, upstream's `ollama:eval`, which runs the legacy evaluator,
//     doctor.mjs has no --ollama flag, and the runner suites still pass, since
//     they start the runner with `node`, not through npm.
//   - A merge of upstream/main resolved the wrong way drops the same hunks, or
//     leaves upstream's `ollama:eval` line beside the fork's.
//   - config/local-paths.txt is what keeps validate-system-paths-coverage.mjs
//     green for the four backend files and the agent rule, and makes the
//     updater stop if a release ever ships a file of the same name (it ships
//     none today, so it does not write to them either way). It only reaches a
//     clone or CI because .gitignore un-ignores it, and that negation only
//     works while it comes after upstream's ignore rule.
//   - .claude/rules/ollama-fork.md is the one place that tells an agent not to
//     run the updater here. A file nobody misses until the updater has run.
//   - A merge updates .agents/skills/career-ops/SKILL.md and not the six
//     copies of it that a checkout without symlinks holds, so six agent tools
//     keep the /career-ops command from before the merge. And an entry git
//     has just written there (a fresh clone, the `git checkout --` that comes
//     before a merge) holds the link's target and nothing else, so that tool
//     has no /career-ops at all until the copies are written again.
//
// It also ties docs/OLLAMA.md to the code. The guide is what the owner follows
// in place of an agent, so a default, a message or a command line that the page
// states and the code no longer has is a defect of its own: the commands that
// were found to go wrong in Windows PowerShell must not come back, and what the
// page quotes must still be what is printed.
//
// Every check reads files. Nothing is started and no network is used.
//
// Run: node --test tests/ollama-fork-guard.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const at = (rel) => join(ROOT, ...rel.split('/'));
const read = (rel) => readFileSync(at(rel), 'utf8');

// The npm surface the docs and .env.example describe. `ollama:eval` used to run
// upstream's ollama-eval.mjs, so its exact command is part of the contract.
const SCRIPTS = {
  ollama: 'node ollama-runner.mjs',
  'ollama:eval': 'node ollama-runner.mjs evaluate',
  'ollama:scan': 'node ollama-runner.mjs scan',
  'ollama:pipeline': 'node ollama-runner.mjs pipeline',
  'ollama:tailor': 'node ollama-runner.mjs tailor',
  'ollama:pdf': 'node ollama-runner.mjs pdf',
  'ollama:apply': 'node ollama-runner.mjs apply',
  'ollama:run': 'node ollama-runner.mjs run',
  'ollama:models': 'node ollama-runner.mjs models',
  'ollama:doctor': 'node ollama-runner.mjs doctor',
};

// Root and lib files upstream does not ship. Listed one by one, like the
// updater's own manifest, so a file that is renamed fails here by name.
const BACKEND_FILES = [
  'ollama-runner.mjs',
  'lib/ollama-config.mjs',
  'lib/ollama-client.mjs',
  'lib/ollama-doctor.mjs',
];

// The fork's rule for agents. Claude Code loads every rule under
// .claude/rules/ that has no `paths` front matter at the start of a session,
// beside CLAUDE.md. It is a file of
// its own because upstream's gate allows CLAUDE.md to hold nothing but the
// `@AGENTS.md` line and one comment (test-all.mjs, #1088): with the rule
// written into CLAUDE.md, two checks of the full suite failed on this fork.
const AGENT_RULE = '.claude/rules/ollama-fork.md';

// What config/local-paths.txt has to declare: the updater's manifest knows
// .claude/skills/ only, so the rule is a file upstream does not ship, like
// the backend.
const DECLARED_FILES = [...BACKEND_FILES, AGENT_RULE];

const ENV_VARS = [
  'OLLAMA_BASE_URL', 'OLLAMA_HOST', 'OLLAMA_ALLOW_REMOTE', 'OLLAMA_MODEL',
  'OLLAMA_MODEL_PREFERENCE', 'OLLAMA_NUM_CTX', 'OLLAMA_NUM_PREDICT', 'OLLAMA_THINK',
  'OLLAMA_KEEP_ALIVE', 'OLLAMA_TEMPERATURE', 'OLLAMA_TIMEOUT_MS', 'OLLAMA_RETRIES',
  'OLLAMA_SEED',
];

const GUIDE = 'docs/OLLAMA.md';

const POINTER_DOCS = [
  'docs/RUNNING_ON_A_BUDGET.md',
  'docs/SCRIPTS.md',
  'docs/PRIVACY.md',
  'docs/SUPPORTED_CLIS.md',
];

const LOCAL_PATHS = 'config/local-paths.txt';
const NEGATION = `!${LOCAL_PATHS}`;

// What upstream ships under a name that begins with `ollama` and this fork
// leaves alone: today the four suites for its own ollama-eval.mjs, no test
// helper and no npm script. (Its one script of that kind is `ollama:eval`, a
// name the fork took over: see SCRIPTS.) A suite, a helper or a script named
// like that which arrives with a merge is upstream's as well, and is added
// here: these sets, not the guide's lists, are what has to grow. An entry is
// spelled the way the failing test prints it.
const UPSTREAM_SUITES = new Set(['ollama-eval', 'ollama-eval-endpoint', 'ollama-eval-exit', 'ollama-profile-context']);
// A path from the checkout root, like 'tests/helpers/ollama-eval-stub.mjs'.
const UPSTREAM_HELPERS = new Set([]);
// The name npm runs it by, like 'ollama:bench'. Never one of the ten in SCRIPTS.
const UPSTREAM_SCRIPTS = new Set([]);

/** Entries of the declaration file, read the way the updater reads it:
 *  one path per line, `#` comments, blank lines and a trailing CR ignored. */
function declaredPaths() {
  return read(LOCAL_PATHS)
    .split('\n')
    .map((line) => line.replace(/\r$/, '').trim())
    .filter((line) => line && !line.startsWith('#'));
}

/** True when `name` appears in `text` as a whole name, so OLLAMA_NUM_CTX does
 *  not count as a mention of OLLAMA_NUM and `ollama` does not match `ollama:eval`. */
function mentions(text, name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(?<![\\w:-])${escaped}(?![\\w:-])`).test(text);
}

/** The lines of a file, each with its number, without a trailing CR. */
function numberedLines(rel) {
  return read(rel).split('\n').map((line, index) => ({ line: line.replace(/\r$/, ''), at: index + 1 }));
}

/** One section of the guide: from the heading that holds `title` to the next
 *  heading of the same or a higher level. '' when there is no such heading. */
function guideSection(title) {
  const lines = read(GUIDE).split('\n');
  const start = lines.findIndex((line) => /^#{2,3} /.test(line) && line.includes(title));
  if (start === -1) return '';
  const level = lines[start].match(/^#+/)[0].length;
  const rest = lines.slice(start + 1);
  const end = rest.findIndex((line) => new RegExp(`^#{2,${level}} `).test(line));
  return (end === -1 ? rest : rest.slice(0, end)).join('\n');
}

/** The names of the suites this fork adds under tests/, without `.test.mjs`. */
function forkSuites() {
  return readdirSync(at('tests'))
    .map((name) => /^(ollama-[\w-]+)\.test\.mjs$/.exec(name)?.[1])
    .filter((name) => name && !UPSTREAM_SUITES.has(name));
}

/** The helpers those suites share, as paths from the checkout root. A `.bak` copy beside one is not a helper. */
function forkTestHelpers() {
  return readdirSync(at('tests/helpers'))
    .filter((name) => /^ollama-.*\.mjs$/.test(name))
    .map((name) => `tests/helpers/${name}`)
    .filter((path) => !UPSTREAM_HELPERS.has(path));
}

/** The `ollama` and `ollama:*` scripts of package.json that upstream did not add: the ones the fork's guide has to show. */
function forkScripts() {
  return Object.keys(JSON.parse(read('package.json')).scripts ?? {})
    .filter((name) => (name === 'ollama' || name.startsWith('ollama:')) && !UPSTREAM_SCRIPTS.has(name));
}

/** The end of a message about a name the guide does not have: where that name goes when upstream, not the fork, added the thing. */
function orUpstreams(set) {
  return `If it came with a merge it is upstream's: add it to ${set} in this file, spelled as it is printed here, not to the guide.`;
}

test('package.json has the ten ollama scripts, each one running ollama-runner.mjs', () => {
  const scripts = JSON.parse(read('package.json')).scripts ?? {};
  const found = Object.fromEntries(Object.keys(SCRIPTS).map((name) => [name, scripts[name]]));
  assert.deepEqual(
    found,
    SCRIPTS,
    'package.json lost or changed an ollama script. If `update-system.mjs apply --force` or a merge '
      + 'replaced the scripts block, put back the ten entries listed in SCRIPTS at the top of this file.',
  );
  // One name runs one command, and each of the ten runs the fork's. Listed as
  // upstream's, a name would no longer have to be in the guide.
  const claimed = Object.keys(SCRIPTS).filter((name) => UPSTREAM_SCRIPTS.has(name));
  assert.deepEqual(claimed, [], `UPSTREAM_SCRIPTS names ${claimed.join(', ')}, which SCRIPTS lists as this fork's: take it out of UPSTREAM_SCRIPTS. `
    + 'That set is for a script upstream added under a name of its own.');
});

test('package.json names each ollama script once', () => {
  // A merge conflict next to `ollama:eval` holds upstream's own line for it in
  // upstream's half. Kept beside the fork's, the file still parses: JSON takes
  // the last of two keys, so the check above passes when the fork's line comes
  // last and `npm run ollama:eval` starts the legacy evaluator when it does not.
  const text = read('package.json');
  // Counted in the scripts block alone: a dependency called `ollama` would be
  // a second `"ollama":` in the file, and is no script. Every line of the block
  // begins with a name, so the block ends at the first line that begins with a
  // closing brace.
  const block = /"scripts"\s*:\s*\{[\s\S]*?\n\s*\}/.exec(text)?.[0] ?? text;
  const twice = Object.keys(SCRIPTS).filter((name) => block.split(`"${name}":`).length - 1 !== 1);
  assert.deepEqual(twice, [], `package.json has more or fewer than one entry for: ${twice.join(', ')}. After a merge conflict, `
    + 'keep the fork\'s ten lines and delete upstream\'s line of the same name (today its one such line is '
    + '`"ollama:eval": "node ollama-eval.mjs"`). A script upstream added under a name of its own stays; '
    + 'if that name begins with ollama, add it to UPSTREAM_SCRIPTS in this file.');
});

test('the script every ollama npm command starts exists', () => {
  assert.ok(existsSync(at('ollama-runner.mjs')), 'ollama-runner.mjs is missing, so every `npm run ollama:*` fails');
});

test('config/local-paths.txt declares the four backend files and the agent rule', () => {
  const declared = declaredPaths();
  const missing = DECLARED_FILES.filter((file) => !declared.includes(file));
  assert.deepEqual(missing, [], `not declared in ${LOCAL_PATHS}: ${missing.join(', ')}. `
    + 'An undeclared file fails validate-system-paths-coverage.mjs once it is committed.');
});

test('config/local-paths.txt lists nothing the system layer ships', () => {
  // docs/ and tests/ are system directories. One entry under either makes
  // update-system.mjs refuse the whole file, which stops `apply` and the
  // coverage guard, not only that one entry.
  const refused = declaredPaths().filter((path) => /^(docs|tests)(\/|$)/.test(path));
  assert.deepEqual(refused, [], `${LOCAL_PATHS} must not list docs/ or tests/ paths: ${refused.join(', ')}`);
});

test('every declared file exists', () => {
  // A declaration for a file that is gone protects nothing and hides the loss:
  // the coverage guard only looks at files that are still tracked.
  const missing = DECLARED_FILES.filter((file) => !existsSync(at(file)));
  assert.deepEqual(missing, [], `declared in ${LOCAL_PATHS} but not on disk: ${missing.join(', ')}`);
});

test('the updater accepts the declaration and treats the backend and the agent rule as user layer', async () => {
  // The real function, not a copy of its rules: apply() reaches it through
  // effectiveUserPaths() and the coverage guard calls it directly. It throws,
  // naming the entry, when a declared path collides with something upstream ships.
  const { localUserPaths } = await import('../update-system.mjs');
  const protectedPaths = localUserPaths(ROOT);
  const unprotected = DECLARED_FILES.filter((file) => !protectedPaths.includes(file));
  assert.deepEqual(unprotected, [], `not user layer for update-system.mjs: ${unprotected.join(', ')}`);
});

test('.gitignore un-ignores config/local-paths.txt after the rule that ignores it', () => {
  const lines = read('.gitignore').split('\n').map((line) => line.replace(/\r$/, '').trim());
  const ignoredAt = lines.lastIndexOf(LOCAL_PATHS);
  const negatedAt = lines.lastIndexOf(NEGATION);
  assert.notEqual(negatedAt, -1, `.gitignore has no "${NEGATION}" line, so the declaration file cannot be committed`);
  // Deleting upstream's rule looks tidy and is a trap: the updater appends any
  // upstream rule it finds missing at the END of the file, after the negation,
  // and the declaration file is ignored again.
  assert.notEqual(ignoredAt, -1, `keep upstream's "${LOCAL_PATHS}" rule in .gitignore, with "${NEGATION}" after it`);
  // Git applies the last pattern that matches, so the negation has to follow the rule.
  assert.ok(
    negatedAt > ignoredAt,
    `"${NEGATION}" (line ${negatedAt + 1}) must come after "${LOCAL_PATHS}" (line ${ignoredAt + 1})`,
  );
});

test('git itself does not ignore config/local-paths.txt', (t) => {
  // The line-order check above cannot see a broader pattern added later, such
  // as `config/*.txt`. Git can. --no-index asks about the rules alone: without
  // it a tracked file is never reported as ignored, whatever the rules say.
  const result = spawnSync('git', ['check-ignore', '-q', '--no-index', LOCAL_PATHS], { cwd: ROOT, encoding: 'utf8' });
  if (result.status === 1) return; // git's answer for "not ignored"
  if (result.status === 0) {
    assert.fail(`git ignores ${LOCAL_PATHS}: a pattern after "${NEGATION}" in .gitignore matches it again`);
  }
  // No git, or not a git checkout (a copied tree): the question cannot be asked here.
  t.skip(`git could not answer (${(result.stderr || result.error?.message || `exit ${result.status}`).trim()})`);
});

test('doctor.mjs still has the --ollama flag', () => {
  assert.ok(
    /--ollama(?![\w-])/.test(read('doctor.mjs')),
    'doctor.mjs does not mention --ollama. It is an upstream file: check whether an update or merge replaced it.',
  );
});

test('the agent rule tells an agent to sync this fork with git and not to run the updater', () => {
  // AGENTS.md, which CLAUDE.md imports, has every agent session check for a
  // release and offer `node update-system.mjs apply --confirm`. Run here, that
  // leaves VERSION at the release and package.json and doctor.mjs behind,
  // stages a snapshot on the fork branch and makes `git merge upstream/main`
  // refuse; the rollback it then offers puts upstream's text back over every
  // fork edit that is not committed. Nothing but this file tells an agent
  // that this checkout is synced another way.
  //
  // A lost or reworded rule comes back from git, and from nowhere before the
  // fork is committed: the guide describes what the updater does here and
  // holds none of the rule's sentences. What they have to say is the list
  // below.
  const restore = 'Restore it from git once it is committed; docs/OLLAMA.md, "Do not run the updater", says what it has to forbid, '
    + 'and the required list in this test what it has to say.';
  assert.ok(existsSync(at(AGENT_RULE)), `${AGENT_RULE} is missing: no agent is told to leave the updater alone. ${restore}`);
  const lines = read(AGENT_RULE).split('\n').map((line) => line.replace(/\r$/, ''));
  // A rule with a `paths:` list in its front matter is loaded only once a file
  // on that list has been opened. This one has to be there from the first
  // message, which is when the update check runs.
  assert.notEqual(lines[0].trim(), '---', `${AGENT_RULE} must have no front matter: a rule limited by \`paths:\` is not loaded at the start of a session`);
  // What the agent is shown, not what the file holds: Claude Code leaves HTML
  // comments out of the CLAUDE.md text it puts in front of the model, and may
  // do the same here, so a rule written inside one could reach nobody.
  const shown = lines.join('\n').replace(/<!--[\s\S]*?-->/g, '');
  const required = [
    'git fetch upstream',
    'git merge upstream/main',
    'Never run or offer',
    'node update-system.mjs apply',
    'node update-system.mjs rollback',
    'npm run update',
    'npm run rollback',
    'update-available',
    'node update-system.mjs dismiss --version',
    // The mode behind `/career-ops update` runs the updater too, and is no
    // part of the section the rule names.
    'modes/update.md',
    // Whole clauses from here on, not single words: a rule that kept the
    // words and said the opposite ("do not run that mode without asking
    // first", "in an unattended run, announce the update and dismiss it")
    // passed while only the words were looked for.
    //
    // Two rules that contradict each other are worse than one: AGENTS.md says
    // to offer the update, so this file has to say which of the two wins.
    'takes precedence over the "Update Check" section of AGENTS.md',
    // Naming the mode does not keep an agent out of it: asked to update, it
    // would open the mode and follow it.
    'do not run that mode: the release is taken with the two git commands above',
    // The check looks at releases and the merge takes main, so "up to date"
    // from the check is no answer to "update".
    'the merge takes upstream\'s `main`, and the check looks at releases only',
    // What stays in force of the section the rule overrides.
    'Three things of that section stay in force',
    '`node update-system.mjs check --force` when the user asks for a check',
    // A subagent or a `claude -p` run has nobody to tell. Without this it
    // would dismiss a release no one has been told of, and no later session
    // announces a dismissed release. The bullet on the announcement says so
    // in its own words too: read alone, "without waiting for an answer"
    // sounds like the unattended case.
    'in an unattended run, say nothing about any update, apply none and dismiss none',
    'Except in an unattended run, when it, or a check the user asked for, reports `update-available`',
    // The announcement comes first, in the reply itself.
    'Say first, in your reply',
    // The dismissal belongs to the turn that announces the release. Left for
    // after the user's answer, it never runs when none comes, and every later
    // session announces the same release.
    'In the same turn, without waiting for an answer, run `node update-system.mjs dismiss --version {remote}`',
    // The way back after an update that ran anyway is `rollback`, which the
    // rule forbids: it has to say where the exception is described.
    'If the updater ran anyway',
    // What a merge needs before and after it on a checkout without symlinks.
    'Keeping the fork in sync with upstream',
  ];
  const missing = required.filter((text) => !shown.includes(text));
  assert.deepEqual(missing, [], `${AGENT_RULE} no longer says: ${missing.join(' | ')}. ${restore}`);
  // The sections the rule sends an agent to exist under those names.
  const guide = read(GUIDE);
  for (const heading of ['## Keeping the fork in sync with upstream', '**If the updater ran anyway**']) {
    assert.ok(guide.includes(heading), `${AGENT_RULE} points at "${heading}", which docs/OLLAMA.md no longer has`);
  }
  // The section the messages above send the owner to still names what the
  // rule has to forbid.
  const section = guideSection('Do not run the updater');
  const unnamed = ['npm run update', 'node update-system.mjs apply', 'node update-system.mjs rollback']
    .filter((text) => !section.includes(text));
  assert.deepEqual(unnamed, [], `the "Do not run the updater" section of docs/OLLAMA.md no longer names: ${unnamed.join(' | ')}. `
    + 'The messages of this test send the owner there for what the rule has to forbid.');
  // The rule is a file of its own because upstream's gate allows CLAUDE.md
  // nothing but the `@AGENTS.md` line. Written back there, it fails two checks
  // of the full gate and none that `--only ollama` runs.
  const wrapper = read('CLAUDE.md');
  const strayed = ['git merge upstream/main', 'update-system.mjs'].filter((text) => wrapper.includes(text));
  assert.deepEqual(strayed, [], `CLAUDE.md holds text of the fork's rule (${strayed.join(', ')}): it belongs in ${AGENT_RULE}`);
});

test('the copies of the /career-ops skill are the skill as it is now', async (t) => {
  // Upstream tracks six entry files, one per agent tool, as symlinks to
  // .agents/skills/career-ops/SKILL.md. A checkout that cannot make symlinks
  // holds a copy of that file in each. `git merge upstream/main` updates the
  // file and not the copies, nothing but the full gate notices, and six tools
  // go on offering the /career-ops of before the merge.
  let entries;
  let canonicalPath;
  try {
    ({ SKILL_ENTRYPOINTS: entries, CANONICAL_SKILL_PATH: canonicalPath } = await import('../scaffolder/bin/skill-entrypoints.mjs'));
  } catch {
    entries = undefined;
  }
  if (!Array.isArray(entries) || typeof canonicalPath !== 'string' || !existsSync(at(canonicalPath))) {
    t.skip('upstream no longer lists its skill entry files in scaffolder/bin/skill-entrypoints.mjs');
    return;
  }
  const canonical = read(canonicalPath);
  // What writes the copies: the last line of the block the guide gives for a
  // merge. Both messages below print it and say where it stands.
  const writeCopies = 'node -e "import(\'./scaffolder/bin/skill-entrypoints.mjs\').then(m=>console.log(m.materializeSkillEntrypoints(\'.\')))"';
  const stale = [];
  const pointers = [];
  for (const { path, pointer } of entries) {
    if (!existsSync(at(path))) continue;
    // A symlink reads as the file it points to, so it is the skill as it is now.
    const text = read(path);
    if (text === canonical) continue;
    // Where git cannot make the symlink it writes the link's target into the
    // file, 43 bytes: on a fresh clone, and at the `git checkout --` that
    // comes before a merge. The line that writes the copies again is easily
    // left out after a merge that stopped on a conflict. Upstream's doctor
    // only warns about such an entry, and the two skill checks of the full
    // gate take it for a link, so this is the check that fails while six
    // tools load that one line as their /career-ops.
    (text.trim() === pointer ? pointers : stale).push(path);
  }
  assert.deepEqual(stale, [], `not the current ${canonicalPath}: ${stale.join(', ')}. Run \`git checkout -- ${stale.join(' ')}\` and then `
    + `\`${writeCopies}\` (docs/OLLAMA.md, "The six skill files").`);
  const one = pointers.length === 1;
  assert.deepEqual(pointers, [], `${pointers.join(', ')} hold${one ? 's' : ''} only the link target, so ${one ? 'that tool loads' : 'those tools load'} `
    + `an empty /career-ops. Run \`${writeCopies}\`, the last line of the block in docs/OLLAMA.md, "The six skill files".`);
  // Both messages name that line of the guide, so the block has to end with it.
  assert.ok(guideSection('The six skill files').replace(/\r/g, '').includes(`${writeCopies}\n\`\`\``),
    'the block in the "The six skill files" section of docs/OLLAMA.md no longer ends with the line that writes the skill copies, '
      + 'and the messages of this test send the owner to its last line');
});

test('.env.example documents every OLLAMA_* variable and sets none of them', () => {
  const example = read('.env.example');
  const undocumented = ENV_VARS.filter((name) => !mentions(example, name));
  assert.deepEqual(undocumented, [], `.env.example does not mention: ${undocumented.join(', ')}`);

  // People copy this file to .env as it is. A live OLLAMA_NUM_CTX line would pin
  // the context window and switch off the runner's automatic sizing, and a live
  // line for any of the runner's own variables (RUNNER_ENV_VARS in
  // lib/ollama-config.mjs) makes a plain `node doctor.mjs` start contacting the
  // server. OLLAMA_HOST and OLLAMA_KEEP_ALIVE do not switch that on, and they
  // stay commented out all the same: a copied file must change nothing.
  const active = example.split('\n').filter((line) => /^\s*(?:export\s+)?OLLAMA_[A-Z0-9_]*\s*=/.test(line));
  assert.deepEqual(active, [], 'every OLLAMA_* line in .env.example must stay commented out');
});

test('docs/OLLAMA.md covers every ollama script and every OLLAMA_* variable', async () => {
  const guide = read(GUIDE);
  const undocumentedScripts = forkScripts().filter((name) => !mentions(guide, `npm run ${name}`));
  assert.deepEqual(undocumentedScripts, [], `docs/OLLAMA.md has no "npm run" line for: ${undocumentedScripts.join(', ')}. `
    + orUpstreams('UPSTREAM_SCRIPTS'));

  const undocumentedVars = ENV_VARS.filter((name) => !mentions(guide, name));
  assert.deepEqual(undocumentedVars, [], `docs/OLLAMA.md does not mention: ${undocumentedVars.join(', ')}`);

  // The list above is typed by hand. The runner's own is the code's, so a
  // variable the runner starts to read has to be added here, and with that to
  // the guide and to .env.example.
  const { RUNNER_ENV_VARS } = await import('../lib/ollama-config.mjs');
  const unlisted = RUNNER_ENV_VARS.filter((name) => !ENV_VARS.includes(name));
  assert.deepEqual(unlisted, [], `the runner reads ${unlisted.join(', ')}: add it to ENV_VARS in this file`);
});

test('docs/OLLAMA.md names every variable that switches the Ollama checks on in a plain doctor run', async () => {
  // `npm run doctor` contacts the server only for someone who set one of the
  // runner's own variables. The page once said "any OLLAMA_* variable", which
  // was true of the code then and sent a reader with OLLAMA_HOST set looking
  // for requests that are no longer made.
  const { RUNNER_ENV_VARS } = await import('../lib/ollama-config.mjs');
  const section = guideSection('`ollama:doctor`');
  assert.notEqual(section, '', 'docs/OLLAMA.md has no `ollama:doctor` section');
  const unnamed = RUNNER_ENV_VARS.filter((name) => !mentions(section, name));
  assert.deepEqual(unnamed, [], `the ollama:doctor section of docs/OLLAMA.md does not name: ${unnamed.join(', ')}`);
  for (const serverOwn of ['OLLAMA_HOST', 'OLLAMA_KEEP_ALIVE']) {
    assert.ok(!RUNNER_ENV_VARS.includes(serverOwn) && mentions(section, serverOwn),
      `the ollama:doctor section must say that ${serverOwn} does not count, and the code must agree`);
  }
});

test('the defaults docs/OLLAMA.md and .env.example state are the ones the code uses', async () => {
  const { DEFAULTS, DEFAULT_MODEL_PREFERENCE, resolveThink } = await import('../lib/ollama-config.mjs');
  const expected = {
    OLLAMA_BASE_URL: DEFAULTS.baseUrl,
    OLLAMA_MODEL_PREFERENCE: DEFAULT_MODEL_PREFERENCE.join(','),
    OLLAMA_NUM_CTX: String(DEFAULTS.numCtx),
    OLLAMA_NUM_PREDICT: String(DEFAULTS.numPredict),
    OLLAMA_THINK: DEFAULTS.think,
    // No default in the code means no keep_alive is sent, and both files have
    // to say "unset": a duration here would be read as what the runner asks for.
    OLLAMA_KEEP_ALIVE: 'keepAlive' in DEFAULTS ? String(DEFAULTS.keepAlive) : 'unset',
    OLLAMA_TEMPERATURE: String(DEFAULTS.temperature),
    OLLAMA_TIMEOUT_MS: String(DEFAULTS.timeoutMs),
    OLLAMA_RETRIES: String(DEFAULTS.retries),
  };

  // The Configuration table of the guide: | `NAME` | default | what it does |
  const inGuide = {};
  for (const line of read(GUIDE).split('\n')) {
    const cells = line.split('|').map((cell) => cell.trim());
    const name = /^`(OLLAMA_[A-Z_]+)`$/.exec(cells[1] ?? '')?.[1];
    if (name && name in expected && !(name in inGuide)) inGuide[name] = cells[2].replace(/`/g, '');
  }
  assert.deepEqual(inGuide, expected, 'the Default column of the Configuration table in docs/OLLAMA.md');

  // .env.example shows each default on a commented line: # NAME=value. Where
  // there is none, the line is an example and has to say so.
  const withDefault = Object.fromEntries(Object.entries(expected).filter(([, value]) => value !== 'unset'));
  const inExample = {};
  for (const line of read('.env.example').split('\n')) {
    const shown = /^#\s*(OLLAMA_[A-Z_]+)=(\S+)/.exec(line);
    if (shown && shown[1] in withDefault) inExample[shown[1]] = shown[2];
  }
  assert.deepEqual(inExample, withDefault, 'the commented defaults in .env.example');
  if (expected.OLLAMA_KEEP_ALIVE === 'unset') {
    assert.match(read('.env.example'), /^#\s*OLLAMA_KEEP_ALIVE=\S+\s+#.*\bunset\b/m,
      '.env.example must say that OLLAMA_KEEP_ALIVE is unset by default');
  }

  // `auto` is the default, so what it means is part of the default.
  assert.equal(resolveThink({ setting: 'auto', capabilities: ['completion', 'thinking'] }), false);
  assert.ok(read(GUIDE).includes('`auto` is off'), 'docs/OLLAMA.md must say that `auto` means thinking off');
  // The usage text is the third place that names the default.
  assert.ok(read('ollama-runner.mjs').includes('(default: auto, which leaves it off)'),
    'the --think line of `node ollama-runner.mjs --help` must say that auto means thinking off');
});

test('docs/OLLAMA.md names every test suite and test helper this fork adds', () => {
  // The sync section lists what a merge of upstream has to leave in place. It
  // went stale once: a suite was split in two and the page went on saying
  // "six test suites" with one of them missing from the list.
  const suites = forkSuites();
  assert.ok(suites.includes('ollama-fork-guard'), 'this suite is one of them');
  const section = guideSection('Keeping the fork in sync with upstream');
  const unnamed = suites.filter((name) => !section.includes(`\`${name}\``));
  assert.deepEqual(unnamed, [], `the sync section of docs/OLLAMA.md does not name: ${unnamed.join(', ')}. ${orUpstreams('UPSTREAM_SUITES')}`);
  const unnamedHelpers = forkTestHelpers().filter((path) => !section.includes(`\`${path}\``));
  assert.deepEqual(unnamedHelpers, [], `the sync section of docs/OLLAMA.md does not name: ${unnamedHelpers.join(', ')}. ${orUpstreams('UPSTREAM_HELPERS')}`);
});

test('no file of the fork reads the process entry path', () => {
  // The rule of tests/main-guard-convention.test.mjs: no tracked .mjs file
  // names that value, in code or in a string, because sixty hand-written "am I
  // the entry script?" checks once grew from it. That suite reads tracked
  // files only. It said nothing while the fork's files were new, and failed
  // the full gate once they were staged: two suites of the fork carried the
  // reference in a script they write for a child process. Its pattern, built
  // from parts so that this file does not hold it either.
  const entryRef = new RegExp(`${['process', 'argv'].join('\\.')}(?:\\[1\\]|\\.at\\(\\s*1\\s*\\))`);
  const offenders = [];
  for (const file of [...BACKEND_FILES, ...forkSuites().map((name) => `tests/${name}.test.mjs`), ...forkTestHelpers()]) {
    for (const { line, at: lineNumber } of numberedLines(file)) {
      if (!/^\s*(\/\/|\*|\/\*)/.test(line) && entryRef.test(line)) offenders.push(`${file}:${lineNumber}`);
    }
  }
  assert.deepEqual(offenders, [], 'use isMainModule() from lib/is-main-module.mjs, or do without the entry path');
});

test('the messages docs/OLLAMA.md quotes are still the ones the code prints', () => {
  // The guide shows these so a reader can match what is on screen. Nothing
  // else ties that prose to the code: a reworded message would leave the page
  // describing a line nobody sees any more. Each entry is a piece of one
  // string of the source, so it is found there as it stands.
  const quotedFrom = [
    {
      files: ['lib/ollama-client.mjs', 'lib/ollama-config.mjs'],
      quoted: [
        'Ollama not reachable at',
        'is not installed in Ollama. Installed chat models:',
        'None of the preferred models are installed',
        'the prompt was truncated. Raise OLLAMA_NUM_CTX.',
        'Reply stopped with done_reason',
        'Reply stopped with done_reason "length" before the answer began: ',
        'the model spent its output budget thinking (',
        'Leave thinking off: drop --think on or OLLAMA_THINK=on. Off is the default.',
        'Ollama returned a tool call instead of a text reply.',
        'answered with a redirect',
        'Redirects are not followed: only the configured address was checked, and requests to it carry your CV.',
        'Set OLLAMA_BASE_URL (or --url) to the Ollama server itself.',
        'Raise OLLAMA_TIMEOUT_MS.',
        'Ollama sent nothing for 5 minutes',
      ],
    },
    {
      files: ['ollama-runner.mjs'],
      quoted: [
        'No tracker yet: created',
        '(ten columns, the last one the posting URL).',
        'is not a job description',
        'never reached the runner',
        'is still the shipped template',
        'Not evaluated. The model stopped at a gate',
        'Stopped early:',
        'postings are read from the public web only',
        'The report advises against applying:',
        'Stopping after the evaluation:',
        'the report advises against applying (',
        'Written without brackets',
        // The header line by which the runner knows a report it saved, and
        // the stops as it words them.
        '**Tool:** Ollama runner (',
        'No visa sponsorship (listed by the runner from the tier work_auth: no_sponsorship)',
        'work authorisation: the report gives the tier No sponsorship',
        'No CV for this one, for that reason.',
        'the report advises against applying',
        'no CV: ',
        '(over the limit: shorten it)',
        'Could not release report reservation',
        'Could not save evaluation',
        '<!-- end of the archived job description -->',
      ],
    },
    {
      files: ['doctor.mjs'],
      quoted: ['Ollama checks are not part of this checkout'],
    },
    {
      // Upstream's, and quoted in the troubleshooting section for the fault
      // of Node's rmSync under a non-ASCII path: the lock that is never
      // released says this, a minute later.
      files: ['tracker-utils.mjs', 'pipeline-lock.mjs'],
      quoted: ['Timed out waiting for tracker lock', 'pipeline lock timeout'],
    },
    {
      // Upstream's, and quoted because the tracker section tells the reader
      // what to do when they appear. A merge that rewords one fails here, and
      // the fix is in the guide.
      files: ['merge-tracker.mjs'],
      quoted: ['URL dedup is INACTIVE', 'No applications.md found'],
    },
  ];
  const guide = read(GUIDE);
  for (const { files, quoted } of quotedFrom) {
    const code = files.map(read).join('\n');
    const goneFromGuide = quoted.filter((text) => !guide.includes(text));
    const goneFromCode = quoted.filter((text) => !code.includes(text));
    assert.deepEqual(goneFromGuide, [], `docs/OLLAMA.md no longer quotes: ${goneFromGuide.join(' | ')}`);
    assert.deepEqual(goneFromCode, [], `${files.join(' and ')} no longer print: ${goneFromCode.join(' | ')}. `
      + 'Update docs/OLLAMA.md and this list.');
  }
});

test('no documented command pipes into `npm run ollama:*` or types a posting in quotes', () => {
  // Both were in the guide, both were followed in Windows PowerShell 5.1, and
  // both ended in a saved report about the wrong thing:
  //   - A pipe into `npm run` makes PowerShell drop the `--`, so npm takes
  //     every option after it (`--no-save` saved a report), and the piped text
  //     reaches the runner in ASCII, with `?` for every other character.
  //   - On Windows npm passes on only the first line of quoted text, so a
  //     pasted posting was evaluated and tracked as its title.
  // A quoted URL is fine, and is how an address with `&` in it has to be typed.
  const offending = [];
  for (const doc of [GUIDE, 'docs/SCRIPTS.md', 'docs/RUNNING_ON_A_BUDGET.md']) {
    for (const { line, at: lineNumber } of numberedLines(doc)) {
      // Two of these docs are upstream's. A line of theirs about a script
      // upstream added is not the fork's to rewrite.
      const piped = /\|\s*npm(?:\.cmd)?\s+run\s+(ollama[\w:-]*)/.exec(line)?.[1];
      if (piped && !UPSTREAM_SCRIPTS.has(piped)) offending.push(`${doc}:${lineNumber} pipes into npm run ${piped}`);
      if (/npm run ollama:(?:eval|run) -- ["'](?!https?:\/\/)/.test(line)) offending.push(`${doc}:${lineNumber} types the posting in quotes`);
    }
  }
  assert.deepEqual(offending, [], 'use --file for a posting in every documented command. A line about a script that came with a merge '
    + 'is upstream\'s: add the script to UPSTREAM_SCRIPTS in this file, spelled as it is printed here.');
});

test('the Path B example of docs/RUNNING_ON_A_BUDGET.md names an input', () => {
  // Upstream's example is the bare command. In this fork the bare command
  // waits for a posting on standard input, which reads as a hang to someone
  // who copied the line.
  const bare = numberedLines('docs/RUNNING_ON_A_BUDGET.md')
    .filter(({ line }) => /^\s*npm run ollama:eval\s*$/.test(line))
    .map(({ at: lineNumber }) => lineNumber);
  assert.deepEqual(bare, [], `docs/RUNNING_ON_A_BUDGET.md shows \`npm run ollama:eval\` with no input on line(s) ${bare.join(', ')}`);
  assert.ok(read('docs/RUNNING_ON_A_BUDGET.md').includes('npm run ollama:eval -- --file'),
    'the Path B example must show a posting given with --file');
});

test('docs/OLLAMA.md says how to get a parseable --json receipt under npm run', () => {
  // npm prints its own banner on standard output ahead of the receipt, so the
  // documented `npm run … -- --json` did not give "one JSON object" there.
  const row = read(GUIDE).split('\n').find((line) => line.startsWith('| `--json`'));
  assert.ok(row, 'docs/OLLAMA.md has no `--json` row in its options table');
  assert.ok(row.includes('--silent'), 'the `--json` row of docs/OLLAMA.md must name --silent');
  assert.ok(read(GUIDE).includes('npm run --silent ollama:models -- --json'),
    'docs/OLLAMA.md must show one command whose output parses as it stands');
});

test('docs/OLLAMA.md says how to see and free what the model holds on the GPU', () => {
  const guide = read(GUIDE);
  for (const command of ['ollama ps', 'ollama stop']) {
    assert.ok(guide.includes(command), `docs/OLLAMA.md does not name \`${command}\``);
  }
});

test('the tracker docs/OLLAMA.md shows is the one the runner creates', () => {
  // The page used to show a nine-column tracker. Every merge of a posting read
  // from a URL into that layout prints "URL dedup is INACTIVE", so a reader
  // who created the file from the page got the warning the runner's own
  // tracker avoids.
  const lines = read(GUIDE).split('\n').map((line) => line.replace(/\r$/, ''));
  const headerAt = lines.findIndex((line) => line.startsWith('| # | Date |'));
  assert.notEqual(headerAt, -1, 'docs/OLLAMA.md shows no tracker header');
  const runner = read('ollama-runner.mjs');
  for (const line of [lines[headerAt], lines[headerAt + 1]]) {
    assert.ok(runner.includes(`'${line}'`), `ollama-runner.mjs does not create a tracker with the line ${line}`);
  }
  assert.ok(lines[headerAt].endsWith('| URL |'), 'the tracker in docs/OLLAMA.md must end with the URL column');
  assert.ok(read(GUIDE).includes('node merge-tracker.mjs --backfill-urls'),
    'docs/OLLAMA.md must say how to add the URL column to a tracker that has nine');
});

test('docs/OLLAMA.md warns that the updater cannot be undone before the fork is committed', () => {
  // `rollback` restores what is committed. Run after an update while the
  // fork's edits were uncommitted, it put upstream's text back over all of
  // them and reported "Rollback complete".
  const section = guideSection('Do not run the updater');
  assert.notEqual(section, '', 'docs/OLLAMA.md has no "Do not run the updater" section');
  assert.match(section, /restore committed state only/, 'the section must say that rollback and --force restore committed state only');
  for (const text of ['.bak', 'refs/backup-pre-update-wip/', 'git merge upstream/main']) {
    assert.ok(section.includes(text), `the "Do not run the updater" section of docs/OLLAMA.md does not mention ${text}`);
  }
  assert.ok(read(GUIDE).includes('Commit the fork\'s work first'),
    'docs/OLLAMA.md must tell the reader to commit the fork before syncing');
});

test('docs/OLLAMA.md holds no home-directory path', () => {
  // test-all.mjs fails the whole suite on this substring in any tracked .md
  // (section 7, absolute paths). A Windows path typed with forward slashes is
  // how it gets into a page written on this machine. Built from parts so that
  // this file does not contain it either.
  const needle = ['', 'Users', ''].join('/');
  const hits = read(GUIDE).split('\n')
    .map((line, index) => (line.includes(needle) ? index + 1 : 0))
    .filter(Boolean);
  assert.deepEqual(hits, [], `docs/OLLAMA.md names a home directory on line(s) ${hits.join(', ')}`);
});

test('docs/OLLAMA.md holds no em dash and no en dash', () => {
  // The owner's check of this page counts both. A message of the code that
  // carries one is described in the page's own words instead of quoted.
  const dashes = new RegExp(`[${String.fromCharCode(0x2013, 0x2014)}]`);
  const hits = numberedLines(GUIDE).filter(({ line }) => dashes.test(line)).map(({ at: lineNumber }) => lineNumber);
  assert.deepEqual(hits, [], `docs/OLLAMA.md has an em dash or an en dash on line(s) ${hits.join(', ')}`);
});

test('the upstream docs this fork edited still point at docs/OLLAMA.md', () => {
  const lost = POINTER_DOCS.filter((doc) => !read(doc).includes('OLLAMA.md'));
  assert.deepEqual(lost, [], `no link to OLLAMA.md left in: ${lost.join(', ')}`);
});

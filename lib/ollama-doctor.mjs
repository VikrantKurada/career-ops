/**
 * ollama-doctor.mjs — the setup check for the local Ollama backend.
 *
 * WHY A CHECK OF ITS OWN
 * ----------------------
 * A server that is not running fails loudly, on the first request. The other
 * ways a local run goes wrong are quiet, and each one costs minutes before it
 * shows, or never shows:
 *   - a model that was never pulled: a 404 after the whole prompt was built;
 *   - a context window the prompt does not fit: Ollama drops what does not fit
 *     and answers from the rest, so the run succeeds and the score is wrong;
 *   - a model another program has loaded at a different window size: every
 *     switch between the two is a full reload (8-20 s idle, over a minute on a
 *     busy machine);
 *   - a model that only partly fits on the GPU: the one probed that way wrote
 *     3 tokens per second, against 130-180 for the models that fit, which
 *     looks like a hang.
 * All four can be read from the server's metadata in well under a second,
 * before anything is loaded or generated.
 *
 * WHAT IT SENDS
 * -------------
 * At most four small requests: the version, the model list, one model's
 * details, the loaded models. No prompt, no CV, no job description, and
 * nothing that makes the server load a model or generate a token. On a GPU
 * that other programs share, a check that started a generation would be the
 * slow, disruptive thing it exists to warn about.
 *
 * WHAT IT READS
 * -------------
 * The lengths of the files an evaluation prompt is built from
 * (measurePromptTokens), so that the window is checked against the owner's
 * own prompt and not a typical one. A CV three times the size of the example
 * moves the runner to the next window size, and a check that still assumed
 * the example would pass a setting the runner refuses. Only the lengths are
 * used; the text goes nowhere.
 *
 * WHO CALLS IT
 * ------------
 * doctor.mjs (`--ollama`, or an install that has set one of the runner's own
 * variables: see ollamaOptIn) and the runner's `doctor` subcommand. Both only
 * print what comes back, so the result has doctor's shape:
 * `{ pass: true, label }`, `{ warn: true, label, fix }` or
 * `{ pass: false, label, fix }`. `fix` is always one line of text here, so a
 * caller can print it without asking whether it is a list.
 *
 * WHAT IT MUST NOT IMPORT
 * -----------------------
 * Anything that reaches an npm package. doctor.mjs loads this file when it
 * gets to its Ollama step, and takes a module that cannot be found there to
 * mean that the copy does not hold the backend. A package missing from
 * node_modules would get the same answer: the checks would be skipped without
 * a word, or `--ollama` would name the wrong thing as missing. doctor is also
 * the script that reports an incomplete install, so what it loads must not
 * need a complete one. For the same reason the two Ollama libs import no
 * package when they load; the client fetches the privacy gate on its first
 * chat request, and this file never makes one.
 *
 * path-resolver.mjs is the one module of upstream's it imports. It reaches no
 * package either, doctor.mjs has loaded it by the time it loads this file,
 * and it is the one place the path of the profile is built: the suite
 * tests/profile-path-override.test.mjs fails on a module that builds its own.
 */

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { getProfilePath } from '../path-resolver.mjs';

import {
  DEFAULT_MODEL_PREFERENCE, NUM_CTX_STEPS, OUTPUT_RESERVE_TOKENS, OllamaConfigError, OllamaModelError, RUNNER_ENV_VARS,
  TYPICAL_PROMPT_TOKENS, evaluationPromptTokens, fitContext, pickModel, readOllamaConfig, resolveBaseUrl, resolveNumPredict,
  resolveThink, sameModel,
} from './ollama-config.mjs';
import { createOllamaClient } from './ollama-client.mjs';

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/**
 * Below this, a thinking model is likely to run out of output before the
 * report ends: thinking tokens count against num_predict. With thinking on,
 * the default model reasoned for 5-8.5K tokens, then needed 8.2-10.8K for the
 * report, and overran a 16000 cap before it reached the score summary.
 */
const MIN_NUM_PREDICT_WITH_THINKING = 24_000;

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/**
 * Settings under the names the runner finds them by.
 *
 * On Windows a variable is found whatever case it was typed in: process.env
 * ignores case there, and the runner reads process.env itself. What these
 * checks are handed is a copy (a spread of the environment, a parsed .env),
 * and a copy is a plain object that keeps each name as it was stored. So
 * `$env:ollama_model = 'llama3.3'` in PowerShell was the model every run used
 * and a name no check here looked up: the report certified a model the runner
 * was not going to use.
 *
 * Callers put the weaker source first (the .env, then the shell, then flags),
 * so when two spellings of one name meet, the later one wins, as it does for
 * the runner. Elsewhere `ollama_model` is simply another variable, which
 * neither the runner nor these checks read.
 *
 * @param {Record<string, string|undefined>|null|undefined} env
 * @returns {Record<string, string|undefined>}
 */
function asTheRunnerReads(env) {
  const settings = env ?? {};
  if (process.platform !== 'win32') return settings;
  return Object.fromEntries(Object.entries(settings).map(([name, value]) => [name.toUpperCase(), value]));
}

// ---------------------------------------------------------------------------
// Opt-in
// ---------------------------------------------------------------------------

const RUNNER_NAMES = new Set(RUNNER_ENV_VARS);

/**
 * Does this install use the Ollama backend?
 *
 * doctor's default run is offline: it is the first thing a new user runs, and
 * most users never start an Ollama server. The Ollama check makes requests, so
 * a plain `node doctor.mjs` runs it only for someone who has set one of the
 * runner's own variables (RUNNER_ENV_VARS in lib/ollama-config.mjs), in the
 * shell or in a .env the runner reads.
 *
 * Only those names count, and not "anything that starts with OLLAMA_". That
 * prefix belongs to Ollama, whose server reads OLLAMA_HOST, OLLAMA_KEEP_ALIVE,
 * OLLAMA_MODELS, OLLAMA_NUM_PARALLEL, OLLAMA_ORIGINS and more with each
 * release. Someone who tunes a server through the environment has said
 * nothing about this runner, and counting their variables made the setup
 * check send requests, and fail over models they never wanted, on machines
 * where career-ops had never talked to Ollama.
 *
 * A name set to nothing does not count. The settings reader treats an empty
 * value as unset, and `OLLAMA_MODEL=` is how a variable is cleared.
 *
 * @param {Record<string, string|undefined>} [env] The process environment.
 * @param {Record<string, string|undefined>} [dotEnvParsed] The .env files the
 *   runner reads, parsed. doctor does not always load them into the
 *   environment.
 * @returns {boolean}
 */
export function ollamaOptIn(env = process.env, dotEnvParsed = {}) {
  return [env, dotEnvParsed].some((source) => Object.entries(asTheRunnerReads(source)).some(
    ([name, value]) => RUNNER_NAMES.has(name) && String(value ?? '').trim() !== '',
  ));
}

// ---------------------------------------------------------------------------
// Prompt size
// ---------------------------------------------------------------------------

/**
 * evaluationPromptTokens() (lib/ollama-config.mjs) for the files of one
 * install, or null when there is no prompt to measure. It is the number the
 * runner sizes its context window with for the same files, by the same
 * function, so a window this check passes is one the runner accepts, and the
 * other way round.
 *
 * The files and the rules are the runner's: modes/_shared.md and
 * modes/oferta.md ship with the code; cv.md and the profile are the owner's
 * and both are required (the runner evaluates nothing without them, and an
 * empty cv.md is no CV); modes/_profile.md and modes/_custom.md are optional.
 * A file that is there and cannot be read gives null as well: the runner
 * would stop on it, so there is no size to report.
 *
 * @param {object} [where]
 * @param {string} [where.codeRoot] The checkout: where modes/ is.
 * @param {string} [where.dataRoot] Where the owner's files are.
 * @param {string} [where.profilePath] The profile, when it is not
 *   `config/profile.yml` under the data root (CAREER_OPS_PROFILE).
 * @returns {number|null}
 */
export function measurePromptTokens({ codeRoot, dataRoot, profilePath } = {}) {
  if (!codeRoot || !dataRoot) return null;
  // Trimmed, as the runner reads them: that also drops a byte-order mark.
  const read = (path) => readFileSync(path, 'utf8').trim();
  const readIfThere = (path) => (existsSync(path) ? read(path) : '');
  try {
    const cv = read(join(dataRoot, 'cv.md'));
    if (!cv) return null;
    return evaluationPromptTokens([
      read(join(codeRoot, 'modes', '_shared.md')),
      read(join(codeRoot, 'modes', 'oferta.md')),
      cv,
      // The resolver's path for a root the caller named: config/profile.yml under it.
      read(profilePath || getProfilePath(dataRoot)),
      readIfThere(join(dataRoot, 'modes', '_profile.md')),
      readIfThere(join(dataRoot, 'modes', '_custom.md')),
    ]);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

const passed = (label) => ({ pass: true, label });
const warned = (label, fix) => ({ warn: true, label, fix });
const failed = (label, fix) => ({ pass: false, label, fix });

/** What an error says, on one line: these become one line of a report. */
const said = (err) => String(err?.message ?? err).split('\n')[0].trim();

/** Two remarks on one line. The first gets a full stop unless it ends in punctuation of its own. */
const withNote = (text, note) => (note ? `${/[.?!]$/.test(text) ? text : `${text}.`} ${note}` : text);

function contextResult({ cfg, fit, promptTokens, modelMaxCtx, think, numPredict }) {
  const prompt = `~${Math.round(promptTokens / 1000)}K prompt + output`;
  const limit = `(model max ${modelMaxCtx ?? 'unknown'})`;
  if (!fit.ok) {
    // One remedy per cause. Told to raise OLLAMA_NUM_CTX, the owner of a model
    // that stops at 32768 would set a number the model cannot honour; told to
    // pick a larger model, the owner of one that holds 262144 would replace it
    // over a setting.
    let fix;
    if (modelMaxCtx !== null && modelMaxCtx < fit.need) {
      fix = `Pick a model with a larger context (e.g. ${DEFAULT_MODEL_PREFERENCE[0]})`;
    } else if (cfg.numCtxExplicit) {
      fix = `Raise OLLAMA_NUM_CTX to ${fit.need} or more, or unset it so the runner sizes the window`;
    } else {
      fix = `Set OLLAMA_NUM_CTX to ${fit.need} or more: the runner does not go past ${NUM_CTX_STEPS[NUM_CTX_STEPS.length - 1]} by itself`;
    }
    // The other way out when thinking was forced on, in the words the runner's
    // own refusal uses: the window then has to hold a reply, not the whole cap.
    if (think === true && numPredict > OUTPUT_RESERVE_TOKENS) {
      fix = withNote(fix, `Or leave thinking off, which is the default: the window then has to hold ${OUTPUT_RESERVE_TOKENS} tokens of `
        + `reply, not the whole output cap of ${numPredict}`);
    }
    return failed(`num_ctx too small: ${fit.reason}`, fix);
  }
  // Not failures: the runner makes the same adjustment and the prompt fits.
  // They are shown because the window is no longer the one the user expects,
  // and its size decides the GPU memory used and whether another program's
  // copy of the model has to be reloaded.
  if (fit.raised) {
    return warned(
      `num_ctx raised from ${cfg.numCtx} to ${fit.numCtx}${fit.clamped ? ', the most this model accepts,' : ''} to fit ${prompt} ${limit}`,
      'Nothing to do: the runner does this by itself while OLLAMA_NUM_CTX is unset. A larger window takes more GPU memory',
    );
  }
  if (fit.clamped) {
    return warned(
      `num_ctx ${cfg.numCtx} is more than this model accepts; using ${fit.numCtx}, which fits ${prompt} ${limit}`,
      cfg.numCtxExplicit
        ? `Set OLLAMA_NUM_CTX to ${fit.numCtx} or less, or leave it unset`
        : 'Nothing to do: the runner uses the model\'s own limit',
    );
  }
  return passed(`num_ctx ${fit.numCtx} fits ${prompt} ${limit}`);
}

function thinkingResult({ cfg, think, numPredict }) {
  // `n/a` is a model without the capability: the request leaves the setting
  // out, whatever OLLAMA_THINK says.
  const state = think === true ? 'on' : think === false ? 'off' : 'n/a';
  const label = `thinking: ${state} (${cfg.think === 'auto' ? 'auto' : 'forced'})`;
  // Only reachable with OLLAMA_NUM_PREDICT set: left alone, the output cap
  // follows the thinking decision.
  if (think === true && numPredict < MIN_NUM_PREDICT_WITH_THINKING) {
    return warned(
      `${label}, but num_predict is ${numPredict}`,
      `Thinking counts against the output cap, and a report needs 8-11K tokens after it. Raise OLLAMA_NUM_PREDICT to ${MIN_NUM_PREDICT_WITH_THINKING} or more, or unset it so the runner sizes the cap itself`,
    );
  }
  return passed(label);
}

/**
 * What to do about a model that is partly in system memory.
 *
 * One fixed remedy ("lower OLLAMA_NUM_CTX or use a smaller model") was wrong
 * in each of the cases it was read in. For a copy another program had loaded
 * at 262144 it blamed a variable that was unset and had played no part. For
 * the runner's own copy it named no floor, and the owner who lowered the
 * window to where the model fits was then refused by the runner, since the
 * prompt no longer did. So the remedy is chosen from what is known: whose
 * window the copy has, and the smallest window the prompt fits in.
 *
 * @param {object} args
 * @param {number|null} args.loadedCtx The window of the loaded copy, when the server says.
 * @param {ReturnType<typeof fitContext>} args.fit
 * @returns {string}
 */
function spillRemedy({ loadedCtx, fit }) {
  const smallerModel = 'a smaller model (node ollama-runner.mjs models)';
  if (fit.ok && loadedCtx > fit.numCtx) {
    // Not the window the runner asks for, and larger: the runner's own copy
    // takes less memory, so whether it fits is not known yet.
    return `Nothing to do yet: the runner's next call reloads the model at ${fit.numCtx}, which takes less memory. `
      + 'Check again after the first run';
  }
  if (fit.ok && loadedCtx === fit.numCtx && fit.numCtx > fit.need) {
    return `Set OLLAMA_NUM_CTX lower, but not below ${fit.need}, or use ${smallerModel}`;
  }
  if (!fit.ok && loadedCtx > fit.need) {
    // The runner sends nothing until the window is corrected, and what it
    // then asks for may be smaller than this copy.
    return 'Nothing to do yet: this copy was not loaded with the current settings, which cannot run an evaluation '
      + '(see the window above). Check again after the first run';
  }
  // What is left: a copy at the smallest window the prompt fits in, or at a
  // smaller one still (any window the runner can use takes at least as much
  // memory), or a server that does not say what it loaded.
  return `Use ${smallerModel}: this prompt needs a window of at least ${fit.need} tokens`;
}

/**
 * Warnings about the copy of the model the server holds right now.
 *
 * `size_vram` below `size` is the server's own statement that part of the
 * model sits in system memory. The share is rounded down, so a model that is
 * 99.6% on the GPU does not read as "100% on GPU: CPU spill". (The client
 * reports both as non-negative numbers, so the division is by a size above
 * zero and the result is below 100.)
 */
function loadedResults({ model, loaded, fit }) {
  const results = [];
  const loadedCtx = loaded.contextLength;
  const spilled = loaded.sizeVram < loaded.size;
  // Skipped when the prompt does not fit: the runner refuses to send it, so
  // there is no next call to announce.
  if (fit.ok && loadedCtx && loadedCtx !== fit.numCtx) {
    let fix;
    if (loadedCtx < fit.need) {
      // A window another program chose is only worth matching if the prompt
      // fits in it; a desktop app's 4096 would truncate every evaluation.
      fix = `Nothing to do: ${loadedCtx} cannot hold the evaluation prompt, so the reload is needed`;
    } else if (spilled) {
      // Nor is one worth matching when the model does not fit on the GPU at
      // it: pinning that size would run every evaluation partly on the CPU.
      fix = `Nothing to do: the reload at ${fit.numCtx} replaces a copy that is partly in system memory`;
    } else {
      fix = `Accept one reload, or set OLLAMA_NUM_CTX=${loadedCtx} so the runner asks for the size that is loaded`;
    }
    results.push(warned(`${model} is loaded at ${loadedCtx}, runner uses ${fit.numCtx}: the next call reloads the model`, fix));
  }
  if (spilled) {
    const share = Math.floor((100 * loaded.sizeVram) / loaded.size);
    results.push(warned(
      `${model} is ${share}% on GPU${loadedCtx ? ` at num_ctx ${loadedCtx}` : ''}: CPU spill`,
      spillRemedy({ loadedCtx, fit }),
    ));
  }
  return results;
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

/**
 * Run the checks in order, appending to `results`. Returns early where the
 * next check has nothing to work with: no usable settings, no server, no
 * model.
 */
async function collect(results, { env, fetchImpl, promptTokens, soft }) {
  const estimate = Number.isFinite(promptTokens) && promptTokens > 0 ? promptTokens : TYPICAL_PROMPT_TOKENS;
  const settings = asTheRunnerReads(env);
  // `soft` is for a caller that was not asked for this check by name. What it
  // turns into a warning is everything that describes the server at this
  // moment: it is off, something else answers on its port, it holds no model
  // the runner would pick, it cannot say what the model is. A server can be
  // any of those on a day when nobody is evaluating anything, and the general
  // setup check must not go red over it.
  const report = soft ? warned : failed;

  // -- Settings. Read with the runner's own reader, so a value the runner
  //    would refuse to start with is refused here, in the same words. Never
  //    softened: unlike a server that is not running right now, a bad value
  //    stays wrong until someone edits it. ---------------------------------
  let cfg;
  let ignoredHost;
  try {
    cfg = readOllamaConfig(settings);
    ({ ignoredHost } = resolveBaseUrl(settings));
  } catch (err) {
    if (!(err instanceof OllamaConfigError)) throw err;
    results.push(failed('Ollama settings cannot be used', said(err)));
    return;
  }
  const client = createOllamaClient({ baseUrl: cfg.baseUrl, fetchImpl });

  // -- 1. Is a server there? -------------------------------------------------
  let version;
  try {
    version = await client.version();
  } catch (err) {
    // The user who set OLLAMA_HOST=0.0.0.0 to make the server listen, and
    // expects every tool to follow that variable, needs to be told it was read
    // and why it was not used.
    const note = ignoredHost
      ? `OLLAMA_HOST=${ignoredHost} is the server bind address and is not used as the client URL`
      : '';
    // Anything but "nothing is listening" keeps its own words: another
    // program on the port, a proxy's error page, or the privacy guard
    // refusing the request, where it is installed over fetch.
    results.push(err?.code === 'UNREACHABLE'
      ? report(`Ollama not reachable at ${cfg.baseUrl}`, withNote('Start Ollama (ollama serve) or set OLLAMA_BASE_URL', note))
      : report(`Ollama at ${cfg.baseUrl} could not be checked`, withNote(said(err), note)));
    return;
  }
  results.push(passed(`Ollama ${version === 'unknown' ? '(version unknown)' : version} reachable at ${cfg.baseUrl}`));

  // -- 2. Is there a model to run? -------------------------------------------
  let picked;
  try {
    picked = pickModel({ requested: cfg.model, preference: cfg.preference, installed: await client.listModels() });
  } catch (err) {
    if (err instanceof OllamaModelError) {
      // The message lists what is installed and names the variable to set.
      results.push(report(
        cfg.model ? `Ollama model: ${cfg.model} is not installed` : 'Ollama model: none of the preferred models is installed',
        said(err),
      ));
    } else {
      results.push(report('Ollama model list could not be read', said(err)));
    }
    return;
  }
  results.push(passed(`Ollama model: ${picked.model} (${picked.reason})`));

  // -- 3 and 4. Context window and thinking. The same decisions, in the same
  //    order, as the runner makes before a request: thinking first, because it
  //    sets the output cap, and the cap is part of what the window must hold.
  //    Their verdicts are not softened: they are about the settings and the
  //    owner's own files, and the runner refuses to send a prompt that does
  //    not fit however the check was started. --------------------------------
  let shown;
  try {
    shown = await client.showModel(picked.model);
  } catch (err) {
    results.push(report(`Ollama model details could not be read for ${picked.model}`, said(err)));
    return;
  }
  const think = resolveThink({ setting: cfg.think, capabilities: shown.capabilities });
  const numPredict = resolveNumPredict({ numPredict: cfg.numPredict, numPredictExplicit: cfg.numPredictExplicit, think });
  const fit = fitContext({
    numCtx: cfg.numCtx,
    explicit: cfg.numCtxExplicit,
    modelMaxCtx: shown.contextLength,
    promptTokens: estimate,
    numPredict,
    // With thinking on, the reasoning and the report share the output cap and
    // the window has to leave room for all of it.
    think,
  });
  results.push(contextResult({ cfg, fit, promptTokens: estimate, modelMaxCtx: shown.contextLength, think, numPredict }));
  results.push(thinkingResult({ cfg, think, numPredict }));

  // -- 5. What is loaded right now. Best effort: it describes a moment, and a
  //    server that cannot list its loaded models has already passed every
  //    check a run depends on. ----------------------------------------------
  let loaded;
  try {
    loaded = (await client.ps()).find((entry) => sameModel(entry.name, picked.model));
  } catch { /* nothing to add */ }
  if (loaded) results.push(...loadedResults({ model: picked.model, loaded, fit }));
}

/**
 * Check that the local Ollama server can run an evaluation.
 *
 * In order: the settings are usable, the server answers, a model can be
 * picked, the context window holds the prompt, how thinking is set, and
 * whether the model is loaded in a way that will cost time. The first three
 * stop the check when they fail, since nothing after them can be decided.
 *
 * It never throws and never ends the process: doctor collects every check
 * before it prints one, so an exception here would cost the user the whole
 * report.
 *
 * @param {object} [options]
 * @param {Record<string, string|undefined>} [options.env] Settings to check.
 *   A caller that reads .env files merges them in first, shell values last.
 *   On Windows the names are read whatever their case, as the runner reads
 *   them.
 * @param {typeof fetch} [options.fetchImpl] Defaults to the global fetch.
 * @param {number|null} [options.promptTokens] Size of the evaluation prompt:
 *   the runner's own count, or measurePromptTokens(). Without a usable number
 *   the size of a prompt built from the example files is assumed.
 * @param {boolean} [options.soft] For a caller that runs this check without
 *   being asked for it by name. Whatever depends on the server as it is right
 *   now is then a warning: it does not answer, it has no model the runner
 *   would pick, its model list or the model's details cannot be read; so is a
 *   check that broke. A setting the runner refuses and a window the prompt
 *   does not fit stay failures: those are wrong until someone edits them.
 * @returns {Promise<Array<{pass: true, label: string} | {warn: true, label: string, fix: string} | {pass: false, label: string, fix: string}>>}
 */
export async function checkOllama({ env = process.env, fetchImpl, promptTokens = TYPICAL_PROMPT_TOKENS, soft = false } = {}) {
  const results = [];
  try {
    await collect(results, { env, fetchImpl, promptTokens, soft });
  } catch (err) {
    // Every failure the checks know about is a result by now. What is left is
    // the unexpected one, and what was found before it is still worth showing.
    results.push((soft ? warned : failed)('Ollama check could not finish', said(err)));
  }
  return results;
}

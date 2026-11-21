/**
 * ollama-cv-latex.mjs — the tailored CV as data, for the local runner's LaTeX
 * path (`node ollama-runner.mjs tailor <report> --latex`).
 *
 * WHY DATA AND NOT A DOCUMENT
 * ---------------------------
 * The HTML path asks the model for the whole document and then refuses any
 * markup that loads or runs something: with a stub in place of the model, an
 * iframe put the text of the data root's .env into a PDF. A LaTeX document
 * cannot be checked that way. One `\input` in a bullet reads a file into the
 * PDF, and turning shell escape off does not stop file reads: TeX Live reads
 * any path by default, and MiKTeX's setting for it has no command-line
 * switch. So the model is asked for the text of the CV as one JSON object
 * (CV_PAYLOAD_SCHEMA), and code writes every command around that text.
 *
 * WHAT THIS PART DOES
 * -------------------
 * It reads a reply into that object (parseCvPayload) and says why a reply is
 * not one (cvPayloadProblem): no JSON object, one cut off before its end, a
 * list for an object, a field of the wrong type, a job without a company, a
 * link that is not a plain web address, too little text. A reason is a
 * lower-case phrase with no full stop, like the HTML path's, because the
 * runner prints it inside "Attempt 1 was not usable (…)".
 *
 * Every string is normalised before any check reads it (cleanText), every
 * string but an address loses the underscores of italics, which no field
 * prints, and a field that prints no bold loses its `**` (normalizeCvPayload),
 * so no spacing, no invisible character, no underscore and no asterisk sets
 * the text checked apart from the text printed. What TeX itself prints
 * otherwise, its ligatures (`--` as an en dash), no check sees;
 * docs/OLLAMA.md tells a template's author how to turn them off. The header of the CV (name, title, contact details) is not
 * the model's to write: readCvHeader() reads it from config/profile.yml.
 *
 * WHERE THE TEXT IS CHECKED
 * -------------------------
 * The checks an HTML CV goes through read HTML: the fact check of
 * verify-cv-facts.mjs, the locked sections and the order of the roles that
 * generate-pdf.mjs enforces, and the runner's link rule and repair. They know
 * none of the macros a .tex is written in, and the final text check of
 * generate-latex.mjs, which reads the .tex, passed one with a swapped
 * employer, wrong dates, an invented degree and an invented skill. So the
 * payload is also written as the plain HTML those checks read
 * (projectCvHtml), with its dates as the template prints them
 * (shownDatesFor), and checked there. That HTML is never rendered. What the
 * link rule and the repair take out of it is read back into the payload
 * (dropPayloadLinks, readSkillsBack), since the .tex is written from the
 * payload.
 *
 * HOW THE .TEX IS WRITTEN
 * -----------------------
 * The look of the CV is a template's: lib/ollama-cv-latex-default.tex, or a
 * pack of the owner's in data/templates/<name>/ (resolveLatexTemplate). The
 * renderer (renderCvTex) writes the text into it as calls to a small set of
 * macros the template defines (\cvEntryHead, \cvItem and the rest), fills its
 * placeholders and drops the blocks that have nothing to show. A template is
 * code, run by the engine with the owner's rights; the text written into it
 * must not be. So every value is escaped once, and a URL goes through
 * sanitizeUrl() and nothing else. escapeLatex() is the defence. Behind it,
 * before anything is substituted, each value is searched for a control
 * sequence the renderer does not write itself (unknownTexCommands), so a
 * command that got past the escaping stops the render instead of reaching
 * the engine. That search sees control sequences only: a raw %, brace, # or ^
 * left by a missed escape is no command to it, though ^^5c is a backslash to
 * TeX. No input reaches its refusal today; it is there for a field added
 * later without its escape.
 *
 * The final text check of generate-latex.mjs reads the finished .tex as text,
 * and read a CV true to cv.md as one with invented tools in three ways: a
 * command inside a macro argument (\textbf, \textless) read as a tool name,
 * and a list of tools with no full stop after it ran on into the next bullet.
 * So bold is written {\bfseries …}, six characters escapeLatex() writes as
 * commands are written back as themselves, and \cvStop{.}, which prints
 * nothing, follows every call that carries text.
 *
 * A hand-edited .tex is to change the layout, never the text: contentMacros()
 * reads the calls that carry text out of one, for the runner to compare with
 * the .tex its payload renders to. It reads those calls and nothing else. A
 * macro the preamble redefines, a call put inside \iffalse…\fi, or a line of
 * text between two calls changes the page and leaves that list as it was,
 * and the final text check, which misses structured facts, is then the only
 * check that reads the change.
 *
 * HOW THE PDF IS MADE
 * -------------------
 * compileLatex() compiles a .tex in a folder made for that compile alone, with
 * a name no one can guess, and removes the folder when the compile ends,
 * whatever happened. The pack is copied in first, never a symbolic link,
 * which could let the engine read or write outside the folder through a name
 * inside it. The engine runs with shell escape off and, for MiKTeX, its
 * package installer off; pdflatex and xelatex twice, since the first pass
 * only measures what the second prints, tectonic once. The engine is looked
 * for on PATH alone: on Windows a program named without a folder is first
 * looked for in the folder it runs in, where a pack could put one of its own.
 * What comes back is the PDF's bytes and its number of pages, read from the
 * last line the engine writes to its log (pageCountFromLog), for the caller
 * to write once it has checked the page limit. A pack is code all the same:
 * the engine runs its class and preamble with the owner's rights.
 *
 * latexEngineProblem() asks the first question of a compile on its own,
 * whether the engine answers --version, of the same program: the runner asks
 * it before it sends the model anything, so that a CV is not tailored only to
 * find no engine to compile it.
 */

import { spawnSync } from 'node:child_process';
import {
  accessSync, constants, cpSync, existsSync, lstatSync, mkdtempSync, readdirSync, readFileSync, statSync, writeFileSync,
} from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { delimiter, dirname, isAbsolute, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { localTectonicPath } from '../generate-latex.mjs';
import { escapeLatex, sanitizeUrl } from './latex-escape.mjs';
import { loadYaml } from './yaml-document.mjs';

// ---------------------------------------------------------------------------
// The schema the model is shown
// ---------------------------------------------------------------------------

// The persona of examples/cv-example.md, cut down to show each part of the
// shape once. Written as an object and printed with JSON.stringify(), so the
// example is always JSON, and a test holds it to the reader below.
const EXAMPLE = {
  summary: [
    'Full-stack AI engineer with 6 years building production ML systems. Led the ML platform at a Series B fintech (2020-2024), '
      + 'scaling from 2 models to 15+ in production.',
  ],
  experience: [
    {
      company: 'TechFin Corp',
      role: 'Senior ML Engineer / ML Platform Lead',
      location: 'Austin, TX',
      dates: '2020-2024',
      blocks: [{
        lead: '',
        intro: '',
        bullets: [
          'Designed real-time **fraud detection** pipeline: Kafka → feature computation → model inference → decision engine. '
            + '99.7% precision at 50ms p99',
          'Reduced model deployment time from 2 weeks to 4 hours with CI/CD pipeline (GitHub Actions + SageMaker)',
        ],
      }],
    },
    {
      company: 'DataStartup Inc',
      role: 'ML Engineer',
      location: 'Remote',
      dates: '2018-2020',
      blocks: [{ lead: '', intro: '', bullets: ['Built NLP pipeline for document classification (BERT fine-tuning, 94% accuracy on legal docs)'] }],
    },
  ],
  projects: {
    intro: '',
    items: [{ name: 'FraudShield', text: 'Real-time fraud detection framework. Kafka Streams + feature store + model serving.', url: '' }],
  },
  education: [{ degree: 'MS Computer Science', institution: 'UT Austin', dates: '2018' }],
  certifications: [],
  skills: [
    { category: 'ML/AI', items: 'PyTorch, TensorFlow, scikit-learn, Hugging Face, LangChain' },
    { category: 'Languages', items: 'Python, Go, TypeScript, SQL' },
  ],
};

/**
 * What the model is told to reply with: the payload's fields, its rules, and
 * one example for another person. The field list holds no brace, so the
 * example is the one JSON object in the text.
 *
 * @type {string}
 */
export const CV_PAYLOAD_SCHEMA = [
  'CV PAYLOAD',
  'Your reply is one JSON object and nothing else: no text before or after it, no code fence, no comment. It holds the text of the '
    + 'CV. The program writes the layout and every LaTeX command, so every value is plain text: no LaTeX, no HTML and no Markdown, '
    + 'except **double asterisks** around a few words to set them in bold, in the summary, a lead, an intro, a bullet, a project '
    + 'text or a certification. The header (name, title, email, phone, links, location) is not part of the reply: the program '
    + 'fills it in from config/profile.yml.',
  '',
  'The fields:',
  'summary: a list of paragraphs.',
  'experience: a list of jobs, newest first by start date. Each job has:',
  '  company: required.',
  '  role: required; the title cv.md gives that job.',
  '  location: "" when cv.md gives none.',
  '  dates: as cv.md writes them.',
  '  blocks: a list of groups of bullets, at least one. Each block has:',
  '    lead: the title cv.md gives a group of bullets, such as a product or a client; else "".',
  '    intro: a sentence before the bullets; else "".',
  '    bullets: a list of strings.',
  '    A block needs an intro or a bullet.',
  'projects: an object. It has:',
  '  intro: a sentence before the list; else "".',
  '  items: a list of projects. Each one has:',
  '    name: required.',
  '    text: the description.',
  '    url: "" unless cv.md gives the address. Then it is the address as cv.md writes it, starting with http:// or https://.',
  'education: a list. Each entry has:',
  '  degree: required.',
  '  institution: required.',
  '  dates: as cv.md writes them.',
  'certifications: a list of strings.',
  'skills: a list. Each entry has:',
  '  category: the name of the group, without a colon.',
  '  items: one string, the items separated by commas.',
  '',
  'A section with nothing to show is an empty list, or for projects an empty items list. At least two sections need content. '
    + 'Never break a line inside a string: write one string per paragraph, bullet or entry.',
  '',
  'An example of the shape, for another person. Take its shape from it, never its text:',
  JSON.stringify(EXAMPLE, null, 2),
].join('\n');

// ---------------------------------------------------------------------------
// Reading a reply
// ---------------------------------------------------------------------------

/**
 * The JSON object in a reply, as text, or null when there is none or it is
 * cut off.
 *
 * A model asked for one JSON object still wraps it now and then: a code fence
 * around it, a sentence before it, a remark after it. The object is found by
 * counting braces from its opening one to the one that closes it, and braces
 * inside a JSON string do not count (a bullet can hold `}` or `\"`), so the
 * fence and the prose fall away by themselves.
 *
 * A brace in the prose before the object ("I left the {placeholders}
 * alone") is no reason to ask again: a span that closes but does not parse is
 * passed over for the next one, and is returned only when none parses, so the
 * reason given is the parser's. A span is never searched inside, so an object
 * nested in a reply that was cut off is not mistaken for the payload: the
 * first span that does not close ends the search with null, whatever closed
 * before it, so a reply cut off after braced prose still reads as cut off.
 * The price of never searching inside a span: an unbalanced brace, or a stray
 * double quote between braces, in the prose before a whole object reads as
 * cut off too, as does one in a remark after an object that does not parse,
 * and either costs a retry.
 *
 * An object that is the first item of a list is returned as that list, so a
 * list is refused as a list, not read as its first item.
 *
 * @param {string} text
 * @returns {string|null}
 */
export function extractJsonObject(text) {
  const source = String(text ?? '');
  let firstClosed = null;
  for (let from = 0; ;) {
    const brace = source.indexOf('{', from);
    if (brace === -1) return firstClosed;
    const before = source.slice(from, brace).trimEnd();
    const start = before.endsWith('[') ? from + before.length - 1 : brace;
    const end = closingIndex(source, start);
    if (end === -1) return null;
    const span = source.slice(start, end + 1);
    try {
      JSON.parse(span);
      return span;
    } catch {
      firstClosed ??= span;
      from = end + 1;
    }
  }
}

/** Where the bracket or brace at `start` is closed, counting outside JSON strings only; -1 when it never is. */
function closingIndex(text, start) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (inString) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') inString = false;
    } else if (char === '"') {
      inString = true;
    } else if (char === '{' || char === '[') {
      depth += 1;
    } else if (char === '}' || char === ']') {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}

/**
 * Why a reply is not a CV payload, or null when it is one.
 *
 * In order: the JSON object is cut out of the reply (extractJsonObject), it
 * parses, each field the schema has is of its type (typeProblem), every string
 * is normalised (normalizeCvPayload), and only then are the required fields
 * looked for, so a bullet of spaces counts as no bullet.
 *
 * @param {string} text A model's reply, or the text of a saved payload.
 * @param {object} [options]
 * @param {number} [options.minTextChars] The text a CV has to carry, as on the HTML path: a reply that is a CV's
 *   shell and little else is asked for again. 0, the default, checks no length.
 * @returns {string|null}
 */
export function cvPayloadProblem(text, { minTextChars = 0 } = {}) {
  return readPayload(text, minTextChars).problem;
}

/**
 * A reply read into a normalised CV payload.
 *
 * @param {string} text
 * @returns {ReturnType<typeof normalizeCvPayload>}
 * @throws {Error} With the sentence cvPayloadProblem() returns for the same text.
 */
export function parseCvPayload(text) {
  const { problem, payload } = readPayload(text, 0);
  if (problem) throw new Error(problem);
  return payload;
}

/** The steps of cvPayloadProblem(), with the payload when there was no problem. */
function readPayload(text, minTextChars) {
  const source = String(text ?? '');
  const json = extractJsonObject(source);
  if (json === null) {
    return { problem: source.includes('{') ? 'the JSON object is cut off before its closing brace' : 'no JSON object was found' };
  }
  let parsed;
  try {
    parsed = JSON.parse(json);
  } catch (err) {
    // V8 quotes part of the reply in some of its messages (Unexpected token
    // 'x', "…" is not valid JSON), and the runner prints the reason as it is:
    // a control character from the reply would reach the terminal.
    return { problem: `the JSON does not parse: ${cleanText(err.message)}` };
  }
  const wrongType = typeProblem(parsed);
  if (wrongType) return { problem: wrongType };
  const payload = normalizeCvPayload(parsed);
  const missing = requiredProblem(payload, minTextChars);
  return missing ? { problem: missing } : { problem: null, payload };
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
//
// Four ways of writing a value are read rather than asked for again, since
// each one costs a whole new generation and means the same: null for a value
// that is not there, a number where text belongs (`"dates": 2018`), one
// string where a list of strings belongs, and for projects a list where the
// object with intro and items belongs (the list is the items, `[]` none).
// Anything else of the wrong type is refused with its path.

/** True for a JSON object (not a list, not null). */
function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** True for what is read as text: a string, a finite number, or nothing. */
function isText(value) {
  return value === null || value === undefined || typeof value === 'string' || (typeof value === 'number' && Number.isFinite(value));
}

/** True for what is read as a list of text: such a list, or one value read as text. */
function isTextList(value) {
  return isText(value) || (Array.isArray(value) && value.every(isText));
}

/** The first of `keys` whose value in `record` is not text, as a problem, or null. */
function textFieldProblem(record, keys, path) {
  const key = keys.find((name) => !isText(record[name]));
  return key ? `${path}.${key} must be text` : null;
}

/** The projects of a parsed reply as an object: a list is read as the items, anything else that is no object as none. */
function projectsOf(value) {
  if (Array.isArray(value)) return { intro: '', items: value };
  return isRecord(value) ? value : {};
}

/** The first problem of a list of objects at `path`, each object checked by `check`, or null. Null items are absent. */
function recordListProblem(value, path, check) {
  if (value === null || value === undefined) return null;
  if (!Array.isArray(value)) return `${path} must be a list of objects`;
  for (const [index, item] of value.entries()) {
    if (item === null || item === undefined) continue;
    if (!isRecord(item)) return `${path}[${index}] must be an object`;
    const problem = check(item, `${path}[${index}]`);
    if (problem) return problem;
  }
  return null;
}

/** The first field of a parsed reply that is not of its type, in the schema's order, or null. */
function typeProblem(root) {
  if (!isRecord(root)) return Array.isArray(root) ? 'the JSON is a list, not one object' : 'the JSON is not one object';
  if (!isTextList(root.summary)) return 'summary must be a list of text';
  const experience = recordListProblem(root.experience, 'experience', (job, path) => (
    textFieldProblem(job, ['company', 'role', 'location', 'dates'], path)
      ?? recordListProblem(job.blocks, `${path}.blocks`, (block, at) => (
        textFieldProblem(block, ['lead', 'intro'], at) ?? (isTextList(block.bullets) ? null : `${at}.bullets must be a list of text`)
      ))
  ));
  if (experience) return experience;
  const { projects } = root;
  const projectProblem = (item, path) => textFieldProblem(item, ['name', 'text', 'url'], path);
  if (Array.isArray(projects)) {
    // Named as written, `projects[0]`: the reply has no `items` to name.
    const problem = recordListProblem(projects, 'projects', projectProblem);
    if (problem) return problem;
  } else if (projects !== null && projects !== undefined) {
    if (!isRecord(projects)) return 'projects must be an object with intro and items';
    const problem = textFieldProblem(projects, ['intro'], 'projects') ?? recordListProblem(projects.items, 'projects.items', projectProblem);
    if (problem) return problem;
  }
  const education = recordListProblem(root.education, 'education', (entry, path) => (
    textFieldProblem(entry, ['degree', 'institution', 'dates'], path)
  ));
  if (education) return education;
  if (!isTextList(root.certifications)) return 'certifications must be a list of text';
  return recordListProblem(root.skills, 'skills', (skill, path) => (
    textFieldProblem(skill, ['category'], path) ?? (isTextList(skill.items) ? null : `${path}.items must be text or a list of text`)
  ));
}

// ---------------------------------------------------------------------------
// Normalising
// ---------------------------------------------------------------------------

/**
 * Control (Cc), format (Cf) and lone surrogate (Cs) characters, and U+FFFD.
 * Format characters are the zero-width ones, the soft hyphen, bidi marks and
 * overrides. None is text a reader should get: TeX reads U+0001 as an invalid
 * character and stops; a zero-width character or a soft hyphen inside a word
 * hides the word from the fact check, which reads the stored string, while the
 * page shows it whole; a bidi override can make a viewer show the text in
 * another order than the one checked. A lone surrogate is written to a file as
 * U+FFFD, which pdflatex cannot print either.
 */
const INVISIBLE_RE = /[\p{Cc}\p{Cf}\p{Cs}\uFFFD]/gu;

/**
 * Spacing, which separates words: JavaScript's `\s` (tab, line breaks,
 * U+00A0, U+2028, U+2029 and the other spaces), and three kinds it leaves
 * out, NEL (U+0085, a line break) and the information separators
 * U+001C-U+001F. Those are control characters too: removed as invisible, they
 * would glue the words on either side together.
 */
const SPACING_RE = /[\s\u0085\u001C-\u001F]+/g;

/**
 * One string of the payload, normalised. In this order: (1) every run of
 * spacing (SPACING_RE) becomes one space; (2) the invisible characters go
 * (INVISIBLE_RE); (3) the ends are trimmed. Spacing comes first because line
 * breaks are control characters too: in review, removing them first made
 * "Built\nKafka" one word. The BOM (U+FEFF) goes before step 1: JavaScript
 * counts it as spacing, but inside a word it is a zero-width character, and
 * "Kaf ka" would hide the word from the fact check as surely as the BOM
 * itself. A removal that leaves two spaces side by side leaves one. A finite
 * number is read as its digits; anything else that is not a string is ''.
 *
 * @param {unknown} value
 * @returns {string}
 */
function cleanText(value) {
  const text = typeof value === 'number' && Number.isFinite(value) ? String(value) : value;
  if (typeof text !== 'string') return '';
  return text.replace(/\uFEFF/g, '').replace(SPACING_RE, ' ').replace(INVISIBLE_RE, '').replace(/ {2,}/g, ' ').trim();
}

/** `**words**`, as the payload marks emphasis. */
const BOLD_RE = /\*\*([^*]+?)\*\*/g;

/**
 * `_words_`, as cv.md writes a title in italics: the pattern
 * normalizeTextForATS() of generate-pdf.mjs reads as italics, copied as it
 * stands. Both underscores sit on a word boundary, and a path separator is
 * none, so snake_case, a_b and docs/_draft_/cv.md are no italics.
 */
const ITALIC_RE = /(?<![\p{L}\p{N}_/\\])_(?=[^_\s])([^_\n]*?[^_\s])_(?![\p{L}\p{N}_/\\])/gu;

/**
 * One round of taking the underscores of italics out, the words left: what
 * normalizeTextForATS() writes as <em>, which it does not do to the
 * underscores of an email address. Around the whole address, they are
 * italics.
 */
function dropItalics(text) {
  return text.replace(ITALIC_RE, (match, inner, at, whole) => {
    const word = /\S*$/.exec(whole.slice(0, at))[0] + /^\S*/.exec(whole.slice(at))[0];
    return word.includes('@') && !inner.includes('@') ? match : inner;
  });
}

/**
 * A string normalised (cleanText) without the markers `drop` takes out, a
 * round at a time until none is left, and the spacing they leave normalised
 * again. Until none is left: `****x****` takes two rounds, and so does
 * `_a _b_ c_`, whose inner pair hides the outer one, which
 * normalizeTextForATS() would then read in the projection while the .tex
 * printed its underscores.
 */
function withoutMarkers(value, drop) {
  let text = cleanText(value);
  for (let before = null; before !== text;) {
    before = text;
    text = drop(text);
  }
  return cleanText(text);
}

/**
 * A string of a field that prints bold, normalised: without the underscores
 * of italics (dropItalics()), its `**` kept for the renderer. No field prints
 * italics, and the checks, which read the projection after
 * normalizeTextForATS(), read the words of `_LLM Eval Toolkit_` without the
 * underscores the PDF printed.
 *
 * @param {unknown} value
 * @returns {string}
 */
function textWithoutItalics(value) {
  return withoutMarkers(value, dropItalics);
}

/**
 * A string of a field that prints no bold, normalised: without its `**` and
 * without the underscores of italics, the words left. cv.md writes a job's
 * title, a project's name and a skills category in bold, and a title in
 * italics, and a model copies them as they stand. The renderer writes `**` as
 * bold only in the fields that print bold; anywhere else the asterisks
 * printed as they came, while the checks read the words without them.
 *
 * @param {unknown} value
 * @returns {string}
 */
function textWithoutEmphasis(value) {
  return withoutMarkers(value, (text) => dropItalics(text.replace(BOLD_RE, (_, inner) => inner)));
}

/** The objects of a list; anything else in it, or a value that is no list, is dropped. */
function records(value) {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

/** A list of text, or one value read as text, as strings normalised by `clean` (cleanText), the empty ones dropped. */
function textList(value, clean = cleanText) {
  return (Array.isArray(value) ? value : [value]).map((item) => clean(item)).filter(Boolean);
}

/**
 * A payload in its one shape: every field of the schema present, in the
 * schema's order, every string normalised (cleanText), keys the schema does
 * not have dropped. Every string but an address loses the underscores of
 * italics around its words (textWithoutItalics()), which no field prints. A
 * field that prints no bold loses the `**` around its words as well
 * (textWithoutEmphasis()): every field but the summary, a lead, an intro, a
 * bullet, the projects intro, a project's text, a certification and an
 * address. So the .json, the projection the checks read and the .tex hold one
 * text. An address is not text: it keeps what it holds.
 *
 * Its limit: an underscore beside a backslash. normalizeTextForATS() keeps
 * `C:\models\_draft_\x` as it is, a path, and so does this. The projection
 * writes the backslash as `&#92;`, which is no path separator to
 * normalizeTextForATS(), and the checks read that pair as italics.
 *
 * Then what holds nothing is dropped: empty strings from a list; a block with
 * no intro and no bullet (a lead alone goes with it: a title with nothing
 * under it); an entry with no text at all; the projects intro when there is
 * no project. `skills[].items` becomes one string (a list is joined with
 * `, `), and `skills[].category` loses its trailing colons: cv.md writes
 * `**Languages:**`, a model copies the colon, and a template that prints
 * `#1: #2` would show two. The colons go after the asterisks, which hid the
 * colon of `**ML/AI:**` from the end. Every trailing colon, with the spaces
 * between them, not only the last: a payload is normalised again on its way
 * (read, checked, rendered), and a step that took one colon per pass would
 * make the category checked depend on how many passes a route makes.
 * Normalising a normalised payload changes nothing.
 *
 * A list where the projects object belongs is read as its items, `[]` as no
 * project. Never throws: any other value of the wrong type is dropped as if
 * absent. cvPayloadProblem() is what says that it was wrong.
 *
 * @param {unknown} payload
 * @returns {{
 *   summary: string[],
 *   experience: {company: string, role: string, location: string, dates: string,
 *     blocks: {lead: string, intro: string, bullets: string[]}[]}[],
 *   projects: {intro: string, items: {name: string, text: string, url: string}[]},
 *   education: {degree: string, institution: string, dates: string}[],
 *   certifications: string[],
 *   skills: {category: string, items: string}[],
 * }}
 */
export function normalizeCvPayload(payload) {
  const root = isRecord(payload) ? payload : {};
  const projects = projectsOf(root.projects);
  // A field that prints bold, and one that prints none.
  const rich = textWithoutItalics;
  const plain = textWithoutEmphasis;
  const items = records(projects.items)
    .map((item) => ({ name: plain(item.name), text: rich(item.text), url: cleanText(item.url) }))
    .filter((item) => item.name || item.text || item.url);
  return {
    summary: textList(root.summary, rich),
    experience: records(root.experience)
      .map((job) => ({
        company: plain(job.company),
        role: plain(job.role),
        location: plain(job.location),
        dates: plain(job.dates),
        blocks: records(job.blocks)
          .map((block) => ({ lead: rich(block.lead), intro: rich(block.intro), bullets: textList(block.bullets, rich) }))
          .filter((block) => block.intro || block.bullets.length > 0),
      }))
      .filter((job) => job.company || job.role || job.location || job.dates || job.blocks.length > 0),
    projects: { intro: items.length > 0 ? rich(projects.intro) : '', items },
    education: records(root.education)
      .map((entry) => ({ degree: plain(entry.degree), institution: plain(entry.institution), dates: plain(entry.dates) }))
      .filter((entry) => entry.degree || entry.institution || entry.dates),
    certifications: textList(root.certifications, rich),
    skills: records(root.skills)
      .map((skill) => ({
        category: plain(skill.category).replace(/[\s:]+$/, ''),
        items: Array.isArray(skill.items) ? textList(skill.items, plain).join(', ') : plain(skill.items),
      }))
      .filter((skill) => skill.category || skill.items),
  };
}

// ---------------------------------------------------------------------------
// Required fields
// ---------------------------------------------------------------------------

/**
 * A project address as it can be written: http or https, and none of the
 * characters sanitizeUrl() deletes (`{ } \ $ ^`) or that end an address in
 * running text (spacing, quotes, angle brackets, `|`, a backquote). With one of
 * the first kind the address the link rule checked would not be the address
 * printed.
 */
const PLAIN_WEB_ADDRESS_RE = /^https?:\/\/[^\s\\{}$^"'<>|`]+$/;

/** ` (label)` after a path in a message, or nothing when there is no label. */
function named(label) {
  return label ? ` (${label})` : '';
}

/** The number of the six sections that have something to show. */
function sectionsWithContent(payload) {
  return [
    payload.summary.length, payload.experience.length, payload.projects.items.length, payload.education.length,
    payload.certifications.length, payload.skills.length,
  ].filter((count) => count > 0).length;
}

/** The first required field a normalised payload lacks, or null. */
function requiredProblem(payload, minTextChars) {
  for (const [index, job] of payload.experience.entries()) {
    const at = `experience[${index}]${named(job.company || job.role)}`;
    if (!job.company) return `${at} has no company`;
    if (!job.role) return `${at} has no role`;
    if (job.blocks.length === 0) return `${at} has no block with an intro or a bullet`;
  }
  for (const [index, item] of payload.projects.items.entries()) {
    const at = `projects.items[${index}]${named(item.name)}`;
    if (!item.name) return `${at} has no name`;
    if (item.url && !PLAIN_WEB_ADDRESS_RE.test(item.url)) return `${at} has an address that is not a plain http or https link`;
  }
  for (const [index, entry] of payload.education.entries()) {
    const at = `education[${index}]${named(entry.institution || entry.degree)}`;
    if (!entry.degree) return `${at} has no degree`;
    if (!entry.institution) return `${at} has no institution`;
  }
  // The HTML path asks for two section titles: a reply with one section is a
  // fragment of a CV, however long that section is.
  if (sectionsWithContent(payload) < 2) return 'the CV has fewer than two sections with content';
  const length = payloadText(payload).length;
  if (minTextChars > 0 && length < minTextChars) {
    return `the CV holds ${length} characters of text where cv.md calls for at least ${minTextChars}`;
  }
  return null;
}

// ---------------------------------------------------------------------------
// The strings of a payload
// ---------------------------------------------------------------------------

/**
 * A payload in its one shape with `fn(value, field)` in place of each string
 * of the schema. A field missing from `payload` is passed as ''.
 */
function mapStrings(payload, fn) {
  const root = isRecord(payload) ? payload : {};
  const projects = projectsOf(root.projects);
  const text = (value, field) => fn(typeof value === 'string' ? value : '', field);
  const list = (value, field) => (Array.isArray(value) ? value : []).map((item) => text(item, field));
  return {
    summary: list(root.summary, 'summary'),
    experience: records(root.experience).map((job) => ({
      company: text(job.company, 'company'),
      role: text(job.role, 'role'),
      location: text(job.location, 'location'),
      dates: text(job.dates, 'dates'),
      blocks: records(job.blocks).map((block) => ({
        lead: text(block.lead, 'lead'),
        intro: text(block.intro, 'intro'),
        bullets: list(block.bullets, 'bullets'),
      })),
    })),
    projects: {
      intro: text(projects.intro, 'intro'),
      items: records(projects.items).map((item) => ({ name: text(item.name, 'name'), text: text(item.text, 'text'), url: text(item.url, 'url') })),
    },
    education: records(root.education).map((entry) => ({
      degree: text(entry.degree, 'degree'),
      institution: text(entry.institution, 'institution'),
      dates: text(entry.dates, 'dates'),
    })),
    certifications: list(root.certifications, 'certifications'),
    skills: records(root.skills).map((skill) => ({ category: text(skill.category, 'category'), items: text(skill.items, 'items') })),
  };
}

/**
 * A copy of a payload with every string of the schema replaced by `fn` of it,
 * the project addresses included. Keys the schema does not have are not
 * copied, and the payload is not changed.
 *
 * `fn` is given the string alone. restoreIfEnabled(text, profileYml), passed
 * as it is, would take anything given after the string for the profile and
 * restore nothing, so a function of more parameters needs a wrapper.
 *
 * @param {ReturnType<typeof normalizeCvPayload>} payload
 * @param {(text: string) => string} fn
 * @returns {ReturnType<typeof normalizeCvPayload>}
 */
export function mapPayloadStrings(payload, fn) {
  return mapStrings(payload, (value) => fn(value));
}

/**
 * The text of a CV a reader sees: every string of the payload but the project
 * addresses, in the schema's order, one space apart. An address is no more
 * text on the page than an href is on the HTML path. What the minimum length
 * of a CV is measured on.
 *
 * @param {ReturnType<typeof normalizeCvPayload>} payload
 * @returns {string}
 */
export function payloadText(payload) {
  const parts = [];
  mapStrings(payload, (value, field) => {
    if (field !== 'url' && value) parts.push(value);
    return value;
  });
  return parts.join(' ');
}

// ---------------------------------------------------------------------------
// The header
// ---------------------------------------------------------------------------

/**
 * The facts of the CV header, from `candidate` in config/profile.yml.
 *
 * They are read by code: the model is never asked for a name or a contact
 * detail, so it has none to get wrong. Each value is normalised like payload
 * text (cleanText) and returned unescaped, '' when the profile has none, for
 * the renderer to escape once. The three addresses are the
 * exception: `emailUrl`, `linkedin.url` and `github.url` have been through
 * sanitizeUrl(), ready for \href. A link's `display` is the profile's own text
 * without its scheme or a trailing slash.
 *
 * A profile that is missing, does not parse or has no `candidate` mapping
 * gives an empty header, never an error: the runner reports a missing profile
 * where it always has. So an empty `name` does not say why it is empty: a
 * caller that has to tell a profile that does not parse from one without a
 * name (the runner's check for candidate.full_name) parses the text with
 * loadYaml itself first, and reports the YAML error.
 *
 * @param {string} profileYmlText The text of config/profile.yml.
 * @returns {{
 *   name: string, firstName: string, lastName: string, title: string, location: string,
 *   email: string, emailUrl: string, phone: string,
 *   linkedin: {url: string, display: string}, github: {url: string, display: string},
 * }}
 */
export function readCvHeader(profileYmlText) {
  let candidate = {};
  try {
    const profile = loadYaml(String(profileYmlText ?? ''));
    if (isRecord(profile) && isRecord(profile.candidate)) candidate = profile.candidate;
  } catch { /* a profile that does not parse gives no header facts */ }
  const name = cleanText(candidate.full_name);
  const words = name ? name.split(' ') : [];
  const email = cleanText(candidate.email);
  const link = (value) => {
    const raw = cleanText(value);
    return { url: sanitizeUrl(raw), display: raw.replace(/^[a-z][a-z\d+.-]*:\/\//i, '').replace(/\/+$/, '') };
  };
  return {
    name,
    firstName: words.length > 1 ? words.slice(0, -1).join(' ') : name,
    lastName: words.length > 1 ? words.at(-1) : '',
    title: cleanText(candidate.title),
    location: cleanText(candidate.location),
    email,
    emailUrl: sanitizeUrl(email),
    phone: cleanText(candidate.phone),
    linkedin: link(candidate.linkedin),
    github: link(candidate.github),
  };
}

// ---------------------------------------------------------------------------
// The projection the checks read
// ---------------------------------------------------------------------------
//
// The markup is the one the checks are known to read, field by field:
//   - A field is a div or a span with its class in double quotes. The fact
//     check finds fields with one pattern that takes nothing else, and a
//     field it does not find it does not check: written as <p class=…> or
//     with single quotes, an invented employer passed.
//   - A skill category is a span.skill-category of its own. Written as text,
//     "Languages: Python" became one claim, "languages: python", that no
//     source holds, and the line blocked.
//   - Within a section the text follows cv.md's order (company, location,
//     role, dates, then per block the lead, the intro and the bullets): the
//     locked-section check compares a section's words, in order, with cv.md's.
//   - Every character that means something in HTML is escaped, the backslash
//     too, as lib/cv-payload-facts.mjs escapes upstream's own projection. The
//     fact check reads a text holding `\begin{document}` as a LaTeX document
//     and drops everything before it.

/** The six sections of a payload, in the order the projection writes them. */
const SECTION_KEYS = ['summary', 'experience', 'projects', 'education', 'certifications', 'skills'];

/** The title of a section cv.md has no heading for. */
const USUAL_TITLES = Object.freeze({
  summary: 'Professional Summary',
  experience: 'Work Experience',
  projects: 'Projects',
  education: 'Education',
  certifications: 'Certifications',
  skills: 'Skills',
});

/** A Markdown heading, its hashes and its text, read the way validateLockedSections() reads cv.md's. */
const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/;

/** Text as HTML text and attribute values: the escape of lib/cv-payload-facts.mjs, `\` included. */
function escapeHtml(value) {
  return String(value ?? '').replace(/&/g, '&amp;').replace(/\\/g, '&#92;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

const HTML_ESCAPES = { amp: '&', '#92': '\\', lt: '<', gt: '>', quot: '"', '#39': '\'' };

/** What escapeHtml() wrote, as it was before. One pass, so `&amp;lt;` comes back as `&lt;`, not as `<`. */
function unescapeHtml(text) {
  return String(text).replace(/&(amp|#92|lt|gt|quot|#39);/g, (_, name) => HTML_ESCAPES[name]);
}

/**
 * The title each section of the projection gets: the heading cv.md gives it,
 * else the usual one (Professional Summary, Work Experience, Projects,
 * Education, Certifications, Skills).
 *
 * The locked-section check of generate-pdf.mjs compares a section of cv.md
 * with the section of the CV that carries the same key, and reads that key
 * from the title. So a heading is taken as validateLockedSections() takes it:
 * of two headings with one key, the shallower, then the first. It is taken
 * only when its escaped form, which is what the projection holds, still
 * carries its key. A heading with a comment in it lost the key that way,
 * because the escaped comment is text and no longer a tag the check skips.
 * cv.md's own headings also keep the projection in the language of the CV:
 * the fact check tells from the text which language a CV is in, and runs its
 * drift check on English ones only.
 *
 * `certifications` is null when cv.md has no heading of its own for them, one
 * the check reads as the certifications section: the projection then writes
 * them at the end of the education section, with no title. That is where
 * cv.md holds the certifications it lists under "Education" or "Education &
 * Certifications", both read as the education section, so a locked education
 * section compares equal. Under a title of their own they would read as taken
 * out of it, with or without them in the payload, and no payload could pass.
 * A certification cv.md does not hold lands there too, where the fact check
 * blocks it and a locked education section fails as well. The limit of the
 * rule: a heading the check reads under a key of its own, such as
 * "Certificates" or "Licenses & Certifications", does not count as theirs,
 * so with education locked, a payload that keeps the certifications listed
 * under it fails the locked-section check.
 *
 * @param {string} cvMarkdown The text of cv.md.
 * @param {(heading: string) => string} sectionKey generate-pdf.mjs's sectionKey(), which folds a heading to its
 *   section's key: "Work Experience" and "Berufserfahrung" to `experience`.
 * @returns {{summary: string, experience: string, projects: string, education: string, certifications: string|null,
 *   skills: string}}
 */
export function sectionTitles(cvMarkdown, sectionKey) {
  const headings = new Map();
  for (const line of String(cvMarkdown ?? '').split(/\r?\n/)) {
    const heading = HEADING_RE.exec(line);
    if (!heading) continue;
    const key = sectionKey(heading[2]);
    const level = heading[1].length;
    if (!SECTION_KEYS.includes(key) || (headings.has(key) && headings.get(key).level <= level)) continue;
    headings.set(key, { level, text: heading[2] });
  }
  const titles = Object.fromEntries(SECTION_KEYS.map((key) => {
    const text = headings.get(key)?.text;
    return [key, text !== undefined && sectionKey(escapeHtml(text)) === key ? text : USUAL_TITLES[key]];
  }));
  if (!headings.has('certifications')) titles.certifications = null;
  return titles;
}

/**
 * The six titles a projection writes, merged key by key: each title `titles`
 * gives as text, else the usual one, and `certifications: null` as given.
 * Spread over the usual titles, a key given as undefined, null or '' would
 * write a section title with no text, and the locked-section check gives the
 * text under such a title to no section: a locked section so titled would
 * read as modified.
 */
function titlesOf(titles) {
  const given = isRecord(titles) ? titles : {};
  return Object.fromEntries(SECTION_KEYS.map((key) => {
    const title = given[key];
    if (key === 'certifications' && title === null) return [key, null];
    return [key, typeof title === 'string' && title.trim() ? title : USUAL_TITLES[key]];
  }));
}

/** A field: a div with its class in double quotes, or nothing when there is no text. */
function field(cls, text) {
  return text ? `<div class="${cls}">${escapeHtml(text)}</div>` : '';
}

/**
 * A job: company, location, role and dates, then each block's lead, intro and bullets, in cv.md's order. The dates as
 * `shownDates` gives them, as the template prints them.
 */
function jobLines(job, shownDates) {
  const lines = ['<div class="job">'];
  for (const line of [field('job-company', job.company), job.location ? `<div>${escapeHtml(job.location)}</div>` : '',
    field('job-role', job.role), field('job-period', shownDates(job.dates))]) {
    if (line) lines.push(line);
  }
  for (const block of job.blocks) {
    if (block.lead) lines.push(`<p>${escapeHtml(block.lead)}</p>`);
    if (block.intro) lines.push(`<p>${escapeHtml(block.intro)}</p>`);
    if (block.bullets.length > 0) lines.push(`<ul>${block.bullets.map((bullet) => `<li>${escapeHtml(bullet)}</li>`).join('')}</ul>`);
  }
  lines.push('</div>');
  return lines;
}

/**
 * A project, its name a link when it has an address. The anchor carries the
 * project's index, so that after the link rule dropPayloadLinks() can tell
 * which project's link stayed without comparing addresses, which the HTML
 * spells escaped.
 */
function projectLine(item, index) {
  const name = item.url ? `<a data-p="${index}" href="${escapeHtml(item.url)}">${escapeHtml(item.name)}</a>` : escapeHtml(item.name);
  return `<div><div>${name}</div>${item.text ? `<p>${escapeHtml(item.text)}</p>` : ''}</div>`;
}

/**
 * A skills line, with its index for readSkillsBack(), the category in a span
 * of its own and the items as bare text after it: the runner's repair takes
 * an item out of a line only when nothing but text follows the category.
 */
function skillLine(skill, index) {
  const category = skill.category ? `<span class="skill-category">${escapeHtml(skill.category)}:</span> ` : '';
  return `<div class="skill-item" data-i="${index}">${category}${escapeHtml(skill.items)}</div>`;
}

/**
 * A payload as the plain HTML the checks of an HTML CV read: the fact check,
 * the locked sections, the order of the roles, the link rule and the repair.
 * Never rendered: the checks read it, and the .tex is written from the
 * payload.
 *
 * Each section starts with a div.section-title and runs to the next one; a
 * section with nothing to show is left out with its title, as the template
 * leaves it out, and nothing follows the last section, because the
 * locked-section check reads the last one to the end of the text. A job is a
 * div.job holding div.job-company, a plain div for the location, div.job-role
 * and div.job-period, then for each block a <p> for the lead, a <p> for the
 * intro and its bullets as <ul><li>. The summary is a <p> per paragraph; a
 * project is its name, linked as <a data-p="<index>" href=…> when it has an
 * address, and its text in a <p>; an education entry is a div.edu-item of
 * div.edu-title, div.edu-org and div.edu-year; a certification is a
 * span.cert-title; a skills line is a div.skill-item data-i="<index>" holding
 * a span.skill-category with the category and a colon, and the items as bare
 * text. An empty field is left out.
 *
 * The payload is normalised first (normalizeCvPayload), and the indexes are
 * those of the normalised payload, which readSkillsBack() and
 * dropPayloadLinks() normalise alike.
 *
 * The dates of a job and of an education entry are written as the template
 * prints them (`shownDates`, from shownDatesFor()): the years alone for one
 * that sets dates=year. The checks then read the dates that print. Read as
 * the payload holds them, a range whose remark the fact check cannot parse
 * passed with a warning while the PDF printed a year cv.md contradicts, and
 * a locked section passed while it printed without its months. The payload
 * keeps its dates as they are written: the .json is what a template prints
 * from.
 *
 * @param {unknown} payload
 * @param {Partial<ReturnType<typeof sectionTitles>>} [titles] From sectionTitles(); a title left out, or given with
 *   no text, is the usual one. `certifications: null` writes the certifications at the end of the education section,
 *   with no title.
 * @param {{shownDates?: (dates: string) => string}} [options] `shownDates`: the dates as the template prints them,
 *   shownDatesFor() of its options. Default as they are written.
 * @returns {string}
 */
export function projectCvHtml(payload, titles, options) {
  const cv = normalizeCvPayload(payload);
  const named = titlesOf(titles);
  const given = options?.shownDates;
  const shownDates = typeof given === 'function' ? (dates) => String(given(dates) ?? '') : (dates) => dates;
  const lines = [];
  const title = (text) => lines.push(`<div class="section-title">${escapeHtml(text)}</div>`);
  if (cv.summary.length > 0) {
    title(named.summary);
    for (const paragraph of cv.summary) lines.push(`<p>${escapeHtml(paragraph)}</p>`);
  }
  if (cv.experience.length > 0) {
    title(named.experience);
    for (const job of cv.experience) lines.push(...jobLines(job, shownDates));
  }
  if (cv.projects.items.length > 0) {
    title(named.projects);
    if (cv.projects.intro) lines.push(`<p>${escapeHtml(cv.projects.intro)}</p>`);
    cv.projects.items.forEach((item, index) => lines.push(projectLine(item, index)));
  }
  const certifications = cv.certifications.map((text) => `<div><span class="cert-title">${escapeHtml(text)}</span></div>`);
  const underEducation = named.certifications === null;
  if (cv.education.length > 0 || (underEducation && certifications.length > 0)) {
    title(named.education);
    for (const entry of cv.education) {
      lines.push(`<div class="edu-item">${field('edu-title', entry.degree)}${field('edu-org', entry.institution)}`
        + `${field('edu-year', shownDates(entry.dates))}</div>`);
    }
    if (underEducation) lines.push(...certifications);
  }
  if (!underEducation && certifications.length > 0) {
    title(named.certifications);
    lines.push(...certifications);
  }
  if (cv.skills.length > 0) {
    title(named.skills);
    cv.skills.forEach((skill, index) => lines.push(skillLine(skill, index)));
  }
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// The way back
// ---------------------------------------------------------------------------

/**
 * A skills line as skillLine() writes it, read after the repair: its index,
 * then its items as the HTML has them. The repair keeps the opening tag and
 * the category as they were and changes only the text after them; a line
 * that lost every item it removes whole. Text that spells such a line is
 * escaped in the projection, so it cannot be taken for one.
 */
const SKILL_LINE_RE = /<div class="skill-item" data-i="(\d+)">(?:<span class="skill-category">[^<]*<\/span>)?([^<]*)<\/div>/g;

/**
 * The payload with the skills lines as the runner's repair left them in its
 * projection.
 *
 * The repair (pruneRejectedEntries() in ollama-runner.mjs) takes out of a
 * skills line an item the fact check rejected, writes a line without the
 * brackets the check misread, and removes a line that lost every item. It
 * works on the projection, and the .tex is written from the payload, so what
 * it did is read back: each line by the index projectCvHtml() gave it, its
 * items unescaped, its category the payload's own. A line the HTML no longer
 * holds was removed, and goes.
 *
 * @param {string} html The projection after the repair.
 * @param {unknown} payload The payload that was projected.
 * @returns {ReturnType<typeof normalizeCvPayload>} A normalised copy; the payload passed in is not changed.
 */
export function readSkillsBack(html, payload) {
  const cv = normalizeCvPayload(payload);
  const lists = new Map();
  for (const [, index, list] of String(html ?? '').matchAll(SKILL_LINE_RE)) lists.set(Number(index), unescapeHtml(list));
  return normalizeCvPayload({
    ...cv,
    skills: cv.skills.flatMap((skill, index) => (lists.has(index) ? [{ category: skill.category, items: lists.get(index) }] : [])),
  });
}

/**
 * The payload without the project links the runner's link rule took off its
 * projection.
 *
 * withoutUnsupportedLinks() in ollama-runner.mjs removes a link whose address
 * none of the candidate's files holds, keeping its text, and names each
 * address as the HTML spells it: `&` as `&amp;`, `'` as `&#39;`. Compared
 * with those names, an unsupported address holding either kept its link in
 * the payload while the run said it was taken off. So the projects are read
 * by index instead: a project keeps its address only when its anchor
 * (`<a data-p="<index>"`, which payload text cannot spell, being escaped) is
 * still in the HTML.
 *
 * @param {unknown} payload The payload that was projected.
 * @param {string} html The projection after withoutUnsupportedLinks().
 * @returns {{payload: ReturnType<typeof normalizeCvPayload>, dropped: string[]}} A normalised copy of the payload,
 *   and the addresses taken off, each once, as the payload has them. The payload passed in is not changed.
 */
export function dropPayloadLinks(payload, html) {
  const cv = normalizeCvPayload(payload);
  const linked = new Set([...String(html ?? '').matchAll(/<a data-p="(\d+)"/g)].map(([, index]) => Number(index)));
  const dropped = [];
  const items = cv.projects.items.map((item, index) => {
    if (!item.url || linked.has(index)) return item;
    if (!dropped.includes(item.url)) dropped.push(item.url);
    return { ...item, url: '' };
  });
  return { payload: { ...cv, projects: { ...cv.projects, items } }, dropped };
}

// ---------------------------------------------------------------------------
// Years only
// ---------------------------------------------------------------------------

/** A year: four digits that are not part of a longer number. */
const YEAR_RE = /(?<!\d)\d{4}(?!\d)/;

/**
 * Where a range of dates is split into its start and its end, the first one
 * met: a dash (one or two hyphens, an en or an em dash) or "to" with spacing
 * on both sides, or a dash written straight after a year and before a word
 * or another year. Not a dash before two digits: "2019-03" is one month.
 */
const RANGE_SEPARATOR_RE = /\s+(?:[-\u2013\u2014]{1,2}|to)\s+|(?<=(?<!\d)\d{4})[-\u2013\u2014]{1,2}(?=\p{L}|\d{4}(?!\d))/iu;

/**
 * The words that open the end of a range: "Present", "Till Date", "Mar". A
 * word is letters, with digits, dots and apostrophes after the first letter,
 * and the words run to the first character that is neither a word nor the
 * space between two: "Present (promoted 2022)" opens with "Present".
 */
const END_WORDS_RE = /^\p{L}[\p{L}\p{M}\p{N}'’.]*(?:[ \t]+\p{L}[\p{L}\p{M}\p{N}'’.]*)*/u;

/**
 * An end that is a year after one word, a month in any language, with a day
 * beside it or not: "Mar 2020", "Sept. 2020", "März 2020",
 * "March 1, 2020", "Q1 2020".
 */
const END_YEAR_RE = /^\p{L}[\p{L}\p{M}\p{N}]*\.?,?\s+(?:\d{1,2}(?:st|nd|rd|th)?[.,]?\s+)?(\d{4})(?!\d)/iu;

/**
 * Words that end a range held today, in the languages career-ops has modes
 * for, in lower case. As the first word after the separator, one of them is
 * the end whatever follows it: read by END_YEAR_RE as a month,
 * "Present, 2022" ended in 2022. A word missing from the list is read as a
 * month when a year follows it, and the checks read that year as it prints
 * (shownDatesFor()), so a year cv.md does not give is found.
 */
const PRESENT_WORDS = new Set([
  'present', 'current', 'currently', 'now', 'today', 'ongoing', 'date',
  'heute', 'jetzt', 'aktuell', 'laufend', 'derzeit', 'dato', 'heden', 'nu', 'nå', 'idag', 'nuværende', 'nåværende', 'pågående',
  'présent', 'aujourd\'hui', 'aujourd’hui', 'actuel', 'actuellement', 'maintenant',
  'presente', 'actualidad', 'actual', 'actualmente', 'hoy', 'atual', 'atualmente', 'hoje', 'oggi', 'attuale', 'attualmente',
  'obecnie', 'teraz', 'dziś', 'dzisiaj', 'nadal', 'настоящее', 'сейчас', 'зараз', 'тепер', 'нині', 'дотепер',
  'günümüz', 'halen', 'şimdi', 'sekarang', 'kini', 'nay', '至今', '迄今', '现在', '現在', '目前', '현재', 'الآن', 'حاليا', 'वर्तमान', 'अब',
]);

/**
 * Where a range ends, as yearsOnly() shows it, from the text after its
 * separator. Text that opens with words ends at those words when the first
 * is one of PRESENT_WORDS, at a year when one word stands before the year, a
 * month, and otherwise at those words, and what follows them is dropped:
 * "Present (promoted 2022)" and "Present, 2022" end at Present, not in 2022,
 * and "Till Date" at Till Date. Text that opens otherwise ends at its first
 * year, or is the end as it is written when it holds none.
 */
function rangeEnd(right) {
  const words = END_WORDS_RE.exec(right)?.[0];
  if (!words) return YEAR_RE.exec(right)?.[0] ?? right;
  const first = words.split(/[ \t]/, 1)[0].normalize('NFC').toLowerCase().replace(/\.+$/, '');
  if (PRESENT_WORDS.has(first)) return words;
  return END_YEAR_RE.exec(right)?.[1] ?? words;
}

/**
 * The years of a range of dates, for a template that shows no months
 * (`% career-ops-latex: dates=year`).
 *
 * The text is split once, at the first separator. The start is the first year
 * before it. The end is the year after it, written alone or after a month
 * ("Mar 2020"), or else the word that ends the range ("Present"), and what
 * follows that word is dropped (rangeEnd()): when the first year after the
 * separator was taken for the end, a job held today with a remark after
 * "Present" printed as one that ended in the year of the remark. A range
 * within one year is that year. Text with no year before the separator is
 * returned as it is.
 *
 * A dash straight after a year splits too, before a word as well as before a
 * year: "2019-Present" is a range, and read as the year 2019 alone it would
 * print as a job that ended that year.
 *
 * Two things it cannot tell from the text. The words around a lone year go
 * with its month: `Expected 2027` shows as `2027`, a degree that reads as
 * finished. And a dash before two digits never splits, so `2019-20` shows as
 * `2019`: an ISO `2011-12` is a month, and an academic year is spelled the
 * same. A template that sets dates=year shows such dates that way. The
 * runner's checks read the dates as this function shows them
 * (shownDatesFor()), so a year that cv.md contradicts is found.
 *
 * @param {string} dates As cv.md writes them: `Mar 2019 – Present`, `2020-2024`.
 * @returns {string} `2019–Present`, `2020–2024`, with an en dash.
 */
export function yearsOnly(dates) {
  const text = String(dates ?? '');
  const separator = RANGE_SEPARATOR_RE.exec(text);
  const start = YEAR_RE.exec(separator ? text.slice(0, separator.index) : text)?.[0];
  if (!start) return text;
  const right = separator ? text.slice(separator.index + separator[0].length).trim() : '';
  const end = rangeEnd(right);
  return end && end !== start ? `${start}\u2013${end}` : start;
}

/**
 * How a template shows the dates of a job and of an entry of the education:
 * as they are written, or, for one that sets `% career-ops-latex: dates=year`,
 * as their years (yearsOnly()). renderCvTex() prints them so, and the runner
 * has the checks read them so (projectCvHtml()): checked as they are written,
 * a year that cv.md contradicts, and a locked section printed without its
 * months, reached the PDF.
 *
 * @param {{dates?: 'full'|'year'}} [options] A template's options, as resolveLatexTemplate() and
 *   readTemplateOptions() give them.
 * @returns {(dates: string) => string}
 */
export function shownDatesFor(options) {
  return options?.dates === 'year' ? yearsOnly : (dates) => String(dates ?? '');
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------
//
// A template is a .tex with placeholders ({{NAME}}, {{EXPERIENCE}}) and
// blocks, the lines from `%%IF KEY` to `%%END KEY`, left out when KEY has
// nothing to show. Its header lines, the comment lines before \documentclass,
// name its engine (`%!TEX TS-program = xelatex`) and the renderer's options
// (`% career-ops-latex: dates=year`). A rendered .tex keeps them, after a line
// 1 of the renderer's own that names the template.

/** The template a CV is written in when none is named. It is lib/ollama-cv-latex-default.tex, not a folder. */
export const DEFAULT_LATEX_TEMPLATE = 'default';

/**
 * The engines a template may name. Not lualatex: it runs Lua code from the
 * document, which reads and writes files and starts programs of its own,
 * whatever the flags of the compile forbid.
 *
 * @type {readonly string[]}
 */
export const LATEX_ENGINES = Object.freeze(['xelatex', 'pdflatex', 'tectonic']);

const LUALATEX_REFUSED = 'lualatex is not supported: it runs Lua code from the document. Use xelatex.';

/** A template's name: the name of its folder under data/templates/, and what line 1 of a rendered .tex gives. */
const TEMPLATE_NAME_RE = /^[a-z0-9][a-z0-9-]*$/;

/** The template of a pack, in its folder. Not cv-template.<name>.tex, a name upstream's own template resolver claims. */
const PACK_FILE = 'cv-latex.tex';

const DEFAULT_TEMPLATE_PATH = fileURLToPath(new URL('./ollama-cv-latex-default.tex', import.meta.url));

/** A placeholder, filled in one pass (renderCvTex). */
const PLACEHOLDER_RE = /\{\{([A-Z_]+)\}\}/g;

/**
 * What a template's author may have meant as a placeholder: a name in double
 * braces, with spaces inside them, in any case, with digits after its first
 * letter or underscore. Every match of PLACEHOLDER_RE is one too, so whatever
 * the renderer would fill is held to the list. Only a name: braces of TeX
 * code (`{{\bfseries #1}}`) are not, nor a group that begins with a digit, a
 * length or a number (`{{0pt}}`), which no placeholder does. A group of one
 * word of letters (`\def\x{{a}}`) is, and is refused.
 */
const PLACEHOLDER_LIKE_RE = /\{\{\s*[A-Za-z_]\w*\s*\}\}/g;

/** The sections of a CV as the placeholders and the blocks of a template name them. */
const SECTION_PLACEHOLDERS = ['SUMMARY', 'EXPERIENCE', 'PROJECTS', 'EDUCATION', 'CERTIFICATIONS', 'SKILLS'];

const PLACEHOLDERS = [
  'NAME', 'FIRST_NAME', 'LAST_NAME', 'TITLE', 'LOCATION', 'EMAIL', 'EMAIL_URL', 'PHONE', 'LINKEDIN_URL', 'LINKEDIN_DISPLAY', 'GITHUB_URL',
  'GITHUB_DISPLAY', 'DATE', 'PAPER', ...SECTION_PLACEHOLDERS,
];

/** What a block can be about: a section, or a fact of the header the profile may leave out. */
const BLOCK_KEYS = [...SECTION_PLACEHOLDERS, 'TITLE', 'EMAIL', 'PHONE', 'LINKEDIN', 'GITHUB', 'LOCATION'];

/**
 * A line that opens or closes a block. Any line whose first word after `%%`
 * is IF or END, in capitals, is taken for one, and refused when it names no
 * block: a misspelt key would otherwise show its lines in every CV, or in
 * none.
 */
const MARKER_RE = /^[ \t]*%%[ \t]*(IF|END)(?=[ \t]|$)(.*)$/;

/** The engine line TeX editors read: `%!TEX TS-program = xelatex`, or `% !TeX program = xelatex`. */
const ENGINE_LINE_RE = /^%\s*!\s*TEX\s+(?:TS-)?program\s*=\s*(.*?)\s*$/i;

/** A line of the renderer's options: `% career-ops-latex: dates=year`. */
const OPTION_LINE_RE = /^%\s*career-ops-latex\s*:\s*(.*?)\s*$/i;

const PAPER_OPTIONS = new Map([['a4', 'a4paper'], ['letter', 'letterpaper']]);

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** A template or a .tex as its lines are read: no byte order mark, LF line ends. */
function normalizedText(text) {
  return String(text ?? '').replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n');
}

/** A line without its comment. A `%` is a comment unless a backslash makes it a character, and `\\` is one control symbol. */
function codeOf(line) {
  for (let index = 0; index < line.length; index += 1) {
    if (line[index] === '\\') index += 1;
    else if (line[index] === '%') return line.slice(0, index);
  }
  return line;
}

/**
 * The header lines of a template or a .tex: the lines before the one with
 * \documentclass in its code, trimmed, with their numbers. A comment that
 * names \documentclass does not end them.
 */
function headerLines(text) {
  const lines = [];
  for (const [index, line] of normalizedText(text).split('\n').entries()) {
    if (/\\documentclass(?![A-Za-z])/.test(codeOf(line))) break;
    lines.push({ number: index + 1, text: line.trim() });
  }
  return lines;
}

/** `a, b or c`. */
function oneOf(items) {
  return `${items.slice(0, -1).join(', ')} or ${items.at(-1)}`;
}

/**
 * What the header lines set: the engine, the dates and the template line 1
 * of a rendered .tex names, with the line each was set on. Or the first
 * problem with them, as `{ line, text }`: an engine that is not one of
 * LATEX_ENGINES, an option or a value the renderer does not know, two
 * options on one line, or one set twice, which would leave it to the order
 * of the lines which one counts.
 */
function readOptions(tex) {
  const options = { engine: 'pdflatex', dates: 'full', template: null };
  const setOn = {};
  for (const { number, text } of headerLines(tex)) {
    const refuse = (detail) => ({ options, setOn, problem: { line: number, text: detail } });
    let key;
    let value;
    const engine = ENGINE_LINE_RE.exec(text);
    if (engine) {
      key = 'engine';
      value = engine[1].toLowerCase();
      if (value === 'lualatex') return refuse(LUALATEX_REFUSED);
      if (!LATEX_ENGINES.includes(value)) return refuse(`the engine line names "${engine[1]}": a template runs on ${oneOf(LATEX_ENGINES)}.`);
    } else {
      const option = OPTION_LINE_RE.exec(text);
      if (!option) continue;
      const pair = /^([^\s=]+)\s*=\s*([^\s=]+)$/.exec(option[1]);
      if (!pair) return refuse(`"${option[1]}" is not one option: write one key=value on each line, such as dates=year.`);
      [, key, value] = pair;
      if (key === 'dates') {
        if (value !== 'full' && value !== 'year') return refuse(`dates is full or year, not "${value}".`);
      } else if (key === 'template') {
        if (!TEMPLATE_NAME_RE.test(value)) return refuse(`"${value}" is not a template name.`);
      } else {
        return refuse(`career-ops-latex has no option "${key}": its options are dates and template.`);
      }
    }
    if (setOn[key]) return refuse(`${key === 'engine' ? 'the engine' : key} was set on line ${setOn[key]} already.`);
    setOn[key] = number;
    options[key] = value;
  }
  return { options, setOn, problem: null };
}

/**
 * The engine a template or a .tex compiles with, and how it shows dates:
 * read from its header lines, the comment lines before \documentclass.
 *
 * `%!TEX TS-program = <engine>` names the engine, one of LATEX_ENGINES;
 * without it the engine is pdflatex. `% career-ops-latex: dates=year` shows
 * the years of each range of dates only (yearsOnly); without it dates are
 * shown as written. `% career-ops-latex: template=<name>` is read too: it is
 * line 1 of a rendered .tex.
 *
 * @param {string} tex
 * @returns {{engine: string, dates: 'full'|'year'}}
 * @throws {Error} On a template problem, as `line <n>: <what is wrong>`: an engine not in LATEX_ENGINES (lualatex
 *   with a sentence of its own), an option or a value the renderer does not know, two options on one line, or one
 *   option set twice.
 */
export function readTemplateOptions(tex) {
  const { options, problem } = readOptions(tex);
  if (problem) throw new Error(`line ${problem.line}: ${problem.text}`);
  return { engine: options.engine, dates: options.dates };
}

/**
 * The template a rendered .tex was made from, as its line 1 names it
 * (`% career-ops-latex: template=<name>`), or null.
 *
 * Line 1 only, and a name only when it is one a template can have, so what
 * this returns can be handed to resolveLatexTemplate() and never names a path
 * outside data/templates/.
 *
 * @param {string} tex
 * @returns {string|null}
 */
export function templateNameOf(tex) {
  const option = OPTION_LINE_RE.exec(normalizedText(tex).split('\n', 1)[0].trim());
  const pair = option && /^template\s*=\s*(\S+)$/.exec(option[1]);
  return pair && TEMPLATE_NAME_RE.test(pair[1]) ? pair[1] : null;
}

/** The first block that is not closed, crossed or about anything, as `{ line, text }`, or null. Blocks may nest. */
function blockProblem(lines) {
  const open = [];
  for (const [index, line] of lines.entries()) {
    const marker = MARKER_RE.exec(line);
    if (!marker) continue;
    const [, word, rest] = marker;
    const key = rest.trim();
    const number = index + 1;
    if (!BLOCK_KEYS.includes(key)) {
      return { line: number, text: `"${line.trim()}" names no block: a block runs from %%IF KEY to %%END KEY, KEY one of ${BLOCK_KEYS.join(', ')}.` };
    }
    if (word === 'IF') {
      open.push({ key, number });
      continue;
    }
    const last = open.pop();
    if (!last) return { line: number, text: `%%END ${key} has no %%IF ${key} before it.` };
    if (last.key !== key) return { line: number, text: `%%END ${key} closes %%IF ${last.key} of line ${last.number}: blocks nest, and never cross.` };
  }
  const unclosed = open.pop();
  return unclosed ? { line: unclosed.number, text: `%%IF ${unclosed.key} has no %%END ${unclosed.key} after it.` } : null;
}

/**
 * The first placeholder that is not one of PLACEHOLDERS written as the
 * renderer fills it, as `{ line, text }`, or null. A placeholder in another
 * spelling (`{{ NAME }}`, `{{Name}}`, `{{PHONE2}}`) is refused too: neither
 * filled nor refused, it stayed in the .tex, and the CV printed its name in
 * place of the candidate's. When it is a placeholder of the list in another
 * spelling, the reason gives the one to write.
 */
function placeholderProblem(lines) {
  for (const [index, line] of lines.entries()) {
    for (const [written] of line.matchAll(PLACEHOLDER_LIKE_RE)) {
      const key = written.slice(2, -2);
      if (PLACEHOLDERS.includes(key)) continue;
      const meant = key.trim().toUpperCase();
      const text = PLACEHOLDERS.includes(meant)
        ? `${written} is not a placeholder of a LaTeX CV: write {{${meant}}}.`
        : `${written} is not a placeholder of a LaTeX CV.`;
      return { line: index + 1, text };
    }
  }
  return null;
}

/**
 * The first problem of a template, as `{ line, text }`, or null: its header
 * lines (readOptions), then a `template=` line, which only the renderer
 * writes, then its blocks, then its placeholders. A .tex rendered from a
 * template carries `template=` on line 1, so a rendered CV copied into a pack
 * as its template is refused: it would print that CV, whatever the payload.
 */
function templateProblem(text) {
  const { options, setOn, problem } = readOptions(text);
  if (problem) return problem;
  if (options.template !== null) {
    return { line: setOn.template, text: 'template= is the renderer\'s to write, on line 1 of each .tex it renders: a template does not carry it.' };
  }
  const lines = normalizedText(text).split('\n');
  return blockProblem(lines) ?? placeholderProblem(lines);
}

/** The packs in data/templates/: each folder with a template name and a cv-latex.tex, sorted. A folder named default is not one. */
function packNames(templatesDir) {
  if (!templatesDir) return [];
  let names;
  try {
    names = readdirSync(templatesDir);
  } catch {
    return [];
  }
  return names
    .filter((name) => TEMPLATE_NAME_RE.test(name) && name !== DEFAULT_LATEX_TEMPLATE && existsSync(join(templatesDir, name, PACK_FILE)))
    .sort();
}

/**
 * The template a CV is written in, found and checked before anything else
 * happens, so a template that cannot be used is refused before the model has
 * been asked for anything.
 *
 * `default` is lib/ollama-cv-latex-default.tex, which has no folder. Any
 * other name is a pack: the folder `{dataRoot}/data/templates/<name>/` with
 * `cv-latex.tex` in it, beside the class, fonts and images it uses, all of
 * which are copied to the build folder with it. A pack is code: the engine
 * runs its class and preamble with the owner's rights.
 *
 * Refused: a name that is not lower-case letters, digits and hyphens (so
 * `../x` never reaches the file system), a pack with no cv-latex.tex, and a
 * template with a problem: an engine or an option the renderer does not
 * know, a placeholder it does not fill, a block that names nothing, is not
 * closed or crosses another. A reason names the file relative to the data
 * root, and ends with the templates there are.
 *
 * @param {string} name
 * @param {object} [options]
 * @param {string} [options.dataRoot] The data root, whose data/templates/ holds the packs.
 * @returns {{ok: true, name: string, path: string, packDir: string|null, text: string,
 *   options: {engine: string, dates: 'full'|'year'}} | {ok: false, reason: string}} `packDir` is null for the default;
 *   `text` has LF line ends and no byte order mark.
 */
export function resolveLatexTemplate(name, { dataRoot } = {}) {
  const templatesDir = dataRoot ? join(dataRoot, 'data', 'templates') : null;
  const refuse = (reason) => ({ ok: false, reason: `${reason} Templates here: ${[DEFAULT_LATEX_TEMPLATE, ...packNames(templatesDir)].join(', ')}.` });
  if (typeof name !== 'string' || !TEMPLATE_NAME_RE.test(name)) {
    return refuse(`${JSON.stringify(String(name))} is not a template name: a name is lower-case letters, digits and hyphens, `
      + 'and begins with a letter or a digit.');
  }
  const builtIn = name === DEFAULT_LATEX_TEMPLATE;
  const shown = builtIn ? 'lib/ollama-cv-latex-default.tex' : `data/templates/${name}/${PACK_FILE}`;
  const path = builtIn ? DEFAULT_TEMPLATE_PATH : templatesDir && join(templatesDir, name, PACK_FILE);
  if (!path || !existsSync(path)) return refuse(`No template "${name}": ${shown} was not found.`);
  let text;
  try {
    text = normalizedText(readFileSync(path, 'utf8'));
  } catch (err) {
    return refuse(`${shown} could not be read (${err.code ?? err.message}).`);
  }
  const problem = templateProblem(text);
  if (problem) return refuse(`${shown}, line ${problem.line}: ${problem.text}`);
  const { engine, dates } = readOptions(text).options;
  return { ok: true, name, path, packDir: builtIn ? null : dirname(path), text, options: { engine, dates } };
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * What escapeLatex() writes as a command for a character, where the final
 * text check reads the command as words: inside a macro argument after
 * "using", `\textless{}` was the tool "kafka \textless" and `$\rightarrow$`
 * the tool "grafana $\rightarrow$ loki", in a CV true to cv.md. The
 * character itself goes in its place, which a T1 font prints (and the arrow
 * once the template declares it), and the tilde becomes `\~{}`, which the
 * check leaves alone. T1 prints `\~{}` as the accent, U+02DC in the text
 * layer of the PDF, unless the template declares it as \textasciitilde, as
 * lib/ollama-cv-latex-default.tex does; xelatex prints `~` without that.
 */
const WRITTEN_BACK = new Map([
  ['\\textless{}', '<'],
  ['\\textgreater{}', '>'],
  ['\\textbar{}', '|'],
  ['$\\rightarrow$', '\u2192'],
  ['$\\pm$', '\u00B1'],
  ['\\textasciitilde{}', '\\~{}'],
]);

/** The sentence break the final text check reads after each call that carries text. The template defines it to print nothing. */
const STOP = '\\cvStop{.}';

/**
 * A value as LaTeX text: each character escaped once, by escapeLatex() one
 * character at a time so that a write-back never matches across two, with
 * the WRITTEN_BACK characters as themselves; then, where `bold` is set,
 * `**words**` as `{\bfseries words}` (never \textbf, which the final check
 * read as the tool "textbf"); then a `[` that opens the value as `{[}`, so
 * that no macro of any template reads it as an option (\item would take
 * "[Beta]" for its label and lose the bullet).
 */
function texText(value, { bold = false } = {}) {
  const escaped = Array.from(String(value ?? ''), (char) => {
    const command = escapeLatex(char);
    return WRITTEN_BACK.get(command) ?? command;
  }).join('');
  const text = bold ? escaped.replace(BOLD_RE, (_, inner) => `{\\bfseries ${inner}}`) : escaped;
  return text.startsWith('[') ? `{[}${text.slice(1)}` : text;
}

/** A call: the macro, then each argument in braces. An empty argument is `{}`, which a template can tell from text. */
function call(macro, args) {
  return `\\${macro}${args.map((arg) => `{${arg}}`).join('')}`;
}

/** An argument of calls, one on each line, after a `%` that keeps the line end from becoming a space; '' when there are none. */
function itemsArgument(lines) {
  return lines.length > 0 ? `%\n${lines.join('\n')}\n` : '';
}

/** The header facts as placeholder values: the text escaped (never bold), the addresses through sanitizeUrl() alone. */
function headerValues(header) {
  const facts = isRecord(header) ? header : {};
  const text = (value) => texText(cleanText(value));
  const address = (value) => sanitizeUrl(cleanText(value));
  const linkedin = isRecord(facts.linkedin) ? facts.linkedin : {};
  const github = isRecord(facts.github) ? facts.github : {};
  return {
    NAME: text(facts.name),
    FIRST_NAME: text(facts.firstName),
    LAST_NAME: text(facts.lastName),
    TITLE: text(facts.title),
    LOCATION: text(facts.location),
    EMAIL: text(facts.email),
    EMAIL_URL: address(facts.emailUrl),
    PHONE: text(facts.phone),
    LINKEDIN_URL: address(linkedin.url),
    LINKEDIN_DISPLAY: text(linkedin.display),
    GITHUB_URL: address(github.url),
    GITHUB_DISPLAY: text(github.display),
  };
}

/** The six sections of a normalised payload, written in the macros of the vocabulary; '' for a section with nothing to show. */
function sectionValues(cv, shownDates) {
  const plain = (value) => texText(value);
  const rich = (value) => texText(value, { bold: true });
  const entries = (list, lines) => list.flatMap((entry, index) => [...(index > 0 ? ['\\cvEntrySep'] : []), ...lines(entry)]).join('\n');
  return {
    SUMMARY: cv.summary.map((paragraph) => `${call('cvSummary', [rich(paragraph)])}${STOP}`).join('\n'),
    EXPERIENCE: entries(cv.experience, (job) => [
      `${call('cvEntryHead', [plain(job.role), plain(job.company), plain(job.location), plain(shownDates(job.dates))])}${STOP}`,
      ...job.blocks.map((block) => call('cvEntryBlock', [
        rich(block.lead),
        rich(block.intro),
        itemsArgument(block.bullets.map((bullet) => `${call('cvItem', [rich(bullet)])}${STOP}`)),
      ])),
    ]),
    PROJECTS: cv.projects.items.length === 0 ? '' : call('cvProjects', [
      rich(cv.projects.intro),
      itemsArgument(cv.projects.items.map((item) => `${call('cvProject', [plain(item.name), sanitizeUrl(item.url), rich(item.text)])}${STOP}`)),
    ]),
    EDUCATION: entries(cv.education, (entry) => [
      `${call('cvEducation', [plain(entry.degree), plain(entry.institution), plain(shownDates(entry.dates))])}${STOP}`,
    ]),
    CERTIFICATIONS: cv.certifications.map((text) => `${call('cvCert', [rich(text)])}${STOP}`).join('\n'),
    SKILLS: cv.skills.map((skill) => `${call('cvSkill', [plain(skill.category), plain(skill.items)])}${STOP}`).join('\n'),
  };
}

/** The lines of a template without its block markers, and without the lines of each block whose key has nothing to show. */
function keepBlocks(lines, shows) {
  const kept = [];
  const open = [];
  for (const line of lines) {
    const marker = MARKER_RE.exec(line);
    if (marker) {
      if (marker[1] === 'IF') open.push(shows(marker[2].trim()));
      else open.pop();
    } else if (open.every(Boolean)) {
      kept.push(line);
    }
  }
  return kept;
}

/**
 * A CV as a .tex: the payload written into a template.
 *
 * In order: the payload is normalised; the template is checked (a template
 * problem throws, as resolveLatexTemplate() refuses it); each value is
 * written: the sections in the macros of the vocabulary, one call per line,
 * with \cvStop{.} after each call that carries text and \cvEntrySep between
 * two entries of experience and of education; the header facts as escaped
 * text; every URL through sanitizeUrl() alone. With `dates=year` the dates of
 * experience and education are shown as years (yearsOnly). Each value is
 * searched for a command the renderer does not write (unknownTexCommands)
 * before anything is substituted. Then the blocks whose key has nothing to
 * show are dropped, and every placeholder is filled in one pass, with one
 * pattern and a function, so that no value is read again as a placeholder or
 * as a replacement pattern (`$'`). Line 1 names the template:
 * `% career-ops-latex: template=<name>`.
 *
 * @param {object} input
 * @param {string} input.template The template's text, as resolveLatexTemplate() returns it.
 * @param {string} input.templateName Its name, written on line 1.
 * @param {unknown} input.payload The CV payload.
 * @param {ReturnType<typeof readCvHeader>} input.header The header facts, from readCvHeader().
 * @param {'a4'|'letter'} input.paper The paper size, filled in as {{PAPER}}: a4paper or letterpaper.
 * @param {Date} [input.date] The day filled in as {{DATE}}, `21 Nov 2026`. Default today.
 * @returns {{tex: string}}
 * @throws {Error} On a template problem, a name that is not a template name, a paper that is neither a4 nor letter,
 *   a date that is not one, or a command the renderer does not write.
 */
export function renderCvTex({ template, templateName, payload, header, paper, date = new Date() } = {}) {
  if (typeof templateName !== 'string' || !TEMPLATE_NAME_RE.test(templateName)) {
    throw new Error(`${JSON.stringify(String(templateName))} is not a template name.`);
  }
  const paperOption = PAPER_OPTIONS.get(paper);
  if (!paperOption) throw new Error(`The paper of a CV is a4 or letter, not ${JSON.stringify(String(paper))}.`);
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) throw new Error('The date of a CV must be a valid Date.');
  const text = normalizedText(template);
  const problem = templateProblem(text);
  if (problem) throw new Error(`The template, line ${problem.line}: ${problem.text}`);
  const cv = normalizeCvPayload(payload);
  const { dates } = readOptions(text).options;
  const values = {
    ...headerValues(header),
    DATE: texText(`${date.getDate()} ${MONTHS[date.getMonth()]} ${date.getFullYear()}`),
    PAPER: paperOption,
    ...sectionValues(cv, shownDatesFor({ dates })),
  };
  const unknown = [...new Set(Object.values(values).flatMap((value) => unknownTexCommands(value)))];
  if (unknown.length > 0) {
    throw new Error(`The CV text would run a LaTeX command the renderer does not write: ${unknown.map((name) => `\\${name}`).join(', ')}.`);
  }
  // A link shows only with text to show, so LinkedIn and GitHub are there when their display text is.
  const shown = { LINKEDIN: 'LINKEDIN_DISPLAY', GITHUB: 'GITHUB_DISPLAY' };
  const body = keepBlocks(text.split('\n'), (key) => values[shown[key] ?? key] !== '')
    .join('\n')
    .replace(PLACEHOLDER_RE, (_, key) => values[key]);
  return { tex: `% career-ops-latex: template=${templateName}\n${body}` };
}

// ---------------------------------------------------------------------------
// What a rendered CV holds
// ---------------------------------------------------------------------------

/**
 * Every command the renderer writes into a value: the eleven macros of the
 * vocabulary, the bold form, and what escapeLatex() and sanitizeUrl() write
 * for a character once the write-backs are done. A control symbol is named by
 * its character.
 */
const RENDERED_COMMANDS = new Set([
  'cvSummary', 'cvEntryHead', 'cvEntryBlock', 'cvItem', 'cvEntrySep', 'cvProjects', 'cvProject', 'cvEducation', 'cvCert', 'cvSkill', 'cvStop',
  'bfseries', 'textbackslash', 'textasciicircum', '~', '{', '}', '_', '&', '%', '$', '#',
]);

/** A control sequence: a backslash with the letters after it, or the one character after it, or the end of the text. */
const CONTROL_SEQUENCE_RE = /\\([A-Za-z]+|[^A-Za-z]|$)/g;

/**
 * The commands in `text` the renderer never writes, each named once, in the
 * order met; [] when there are none.
 *
 * A check behind the escaping that the text of a CV is inert. Escaping is
 * what stands between the text and the engine, and shell escape off does not
 * stop a file read: one value that missed escapeLatex() could put any file
 * the owner can read into a PDF that goes to a recruiter. So renderCvTex()
 * runs this on every value it substitutes and refuses to render when it
 * finds anything, \href among it: the renderer writes no link of its own,
 * only the address a template's macro links.
 *
 * It reads control sequences, and nothing else: a raw `%`, `{`, `}`, `#` or
 * `^` is none, so `50% done } ^^5cinput{x} #1` gives [] although TeX reads
 * `^^5c` as a backslash. It catches a missed escape of a backslash, not of
 * those characters; escapeLatex() is the only defence there.
 *
 * @param {string} text
 * @returns {string[]} The names without their backslash: `input`, and `\` for the line break `\\`.
 */
export function unknownTexCommands(text) {
  const unknown = [];
  for (const [, name] of String(text ?? '').matchAll(CONTROL_SEQUENCE_RE)) {
    if (!RENDERED_COMMANDS.has(name) && !unknown.includes(name)) unknown.push(name);
  }
  return unknown;
}

/** The macros of the vocabulary that carry text, with the number of arguments of each. */
const TEXT_MACROS = new Map([
  ['cvSummary', 1], ['cvEntryHead', 4], ['cvEntryBlock', 3], ['cvItem', 1], ['cvProjects', 2], ['cvProject', 3], ['cvEducation', 3],
  ['cvCert', 1], ['cvSkill', 2],
]);

/** The argument that holds the calls of a macro's items: the third of \cvEntryBlock, the second of \cvProjects. */
const ITEMS_ARGUMENT = new Map([['cvEntryBlock', 2], ['cvProjects', 1]]);

const LETTERS_RE = /[A-Za-z]+/y;

/**
 * A .tex without its comments, as TeX reads it: a `%` that no backslash makes
 * a character ends the line there, and takes the line end and the spacing
 * that opens the next line with it.
 */
function withoutComments(text) {
  const kept = [];
  let from = 0;
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === '\\') {
      index += 1;
    } else if (text[index] === '%') {
      kept.push(text.slice(from, index));
      const lineEnd = text.indexOf('\n', index);
      if (lineEnd === -1) return kept.join('');
      index = lineEnd;
      while (text[index + 1] === ' ' || text[index + 1] === '\t') index += 1;
      from = index + 1;
    }
  }
  kept.push(text.slice(from));
  return kept.join('');
}

/** Where the group that opens at `open` closes, counting braces but not `\{` and `\}`; -1 when it does not. */
function groupEnd(text, open) {
  let depth = 0;
  for (let index = open; index < text.length; index += 1) {
    if (text[index] === '\\') {
      index += 1;
    } else if (text[index] === '{') {
      depth += 1;
    } else if (text[index] === '}') {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}

/** The calls of TEXT_MACROS in `text`, each with its arguments, appended to `calls`; the calls of an items argument after their macro. */
function collectCalls(text, calls) {
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] !== '\\') continue;
    LETTERS_RE.lastIndex = index + 1;
    const name = LETTERS_RE.exec(text)?.[0];
    if (!name) {
      // A control symbol, \{ and \} among them: no brace of a group.
      index += 1;
      continue;
    }
    index += name.length;
    const arity = TEXT_MACROS.get(name);
    if (arity === undefined) continue;
    const args = [];
    let next = index + 1;
    while (args.length < arity) {
      while (/\s/.test(text[next] ?? '')) next += 1;
      if (text[next] !== '{') break;
      const close = groupEnd(text, next);
      const end = close === -1 ? text.length : close;
      args.push(text.slice(next + 1, end));
      next = end + 1;
    }
    index = next - 1;
    const items = ITEMS_ARGUMENT.get(name);
    calls.push(`\\${name}${args.map((arg, at) => `{${at === items ? '\u2026' : arg.replace(/\s+/g, ' ').trim()}}`).join('')}`);
    if (items !== undefined && args[items] !== undefined) collectCalls(args[items], calls);
  }
}

/**
 * The text of a .tex: each call of a macro that carries text, with its
 * arguments, in the order of the body.
 *
 * What the runner compares between a hand-edited .tex and the .tex its
 * payload renders to, since a .tex may change the layout of a CV and never
 * its text: the payload is what the checks read. So what a call of
 * \cvSummary, \cvEntryHead, \cvEntryBlock, \cvItem, \cvProjects, \cvProject,
 * \cvEducation, \cvCert or \cvSkill holds counts, and nothing else does: the
 * preamble, comments, what follows \end{document}, \cvEntrySep and \cvStop
 * (which print no text), other commands (\vspace), and how an argument is
 * spaced or broken across lines. The items of a block or of the projects are
 * calls of their own, after their macro, whose items argument is written
 * `{…}`, so a line of layout between two bullets is no change of text. An
 * argument is read to its matching brace; `\{` and `\}` are text.
 *
 * The limit: it reads what the calls hold, not what the page prints, so an
 * edit that changes only what they print gives the same list as the .tex
 * before it: a macro redefined in the preamble, in any spelling; one
 * redefined in the body through \csname (\expandafter\def\csname
 * cvEntryHead\endcsname…), or any macro a call prints with, such as \cvStop
 * or \textbf, redefined there by \def, \let or \renewcommand; a call put
 * inside \iffalse…\fi with other text after it; or a line of prose between
 * two calls. Only a macro of the list above redefined in the body under its
 * own name (\def\cvEntryHead, \let\cvItem) shows, as one call more.
 *
 * @param {string} tex
 * @returns {string[]} Calls such as `\cvEntryHead{ML Engineer}{Northwind Robotics}{Remote}{2018-2020}`.
 */
export function contentMacros(tex) {
  const source = withoutComments(normalizedText(tex));
  const begin = /\\begin\s*\{document\}/.exec(source);
  const fromBody = begin ? source.slice(begin.index + begin[0].length) : source;
  const end = /\\end\s*\{document\}/.exec(fromBody);
  const calls = [];
  collectCalls(end ? fromBody.slice(0, end.index) : fromBody, calls);
  return calls;
}

// ---------------------------------------------------------------------------
// Compiling
// ---------------------------------------------------------------------------
//
// A .tex is compiled in a folder made for that compile alone, never in the
// pack or in output/: the pack is copied in beside the .tex, the engine runs
// there, and the folder is removed when the compile ends. The caller gets the
// PDF's bytes and writes them where they belong once it has checked the page
// limit, so a CV over the limit never takes the place of an earlier PDF.

/** The start of a build folder's name. mkdtempSync() adds six random characters, so no folder can be made ready for a compile beforehand. */
const BUILD_FOLDER_PREFIX = 'career-ops-latex-';

/**
 * What the engine writes and the compile reads. A pack copied from a folder
 * in which a CV was compiled by hand can bring its own, which would be taken
 * for the engine's: they are deleted before the first pass.
 */
const OUTPUT_FILES = Object.freeze(['cv.pdf', 'cv.log', 'cv.aux']);

/** The flags of each pass of pdflatex and xelatex: no shell escape, no wait for an answer on an error, and an end to the pass at the first. */
const PASS_FLAGS = Object.freeze(['-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error']);

/** tectonic's one call: --untrusted turns off shell escape and its other unsafe features; the log and the .aux are kept for the page count. */
const TECTONIC_ARGS = Object.freeze(['--untrusted', '--keep-logs', '--keep-intermediates', '--outdir', '.', 'cv.tex']);

/** Where TeX breaks a line of its log and goes on in the next: after 79 bytes in pdfTeX, after 79 characters in XeTeX. */
const LOG_LINE_LENGTH = 79;

/** The line an engine writes after all the text: `Output written on cv.pdf (2 pages, 14137 bytes).`; xelatex `(1 page).`, tectonic `cv.xdv`. */
const OUTPUT_WRITTEN_RE = /Output written on .*? \((\d+) pages?\b/g;

/** The number of pages LaTeX writes to the .aux: `\gdef \@abspage@last{2}`. */
const LAST_PAGE_RE = /\\@abspage@last\{(\d+)\}/g;

/** `Missing character: There is no – in font ec-lmr10!`; newer engines write the code after the character, `– (U+2013)`. */
const MISSING_CHARACTER_RE = /Missing character: There is no (.+?)(?: \(U\+([\dA-F]{4,6})\))? in font /g;

/** The line every fatal error ends with, which says no more than that it was one. */
const FATAL_ERROR_RE = /^!\s+==> Fatal error occurred/;

const UNICODE_HINT = 'The template\'s engine cannot print that character: replace it in the .json, or use a template for xelatex.';

/**
 * A log's lines as TeX meant them: a line TeX broke at LOG_LINE_LENGTH is
 * joined to the line it went on in. A broken line is exactly that long, in
 * bytes or in characters; a longer one, such as the first line with the
 * engine's name, was written whole.
 */
function logLines(text) {
  const lines = [];
  let pending = '';
  for (const line of String(text ?? '').replace(/\r\n?/g, '\n').split('\n')) {
    pending += line;
    if (line.length === LOG_LINE_LENGTH || Buffer.byteLength(line, 'utf8') === LOG_LINE_LENGTH) continue;
    lines.push(pending);
    pending = '';
  }
  if (pending) lines.push(pending);
  return lines;
}

/**
 * The number of pages of a compile, as its log or its .aux gives it, or null
 * when neither does.
 *
 * From the log, `Output written on cv.pdf (2 pages, 14137 bytes).` (xelatex
 * `(1 page).`, tectonic `cv.xdv`), and the last such line: TeX copies the
 * text of an overfull line into the log, so a line of the CV that reads like
 * it can come first, and the engine writes its own after all the text. A line
 * TeX broke at 79 columns is read whole. Without one, `\@abspage@last{N}`
 * from the .aux, the last one.
 *
 * @param {string} log The text of cv.log.
 * @param {string} [aux] The text of cv.aux.
 * @returns {number|null}
 */
export function pageCountFromLog(log, aux = '') {
  let pages = null;
  for (const line of logLines(log)) {
    for (const [, count] of line.matchAll(OUTPUT_WRITTEN_RE)) pages = Number(count);
  }
  if (pages !== null) return pages;
  const last = [...String(aux ?? '').matchAll(LAST_PAGE_RE)].at(-1);
  return last ? Number(last[1]) : null;
}

/** A character as a message can show it: itself, or its code (U+200B) when it prints as nothing or as spacing. */
function shownCharacter(text) {
  if (cleanText(text) === text) return text;
  return Array.from(text, (char) => `U+${char.codePointAt(0).toString(16).toUpperCase().padStart(4, '0')}`).join(' ');
}

/** The characters a log says the font has no glyph for, each once, in the order met. */
function missingGlyphsOf(log) {
  const glyphs = [];
  for (const line of logLines(log)) {
    for (const [, written, code] of line.matchAll(MISSING_CHARACTER_RE)) {
      const point = code ? Number.parseInt(code, 16) : Number.NaN;
      const glyph = shownCharacter(point <= 0x10FFFF ? String.fromCodePoint(point) : written);
      if (glyph && !glyphs.includes(glyph)) glyphs.push(glyph);
    }
  }
  return glyphs;
}

/**
 * The lines that say why a pass stopped: TeX's error lines, which begin with
 * `!`, each with the line of the .tex it stopped on after it (`l.97 …`), and
 * tectonic's, which begin with `error:`. Not the line every fatal error ends
 * with. Each once, its spacing normalised and with no control character, as
 * it is printed in a terminal.
 */
function errorLinesOf(text) {
  const found = [];
  let afterError = false;
  for (const line of logLines(text)) {
    const trimmed = line.trim();
    let keep = false;
    if (trimmed.startsWith('!')) {
      keep = !FATAL_ERROR_RE.test(trimmed);
      afterError = keep;
    } else if (trimmed.startsWith('error:')) {
      keep = true;
      afterError = false;
    } else if (afterError && /^l\.\d+\s/.test(trimmed)) {
      keep = true;
      afterError = false;
    }
    const shown = keep ? cleanText(trimmed) : '';
    if (shown && !found.includes(shown)) found.push(shown);
  }
  return found;
}

/** A file's text, or '' when there is none or it cannot be read. */
function readText(path) {
  try {
    return readFileSync(path, 'utf8');
  } catch {
    return '';
  }
}

/** A time limit as a message gives it: `120 s`, or `500 ms` under a second. */
function within(ms) {
  return ms >= 1000 ? `${Math.round(ms / 1000)} s` : `${ms} ms`;
}

/**
 * Where a program named without a folder is on PATH, or null: the first
 * folder of PATH given in full that holds it, on Windows as `<name>.com` or
 * `<name>.exe`, the two spawnSync() looks for. A folder of PATH not given in
 * full would be read from the folder the runner was started in.
 */
function onPath(name) {
  const windows = process.platform === 'win32';
  const files = windows ? [`${name}.com`, `${name}.exe`] : [name];
  for (const entry of String(process.env.PATH ?? '').split(delimiter)) {
    const folder = entry.trim().replace(/^"(.*)"$/, '$1');
    if (!isAbsolute(folder)) continue;
    for (const file of files) {
      const path = join(folder, file);
      try {
        if (!statSync(path).isFile()) continue;
        if (!windows) accessSync(path, constants.X_OK);
        return path;
      } catch { /* not in this folder */ }
    }
  }
  return null;
}

/**
 * The default `run` of compileLatex(): spawnSync() with the window of a
 * console program hidden and room for a long log on the terminal, but with a
 * program named without a folder looked for on PATH alone. spawnSync() on
 * Windows looks for one in the folder it is to run in first, unless the
 * environment holds NoDefaultCurrentDirectoryInExePath, which a shell does
 * not set; that folder is the build folder, where a pack, or the first pass
 * of the engine, could put a pdflatex.exe of its own.
 */
function runProgram(bin, args, { cwd, timeout }) {
  const program = isAbsolute(bin) ? bin : onPath(bin);
  if (!program) {
    return { status: null, stdout: '', stderr: '', error: Object.assign(new Error(`${bin} was not found on PATH`), { code: 'ENOENT' }) };
  }
  const { status, stdout, stderr, error } = spawnSync(program, args, {
    cwd, timeout, encoding: 'utf8', windowsHide: true, maxBuffer: 32 * 1024 * 1024,
  });
  return { status, stdout: stdout ?? '', stderr: stderr ?? '', error };
}

/**
 * The default `remove` of compileLatex(). rm() of node:fs/promises, not
 * rmSync(): on Node 24.11 for Windows rmSync() removes nothing under a path
 * with a non-ASCII character, such as a temporary folder under a user name
 * with an accent, and reports no error, which left the build folder and the
 * copy of the CV in it behind.
 */
function removeFolder(dir) {
  return rm(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
}

/** Why a pack entry is not copied: a symbolic link, or something that is neither a file nor a folder. */
function packEntryRefusal(packDir, source, stat) {
  const entry = cleanText(relative(packDir, source).split(sep).join('/'));
  if (!stat.isSymbolicLink()) return `The template pack holds ${entry}, which is neither a file nor a folder.`;
  return entry
    ? `The template pack holds a symbolic link, ${entry}, which is not copied to the build folder. Put the file or folder itself in the pack.`
    : 'The template pack is a symbolic link, which is not copied to the build folder. Put the folder itself in its place.';
}

/** Why an engine is taken for not installed, with the line of a template that chooses it. */
function noEngineReason(engine) {
  const install = engine === 'tectonic' ? ' Install it with: node install-latex-engine.mjs' : '';
  return `The template's engine, ${engine}, is not installed: ${engine} --version did not run. A template names its engine on the line `
    + `%!TEX TS-program = ${engine} (pdflatex when it has none).${install}`;
}

/** True for a call of `run` that ran out of time and was stopped. */
function timedOut(result) {
  return result?.error?.code === 'ETIMEDOUT';
}

/**
 * A value as a message names it, with no method of it called: a string as it
 * is, null as null, anything else by its type. String() of an object without
 * a prototype throws, and the compile reports what went wrong, never throws it.
 */
function valueName(value) {
  if (typeof value === 'string') return value;
  return value === null ? 'null' : typeof value;
}

/**
 * What went wrong, as a message names it: the error's code, else its message,
 * else the thrown value when that is a string or a number. No method of it is
 * called, for valueName()'s reason.
 */
function errorWord(err) {
  for (const word of [err?.code, err?.message, err]) {
    const text = typeof word === 'string' || Number.isFinite(word) ? cleanText(word) : '';
    if (text) return text;
  }
  return 'no reason given';
}

/** Why nothing is run for `engine`, as a failure of compileLatex(), or null: an engine that is not one of LATEX_ENGINES. */
function engineRefusal(engine) {
  if (LATEX_ENGINES.includes(engine)) return null;
  const reason = engine === 'lualatex'
    ? LUALATEX_REFUSED
    : `${JSON.stringify(cleanText(valueName(engine)))} is not an engine a CV compiles with: a template runs on ${oneOf(LATEX_ENGINES)}.`;
  return { ok: false, code: 'COMPILE', reason };
}

/**
 * The program run for an engine: for tectonic the one install-latex-engine.mjs
 * puts beside the scripts, when it is there; else the engine's name, which
 * runProgram() looks for on PATH alone.
 */
function engineProgram(engine) {
  return engine === 'tectonic' && existsSync(localTectonicPath()) ? localTectonicPath() : engine;
}

/** The failure an answer to `<engine> --version` shows, or null for an engine's answer: out of time, or none an engine gives. */
function versionFailure(version, engine, timeoutMs) {
  if (timedOut(version)) return { ok: false, code: 'TIMEOUT', reason: `${engine} --version did not answer within ${within(timeoutMs)}.` };
  if (version?.error || version?.status !== 0) return { ok: false, code: 'NO_ENGINE', reason: noEngineReason(engine) };
  return null;
}

/**
 * Whether a template's engine is there to compile with, for the runner to ask
 * before it sends the model anything: null when the engine answers
 * `--version`, the question compileLatex() asks first, put to the program
 * compileLatex() runs, so that what answers here is what a compile runs.
 * Else the failure compileLatex() would give for it: NO_ENGINE, the engine
 * did not answer (not installed, or broken); TIMEOUT, it took longer than
 * `timeoutMs`; COMPILE, it is not one of LATEX_ENGINES, or the question could
 * not be asked because `run` threw, which installing an engine would not
 * mend. It is asked in the system's temporary folder, and no folder is made
 * for it. Nothing is thrown.
 *
 * @param {string} engine A template's engine, as readTemplateOptions() gives it.
 * @param {object} [options]
 * @param {(bin: string, args: string[], options: {cwd: string, timeout: number}) => {status: number|null, stdout: string,
 *   stderr: string, error?: {code?: string, message?: string}}} [options.run] As compileLatex()'s `run`. Default
 *   spawnSync(), the program looked for on PATH alone.
 * @param {number} [options.timeoutMs] What the call may take. Default two minutes, as for each call of a compile.
 * @returns {Promise<null | {ok: false, code: 'NO_ENGINE'|'TIMEOUT'|'COMPILE', reason: string}>}
 */
export async function latexEngineProblem(engine, options) {
  const { run = runProgram, timeoutMs = 120000 } = options ?? {};
  const refused = engineRefusal(engine);
  if (refused) return refused;
  try {
    const version = await run(engineProgram(engine), ['--version'], { cwd: tmpdir(), timeout: timeoutMs });
    return versionFailure(version, engine, timeoutMs);
  } catch (err) {
    return { ok: false, code: 'COMPILE', reason: `The engine check stopped (${errorWord(err)}).` };
  }
}

/** The failure of a pass, or null when it ran through: out of time, or stopped with the lines that say why. */
function passFailure(result, { dir, engine, timeoutMs, pass }) {
  const which = pass ? ` pass ${pass} of 2` : '';
  if (timedOut(result)) {
    return { ok: false, code: 'TIMEOUT', reason: `${engine} did not finish${which} within ${within(timeoutMs)}, and was stopped.` };
  }
  if (!result?.error && result?.status === 0) return null;
  let lines = errorLinesOf(readText(join(dir, 'cv.log')));
  if (lines.length === 0) lines = errorLinesOf(`${result?.stdout ?? ''}\n${result?.stderr ?? ''}`);
  let how = 'with no exit code';
  if (result?.error) how = errorWord(result.error);
  else if (Number.isInteger(result?.status)) how = `exit code ${result.status}`;
  const stopped = `${engine} stopped${pass ? ` on${which}` : ''} (${how})`;
  const hint = lines.some((line) => line.includes('Unicode character')) ? ` ${UNICODE_HINT}` : '';
  return { ok: false, code: 'COMPILE', reason: lines.length > 0 ? `${stopped}: ${lines.join(' ')}${hint}` : `${stopped}, and gave no error line.` };
}

/** The steps of compileLatex() in the build folder `dir`, each failure returned. What throws is a failure compileLatex() reports. */
async function compileIn(dir, { tex, packDir, engine, timeoutMs, run }) {
  const fail = (code, reason) => ({ ok: false, code, reason });
  if (packDir) {
    let refused = null;
    try {
      cpSync(packDir, dir, {
        recursive: true,
        filter: (source) => {
          const stat = lstatSync(source);
          if (stat.isFile() || stat.isDirectory()) return true;
          refused = packEntryRefusal(packDir, source, stat);
          throw new Error(refused);
        },
      });
    } catch (err) {
      return fail('COMPILE', refused ?? `The template pack could not be copied to the build folder (${errorWord(err)}).`);
    }
  }
  for (const name of OUTPUT_FILES) await rm(join(dir, name), { recursive: true, force: true });
  const stale = OUTPUT_FILES.find((name) => existsSync(join(dir, name)));
  if (stale) return fail('COMPILE', `The template pack's ${stale} could not be taken out of the build folder.`);
  try {
    writeFileSync(join(dir, 'cv.tex'), tex);
  } catch (err) {
    return fail('COMPILE', `cv.tex could not be written to the build folder (${errorWord(err)}).`);
  }

  const program = engineProgram(engine);
  const call = (args) => run(program, [...args], { cwd: dir, timeout: timeoutMs });
  const version = await call(['--version']);
  const unanswered = versionFailure(version, engine, timeoutMs);
  if (unanswered) return unanswered;
  if (engine === 'tectonic') {
    const stopped = passFailure(await call(TECTONIC_ARGS), { dir, engine, timeoutMs, pass: 0 });
    if (stopped) return stopped;
  } else {
    const miktex = /MiKTeX/i.test(`${version.stdout ?? ''}\n${version.stderr ?? ''}`);
    const args = [...PASS_FLAGS, ...(miktex ? ['-disable-installer'] : []), 'cv.tex'];
    for (const pass of [1, 2]) {
      const stopped = passFailure(await call(args), { dir, engine, timeoutMs, pass });
      if (stopped) return stopped;
    }
  }

  const log = readText(join(dir, 'cv.log'));
  if (!existsSync(join(dir, 'cv.pdf'))) {
    return fail('COMPILE', `${engine} finished without writing a PDF${/No pages of output/.test(log) ? ': the document has no pages' : ''}.`);
  }
  const pages = pageCountFromLog(log, readText(join(dir, 'cv.aux')));
  if (pages === null) return fail('PAGES', `${engine} wrote a PDF, but neither cv.log nor cv.aux gives its number of pages.`);
  return { ok: true, pdf: readFileSync(join(dir, 'cv.pdf')), pages, missingGlyphs: missingGlyphsOf(log), engine };
}

/**
 * A .tex compiled to a PDF, in a build folder of its own.
 *
 * The folder is made with mkdtempSync() under `tmpRoot`, so its name cannot
 * be guessed, and everything after runs inside try … finally, which removes
 * it. The pack folder is copied in first: a symbolic link in it, which could
 * let the engine read or write outside the build folder through a name
 * inside it, is refused, and so is anything else that is neither a file nor
 * a folder. A cv.pdf, cv.log or cv.aux the pack brought is deleted, so only
 * what the engine writes is read. The .tex is written as cv.tex. The engine
 * has to answer --version. pdflatex and xelatex then run twice, as the
 * second pass prints what the first one measured, with shell escape off, no
 * wait for an answer on an error, the first error ending the pass, and for
 * MiKTeX the package installer off, so a package the .tex asks for is not
 * fetched in the middle of a compile. tectonic runs once, untrusted, and is
 * the one install-latex-engine.mjs puts beside the scripts when it is there.
 * Each call may take `timeoutMs`.
 *
 * The page count is the last one the log gives (pageCountFromLog).
 * `missingGlyphs` are the characters the log says the font has no glyph for:
 * the engine leaves them out of the PDF and exits 0 all the same.
 *
 * Nothing is thrown: a failure is returned with its code. NO_ENGINE, the
 * engine did not answer --version; COMPILE, the engine is not one of
 * LATEX_ENGINES (no input, or null for it, names none), the folder, the pack
 * or the .tex could not be made ready, a pass failed (the reason holds the
 * error lines of its log), no PDF was written, or `run` threw; TIMEOUT, a
 * call ran out of time and was stopped; PAGES, a PDF without a page count.
 * When the folder is still there after its removal, `leftover` names it: it
 * holds a copy of the CV. latexEngineProblem() asks the first question of a
 * compile alone, for the runner to ask before it sends the model anything.
 *
 * @param {object} [input]
 * @param {string} input.tex The .tex to compile.
 * @param {string|null} [input.packDir] The pack folder, whose files go beside the .tex; null for the default template.
 * @param {'xelatex'|'pdflatex'|'tectonic'} input.engine
 * @param {number} [input.timeoutMs] What each call of the engine may take. Default two minutes.
 * @param {(bin: string, args: string[], options: {cwd: string, timeout: number}) => {status: number|null, stdout: string,
 *   stderr: string, error?: {code?: string, message?: string}}} [input.run] Runs a program and waits for it; one that ran
 *   out of time gives `{ status: null, error: { code: 'ETIMEDOUT' } }`. Default spawnSync(), the program looked for on
 *   PATH alone.
 * @param {string} [input.tmpRoot] The folder the build folder is made in. Default the system's temporary folder.
 * @param {(dir: string) => unknown} [input.remove] Removes the build folder. Default rm() of node:fs/promises.
 * @returns {Promise<{ok: true, pdf: Buffer, pages: number, missingGlyphs: string[], engine: string, leftover?: string}
 *   | {ok: false, code: 'NO_ENGINE'|'COMPILE'|'TIMEOUT'|'PAGES', reason: string, leftover?: string}>}
 */
export async function compileLatex(input) {
  // Read here, not in the parameter list: a parameter default stands in for
  // undefined alone, and destructuring null there threw.
  const { tex, packDir = null, engine, timeoutMs = 120000, run = runProgram, tmpRoot, remove = removeFolder } = input ?? {};
  const refused = engineRefusal(engine);
  if (refused) return refused;
  if (typeof tex !== 'string') return { ok: false, code: 'COMPILE', reason: 'There is no LaTeX source to compile.' };
  let dir;
  try {
    dir = mkdtempSync(join(tmpRoot ?? tmpdir(), BUILD_FOLDER_PREFIX));
  } catch (err) {
    return { ok: false, code: 'COMPILE', reason: `No build folder could be made for the compile (${errorWord(err)}).` };
  }
  let result;
  try {
    result = await compileIn(dir, { tex, packDir, engine, timeoutMs, run });
  } catch (err) {
    result = { ok: false, code: 'COMPILE', reason: `The compile stopped (${errorWord(err)}).` };
  } finally {
    try {
      await remove(dir);
    } catch { /* what is left is named below */ }
    if (existsSync(dir)) result = { ...result, leftover: dir };
  }
  return result;
}

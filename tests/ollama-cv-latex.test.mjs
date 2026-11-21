// tests/ollama-cv-latex.test.mjs — the tailored CV as data, for the local
// runner's LaTeX path (lib/ollama-cv-latex.mjs).
//
// That path asks the model for the CV's text as one JSON object and writes
// every LaTeX command itself. What this suite pins is how a reply becomes that
// object, and what is refused on the way:
//
//   - A model asked for one JSON object still wraps it now and then: a code
//     fence, a sentence before it, a remark after it. The object is cut out of
//     the reply, and a brace in the prose around it is not taken for it.
//   - A reply cut off before the object ends is refused, never read up to the
//     cut: a CV without its last sections would pass every check after it.
//   - Every string is normalised before anything reads it, in this order:
//     spacing first, then the invisible characters. In review, removing
//     control characters first made "Built\nKafka" one word. A zero-width or
//     a bidi character makes the text printed differ from the text checked,
//     and TeX reads U+0001 as an invalid character and stops.
//   - A link is a plain http or https address. sanitizeUrl() deletes
//     `{ } \ $ ^` from an address, so one holding them would be checked as one
//     address and printed as another.
//
// The header is not the model's to write: readCvHeader() reads it from the
// profile, and a profile without a phone, a title or links leaves them out.
//
// The text is checked where an HTML CV's text is checked. The payload is also
// written as the plain HTML the existing checks read (projectCvHtml), and this
// suite runs those checks themselves: the fact check of verify-cv-facts.mjs,
// the locked sections of generate-pdf.mjs, the order of the roles, and the
// runner's link rule and repair. A field in a tag or a class they do not read
// passes unchecked, and a backslash written as itself hides everything before
// it from the fact check. What the link rule and the repair change is read
// back into the payload (dropPayloadLinks, readSkillsBack), because the .tex
// is written from the payload, not from that HTML.
//
// The .tex is the payload written into a template (renderCvTex), and its text
// has to stay text: every value is escaped once, an address goes through
// sanitizeUrl() alone, and a value holding a command the renderer does not
// write stops the render (unknownTexCommands). A template is refused before
// anything is rendered when its engine, a placeholder or a block is one the
// renderer does not know (resolveLatexTemplate). The final text check of
// generate-latex.mjs runs here too, the real one, on rendered CVs: it read a
// CV true to cv.md as invented when the bold was \textbf, when a character
// was written as a command inside an argument, and when nothing ended a list
// of tools at the end of a bullet. contentMacros() reads the text out of a
// .tex, which a hand edit may lay out anew but not reword.
//
// The .tex is compiled in a folder made for that compile alone
// (compileLatex), never in the pack or in output/: the pack is copied in
// without a link that reaches outside it, the engine runs with shell escape
// off, and the folder goes when the compile ends, whatever happened. The
// engine is a stand-in here that records what it is asked and writes what a
// test gives it, and in one test a real one, on a machine that has one: the
// default template compiles the persona CV with every character the escaping
// touches, and no character goes missing, under pdflatex or tectonic. The
// page count is read from the last line the engine writes to its log, since
// TeX copies the text of an overfull line into the log too. Whether the
// engine is there can be asked before a compile (latexEngineProblem), of the
// program a compile runs, and gets the answer a compile would give.
//
// Everything below is fictional: the persona of examples/cv-example.md and
// config/profile.example.yml.
//
// Run:  node --test tests/ollama-cv-latex.test.mjs
import { after, describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, symlinkSync, writeFileSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toolRequired } from './required-tools.mjs';

const ROOT = fileURLToPath(new URL('../', import.meta.url));

// `node test-all.mjs --needs latex` runs this suite with
// CAREER_OPS_REQUIRE_LATEX=1, so that the real compile fails where no engine
// is installed instead of skipping. Read before the variables go below.
const LATEX_REQUIRED = toolRequired('latex');

// Nothing here may follow a variable of the developer's shell to a real data
// root, in this part of the suite or in the parts that load upstream's checks:
// every CAREER_OPS_* and PRIVACY_* variable goes, and the root is a folder of
// the suite's own. The module under test, and what it imports, loads after.
for (const name of Object.keys(process.env)) {
  if (/^(CAREER_OPS_|PRIVACY_)/i.test(name)) delete process.env[name];
}
const SANDBOX = mkdtempSync(join(tmpdir(), 'ollama cv latex suite-'));
// rm() of node:fs/promises: on Node 24.11 for Windows, rmSync() removes nothing
// under a path with a non-ASCII character and reports no error.
after(() => rm(SANDBOX, { recursive: true, force: true, maxRetries: 10 }));
process.env.CAREER_OPS_ROOT = SANDBOX;
process.env.CAREER_OPS_DATA_DIR = '';

const {
  CV_PAYLOAD_SCHEMA, DEFAULT_LATEX_TEMPLATE, LATEX_ENGINES, compileLatex, contentMacros, cvPayloadProblem, dropPayloadLinks,
  extractJsonObject, latexEngineProblem, mapPayloadStrings, normalizeCvPayload, pageCountFromLog, parseCvPayload, payloadText,
  projectCvHtml, readCvHeader, readSkillsBack, readTemplateOptions, renderCvTex, resolveLatexTemplate, sectionTitles, shownDatesFor,
  templateNameOf, unknownTexCommands, yearsOnly,
} = await import('../lib/ollama-cv-latex.mjs');
const { walkTree } = await import('../lib/walk-tree.mjs');
// The checks a projection goes through, the real ones: the locked sections and
// the ATS normalisation of generate-pdf.mjs, the fact check of
// verify-cv-facts.mjs, the order of the roles, and the runner's link rule and
// repair. Then from generate-latex.mjs the final text check, which reads the
// rendered .tex, where install-latex-engine.mjs puts tectonic, and the engine
// the real compile runs on. generate-pdf.mjs creates output/ under the data
// root as it loads, so they load only now, with the root pinned to the
// sandbox.
const [
  { normalizeTextForATS, sectionKey, validateLockedSections },
  factGate,
  { validateCvExperienceOrder },
  { checkReading, pruneRejectedEntries, withoutUnsupportedLinks },
  { localTectonicPath, resolveLatexEngine, runLatexFactGate },
  { sanitizeUrl },
] = await Promise.all([
  import('../generate-pdf.mjs'),
  import('../verify-cv-facts.mjs'),
  import('../cv-experience-order.mjs'),
  import('../ollama-runner.mjs'),
  import('../generate-latex.mjs'),
  import('../lib/latex-escape.mjs'),
]);

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

// The CV of examples/cv-example.md as a payload, word for word. It is already
// in the shape normalizeCvPayload() returns.
const PERSONA = {
  summary: [
    'Full-stack AI engineer with 6 years building production ML systems. Led the ML platform at a Series B fintech (2020-2024), '
      + 'scaling from 2 models to 15+ in production. Built real-time fraud detection (99.7% precision, $2M/year saved), '
      + 'recommendation engine (18% uplift), and an internal MLOps platform serving 4 engineering teams.',
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
          'Led ML platform team (3 engineers), built internal MLOps tooling: model registry, A/B testing framework, feature store',
          'Designed real-time fraud detection pipeline: Kafka → feature computation → model inference → decision engine. '
            + '99.7% precision at 50ms p99',
          'Built recommendation engine for lending products: collaborative filtering + LLM reranking. 18% conversion uplift',
          'Reduced model deployment time from 2 weeks to 4 hours with CI/CD pipeline (GitHub Actions + SageMaker)',
          'Established model monitoring: drift detection, performance dashboards (Grafana), automated retraining triggers',
        ],
      }],
    },
    {
      company: 'DataStartup Inc',
      role: 'ML Engineer',
      location: 'Remote',
      dates: '2018-2020',
      blocks: [{
        lead: '',
        intro: '',
        bullets: [
          'Built NLP pipeline for document classification (BERT fine-tuning, 94% accuracy on legal docs)',
          'Implemented search ranking with learning-to-rank models',
          'Set up experiment tracking with MLflow and model versioning',
        ],
      }],
    },
  ],
  projects: {
    intro: '',
    items: [
      {
        name: 'FraudShield',
        text: '(Open Source) Real-time fraud detection framework. Kafka Streams + feature store + model serving. 500+ GitHub stars',
        url: '',
      },
      {
        name: 'LLM Eval Toolkit',
        text: '(Open Source) Evaluation framework for LLM applications. Supports custom metrics, regression testing, CI integration',
        url: '',
      },
    ],
  },
  education: [
    { degree: 'MS Computer Science', institution: 'UT Austin', dates: '2018' },
    { degree: 'BS Computer Science', institution: 'UC Berkeley', dates: '2016' },
  ],
  certifications: [],
  skills: [
    { category: 'ML/AI', items: 'PyTorch, TensorFlow, scikit-learn, Hugging Face, LangChain' },
    { category: 'MLOps', items: 'SageMaker, MLflow, Kubeflow, Airflow, Feature Store' },
    { category: 'Infra', items: 'Kubernetes, Kafka, Redis, PostgreSQL, AWS' },
    { category: 'Languages', items: 'Python, Go, TypeScript, SQL' },
  ],
};

/** A copy of the persona payload, changed by `change`. */
function withChange(change) {
  const copy = structuredClone(PERSONA);
  change(copy);
  return copy;
}

// The persona with text in every field of the schema, so a walk that misses
// one field shows.
const FULL = withChange((cv) => {
  cv.experience[0].blocks[0].lead = 'ML platform';
  cv.experience[0].blocks[0].intro = 'A platform team of three.';
  cv.projects.intro = 'Two open-source projects.';
  cv.projects.items[0].url = 'https://github.com/alexchen/fraudshield';
  cv.certifications = ['Certified Kubernetes Administrator'];
});

/** cvPayloadProblem() for a payload sent as a reply would send it. */
const problemOf = (payload, options) => cvPayloadProblem(JSON.stringify(payload), options);

const EMPTY_PAYLOAD = { summary: [], experience: [], projects: { intro: '', items: [] }, education: [], certifications: [], skills: [] };

// ---------------------------------------------------------------------------
// Reading a reply
// ---------------------------------------------------------------------------

describe('reading a reply', () => {
  test('a reply in a code fence is read', () => {
    for (const fence of ['```json', '```JSON', '```']) {
      const text = `${fence}\n${JSON.stringify(PERSONA, null, 2)}\n\`\`\``;
      assert.equal(cvPayloadProblem(text), null, fence);
      assert.deepEqual(parseCvPayload(text), PERSONA, fence);
    }
  });

  test('a sentence before the object and a remark after it are left out', () => {
    const text = `Here is the tailored CV as one JSON object:\n\n${JSON.stringify(PERSONA)}\n\nI kept every role of cv.md, newest first.`;
    assert.equal(cvPayloadProblem(text), null);
    assert.deepEqual(parseCvPayload(text), PERSONA);
  });

  test('a brace or a bracket in the prose around the object is not taken for it', () => {
    const text = `I left the {placeholders} alone, as rule [1] says.\n${JSON.stringify(PERSONA)}\nNothing {more} to add.`;
    assert.deepEqual(parseCvPayload(text), PERSONA);
    assert.equal(extractJsonObject('See [1]: {"summary": ["x"]} and {"y": 2}'), '{"summary": ["x"]}');
  });

  test('a reply cut off before the object ends is refused, not read up to the cut', () => {
    const json = JSON.stringify(PERSONA);
    for (const cut of [json.length - 1, json.indexOf('"education"'), json.indexOf('Kafka'), 1]) {
      assert.equal(extractJsonObject(json.slice(0, cut)), null, `cut at ${cut}`);
      assert.equal(cvPayloadProblem(json.slice(0, cut)), 'the JSON object is cut off before its closing brace', `cut at ${cut}`);
    }
    // A complete object inside the one that was cut off is not the payload.
    assert.equal(extractJsonObject('{"experience": [{"company": "Northwind Robotics"}'), null);
    // Nor is a pair of braces in the prose before it: the cut is still a cut,
    // not a parse error of "{name}".
    const prose = `I kept the {name} of each role as cv.md writes it.\n${json.slice(0, json.indexOf('"education"'))}`;
    assert.equal(extractJsonObject(prose), null);
    assert.equal(cvPayloadProblem(prose), 'the JSON object is cut off before its closing brace');
  });

  test('a reply with no JSON object is refused', () => {
    for (const text of ['', 'I cannot tailor this CV.', '["summary", "experience"]', undefined]) {
      assert.equal(cvPayloadProblem(text), 'no JSON object was found', String(text));
    }
  });

  test('a list is refused as a list, not read as its first item', () => {
    const list = `[${JSON.stringify(PERSONA)}]`;
    assert.equal(extractJsonObject(list), list);
    assert.equal(cvPayloadProblem(list), 'the JSON is a list, not one object');
    assert.equal(cvPayloadProblem(`\`\`\`json\n[\n  ${JSON.stringify(PERSONA)}\n]\n\`\`\``), 'the JSON is a list, not one object');
  });

  test('JSON that does not parse is refused with the parser\'s reason', () => {
    const problem = cvPayloadProblem('{"summary": ["One."] "skills": []}');
    assert.match(problem, /^the JSON does not parse: \S/);
  });

  test('the parser\'s reason carries no control character of the reply', () => {
    // V8 quotes part of the reply in some of its messages, and the runner
    // prints the reason to the terminal as it is.
    for (const text of ['{"summary": ["One."], "skills": \u001B}', '{"summary": [One \u001B\u0007 two]}', '{"a": tru\u001Be}']) {
      const problem = cvPayloadProblem(text);
      assert.match(problem, /^the JSON does not parse: \S/, JSON.stringify(text));
      assert.doesNotMatch(problem, /\p{Cc}/u, JSON.stringify(text));
    }
  });

  test('braces and escaped quotes inside a string are text, not structure', () => {
    const payload = withChange((cv) => {
      cv.summary = ['Parsed {"key": "value"} records } and a lone " quote', 'A Windows path that ends with C:\\'];
      cv.experience[1].blocks[0].bullets.push('Matched } and { and \\" in a log line');
    });
    const text = `${JSON.stringify(payload)}\n}`;
    assert.equal(extractJsonObject(text), JSON.stringify(payload));
    assert.deepEqual(parseCvPayload(text), payload);
  });

  test('parseCvPayload throws the sentence cvPayloadProblem returns', () => {
    for (const text of ['no JSON here', '{"summary": ["One."]', `[${JSON.stringify(PERSONA)}]`, JSON.stringify({ summary: ['One.'] })]) {
      assert.throws(() => parseCvPayload(text), { message: cvPayloadProblem(text) });
    }
  });
});

// ---------------------------------------------------------------------------
// What a payload must hold
// ---------------------------------------------------------------------------

describe('what a payload must hold', () => {
  test('the persona CV is a payload', () => {
    assert.equal(problemOf(PERSONA), null);
    assert.equal(problemOf(FULL), null);
  });

  test('each required field is named when it is missing', () => {
    const cases = [
      [(cv) => { cv.experience[0].company = ''; }, 'experience[0] (Senior ML Engineer / ML Platform Lead) has no company'],
      [(cv) => { cv.experience[1].role = ' '; }, 'experience[1] (DataStartup Inc) has no role'],
      [(cv) => { cv.experience[0].blocks = []; }, 'experience[0] (TechFin Corp) has no block with an intro or a bullet'],
      [(cv) => { cv.projects.items[1].name = ''; }, 'projects.items[1] has no name'],
      [(cv) => { cv.education[0].degree = ''; }, 'education[0] (UT Austin) has no degree'],
      [(cv) => { cv.education[1].institution = ''; }, 'education[1] (BS Computer Science) has no institution'],
    ];
    for (const [change, problem] of cases) assert.equal(problemOf(withChange(change)), problem);
  });

  test('a CV with one section is refused, as the HTML path refuses one section title', () => {
    assert.equal(problemOf({ summary: PERSONA.summary }), 'the CV has fewer than two sections with content');
    assert.equal(problemOf({ skills: PERSONA.skills, certifications: [] }), 'the CV has fewer than two sections with content');
    assert.equal(problemOf({ summary: PERSONA.summary, skills: PERSONA.skills }), null);
  });

  test('a bullet of spaces is no content', () => {
    assert.equal(problemOf({ experience: [{ blocks: [{ bullets: [' '] }] }] }), 'the CV has fewer than two sections with content');
    assert.equal(
      problemOf({
        summary: PERSONA.summary,
        experience: [{ company: 'Northwind Robotics', role: 'ML Engineer', blocks: [{ bullets: [' ', '\t'] }] }],
      }),
      'experience[0] (Northwind Robotics) has no block with an intro or a bullet',
    );
  });

  test('a link is a plain http or https address, or none', () => {
    const refused = [
      'https://example.com/fraud shield', 'https://example.com/{x}', 'https://example.com/a\\b', 'https://example.com/"x"',
      'https://example.com/<x>', 'javascript:alert(1)', 'https://example.com/$x', 'https://example.com/^x',
      "https://example.com/it's", 'https://example.com/a|b', 'https://example.com/a`b', 'ftp://example.com/x',
      'github.com/alexchen/fraudshield',
    ];
    for (const url of refused) {
      assert.equal(problemOf(withChange((cv) => { cv.projects.items[0].url = url; })),
        'projects.items[0] (FraudShield) has an address that is not a plain http or https link', url);
    }
    for (const url of ['https://github.com/alexchen/fraudshield', 'http://example.com/a?b=1&c=2#d', 'https://example.com/~alex/100%25']) {
      assert.equal(problemOf(withChange((cv) => { cv.projects.items[0].url = url; })), null, url);
    }
  });

  test('the minimum length is measured on the text a reader sees', () => {
    const length = payloadText(PERSONA).length;
    assert.equal(problemOf(PERSONA, { minTextChars: length }), null);
    assert.equal(problemOf(PERSONA, { minTextChars: length + 1 }),
      `the CV holds ${length} characters of text where cv.md calls for at least ${length + 1}`);
    // An address is not text on the page, as an href is not on the HTML path.
    const linked = withChange((cv) => { cv.projects.items[0].url = 'https://github.com/alexchen/fraudshield'; });
    assert.equal(payloadText(linked), payloadText(PERSONA));
  });

  test('a field of the wrong type is named with its path', () => {
    const cases = [
      [{ summary: { text: 'x' } }, 'summary must be a list of text'],
      [{ summary: ['x', { text: 'y' }] }, 'summary must be a list of text'],
      [{ experience: { company: 'x' } }, 'experience must be a list of objects'],
      [{ experience: ['TechFin Corp'] }, 'experience[0] must be an object'],
      [{ experience: [{ company: ['TechFin Corp'] }] }, 'experience[0].company must be text'],
      [{ experience: [{ company: 'TechFin Corp', blocks: { bullets: [] } }] }, 'experience[0].blocks must be a list of objects'],
      [{ experience: [{ blocks: [{ bullets: [['x']] }] }] }, 'experience[0].blocks[0].bullets must be a list of text'],
      [{ experience: [{ blocks: [{ lead: true }] }] }, 'experience[0].blocks[0].lead must be text'],
      [{ projects: 'FraudShield' }, 'projects must be an object with intro and items'],
      [{ projects: ['FraudShield'] }, 'projects[0] must be an object'],
      [{ projects: [{ name: ['FraudShield'] }] }, 'projects[0].name must be text'],
      [{ projects: { items: { name: 'x' } } }, 'projects.items must be a list of objects'],
      [{ projects: { intro: ['x'] } }, 'projects.intro must be text'],
      [{ projects: { items: [{ url: {} }] } }, 'projects.items[0].url must be text'],
      [{ education: [{ degree: false }] }, 'education[0].degree must be text'],
      [{ certifications: [['Certified Kubernetes Administrator']] }, 'certifications must be a list of text'],
      [{ skills: [{ category: { name: 'x' } }] }, 'skills[0].category must be text'],
      [{ skills: [{ category: 'Languages', items: { a: 'Python' } }] }, 'skills[0].items must be text or a list of text'],
    ];
    for (const [change, problem] of cases) assert.equal(problemOf({ ...PERSONA, ...change }), problem, problem);
  });

  test('a list of projects is read as the items, and an empty list as no project', () => {
    // Each refusal would cost a whole new generation, for a form that can mean
    // one thing only.
    assert.deepEqual(parseCvPayload(JSON.stringify({ ...PERSONA, projects: PERSONA.projects.items })), PERSONA);
    assert.deepEqual(parseCvPayload(JSON.stringify({ ...PERSONA, projects: [] })), { ...PERSONA, projects: { intro: '', items: [] } });
    assert.equal(payloadText({ ...PERSONA, projects: PERSONA.projects.items }), payloadText(PERSONA));
    assert.deepEqual(mapPayloadStrings({ ...FULL, projects: FULL.projects.items }, (text) => text).projects, { ...FULL.projects, intro: '' });
  });

  test('a value written null is absent, a number is its digits, and one string is a list of one', () => {
    const lenient = {
      summary: 'One paragraph of summary.',
      experience: [{
        company: 'Northwind Robotics',
        role: 'ML Engineer',
        location: null,
        dates: 2024,
        blocks: [{ lead: null, intro: null, bullets: 'Built the ranking service' }, null],
      }],
      projects: null,
      education: [{ degree: 'MS Computer Science', institution: 'UT Austin', dates: 2018 }],
      certifications: null,
      skills: [{ category: 'Languages', items: 'Python, Go' }, null],
    };
    assert.deepEqual(parseCvPayload(JSON.stringify(lenient)), {
      summary: ['One paragraph of summary.'],
      experience: [{
        company: 'Northwind Robotics',
        role: 'ML Engineer',
        location: '',
        dates: '2024',
        blocks: [{ lead: '', intro: '', bullets: ['Built the ranking service'] }],
      }],
      projects: { intro: '', items: [] },
      education: [{ degree: 'MS Computer Science', institution: 'UT Austin', dates: '2018' }],
      certifications: [],
      skills: [{ category: 'Languages', items: 'Python, Go' }],
    });
  });

  test('keys the schema does not have are dropped', () => {
    const extra = {
      template: 'default',
      header: { name: 'Alex Chen' },
      ...PERSONA,
      experience: [
        { ...PERSONA.experience[0], seniority: 'senior', blocks: [{ ...PERSONA.experience[0].blocks[0], tags: ['ml'] }] },
        PERSONA.experience[1],
      ],
      projects: { ...PERSONA.projects, heading: 'Projects' },
    };
    assert.deepEqual(parseCvPayload(JSON.stringify(extra)), PERSONA);
  });
});

// ---------------------------------------------------------------------------
// Normalising payload text
// ---------------------------------------------------------------------------

describe('normalising payload text', () => {
  test('line breaks, tabs and other spacing become one space', () => {
    // NEL and the information separators U+001C-U+001F are not `\s` in
    // JavaScript: removed as control characters, they would glue a and b.
    const spacing = [
      'a\nb', 'a\tb', 'a\r\nb', 'a \n\t b', 'a\u00A0b', 'a\u2028b', 'a\u2029b', '  a b  ', 'a\u0085b', 'a\u001Cb', 'a\u001Fb',
      'a\u001E\u0085 b',
    ];
    for (const raw of spacing) {
      assert.deepEqual(normalizeCvPayload({ summary: [raw] }).summary, ['a b'], JSON.stringify(raw));
    }
  });

  test('control, format and bidi characters are removed', () => {
    const cases = [
      ['Kafka\u0001 pipelines', 'Kafka pipelines'],
      ['Kaf\u200Bka', 'Kafka'],
      ['\u202Eexamples', 'examples'],
      ['Kaf\u00ADka', 'Kafka'],
      ['\u2066Kafka\u2069', 'Kafka'],
      ['Kafka\u007F\u0085', 'Kafka'],
      ['Kafka\uFFFD', 'Kafka'],
      ['Kafka\uD800', 'Kafka'],
      // The BOM is `\s` in JavaScript, yet inside a word it is a zero-width
      // character: removed, not made a space.
      ['Kaf\uFEFFka', 'Kafka'],
      ['\uFEFFKafka pipelines', 'Kafka pipelines'],
      // A removal never leaves two spaces side by side.
      ['Kafka \u200B pipelines', 'Kafka pipelines'],
      ['Kafka \uFEFF pipelines', 'Kafka pipelines'],
    ];
    for (const [raw, clean] of cases) {
      assert.deepEqual(normalizeCvPayload({ summary: [raw] }).summary, [clean], JSON.stringify(raw));
    }
  });

  test('every string of the schema is normalised, the address included', () => {
    const messy = mapPayloadStrings(FULL, (text) => `\u200B ${text.replace(/ /g, '\n')}\t`);
    assert.deepEqual(normalizeCvPayload(messy), FULL);
  });

  test('blocks, entries and strings with nothing in them are dropped', () => {
    const sparse = {
      summary: ['', 'One paragraph.', ' '],
      experience: [
        { company: '', role: '', location: '', dates: '', blocks: [] },
        {
          company: 'Northwind Robotics',
          role: 'ML Engineer',
          blocks: [
            // A lead with nothing under it goes with its block.
            { lead: 'Search', intro: '', bullets: [] },
            { lead: '', intro: 'Ranking work.', bullets: [] },
            { bullets: ['', 'Built ranking'] },
          ],
        },
      ],
      projects: { intro: 'Open source.', items: [{ name: '', text: '', url: '' }] },
      education: [{ degree: ' ', institution: '', dates: '' }],
      certifications: ['', ' '],
      skills: [{ category: ':', items: [] }, { category: 'Languages', items: ['Python', '', 'Go'] }],
    };
    assert.deepEqual(normalizeCvPayload(sparse), {
      summary: ['One paragraph.'],
      experience: [{
        company: 'Northwind Robotics',
        role: 'ML Engineer',
        location: '',
        dates: '',
        blocks: [{ lead: '', intro: 'Ranking work.', bullets: [] }, { lead: '', intro: '', bullets: ['Built ranking'] }],
      }],
      projects: { intro: '', items: [] },
      education: [],
      certifications: [],
      skills: [{ category: 'Languages', items: 'Python, Go' }],
    });
  });

  test('a skills category loses its trailing colons, so Languages: and Languages read alike', () => {
    const category = (value) => normalizeCvPayload({ skills: [{ category: value, items: 'Python, Go' }] }).skills[0].category;
    assert.equal(category('Languages:'), 'Languages');
    assert.equal(category('Languages'), 'Languages');
    assert.equal(category('Languages :'), 'Languages');
    // Every one, not one per pass: a payload is normalised again on its way,
    // and the category checked must not depend on how many times.
    assert.equal(category('Ratios::'), 'Ratios');
    assert.equal(category('Tools: :'), 'Tools');
    assert.equal(category('Ratios: 1:2'), 'Ratios: 1:2');
    assert.equal(category(': :'), '');
  });

  test('skill items given as a list become one string', () => {
    assert.deepEqual(
      normalizeCvPayload({ skills: [{ category: 'Languages', items: ['Python', ' Go ', 'TypeScript'] }] }).skills,
      [{ category: 'Languages', items: 'Python, Go, TypeScript' }],
    );
  });

  test('**…** is dropped from every field that prints no bold, and kept in the fields that print it', () => {
    // A field outside the bold ones printed its asterisks as they came, while
    // the checks, which read the projection after normalizeTextForATS(), read
    // the words without them: the text checked was not the text printed.
    const marked = {
      summary: ['**summary** text'],
      experience: [{
        company: '**company** text',
        role: '**role** text',
        location: '**location** text',
        dates: '**dates** text',
        blocks: [{ lead: '**lead** text', intro: '**intro** text', bullets: ['**bullet** text'] }],
      }],
      projects: { intro: '**pintro** text', items: [{ name: '**name** text', text: '**ptext** text', url: '' }] },
      education: [{ degree: '**degree** text', institution: '**institution** text', dates: '**edates** text' }],
      certifications: ['**cert** text'],
      skills: [{ category: '**category** text', items: '**items** text' }],
    };
    assert.deepEqual(normalizeCvPayload(marked), {
      summary: ['**summary** text'],
      experience: [{
        company: 'company text',
        role: 'role text',
        location: 'location text',
        dates: 'dates text',
        blocks: [{ lead: '**lead** text', intro: '**intro** text', bullets: ['**bullet** text'] }],
      }],
      projects: { intro: '**pintro** text', items: [{ name: 'name text', text: '**ptext** text', url: '' }] },
      education: [{ degree: 'degree text', institution: 'institution text', dates: 'edates text' }],
      certifications: ['**cert** text'],
      skills: [{ category: 'category text', items: 'items text' }],
    });
  });

  test('cv.md\'s bold title, project name and skills category come through as text, and the category then loses its colon', () => {
    // examples/cv-example.md writes each of these in bold, and a model copies them.
    const copied = normalizeCvPayload({
      experience: [{ company: 'TechFin Corp', role: '**Senior ML Engineer / ML Platform Lead**', blocks: [{ bullets: ['Built ranking'] }] }],
      projects: { items: [{ name: '**FraudShield**', text: 'A framework.' }] },
      skills: [{ category: '**ML/AI:**', items: 'PyTorch' }, { category: '**Languages**:', items: ['**Python**', 'Go'] }],
    });
    assert.equal(copied.experience[0].role, 'Senior ML Engineer / ML Platform Lead');
    assert.equal(copied.projects.items[0].name, 'FraudShield');
    assert.deepEqual(copied.skills, [{ category: 'ML/AI', items: 'PyTorch' }, { category: 'Languages', items: 'Python, Go' }]);
    // The spacing a marker leaves goes with it, a marker inside a marker goes
    // too, and a payload normalised again stays as it is.
    for (const [raw, plain] of [['** Lead ** engineer', 'Lead engineer'], ['a ** b ** c', 'a b c'], ['****Kafka****', 'Kafka'], ['** **', '']]) {
      const once = normalizeCvPayload({ education: [{ degree: raw, institution: 'UT Austin' }] });
      assert.equal(once.education[0].degree, plain, raw);
      assert.deepEqual(normalizeCvPayload(once), once, raw);
    }
  });

  test('_…_ around words, cv.md\'s italics, is dropped from every field but the address, so the checks read the text the .tex prints', () => {
    // normalizeTextForATS() reads `_LLM Eval Toolkit_` as italics, so the
    // checks read the words alone, while the renderer escaped the underscores
    // and the PDF printed them. No field of the .tex prints italics.
    const marked = mapPayloadStrings(FULL, (text) => (text ? `_${text}_ text` : text));
    const expected = mapPayloadStrings(FULL, (text) => (text ? `${text} text` : text));
    // An address is no text, and keeps what it holds.
    marked.projects.items[0].url = 'https://example.com/a?q=_x_';
    expected.projects.items[0].url = 'https://example.com/a?q=_x_';
    const once = normalizeCvPayload(marked);
    assert.deepEqual(once, normalizeCvPayload(expected));
    assert.doesNotMatch(normalizeTextForATS(projectCvHtml(once)).html, /<em>/);
    assert.deepEqual(normalizeCvPayload(once), once);
    // The rule is normalizeTextForATS()'s own, so what it keeps as written, the
    // payload keeps: an underscore inside a word, beside a path separator, in
    // an email address, or doubled.
    const field = (text) => normalizeCvPayload({ summary: [text], skills: [{ category: text, items: 'Python' }] });
    const cases = [
      ['Built _LLM Eval Toolkit_ in Go', 'Built LLM Eval Toolkit in Go'],
      ['**_FraudShield_** and _**Kafka**_', '**FraudShield** and **Kafka**'],
      ['Wrote _a _nested_ title_', 'Wrote a nested title'],
      ['Mail _jane@example.com_', 'Mail jane@example.com'],
      ['feature_store and snake_case_names', 'feature_store and snake_case_names'],
      ['docs/_draft_/cv.md and C:\\models\\_draft_\\x', 'docs/_draft_/cv.md and C:\\models\\_draft_\\x'],
      ['__init__ and _private', '__init__ and _private'],
      ['Mail _jane_@example.com', 'Mail _jane_@example.com'],
    ];
    for (const [raw, plain] of cases) {
      const cv = field(raw);
      assert.equal(cv.summary[0], plain, raw);
      assert.equal(cv.skills[0].category, plain.replace(/\*\*/g, ''), raw);
      assert.deepEqual(normalizeCvPayload(cv), cv, raw);
      // What is taken out is what normalizeTextForATS() reads as italics, and it finds none left.
      assert.equal(normalizeTextForATS(`<p>${raw}</p>`).html.includes('<em>'), raw !== plain, raw);
      assert.doesNotMatch(normalizeTextForATS(`<p>${cv.summary[0]}</p>`).html, /<em>/, raw);
    }
  });

  test('anything that is not a payload normalises to an empty one', () => {
    for (const value of [undefined, null, 'cv', [], {}, { experience: 'x', projects: [], skills: {} }]) {
      assert.deepEqual(normalizeCvPayload(value), EMPTY_PAYLOAD, JSON.stringify(value));
    }
  });

  test('normalising twice changes nothing, and the input is left as it was', () => {
    const messy = mapPayloadStrings(FULL, (text) => ` ${text}\n`);
    messy.summary.push('Kaf\uFEFFka \u200B a\u0085b \u2066c\u2069 \u001Fd :');
    messy.skills.push({ category: 'Ratios::', items: 'Python' }, { category: 'Tools: :', items: ['Go', ' Rust '] });
    const before = structuredClone(messy);
    const once = normalizeCvPayload(messy);
    assert.deepEqual(normalizeCvPayload(once), once);
    assert.deepEqual(messy, before);
  });
});

// ---------------------------------------------------------------------------
// The strings of a payload
// ---------------------------------------------------------------------------

describe('the strings of a payload', () => {
  /** `value` with every string in it wrapped in angle brackets. */
  const wrapAll = (value) => {
    if (typeof value === 'string') return `<${value}>`;
    if (Array.isArray(value)) return value.map(wrapAll);
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, wrapAll(item)]));
    return value;
  };

  test('mapPayloadStrings reaches every field and changes nothing else', () => {
    const before = structuredClone(FULL);
    assert.deepEqual(mapPayloadStrings(FULL, (text) => `<${text}>`), wrapAll(FULL));
    assert.deepEqual(FULL, before);
  });

  test('mapPayloadStrings passes the string alone', () => {
    // restoreIfEnabled(text, profileYml), passed as it is, would take a second
    // argument for the profile and restore nothing.
    const seen = [];
    mapPayloadStrings(FULL, (...args) => { seen.push(args.length); return args[0]; });
    assert.ok(seen.length > 0);
    assert.deepEqual([...new Set(seen)], [1]);
  });

  test('payloadText reads every string but the addresses, in order, one space apart', () => {
    const small = {
      summary: ['One.', 'Two.'],
      experience: [{
        company: 'Northwind Robotics',
        role: 'ML Engineer',
        location: 'Remote',
        dates: '2018-2020',
        blocks: [{ lead: 'Search', intro: 'Ranking.', bullets: ['Built ranking.', 'Ran tests.'] }],
      }],
      projects: { intro: 'Open source.', items: [{ name: 'FraudShield', text: 'A framework.', url: 'https://github.com/alexchen/fraudshield' }] },
      education: [{ degree: 'MS Computer Science', institution: 'UT Austin', dates: '2018' }],
      certifications: ['Certified Kubernetes Administrator'],
      skills: [{ category: 'Languages', items: 'Python, Go' }],
    };
    assert.equal(payloadText(small), 'One. Two. Northwind Robotics ML Engineer Remote 2018-2020 Search Ranking. Built ranking. Ran tests. '
      + 'Open source. FraudShield A framework. MS Computer Science UT Austin 2018 Certified Kubernetes Administrator Languages Python, Go');
    assert.equal(payloadText(EMPTY_PAYLOAD), '');
  });
});

// ---------------------------------------------------------------------------
// The CV header
// ---------------------------------------------------------------------------

describe('the CV header from the profile', () => {
  const FULL_PROFILE = [
    'candidate:',
    '  full_name: "Jane Smith"',
    '  title: "Senior Backend Engineer"',
    '  email: "jane@example.com"',
    '  phone: "+1-555-0123"',
    '  location: "San Francisco, CA"',
    '  linkedin: "linkedin.com/in/janesmith"',
    '  github: "github.com/janesmith"',
    '',
  ].join('\n');
  const FULL_HEADER = {
    name: 'Jane Smith',
    firstName: 'Jane',
    lastName: 'Smith',
    title: 'Senior Backend Engineer',
    location: 'San Francisco, CA',
    email: 'jane@example.com',
    emailUrl: 'mailto:jane@example.com',
    phone: '+1-555-0123',
    linkedin: { url: 'https://linkedin.com/in/janesmith', display: 'linkedin.com/in/janesmith' },
    github: { url: 'https://github.com/janesmith', display: 'github.com/janesmith' },
  };
  const EMPTY_HEADER = {
    name: '',
    firstName: '',
    lastName: '',
    title: '',
    location: '',
    email: '',
    emailUrl: '',
    phone: '',
    linkedin: { url: '', display: '' },
    github: { url: '', display: '' },
  };

  test('every fact is read from candidate', () => {
    assert.deepEqual(readCvHeader(FULL_PROFILE), FULL_HEADER);
  });

  test('the example profile gives its header, without the title it leaves commented out', () => {
    assert.deepEqual(readCvHeader(readFileSync(join(ROOT, 'config', 'profile.example.yml'), 'utf8')), { ...FULL_HEADER, title: '' });
  });

  test('a fact the profile leaves out or leaves blank is empty, and the others stay', () => {
    const outputs = { title: ['title'], email: ['email', 'emailUrl'], phone: ['phone'], location: ['location'], linkedin: ['linkedin'], github: ['github'] };
    for (const [key, fields] of Object.entries(outputs)) {
      const expected = { ...FULL_HEADER };
      for (const field of fields) expected[field] = EMPTY_HEADER[field];
      const without = FULL_PROFILE.split('\n').filter((line) => !line.startsWith(`  ${key}:`)).join('\n');
      assert.deepEqual(readCvHeader(without), expected, `without ${key}`);
      const blank = FULL_PROFILE.replace(new RegExp(`^  ${key}: .*$`, 'm'), `  ${key}: ""`);
      assert.deepEqual(readCvHeader(blank), expected, `${key} blank`);
    }
  });

  test('a one-word name is the first name, with no last name', () => {
    const one = readCvHeader('candidate:\n  full_name: "Jane"\n');
    assert.deepEqual([one.name, one.firstName, one.lastName], ['Jane', 'Jane', '']);
    const three = readCvHeader('candidate:\n  full_name: "Jane Ann Smith"\n');
    assert.deepEqual([three.name, three.firstName, three.lastName], ['Jane Ann Smith', 'Jane Ann', 'Smith']);
  });

  test('values come back unescaped with their spacing normalised, and links through sanitizeUrl', () => {
    const header = readCvHeader([
      'candidate:',
      '  full_name: "Jane\\nSmith"',
      "  title: 'R&D_lead 100% {core} #1 $x ~y ^z \\path'",
      '  location: "San Francisco,\\nCA"',
      '  linkedin: "https://www.linkedin.com/in/jane%20smith/"',
      '  github: "http://github.com/janesmith/"',
      '  phone: 5550123',
      '',
    ].join('\n'));
    assert.deepEqual([header.name, header.firstName, header.lastName], ['Jane Smith', 'Jane', 'Smith']);
    assert.equal(header.title, 'R&D_lead 100% {core} #1 $x ~y ^z \\path');
    assert.equal(header.location, 'San Francisco, CA');
    // The address is ready for \href; what is shown is the profile's own text.
    assert.deepEqual(header.linkedin, { url: 'https://www.linkedin.com/in/jane\\%20smith/', display: 'www.linkedin.com/in/jane%20smith' });
    assert.deepEqual(header.github, { url: 'http://github.com/janesmith/', display: 'github.com/janesmith' });
    assert.equal(header.phone, '5550123');
  });

  test('a profile that is missing, does not parse or has no candidate mapping gives an empty header', () => {
    const profiles = [
      undefined, '', '# comments only\n', 'candidate: [unclosed\n', 'cv:\n  output_format: latex\n', 'candidate: Jane Smith\n',
      '- a\n- list\n',
    ];
    for (const text of profiles) assert.deepEqual(readCvHeader(text), EMPTY_HEADER, JSON.stringify(text));
  });
});

// ---------------------------------------------------------------------------
// The schema the model is shown
// ---------------------------------------------------------------------------

describe('the schema shown to the model', () => {
  test('its example is a payload this module accepts', () => {
    // The example is the one JSON object in the text, so the reader finds it
    // there. A model shown a shape the reader refuses would be asked again on
    // every run.
    const example = parseCvPayload(CV_PAYLOAD_SCHEMA);
    assert.equal(example.experience[0].company, 'TechFin Corp');
    assert.equal(cvPayloadProblem(CV_PAYLOAD_SCHEMA), null);
  });

  test('it describes every field and asks for one JSON object', () => {
    const fields = [
      'summary', 'experience', 'company', 'role', 'location', 'dates', 'blocks', 'lead', 'intro', 'bullets', 'projects', 'items',
      'name', 'text', 'url', 'education', 'degree', 'institution', 'certifications', 'skills', 'category',
    ];
    for (const field of fields) assert.match(CV_PAYLOAD_SCHEMA, new RegExp(`^\\s*${field}: `, 'm'), field);
    assert.match(CV_PAYLOAD_SCHEMA, /one JSON object/);
    // A raw line break inside a JSON string does not parse, so the schema
    // never invites one: each would cost a whole new generation.
    assert.match(CV_PAYLOAD_SCHEMA, /Never break a line inside a string/);
  });

  test('it asks for the jobs newest first by start date, the order the order check holds a CV to, not cv.md\'s order', () => {
    // validateCvExperienceOrder() blocks a CV whose jobs are not newest first
    // by start date, in cv.md's order as much as in any other, and the rules
    // sent beside the schema say so: a schema asking for cv.md's order
    // contradicted them, and a cv.md listing a later role second cost a
    // second request.
    assert.match(CV_PAYLOAD_SCHEMA, /^experience: a list of jobs, newest first by start date\. /m);
    assert.doesNotMatch(CV_PAYLOAD_SCHEMA, /as cv\.md orders them/);
  });
});

// ---------------------------------------------------------------------------
// The projection the checks read
// ---------------------------------------------------------------------------

const PERSONA_CV = readFileSync(join(ROOT, 'examples', 'cv-example.md'), 'utf8');

/** The section titles of a cv.md that uses the usual headings, or has none. */
const USUAL_TITLES = {
  summary: 'Professional Summary',
  experience: 'Work Experience',
  projects: 'Projects',
  education: 'Education',
  certifications: 'Certifications',
  skills: 'Skills',
};

/**
 * A data root of its own holding `cvText` as cv.md, and the files the fact
 * check reads there, named as the runner names them (factGateInputs()):
 * cv.md, an article-digest.md and a config/cv-facts.json, neither of them
 * written.
 */
function checkRoot(cvText) {
  const root = mkdtempSync(join(SANDBOX, 'checks-'));
  writeFileSync(join(root, 'cv.md'), cvText);
  return {
    root,
    inputs: { sourcePaths: [join(root, 'cv.md'), join(root, 'article-digest.md')], configPath: join(root, 'config', 'cv-facts.json') },
  };
}

const PERSONA_ROOT = checkRoot(PERSONA_CV);

/** The projection of a payload, titled from `cv` as the runner titles it. */
const projected = (payload, cv = PERSONA_CV) => projectCvHtml(payload, sectionTitles(cv, sectionKey));

/** The fact check's verdict on a projection, handed over as the runner hands it: after ATS normalisation. */
const factCheck = (html, { inputs } = PERSONA_ROOT) => factGate.verifyFacts(normalizeTextForATS(html).html, inputs);

/** The facts a fact-check result rejects, as `kind=value`. */
const rejectedFacts = (result) => result.unsupportedFacts.map(({ kind, value }) => `${kind}=${value}`);

describe('the section titles of a projection', () => {
  test('each section is titled with the heading cv.md gives it', () => {
    const cv = [
      '# CV -- Alex Chen', '## Summary', 'One line.', '## Experience', '### TechFin Corp -- Austin, TX', '## Selected Projects',
      '## Education', '## Certifications', '## Technical Skills', '',
    ].join('\n\n');
    assert.deepEqual(sectionTitles(cv, sectionKey), {
      summary: 'Summary',
      experience: 'Experience',
      projects: 'Selected Projects',
      education: 'Education',
      certifications: 'Certifications',
      skills: 'Technical Skills',
    });
  });

  test('a section cv.md has no heading for gets the usual title, and certifications none', () => {
    // Certifications with no heading of their own close the education section
    // of the projection, as the next test sets out.
    const usual = { ...USUAL_TITLES, certifications: null };
    assert.deepEqual(sectionTitles(PERSONA_CV, sectionKey), usual);
    for (const cv of ['', undefined, '# CV -- Alex Chen\n\nNo sections yet.\n']) {
      assert.deepEqual(sectionTitles(cv, sectionKey), usual, String(cv));
    }
    assert.deepEqual(sectionTitles('## Summary\n\n## Constructor\n\n## Interests\n', sectionKey), { ...usual, summary: 'Summary' });
  });

  test('certifications with no heading of their own in cv.md have no title of their own', () => {
    // generate-pdf.mjs reads "Education" and "Education & Certifications"
    // alike as the education section, so a locked education section holds the
    // certifications cv.md lists under either.
    for (const heading of ['Education', 'Education & Certifications', 'Wykształcenie i certyfikaty']) {
      const titles = sectionTitles(`## ${heading}\n\n- MS Computer Science, UT Austin (2018)\n- Certified Kubernetes Administrator\n`,
        sectionKey);
      assert.deepEqual([titles.education, titles.certifications], [heading, null], heading);
    }
    // A heading of their own is theirs.
    const both = sectionTitles('## Education & Certifications\n\n## Certifications\n', sectionKey);
    assert.deepEqual([both.education, both.certifications], ['Education & Certifications', 'Certifications']);
    // Only a heading the check reads as the certifications section counts: it
    // reads "Certificates" under a key of its own, the limit of this rule.
    assert.equal(sectionKey('Certificates'), 'certificates');
    assert.equal(sectionTitles('## Education\n\n## Certificates\n', sectionKey).certifications, null);
  });

  test('of two headings for one section the shallower, then the first, is taken, as the locked-section check takes it', () => {
    assert.equal(sectionTitles('### Skills\n\n## Technical Skills\n', sectionKey).skills, 'Technical Skills');
    assert.equal(sectionTitles('## Skills\n\n## Technical Skills\n', sectionKey).skills, 'Skills');
  });

  test('a heading that would not name its section once written into the projection gives way to the usual title', () => {
    // generate-pdf.mjs reads a section's name from the HTML, where the comment
    // is escaped text, no longer a tag it skips.
    assert.equal(sectionKey('Technical Skills <!-- tools -->'), 'skills');
    assert.equal(sectionTitles('## Technical Skills <!-- tools -->\n', sectionKey).skills, 'Skills');
  });
});

describe('the projection of a payload', () => {
  const SMALL = {
    summary: ['One paragraph.'],
    experience: [{
      company: 'Northwind Robotics',
      role: 'ML Engineer',
      location: 'Remote',
      dates: '2018-2020',
      blocks: [{ lead: 'Search', intro: 'Ranking work.', bullets: ['Built ranking', 'Ran tests'] }],
    }],
    projects: {
      intro: 'Open source.',
      items: [
        { name: 'FraudShield', text: 'A framework.', url: 'https://github.com/alexchen/fraudshield' },
        { name: 'LLM Eval Toolkit', text: '', url: '' },
      ],
    },
    education: [{ degree: 'MS Computer Science', institution: 'UT Austin', dates: '2018' }],
    certifications: ['Certified Kubernetes Administrator'],
    skills: [{ category: 'Languages', items: 'Python, Go' }],
  };

  test('each field is written in the markup the checks read', () => {
    // Fields in a div or a span with the class in double quotes: the fact
    // check reads no other tag and no other quoting, and passes what it does
    // not read.
    assert.equal(projectCvHtml(SMALL, USUAL_TITLES), [
      '<div class="section-title">Professional Summary</div>',
      '<p>One paragraph.</p>',
      '<div class="section-title">Work Experience</div>',
      '<div class="job">',
      '<div class="job-company">Northwind Robotics</div>',
      '<div>Remote</div>',
      '<div class="job-role">ML Engineer</div>',
      '<div class="job-period">2018-2020</div>',
      '<p>Search</p>',
      '<p>Ranking work.</p>',
      '<ul><li>Built ranking</li><li>Ran tests</li></ul>',
      '</div>',
      '<div class="section-title">Projects</div>',
      '<p>Open source.</p>',
      '<div><div><a data-p="0" href="https://github.com/alexchen/fraudshield">FraudShield</a></div><p>A framework.</p></div>',
      '<div><div>LLM Eval Toolkit</div></div>',
      '<div class="section-title">Education</div>',
      '<div class="edu-item"><div class="edu-title">MS Computer Science</div><div class="edu-org">UT Austin</div>'
        + '<div class="edu-year">2018</div></div>',
      '<div class="section-title">Certifications</div>',
      '<div><span class="cert-title">Certified Kubernetes Administrator</span></div>',
      '<div class="section-title">Skills</div>',
      '<div class="skill-item" data-i="0"><span class="skill-category">Languages:</span> Python, Go</div>',
    ].join('\n'));
  });

  test('without titles it uses the usual ones', () => {
    assert.equal(projectCvHtml(SMALL), projectCvHtml(SMALL, USUAL_TITLES));
    assert.equal(projectCvHtml(SMALL, { summary: 'Summary' }), projectCvHtml(SMALL, { ...USUAL_TITLES, summary: 'Summary' }));
  });

  test('a title given with no text is the usual one, so no section goes untitled', () => {
    // Written as given, it would be a section title with no text, and the
    // locked-section check gives the text under such a title to no section:
    // a locked section so titled reads as modified. Only certifications take
    // null, which puts them under education.
    for (const key of Object.keys(USUAL_TITLES)) {
      for (const value of [undefined, '', ' ', ...(key === 'certifications' ? [] : [null])]) {
        assert.equal(projectCvHtml(SMALL, { [key]: value }), projectCvHtml(SMALL), `${key}: ${JSON.stringify(value)}`);
      }
    }
    assert.equal(projectCvHtml(SMALL, null), projectCvHtml(SMALL));
    const titles = { ...sectionTitles(PERSONA_CV, sectionKey), skills: undefined };
    validateLockedSections(projectCvHtml(PERSONA, titles), PERSONA_CV, ['skills']);
  });

  test('a section with nothing to show is left out with its title, and so is an empty field', () => {
    const sparse = {
      experience: [{ company: 'Northwind Robotics', role: 'ML Engineer', blocks: [{ bullets: ['Built ranking'] }] }],
      projects: { intro: 'Not shown without a project.', items: [] },
      education: [{ degree: 'MS Computer Science', institution: 'UT Austin' }],
    };
    assert.equal(projectCvHtml(sparse), [
      '<div class="section-title">Work Experience</div>',
      '<div class="job">',
      '<div class="job-company">Northwind Robotics</div>',
      '<div class="job-role">ML Engineer</div>',
      '<ul><li>Built ranking</li></ul>',
      '</div>',
      '<div class="section-title">Education</div>',
      '<div class="edu-item"><div class="edu-title">MS Computer Science</div><div class="edu-org">UT Austin</div></div>',
    ].join('\n'));
    assert.equal(projectCvHtml(EMPTY_PAYLOAD), '');
  });

  test('certifications without a title of their own close the education section', () => {
    const titles = { ...USUAL_TITLES, education: 'Education & Certifications', certifications: null };
    const html = projectCvHtml({ education: SMALL.education, certifications: SMALL.certifications, skills: SMALL.skills }, titles);
    assert.equal(html, [
      '<div class="section-title">Education &amp; Certifications</div>',
      '<div class="edu-item"><div class="edu-title">MS Computer Science</div><div class="edu-org">UT Austin</div>'
        + '<div class="edu-year">2018</div></div>',
      '<div><span class="cert-title">Certified Kubernetes Administrator</span></div>',
      '<div class="section-title">Skills</div>',
      '<div class="skill-item" data-i="0"><span class="skill-category">Languages:</span> Python, Go</div>',
    ].join('\n'));
    // With no degree to show they are the education section by themselves.
    assert.match(projectCvHtml({ certifications: SMALL.certifications, skills: SMALL.skills }, titles),
      /^<div class="section-title">Education &amp; Certifications<\/div>\n<div><span class="cert-title">/);
  });

  test('text is escaped, a backslash too, so no payload text reads as markup or as a LaTeX document', () => {
    const html = projectCvHtml({
      summary: ['R&D <b>work</b> on "agents", it\'s C:\\tools and \\begin{document}'],
      experience: [{ company: 'A&B <Labs>', role: 'ML Engineer', blocks: [{ bullets: ['Cut p99 < 50ms & cost > 10%'] }] }],
    });
    assert.ok(html.includes('<p>R&amp;D &lt;b&gt;work&lt;/b&gt; on &quot;agents&quot;, it&#39;s C:&#92;tools and &#92;begin{document}</p>'), html);
    assert.ok(html.includes('<div class="job-company">A&amp;B &lt;Labs&gt;</div>'), html);
    assert.ok(html.includes('<li>Cut p99 &lt; 50ms &amp; cost &gt; 10%</li>'), html);
    assert.ok(!html.includes('\\'), 'no backslash is written as itself');
    // Every tag is one the projection writes.
    const tagsOf = (text) => [...new Set([...text.matchAll(/<\/?([a-z]+)/g)].map(([, tag]) => tag))].sort();
    assert.deepEqual(tagsOf(html), ['div', 'li', 'p', 'ul']);
    // So in every field of the schema, the name and text of a project with a
    // link and of one without among them: a field written as itself would add
    // a <b> here. Empty strings stay empty, so the second project keeps no link.
    const tagged = mapPayloadStrings(FULL, (text) => (text ? `${text} <b>x</b>` : text));
    assert.deepEqual(tagsOf(projectCvHtml(tagged)), ['a', 'div', 'li', 'p', 'span', 'ul']);
  });

  test('the payload is normalised first, so an index counts the lines that are shown', () => {
    const messy = {
      summary: ['', '  One\nparagraph. '],
      projects: [{ name: '', text: '', url: '' }, { name: 'FraudShield', text: '', url: 'https://github.com/alexchen/fraudshield' }],
      skills: [{ category: '', items: '' }, { category: 'Languages:', items: ['Python', 'Go'] }],
    };
    const html = projectCvHtml(messy);
    assert.equal(html, projectCvHtml(normalizeCvPayload(messy)));
    assert.ok(html.includes('<div class="skill-item" data-i="0"><span class="skill-category">Languages:</span> Python, Go</div>'), html);
    assert.ok(html.includes('<a data-p="0" href="https://github.com/alexchen/fraudshield">FraudShield</a>'), html);
  });

  test('a title, a name or a category cv.md writes in bold is one text in the payload, the projection and the .tex', () => {
    const copied = withChange((cv) => {
      cv.experience[0].role = `**${cv.experience[0].role}**`;
      cv.projects.items[0].name = '**FraudShield**';
      cv.skills[0].category = '**ML/AI:**';
    });
    const payload = normalizeCvPayload(copied);
    assert.deepEqual(payload, PERSONA);
    const html = projectCvHtml(copied);
    assert.equal(html, projectCvHtml(PERSONA));
    assert.ok(html.includes('<div class="job-role">Senior ML Engineer / ML Platform Lead</div>'), html);
    assert.ok(html.includes('<span class="skill-category">ML/AI:</span> PyTorch'), html);
    const tex = render(copied, { template: '{{EXPERIENCE}}\n{{PROJECTS}}\n{{SKILLS}}' });
    assert.ok(!tex.includes('*'), tex);
    for (const call of ['\\cvEntryHead{Senior ML Engineer / ML Platform Lead}{TechFin Corp}', '\\cvProject{FraudShield}{}',
      '\\cvSkill{ML/AI}{PyTorch,']) {
      assert.ok(tex.includes(call), call);
    }
  });

  test('the dates are projected as the template shows them, so the checks read the dates that print', () => {
    const payload = withChange((cv) => {
      cv.experience[1].dates = 'Mar 2019 \u2013 Present (promoted 2022)';
      cv.education[0].dates = 'Sep 2014 \u2013 Jan 2017';
    });
    const years = projectCvHtml(payload, USUAL_TITLES, { shownDates: yearsOnly });
    for (const field of ['<div class="job-period">2020\u20132024</div>', '<div class="job-period">2019\u2013Present</div>',
      '<div class="edu-year">2014\u20132017</div>', '<div class="edu-year">2016</div>']) {
      assert.ok(years.includes(field), `${field} in ${years}`);
    }
    // As a template that shows years shows them, and as written for one that shows them in full, or without one.
    assert.equal(projectCvHtml(payload, USUAL_TITLES, { shownDates: shownDatesFor({ dates: 'year' }) }), years);
    const full = projectCvHtml(payload, USUAL_TITLES);
    assert.ok(full.includes('<div class="job-period">Mar 2019 \u2013 Present (promoted 2022)</div>'), full);
    assert.equal(projectCvHtml(payload, USUAL_TITLES, { shownDates: shownDatesFor({ dates: 'full' }) }), full);
    assert.equal(projectCvHtml(payload, USUAL_TITLES, {}), full);
    // The template's own function prints the same: renderCvTex() takes it from there.
    const tex = render(payload, { template: '% career-ops-latex: dates=year\n{{EXPERIENCE}}\n{{EDUCATION}}' });
    assert.ok(tex.includes('{Remote}{2019\u2013Present}\\cvStop{.}') && tex.includes('{UT Austin}{2014\u20132017}\\cvStop{.}'), tex);
  });
});

describe('the checks on a projection', () => {
  test('the persona CV, word for word as cv.md has it, passes every check', () => {
    const html = projected(PERSONA);
    const result = factCheck(html);
    assert.equal(result.verdict, 'pass', JSON.stringify(result, null, 2));
    validateLockedSections(html, PERSONA_CV, ['summary', 'experience', 'projects', 'education', 'skills']);
    validateCvExperienceOrder(html);
  });

  test('an invented employer, degree, certification or skill, or wrong dates, each block', () => {
    const cases = [
      [(cv) => { cv.experience[0].company = 'Northwind Robotics'; }, 'employer=northwind robotics'],
      [(cv) => { cv.education[0].degree = 'PhD Computer Science'; }, 'education=phd computer science'],
      [(cv) => { cv.certifications = ['Certified Kubernetes Administrator']; }, 'certification=certified kubernetes administrator'],
      [(cv) => { cv.skills[2].items += ', Terraform'; }, 'tool=terraform'],
      // The dates of the other role: every string here is the persona's own.
      [(cv) => { cv.experience[0].dates = '2018-2020'; }, 'employment-dates=techfin corp: 2018-2020'],
    ];
    for (const [change, finding] of cases) {
      const result = factCheck(projected(withChange(change)));
      assert.equal(result.verdict, 'block', finding);
      assert.ok(rejectedFacts(result).includes(finding), `${finding} among: ${rejectedFacts(result).join(', ')}`);
    }
  });

  test('roles listed oldest first fail the order check', () => {
    validateCvExperienceOrder(projected(PERSONA));
    const reversed = withChange((cv) => { cv.experience.reverse(); });
    assert.throws(() => validateCvExperienceOrder(projected(reversed)), /not in reverse-chronological order/);
  });

  test('with the years a template shows, the fact check reads the dates that print: an end year cv.md does not have blocks', () => {
    // The check reads no date in "Dec 2021 (contract)" and lets it pass with a
    // warning; a template that shows years prints 2018-2021 for it, which
    // cv.md's 2018-2020 contradicts. Checked as written, that year reached the PDF.
    const payload = withChange((cv) => { cv.experience[1].dates = 'Jan 2018 – Dec 2021 (contract)'; });
    const titles = sectionTitles(PERSONA_CV, sectionKey);
    const written = factCheck(projectCvHtml(payload, titles));
    assert.notEqual(written.verdict, 'block', JSON.stringify(written, null, 2));
    const printed = factCheck(projectCvHtml(payload, titles, { shownDates: yearsOnly }));
    assert.equal(printed.verdict, 'block', JSON.stringify(printed, null, 2));
    assert.ok(rejectedFacts(printed).some((fact) => /^employment-dates=datastartup inc: 2018.2021$/.test(fact)), rejectedFacts(printed).join(', '));
    // The persona's own dates print as years cv.md has.
    assert.equal(factCheck(projectCvHtml(PERSONA, titles, { shownDates: yearsOnly })).verdict, 'pass');
  });

  test('a locked section printed in years passes only where cv.md writes years: dates with months read as a changed section', () => {
    const titles = sectionTitles(PERSONA_CV, sectionKey);
    // The persona's dates are years, and the dash between them is no word to the check.
    validateLockedSections(projectCvHtml(PERSONA, titles, { shownDates: yearsOnly }), PERSONA_CV, ['experience', 'education']);
    assert.equal(PERSONA_CV.split('\n2018-2020').length - 1, 1, 'the persona CV no longer dates its second job 2018-2020');
    const withMonths = PERSONA_CV.replace('\n2018-2020', '\nJan 2018 - Dec 2020');
    const payload = withChange((cv) => { cv.experience[1].dates = 'Jan 2018 - Dec 2020'; });
    validateLockedSections(projectCvHtml(payload, titles), withMonths, ['experience']);
    assert.throws(() => validateLockedSections(projectCvHtml(payload, titles, { shownDates: yearsOnly }), withMonths, ['experience']),
      /Locked section 'experience' was modified during tailoring/);
  });

  test('a locked section passes as cv.md has it, and fails once a bullet is reworded', () => {
    validateLockedSections(projected(PERSONA), PERSONA_CV, ['experience']);
    const reworded = withChange((cv) => { cv.experience[1].blocks[0].bullets[1] = 'Shipped search ranking with learning-to-rank models'; });
    assert.throws(() => validateLockedSections(projected(reworded), PERSONA_CV, ['experience']),
      /Locked section 'experience' was modified during tailoring/);
  });

  test('a job intro, a lead and its intro come in the order cv.md writes them', () => {
    const cv = [
      '## Work Experience', '### Northwind Robotics -- Remote', '**ML Engineer**\n2018-2020', 'A team of four in the robotics group.',
      '**Search**', 'Ranking for the parts store.', '- Built ranking\n- Ran tests', '## Skills', '- **Languages:** Python, Go', '',
    ].join('\n\n');
    const payload = {
      experience: [{
        company: 'Northwind Robotics',
        role: 'ML Engineer',
        location: 'Remote',
        dates: '2018-2020',
        blocks: [
          { lead: '', intro: 'A team of four in the robotics group.', bullets: [] },
          { lead: 'Search', intro: 'Ranking for the parts store.', bullets: ['Built ranking', 'Ran tests'] },
        ],
      }],
      skills: [{ category: 'Languages', items: 'Python, Go' }],
    };
    validateLockedSections(projected(payload, cv), cv, ['experience', 'skills']);
  });

  test('certifications under cv.md\'s education heading are part of a locked education section', () => {
    // A plain "Education" heading as much as "Education & Certifications":
    // generate-pdf.mjs reads both as the education section.
    for (const heading of ['Education', 'Education & Certifications']) {
      const cv = PERSONA_CV.replace('## Education\n', `## ${heading}\n`)
        .replace('(2016)\n', '(2016)\n- Certified Kubernetes Administrator\n');
      assert.equal(cv.match(new RegExp(`^## ${heading}$|^- Certified Kubernetes Administrator$`, 'gm'))?.length, 2,
        'the persona CV changed');
      const payload = withChange((p) => { p.certifications = ['Certified Kubernetes Administrator']; });
      validateLockedSections(projected(payload, cv), cv, ['education']);
      // Under a title of their own they would read as taken out of it, and
      // left out of the payload the section misses a line cv.md has.
      for (const changed of [projectCvHtml(payload, USUAL_TITLES), projected(PERSONA, cv)]) {
        assert.throws(() => validateLockedSections(changed, cv, ['education']),
          /Locked section 'education' was modified during tailoring/, heading);
      }
      // There they are still read as certifications. (Under the merged heading
      // the fact check warns that it cannot pair degree and school.)
      const merged = checkRoot(cv);
      assert.notEqual(factCheck(projected(payload, cv), merged).verdict, 'block', heading);
      const invented = withChange((p) => { p.certifications = ['Certified Kubernetes Administrator', 'Certified Robotics Engineer']; });
      assert.ok(rejectedFacts(factCheck(projected(invented, cv), merged)).includes('certification=certified robotics engineer'), heading);
    }
  });

  test('a backslash in the text leaves the whole projection to the fact check', () => {
    // verify-cv-facts.mjs reads a text that holds \begin{document} as a LaTeX
    // document, and drops everything before it. Put in a project's text, a
    // line of its own: in a list of bullets that holds "94%", the check first
    // cut the line at the % as a LaTeX comment, and the \begin{document} with it.
    const payload = withChange((cv) => {
      cv.experience[0].company = 'Northwind Robotics';
      cv.projects.items[1].text += ' Its report template opens with \\begin{document}';
    });
    const html = projected(payload);
    assert.ok(!html.includes('\\'), 'no backslash is written as itself');
    assert.ok(rejectedFacts(factCheck(html)).includes('employer=northwind robotics'));
    // Written as itself, the backslash would hide the employer from the check.
    assert.ok(!rejectedFacts(factCheck(html.replaceAll('&#92;', '\\'))).includes('employer=northwind robotics'));
  });
});

describe('reading repairs back into the payload', () => {
  test('a skill line the repair cut is read back, and one it removed is gone', () => {
    const payload = withChange((cv) => {
      cv.skills[2].items += ', Terraform';
      cv.skills.splice(3, 0, { category: 'Cloud', items: 'Pulumi, Terraform' });
    });
    const html = projected(payload);
    const result = factCheck(html);
    assert.deepEqual(rejectedFacts(result).sort(), ['tool=pulumi', 'tool=terraform']);
    const repaired = pruneRejectedEntries(html, result.unsupportedFacts.map(({ value }) => value),
      checkReading({ gate: factGate, checks: { normalizeTextForATS } }));
    assert.equal(repaired.removed.length, 3);
    const back = readSkillsBack(repaired.html, payload);
    assert.deepEqual(back, PERSONA);
    assert.equal(factCheck(projected(back)).verdict, 'pass');
  });

  test('a skill line comes back with its characters as written', () => {
    const payload = {
      summary: ['One paragraph.'],
      skills: [{ category: 'R&D', items: 'C++ <beta>, "Kafka" Streams, the team\'s tools, a\\b, Q&amp;A' }, { category: '', items: 'Go' }],
    };
    const back = readSkillsBack(projectCvHtml(payload), payload);
    assert.deepEqual(back.skills, payload.skills);
    assert.deepEqual(back, normalizeCvPayload(payload));
  });

  test('only a skill line the projection wrote is read, and a line missing from the HTML is gone', () => {
    // Text that spells a skill line is escaped, so it cannot stand in for one:
    // here for line 2, taken out of the HTML as the repair takes out a line
    // that lost every item. Written as itself, the summary would bring back
    // "Infra: Terraform".
    const forged = '<div class="skill-item" data-i="2"><span class="skill-category">Infra:</span> Terraform</div>';
    const payload = withChange((cv) => { cv.summary.push(forged); });
    const line = `\n<div class="skill-item" data-i="2"><span class="skill-category">Infra:</span> ${PERSONA.skills[2].items}</div>`;
    const html = projected(payload);
    assert.ok(html.includes(line), html);
    assert.deepEqual(readSkillsBack(html.replace(line, ''), payload).skills, PERSONA.skills.filter((_, index) => index !== 2));
    assert.deepEqual(readSkillsBack('', PERSONA).skills, []);
  });

  test('a link the link rule takes off loses its address, whatever the address holds', () => {
    const payload = withChange((cv) => {
      cv.projects.items[0].url = 'https://github.com/alexchen/fraudshield?tab=readme&x=1';
      cv.projects.items[1].url = 'https://example.com/alex\'s-tools';
      cv.projects.items.push({ name: 'Notes', text: 'Notes on ML systems.', url: 'https://alexchen.dev' });
    });
    const before = structuredClone(payload);
    const { html, unlinked } = withoutUnsupportedLinks(projected(payload), PERSONA_CV);
    const { payload: kept, dropped } = dropPayloadLinks(payload, html);
    assert.deepEqual(kept.projects.items.map(({ url }) => url), ['', '', 'https://alexchen.dev']);
    assert.deepEqual(dropped, ['https://github.com/alexchen/fraudshield?tab=readme&x=1', 'https://example.com/alex\'s-tools']);
    // The link rule names each address as the HTML spells it: matched against
    // those, the payload's own spelling would have kept both links.
    assert.equal(unlinked.length, 2);
    assert.ok(dropped.every((url) => !unlinked.includes(url)), unlinked.join(', '));
    assert.deepEqual(payload, before);
  });

  test('an address the link rule takes off is named once, however many projects carry it', () => {
    // As the link rule names it: once, in the order first met. Every project
    // that carried it loses it.
    const payload = withChange((cv) => {
      cv.projects.items[0].url = 'https://github.com/alexchen/ml-tools';
      cv.projects.items[1].url = 'https://github.com/alexchen/llm-eval';
      cv.projects.items.push({ name: 'Notes', text: 'Notes on ML systems.', url: 'https://github.com/alexchen/ml-tools' });
    });
    const { html, unlinked } = withoutUnsupportedLinks(projected(payload), PERSONA_CV);
    assert.deepEqual(unlinked, ['https://github.com/alexchen/ml-tools', 'https://github.com/alexchen/llm-eval']);
    const { payload: kept, dropped } = dropPayloadLinks(payload, html);
    assert.deepEqual(kept.projects.items.map(({ url }) => url), ['', '', '']);
    assert.deepEqual(dropped, ['https://github.com/alexchen/ml-tools', 'https://github.com/alexchen/llm-eval']);
  });

  test('an anchor spelled out in the text cannot keep a link', () => {
    // The forged anchor names project 0 and an address the persona's CV holds.
    // Written as itself, the link rule would keep it, and with it project 0's
    // own address, which the CV does not hold.
    const forged = '<a data-p="0" href="https://alexchen.dev">FraudShield</a>';
    const places = {
      'in a project\'s text': (cv) => { cv.projects.items[0].text += ` ${forged}`; },
      'as the name of a project without a link': (cv) => { cv.projects.items[1].name = forged; },
    };
    for (const [where, place] of Object.entries(places)) {
      const payload = withChange((cv) => {
        cv.projects.items[0].url = 'https://github.com/alexchen/fraudshield';
        place(cv);
      });
      const projection = projected(payload);
      assert.ok(projection.includes('&lt;a data-p=&quot;0&quot; href=&quot;https://alexchen.dev&quot;&gt;'), `${where}: ${projection}`);
      const { html } = withoutUnsupportedLinks(projection, PERSONA_CV);
      const { payload: kept, dropped } = dropPayloadLinks(payload, html);
      assert.equal(kept.projects.items[0].url, '', where);
      assert.deepEqual(dropped, ['https://github.com/alexchen/fraudshield'], where);
    }
  });

  test('a payload whose links all stay comes back as it was', () => {
    const linked = withChange((cv) => { cv.projects.items[0].url = 'https://alexchen.dev'; });
    const { html } = withoutUnsupportedLinks(projected(linked), PERSONA_CV);
    assert.deepEqual(dropPayloadLinks(linked, html), { payload: linked, dropped: [] });
    assert.deepEqual(dropPayloadLinks(PERSONA, projected(PERSONA)), { payload: PERSONA, dropped: [] });
  });
});

// ---------------------------------------------------------------------------
// Years only
// ---------------------------------------------------------------------------

describe('years only', () => {
  test('a range keeps its years, and a range within one year is that year', () => {
    const cases = [
      ['Mar 2019 \u2013 Present', '2019\u2013Present'],
      ['Feb 2017 \u2013 Nov 2017', '2017'],
      ['Sep 2014 \u2013 Jan 2017', '2014\u20132017'],
      ['2020-2024', '2020\u20132024'],
      ['2018', '2018'],
    ];
    for (const [dates, years] of cases) assert.equal(yearsOnly(dates), years, dates);
  });

  test('each way of writing a range is split once, at its first separator', () => {
    const ranges = [
      '2018 - 2020', '2018 \u2014 2020', '2018 to 2020', '2018 TO 2020', '2018\u20132020', '2018 -- 2020', '2018\u20142020',
      '2018--2020', 'Jan 2018-Mar 2020', 'Jan 2018 \u2013 Mar 2020 (remote)', '2018 \u2013 2020 \u2013 2022',
    ];
    for (const dates of ranges) assert.equal(yearsOnly(dates), '2018\u20132020', dates);
    // An end with no year is kept as it is written: the start year alone
    // would read as a job that ended in that year.
    assert.equal(yearsOnly('2018-Present'), '2018\u2013Present');
    assert.equal(yearsOnly('Jan 2018 to now'), '2018\u2013now');
  });

  test('a date with no year before its separator is left as it is, and one month is its year', () => {
    for (const dates of ['Present', 'Q3 \u2013 Q4', '', 'since spring', 'Summer \u2013 2019']) assert.equal(yearsOnly(dates), dates, dates);
    assert.equal(yearsOnly('2019-03'), '2019');
    assert.equal(yearsOnly('2019-03 \u2013 2021-06'), '2019\u20132021');
  });

  test('a word after the separator ends the range, and what follows that word is dropped', () => {
    // The first year after the separator was taken for the end, so a job held
    // today, with a remark after "Present", printed as one that ended then.
    const cases = [
      ['Mar 2020 \u2013 Present (promoted 2022)', '2020\u2013Present'],
      ['Mar 2020 - Present, promoted in 2022', '2020\u2013Present'],
      ['2020-Present (promoted 2022)', '2020\u2013Present'],
      ['Jan 2018 \u2013 Till Date', '2018\u2013Till Date'],
      // A word that ends a range held today is no month, whatever year follows it.
      ['Mar 2020 - Present, 2022', '2020\u2013Present'],
      ['Mar 2020 - Present 2022', '2020\u2013Present'],
      ['Jan 2018 to now, 2020', '2018\u2013now'],
      ['Jan 2018 \u2013 Present. 2020', '2018\u2013Present.'],
      ['M\u00e4rz 2019 \u2013 heute 2022', '2019\u2013heute'],
      ['Mar 2019 \u2013 Actualidad, 2022', '2019\u2013Actualidad'],
      // A month before the year that ends the range is no such word, in any language and with a day beside it.
      ['Jan 2018 \u2013 Mar. 2020 (remote)', '2018\u20132020'],
      ['Jan 2018 \u2013 M\u00e4rz 2020', '2018\u20132020'],
      ['Jan 2018 \u2013 March 1, 2020', '2018\u20132020'],
      ['Jan 2018 \u2013 March, 2020', '2018\u20132020'],
      ['Jan 2018 \u2013 Q1 2020', '2018\u20132020'],
      ['Jan 2018 \u2013 1 Mar 2020', '2018\u20132020'],
    ];
    for (const [dates, years] of cases) assert.equal(yearsOnly(dates), years, dates);
  });
});

// ---------------------------------------------------------------------------
// The header lines of a template
// ---------------------------------------------------------------------------

describe('the header lines of a template', () => {
  const ARTICLE = '\\documentclass{article}\n\\begin{document}\nx\n\\end{document}\n';
  const withLines = (...lines) => `${lines.join('\n')}\n${ARTICLE}`;

  test('the engines, and the template a CV is written in when none is named', () => {
    assert.deepEqual(LATEX_ENGINES, ['xelatex', 'pdflatex', 'tectonic']);
    assert.equal(DEFAULT_LATEX_TEMPLATE, 'default');
  });

  test('a template with no header lines runs on pdflatex and shows dates in full', () => {
    assert.deepEqual(readTemplateOptions(ARTICLE), { engine: 'pdflatex', dates: 'full' });
    assert.deepEqual(readTemplateOptions(withLines('% A CV.', '%')), { engine: 'pdflatex', dates: 'full' });
    assert.deepEqual(readTemplateOptions(''), { engine: 'pdflatex', dates: 'full' });
  });

  test('the engine line names one of the engines, as TeX editors write it', () => {
    for (const engine of LATEX_ENGINES) {
      assert.deepEqual(readTemplateOptions(withLines(`%!TEX TS-program = ${engine}`)), { engine, dates: 'full' }, engine);
    }
    assert.equal(readTemplateOptions(withLines('% !TeX TS-program = XeLaTeX')).engine, 'xelatex');
    assert.equal(readTemplateOptions(withLines('%!TEX program = tectonic')).engine, 'tectonic');
  });

  test('lualatex is refused, and so is any other engine', () => {
    assert.throws(() => readTemplateOptions(withLines('% A CV.', '%!TEX TS-program = lualatex')),
      { message: 'line 2: lualatex is not supported: it runs Lua code from the document. Use xelatex.' });
    // The line names an engine, never a command: no option rides on it.
    for (const value of ['latexmk', 'xelatex -shell-escape', 'luatex', '']) {
      assert.throws(() => readTemplateOptions(withLines(`%!TEX TS-program = ${value}`)),
        { message: `line 1: the engine line names "${value}": a template runs on xelatex, pdflatex or tectonic.` }, value);
    }
  });

  test('dates is full or year', () => {
    assert.equal(readTemplateOptions(withLines('% career-ops-latex: dates=year')).dates, 'year');
    assert.equal(readTemplateOptions(withLines('% career-ops-latex: dates=full')).dates, 'full');
    assert.equal(readTemplateOptions(withLines('%career-ops-latex:dates = year')).dates, 'year');
    assert.throws(() => readTemplateOptions(withLines('% career-ops-latex: dates=month')),
      { message: 'line 1: dates is full or year, not "month".' });
  });

  test('an unknown option, two on one line, none, or one set twice is a template problem', () => {
    const cases = [
      [['% career-ops-latex: colour=blue'], 'line 1: career-ops-latex has no option "colour": its options are dates and template.'],
      [['% career-ops-latex: dates=year template=plain'], 'line 1: "dates=year template=plain" is not one option: write one key=value '
        + 'on each line, such as dates=year.'],
      [['% career-ops-latex:'], 'line 1: "" is not one option: write one key=value on each line, such as dates=year.'],
      [['% career-ops-latex: template=../x'], 'line 1: "../x" is not a template name.'],
      [['% career-ops-latex: dates=year', '% career-ops-latex: dates=full'], 'line 2: dates was set on line 1 already.'],
      [['%!TEX TS-program = xelatex', '%!TEX TS-program = pdflatex'], 'line 2: the engine was set on line 1 already.'],
    ];
    for (const [lines, message] of cases) assert.throws(() => readTemplateOptions(withLines(...lines)), { message }, message);
  });

  test('only the comment lines before \\documentclass are read', () => {
    const late = ARTICLE.replace('\\begin{document}', '%!TEX TS-program = lualatex\n% career-ops-latex: colour=blue\n\\begin{document}');
    assert.deepEqual(readTemplateOptions(late), { engine: 'pdflatex', dates: 'full' });
    // A comment that names \documentclass does not end them.
    assert.equal(readTemplateOptions(withLines('% From \\documentclass{article}, cut down.', '% career-ops-latex: dates=year')).dates, 'year');
    // A BOM and CRLF line ends, as an editor on Windows may save a template.
    assert.deepEqual(readTemplateOptions(`\uFEFF%!TEX TS-program = xelatex\r\n% career-ops-latex: dates=year\r\n${ARTICLE}`),
      { engine: 'xelatex', dates: 'year' });
  });

  test('templateNameOf reads line 1 only, and gives a name only', () => {
    assert.equal(templateNameOf(`% career-ops-latex: template=plain\n${ARTICLE}`), 'plain');
    assert.equal(templateNameOf(`\uFEFF% career-ops-latex: template=plain\r\n${ARTICLE}`), 'plain');
    assert.equal(templateNameOf(`%!TEX TS-program = xelatex\n% career-ops-latex: template=plain\n${ARTICLE}`), null);
    for (const name of ['../x', 'A', 'a.b', '', 'a b', '-x']) {
      assert.equal(templateNameOf(`% career-ops-latex: template=${name}\n${ARTICLE}`), null, name);
    }
    assert.equal(templateNameOf(`% career-ops-latex: dates=year\n${ARTICLE}`), null);
    for (const text of ['', undefined, ARTICLE]) assert.equal(templateNameOf(text), null, String(text));
  });
});

// ---------------------------------------------------------------------------
// Resolving a template
// ---------------------------------------------------------------------------

const DEFAULT_TEMPLATE_TEXT = readFileSync(join(ROOT, 'lib', 'ollama-cv-latex-default.tex'), 'utf8').replace(/\r\n/g, '\n');

/** A data root of its own with a pack for each entry: its cv-latex.tex, or for null a folder without one. */
function packRoot(packs = {}) {
  const root = mkdtempSync(join(SANDBOX, 'packs-'));
  for (const [name, text] of Object.entries(packs)) {
    const dir = join(root, 'data', 'templates', name);
    mkdirSync(dir, { recursive: true });
    if (text !== null) writeFileSync(join(dir, 'cv-latex.tex'), text);
  }
  return root;
}

const PLAIN_PACK = [
  '%!TEX TS-program = xelatex', '% career-ops-latex: dates=year', '\\documentclass{article}', '\\begin{document}', '{{NAME}}',
  '{{EXPERIENCE}}', '\\end{document}', '',
].join('\n');

describe('resolving a template', () => {
  test('default is the template in lib/, with no folder to copy', () => {
    const resolved = resolveLatexTemplate('default', { dataRoot: packRoot() });
    assert.equal(resolved.ok, true, resolved.reason);
    assert.deepEqual([resolved.name, resolved.path, resolved.packDir, resolved.text],
      ['default', join(ROOT, 'lib', 'ollama-cv-latex-default.tex'), null, DEFAULT_TEMPLATE_TEXT]);
    assert.deepEqual(resolved.options, { engine: 'pdflatex', dates: 'full' });
    // It needs no data root.
    assert.equal(resolveLatexTemplate('default').ok, true);
  });

  test('a pack is data/templates/<name>/cv-latex.tex, and the folder it is in goes with it', () => {
    const dataRoot = packRoot({ plain: PLAIN_PACK });
    const resolved = resolveLatexTemplate('plain', { dataRoot });
    assert.equal(resolved.ok, true, resolved.reason);
    const folder = join(dataRoot, 'data', 'templates', 'plain');
    assert.deepEqual([resolved.name, resolved.path, resolved.packDir, resolved.text], ['plain', join(folder, 'cv-latex.tex'), folder, PLAIN_PACK]);
    assert.deepEqual(resolved.options, { engine: 'xelatex', dates: 'year' });
  });

  test('a folder named default does not take the place of the template in lib/', () => {
    const resolved = resolveLatexTemplate('default', { dataRoot: packRoot({ default: PLAIN_PACK }) });
    assert.deepEqual([resolved.ok, resolved.packDir, resolved.options.engine], [true, null, 'pdflatex']);
  });

  test('a name that is not one is refused, and the reason lists the templates there are', () => {
    // modern is a folder without cv-latex.tex, such as a pack of HTML templates.
    const dataRoot = packRoot({ plain: PLAIN_PACK, modern: null });
    for (const name of ['../x', 'A', '', 'a.b', 'a/b', 'a\\b', ' plain', 'plain\n', '-x', undefined, null, 7]) {
      const { ok, reason } = resolveLatexTemplate(name, { dataRoot });
      assert.equal(ok, false, String(name));
      assert.match(reason, /^".*" is not a template name: a name is lower-case letters, digits and hyphens, and begins with a letter or a digit\. /,
        String(name));
      assert.ok(reason.endsWith(' Templates here: default, plain.'), reason);
      assert.doesNotMatch(reason, /\p{Cc}/u, String(name));
    }
  });

  test('a pack that is not there is refused, naming the file looked for', () => {
    const dataRoot = packRoot({ plain: PLAIN_PACK, modern: null });
    assert.deepEqual(resolveLatexTemplate('fancy', { dataRoot }), {
      ok: false,
      reason: 'No template "fancy": data/templates/fancy/cv-latex.tex was not found. Templates here: default, plain.',
    });
    assert.equal(resolveLatexTemplate('modern', { dataRoot }).reason,
      'No template "modern": data/templates/modern/cv-latex.tex was not found. Templates here: default, plain.');
    assert.equal(resolveLatexTemplate('plain', { dataRoot: packRoot() }).reason,
      'No template "plain": data/templates/plain/cv-latex.tex was not found. Templates here: default.');
    assert.equal(resolveLatexTemplate('plain').ok, false);
  });

  test('each template problem is refused, by the resolver and by the renderer', () => {
    const body = (...lines) => ['\\documentclass{article}', '\\begin{document}', ...lines, '\\end{document}', ''].join('\n');
    const keys = 'SUMMARY, EXPERIENCE, PROJECTS, EDUCATION, CERTIFICATIONS, SKILLS, TITLE, EMAIL, PHONE, LINKEDIN, GITHUB, LOCATION';
    const cases = [
      [body('{{AWARDS}}'), 'line 3: {{AWARDS}} is not a placeholder of a LaTeX CV.'],
      [body('%%IF AWARDS', 'x', '%%END AWARDS'), `line 3: "%%IF AWARDS" names no block: a block runs from %%IF KEY to %%END KEY, KEY one of ${keys}.`],
      [body('%%IF', 'x'), `line 3: "%%IF" names no block: a block runs from %%IF KEY to %%END KEY, KEY one of ${keys}.`],
      [body('%%END OF THE HEADER'), `line 3: "%%END OF THE HEADER" names no block: a block runs from %%IF KEY to %%END KEY, KEY one of ${keys}.`],
      [body('%%IF SKILLS', '{{SKILLS}}'), 'line 3: %%IF SKILLS has no %%END SKILLS after it.'],
      [body('{{SKILLS}}', '%%END SKILLS'), 'line 4: %%END SKILLS has no %%IF SKILLS before it.'],
      [body('%%IF SKILLS', '%%IF TITLE', '%%END SKILLS', '%%END TITLE'), 'line 5: %%END SKILLS closes %%IF TITLE of line 4: blocks nest, '
        + 'and never cross.'],
      [`%!TEX TS-program = lualatex\n${body()}`, 'line 1: lualatex is not supported: it runs Lua code from the document. Use xelatex.'],
      [`%!TEX TS-program = latexmk\n${body()}`, 'line 1: the engine line names "latexmk": a template runs on xelatex, pdflatex or tectonic.'],
      [`% career-ops-latex: dates=month\n${body()}`, 'line 1: dates is full or year, not "month".'],
      [`% career-ops-latex: colour=blue\n${body()}`, 'line 1: career-ops-latex has no option "colour": its options are dates and template.'],
      // A rendered .tex copied in as a template would print that CV, whatever the payload.
      [`% career-ops-latex: template=plain\n${body()}`, 'line 1: template= is the renderer\'s to write, on line 1 of each .tex it renders: '
        + 'a template does not carry it.'],
    ];
    for (const [text, detail] of cases) {
      const { ok, reason } = resolveLatexTemplate('bad', { dataRoot: packRoot({ bad: text }) });
      assert.equal(ok, false, detail);
      assert.ok(reason.startsWith(`data/templates/bad/cv-latex.tex, ${detail}`), `${detail}\n${reason}`);
      assert.ok(reason.endsWith(' Templates here: default, bad.'), reason);
      assert.throws(() => renderCvTex({ template: text, templateName: 'bad', payload: PERSONA, header: {}, paper: 'letter' }),
        { message: `The template, ${detail}` }, detail);
    }
  });

  test('a placeholder written with spaces, in lower case or with a digit is refused too, where it would print as its name', () => {
    // Neither filled nor refused, {{ NAME }} stayed in the .tex, and the CV's
    // header read NAME. The slots of upstream's templates take that spelling.
    const body = (...lines) => ['\\documentclass{article}', '\\begin{document}', ...lines, '\\end{document}', ''].join('\n');
    const cases = [
      [body('{{ NAME }}'), 'line 3: {{ NAME }} is not a placeholder of a LaTeX CV: write {{NAME}}.'],
      [body('{{Name}}'), 'line 3: {{Name}} is not a placeholder of a LaTeX CV: write {{NAME}}.'],
      [body('\\cvContact{{{ phone }}}'), 'line 3: {{ phone }} is not a placeholder of a LaTeX CV: write {{PHONE}}.'],
      [body('{{PHONE2}}'), 'line 3: {{PHONE2}} is not a placeholder of a LaTeX CV.'],
      [body('{{NAME}} {{_X}}'), 'line 3: {{_X}} is not a placeholder of a LaTeX CV.'],
    ];
    for (const [text, detail] of cases) {
      const { ok, reason } = resolveLatexTemplate('bad', { dataRoot: packRoot({ bad: text }) });
      assert.equal(ok, false, detail);
      assert.ok(reason.startsWith(`data/templates/bad/cv-latex.tex, ${detail}`), `${detail}\n${reason}`);
      assert.throws(() => renderCvTex({ template: text, templateName: 'bad', payload: PERSONA, header: {}, paper: 'letter' }),
        { message: `The template, ${detail}` }, detail);
    }
    // Braces of TeX code around a placeholder, or around anything but a name, are no placeholder of their own. Nor
    // is a group of a length or a number: no placeholder begins with a digit.
    const code = body('\\hypersetup{pdfauthor={{{FIRST_NAME}} {{LAST_NAME}}}}', '\\cvContact{{{LOCATION}}}%', '\\newcommand{\\x}[1]{{\\bfseries #1}}',
      '\\setlength{\\parindent}{{0pt}}', '\\setcounter{secnumdepth}{{2}}');
    const resolved = resolveLatexTemplate('code', { dataRoot: packRoot({ code }) });
    assert.equal(resolved.ok, true, resolved.reason);
    // A name alone in double braces is one, in TeX code too; the guide's way
    // around it, the line broken after the braces with a %, is the same code.
    assert.match(resolveLatexTemplate('group', { dataRoot: packRoot({ group: body('\\def\\x{{a}}') }) }).reason,
      /line 3: \{\{a\}\} is not a placeholder of a LaTeX CV\./);
    const broken = resolveLatexTemplate('broken', { dataRoot: packRoot({ broken: body('\\def\\x{{%', 'a}}') }) });
    assert.equal(broken.ok, true, broken.reason);
  });
});

// ---------------------------------------------------------------------------
// Rendering a CV
// ---------------------------------------------------------------------------

const JANE_PROFILE = [
  'candidate:', '  full_name: "Jane Smith"', '  title: "Senior Backend Engineer"', '  email: "jane@example.com"', '  phone: "+1-555-0123"',
  '  location: "San Francisco, CA"', '  linkedin: "linkedin.com/in/janesmith"', '  github: "github.com/janesmith"', '',
].join('\n');
const JANE = readCvHeader(JANE_PROFILE);
const NAME_ONLY = readCvHeader('candidate:\n  full_name: "Jane Smith"\n');
const DAY = new Date(2026, 10, 21);
const SIX_SECTIONS = ['{{SUMMARY}}', '{{EXPERIENCE}}', '{{PROJECTS}}', '{{EDUCATION}}', '{{CERTIFICATIONS}}', '{{SKILLS}}'].join('\n');
const BLOCK_KEYS = ['SUMMARY', 'EXPERIENCE', 'PROJECTS', 'EDUCATION', 'CERTIFICATIONS', 'SKILLS', 'TITLE', 'EMAIL', 'PHONE', 'LINKEDIN', 'GITHUB',
  'LOCATION'];

/** The .tex of a payload, in the default template for Jane Smith on letter paper unless told otherwise. */
function render(payload, { template = DEFAULT_TEMPLATE_TEXT, templateName = 'default', header = JANE, paper = 'letter', date = DAY } = {}) {
  return renderCvTex({ template, templateName, payload, header, paper, date }).tex;
}

/** What a template of one placeholder renders to, without line 1. */
function filled(placeholder, payload = PERSONA, options = {}) {
  return render(payload, { ...options, template: `{{${placeholder}}}` }).split('\n').slice(1).join('\n');
}

/** A header from JANE_PROFILE without the line of `key`. */
const janeWithout = (key) => readCvHeader(JANE_PROFILE.split('\n').filter((line) => !line.startsWith(`  ${key}:`)).join('\n'));

describe('rendering a CV', () => {
  test('line 1 names the template, and the template follows it', () => {
    const tex = render(PERSONA);
    const [first, ...rest] = tex.split('\n');
    assert.equal(first, '% career-ops-latex: template=default');
    assert.equal(templateNameOf(tex), 'default');
    assert.ok(rest.join('\n').startsWith('%!TEX TS-program = pdflatex\n% career-ops-latex: dates=full\n\\documentclass[letterpaper,11pt]{article}\n'),
      rest.slice(0, 3).join('\n'));
    // A rendered .tex keeps the header lines of its template, and they read alike.
    assert.deepEqual(readTemplateOptions(tex), { engine: 'pdflatex', dates: 'full' });
    assert.equal(templateNameOf(render(PERSONA, { templateName: 'plain' })), 'plain');
    for (const templateName of ['../x', 'A', '', undefined]) {
      assert.throws(() => renderCvTex({ template: DEFAULT_TEMPLATE_TEXT, templateName, payload: PERSONA, header: JANE, paper: 'letter' }),
        { message: /is not a template name/ }, String(templateName));
    }
  });

  test('each header placeholder is filled from the profile, and the date and the paper by the caller', () => {
    const keys = ['NAME', 'FIRST_NAME', 'LAST_NAME', 'TITLE', 'LOCATION', 'EMAIL', 'EMAIL_URL', 'PHONE', 'LINKEDIN_URL', 'LINKEDIN_DISPLAY',
      'GITHUB_URL', 'GITHUB_DISPLAY', 'DATE', 'PAPER'];
    assert.deepEqual(Object.fromEntries(keys.map((key) => [key, filled(key)])), {
      NAME: 'Jane Smith',
      FIRST_NAME: 'Jane',
      LAST_NAME: 'Smith',
      TITLE: 'Senior Backend Engineer',
      LOCATION: 'San Francisco, CA',
      EMAIL: 'jane@example.com',
      EMAIL_URL: 'mailto:jane@example.com',
      PHONE: '+1-555-0123',
      LINKEDIN_URL: 'https://linkedin.com/in/janesmith',
      LINKEDIN_DISPLAY: 'linkedin.com/in/janesmith',
      GITHUB_URL: 'https://github.com/janesmith',
      GITHUB_DISPLAY: 'github.com/janesmith',
      DATE: '21 Nov 2026',
      PAPER: 'letterpaper',
    });
    assert.equal(filled('PAPER', PERSONA, { paper: 'a4' }), 'a4paper');
    assert.equal(filled('DATE', PERSONA, { date: new Date(2027, 0, 5) }), '5 Jan 2027');
    assert.throws(() => render(PERSONA, { paper: 'legal' }), { message: /paper/ });
    assert.throws(() => render(PERSONA, { date: new Date('not a date') }), { message: /date/ });
  });

  test('the sections are written in the macros of the vocabulary, one call per line', () => {
    const small = {
      summary: ['One paragraph.'],
      experience: [{
        company: 'Northwind Robotics',
        role: 'ML Engineer',
        location: 'Remote',
        dates: '2018-2020',
        blocks: [{ lead: 'Search', intro: 'Ranking work.', bullets: ['Built ranking', 'Ran tests'] }],
      }],
      projects: {
        intro: 'Open source.',
        items: [
          { name: 'FraudShield', text: 'A framework.', url: 'https://github.com/alexchen/fraudshield' },
          { name: 'LLM Eval Toolkit', text: '', url: '' },
        ],
      },
      education: [{ degree: 'MS Computer Science', institution: 'UT Austin', dates: '2018' }],
      certifications: ['Certified Kubernetes Administrator'],
      skills: [{ category: 'Languages', items: 'Python, Go' }],
    };
    assert.equal(render(small, { template: SIX_SECTIONS }), [
      '% career-ops-latex: template=default',
      '\\cvSummary{One paragraph.}\\cvStop{.}',
      '\\cvEntryHead{ML Engineer}{Northwind Robotics}{Remote}{2018-2020}\\cvStop{.}',
      '\\cvEntryBlock{Search}{Ranking work.}{%',
      '\\cvItem{Built ranking}\\cvStop{.}',
      '\\cvItem{Ran tests}\\cvStop{.}',
      '}',
      '\\cvProjects{Open source.}{%',
      '\\cvProject{FraudShield}{https://github.com/alexchen/fraudshield}{A framework.}\\cvStop{.}',
      '\\cvProject{LLM Eval Toolkit}{}{}\\cvStop{.}',
      '}',
      '\\cvEducation{MS Computer Science}{UT Austin}{2018}\\cvStop{.}',
      '\\cvCert{Certified Kubernetes Administrator}\\cvStop{.}',
      '\\cvSkill{Languages}{Python, Go}\\cvStop{.}',
    ].join('\n'));
  });

  test('a block is kept when its key has something to show, and dropped with its lines when not', () => {
    const template = BLOCK_KEYS.map((key) => `%%IF ${key}\nshown ${key.toLowerCase()}\n%%END ${key}`).join('\n');
    const shown = (payload, header = JANE) => render(payload, { template, header }).split('\n').slice(1).filter(Boolean);
    const all = BLOCK_KEYS.map((key) => `shown ${key.toLowerCase()}`);
    assert.deepEqual(shown(FULL), all);
    assert.deepEqual(shown(EMPTY_PAYLOAD, NAME_ONLY), []);
    const fullWithout = (change) => {
      const copy = structuredClone(FULL);
      change(copy);
      return copy;
    };
    const emptied = {
      SUMMARY: [fullWithout((cv) => { cv.summary = []; }), JANE],
      EXPERIENCE: [fullWithout((cv) => { cv.experience = []; }), JANE],
      PROJECTS: [fullWithout((cv) => { cv.projects.items = []; }), JANE],
      EDUCATION: [fullWithout((cv) => { cv.education = []; }), JANE],
      CERTIFICATIONS: [fullWithout((cv) => { cv.certifications = []; }), JANE],
      SKILLS: [fullWithout((cv) => { cv.skills = []; }), JANE],
      TITLE: [FULL, janeWithout('title')],
      EMAIL: [FULL, janeWithout('email')],
      PHONE: [FULL, janeWithout('phone')],
      LINKEDIN: [FULL, janeWithout('linkedin')],
      GITHUB: [FULL, janeWithout('github')],
      LOCATION: [FULL, janeWithout('location')],
    };
    for (const [key, [payload, header]] of Object.entries(emptied)) {
      assert.deepEqual(shown(payload, header), all.filter((line) => line !== `shown ${key.toLowerCase()}`), key);
    }
  });

  test('blocks nest: an inner block shows only within an outer one that shows', () => {
    const template = '%%IF EXPERIENCE\nexperience\n%%IF TITLE\ntitle\n%%END TITLE\n%%END EXPERIENCE\nafter';
    const lines = (payload, header = JANE) => render(payload, { template, header }).split('\n').slice(1);
    assert.deepEqual(lines(PERSONA), ['experience', 'title', 'after']);
    assert.deepEqual(lines(PERSONA, janeWithout('title')), ['experience', 'after']);
    assert.deepEqual(lines(withChange((cv) => { cv.experience = []; })), ['after']);
  });

  test('payload text is escaped once, so the LaTeX specials print as themselves', () => {
    const payload = withChange((cv) => {
      cv.experience[1].blocks[0].bullets.push('& % $ # _ { } ~ ^ \\', 'Wrote \\begin{document} and \\end{document} as text');
    });
    const tex = render(payload);
    assert.ok(tex.includes('\\cvItem{\\& \\% \\$ \\# \\_ \\{ \\} \\~{} \\textasciicircum{} \\textbackslash{}}\\cvStop{.}'), filled('EXPERIENCE', payload));
    assert.ok(tex.includes('\\cvItem{Wrote \\textbackslash{}begin\\{document\\} and \\textbackslash{}end\\{document\\} as text}'));
    // So the document has one body, the template's.
    assert.equal(tex.match(/\\begin\{document\}/g).length, 1);
    assert.equal(tex.match(/\\end\{document\}/g).length, 1);
  });

  test('six characters the final check misreads as commands are written as themselves', () => {
    const payload = withChange((cv) => { cv.experience[1].blocks[0].bullets.push('a < b > c | d \u2192 e \u00B1 f ~ g'); });
    const experience = filled('EXPERIENCE', payload);
    assert.ok(experience.includes('\\cvItem{a < b > c | d \u2192 e \u00B1 f \\~{} g}\\cvStop{.}'), experience);
    const commands = ['\\textless', '\\textgreater', '\\textbar', '\\rightarrow', '\\pm', '\\textasciitilde'];
    assert.deepEqual(commands.filter((command) => experience.includes(command)), []);
  });

  test('payload text is normalised first: no line break, control or bidi character reaches the .tex', () => {
    const payload = withChange((cv) => { cv.experience[1].blocks[0].bullets.push('Built\nKafka\u202E\u0001 pipe\u200Blines\tfast'); });
    assert.ok(filled('EXPERIENCE', payload).includes('\\cvItem{Built Kafka pipelines fast}\\cvStop{.}'));
  });

  test('**bold** is {\\bfseries …} in the summary, a lead, an intro, a bullet, a project text and a certification, and dropped elsewhere', () => {
    const marked = {
      summary: ['**summary** text'],
      experience: [{
        company: '**company** text',
        role: '**role** text',
        location: '**location** text',
        dates: '**dates** text',
        blocks: [{ lead: '**lead** text', intro: '**intro** text', bullets: ['**bullet** text'] }],
      }],
      projects: { intro: '**pintro** text', items: [{ name: '**name** text', text: '**ptext** text', url: '' }] },
      education: [{ degree: '**degree** text', institution: '**institution** text', dates: '**edates** text' }],
      certifications: ['**cert** text'],
      skills: [{ category: '**category** text', items: '**items** text' }],
    };
    const tex = render(marked, { template: SIX_SECTIONS, header: { ...JANE, title: '**title** text' } });
    for (const word of ['summary', 'lead', 'intro', 'bullet', 'pintro', 'ptext', 'cert']) assert.ok(tex.includes(`{\\bfseries ${word}} text`), word);
    // Printed as the text the checks read: without the asterisks, which they never see.
    for (const word of ['company', 'role', 'location', 'dates', 'name', 'degree', 'institution', 'edates', 'category', 'items']) {
      assert.ok(tex.includes(`{${word} text}`), word);
    }
    assert.ok(!tex.includes('*'), tex);
    assert.ok(!tex.includes('\\textbf'), tex);
    // The header is the profile's, written as the profile has it, and never bold.
    assert.equal(filled('TITLE', PERSONA, { header: { ...JANE, title: '**Lead** engineer' } }), '**Lead** engineer');
  });

  test('an argument that opens with [ opens with {[}, so no macro reads it as an option', () => {
    const payload = withChange((cv) => {
      cv.summary = ['[Draft] A summary.'];
      cv.experience[1].company = '[Stealth] Labs';
      cv.experience[1].blocks[0].bullets.push('[Beta] Shipped the ranking service', 'Shipped [beta] ranking');
      cv.skills[3].items = '[core] Python, Go';
    });
    const tex = render(payload, { header: { ...JANE, title: '[Interim] Head of ML' } });
    for (const text of ['\\cvSummary{{[}Draft] A summary.}', '{{[}Stealth] Labs}', '\\cvItem{{[}Beta] Shipped the ranking service}',
      '\\cvItem{Shipped [beta] ranking}', '\\cvSkill{Languages}{{[}core] Python, Go}', '{\\large {[}Interim] Head of ML}']) {
      assert.ok(tex.includes(text), text);
    }
  });

  test('an empty value is written {}', () => {
    const sparse = {
      experience: [{ company: 'Northwind Robotics', role: 'ML Engineer', blocks: [{ intro: 'Ranking work.' }, { bullets: ['Built ranking'] }] }],
      projects: { items: [{ name: 'LLM Eval Toolkit' }] },
      education: [{ degree: 'MS Computer Science', institution: 'UT Austin' }],
      skills: [{ items: 'Go' }],
    };
    assert.equal(filled('EXPERIENCE', sparse), [
      '\\cvEntryHead{ML Engineer}{Northwind Robotics}{}{}\\cvStop{.}',
      '\\cvEntryBlock{}{Ranking work.}{}',
      '\\cvEntryBlock{}{}{%',
      '\\cvItem{Built ranking}\\cvStop{.}',
      '}',
    ].join('\n'));
    assert.equal(filled('PROJECTS', sparse), '\\cvProjects{}{%\n\\cvProject{LLM Eval Toolkit}{}{}\\cvStop{.}\n}');
    assert.equal(filled('EDUCATION', sparse), '\\cvEducation{MS Computer Science}{UT Austin}{}\\cvStop{.}');
    assert.equal(filled('SKILLS', sparse), '\\cvSkill{}{Go}\\cvStop{.}');
    assert.ok(!render(sparse, { template: SIX_SECTIONS }).includes('{ }'));
  });

  test('\\cvEntrySep stands between two entries of experience and of education, never after the last', () => {
    const experience = filled('EXPERIENCE').split('\n');
    assert.equal(experience.filter((line) => line === '\\cvEntrySep').length, 1);
    const at = experience.indexOf('\\cvEntrySep');
    assert.equal(experience[at - 1], '}');
    assert.ok(experience[at + 1].startsWith('\\cvEntryHead{ML Engineer}{DataStartup Inc}'), experience[at + 1]);
    assert.deepEqual(filled('EDUCATION').split('\n'), [
      '\\cvEducation{MS Computer Science}{UT Austin}{2018}\\cvStop{.}',
      '\\cvEntrySep',
      '\\cvEducation{BS Computer Science}{UC Berkeley}{2016}\\cvStop{.}',
    ]);
    const one = withChange((cv) => { cv.experience.pop(); cv.education.pop(); });
    assert.ok(!render(one, { template: SIX_SECTIONS }).includes('\\cvEntrySep'));
    for (const key of ['SUMMARY', 'PROJECTS', 'CERTIFICATIONS', 'SKILLS']) assert.ok(!filled(key, FULL).includes('\\cvEntrySep'), key);
  });

  test('\\cvStop{.} follows each call that carries text, and only those', () => {
    const lines = render(FULL, { template: SIX_SECTIONS }).split('\n');
    const carrying = lines.filter((line) => /^\\cv(?:Summary|EntryHead|Item|Project|Education|Cert|Skill)\{/.test(line));
    assert.equal(carrying.length, 20);
    for (const line of carrying) assert.ok(line.endsWith('}\\cvStop{.}'), line);
    for (const line of lines.filter((text) => /^\\cv(?:EntryBlock|Projects)\{/.test(text))) assert.ok(!line.includes('\\cvStop'), line);
    assert.equal(lines.join('\n').split('\\cvStop{.}').length - 1, carrying.length);
  });

  test('an address goes through sanitizeUrl only: never escaped as text, never bold', () => {
    const url = 'https://example.com/a_b/**x**?q=1&r=2%20#top~1';
    const projects = filled('PROJECTS', withChange((cv) => { cv.projects.items[0].url = url; }));
    assert.equal(sanitizeUrl(url), 'https://example.com/a_b/**x**?q=1\\&r=2\\%20\\#top\\~1');
    assert.ok(projects.includes(`\\cvProject{FraudShield}{${sanitizeUrl(url)}}{`), projects);
    const header = readCvHeader('candidate:\n  full_name: "Jane Smith"\n  email: "jane_smith@example.com"\n  linkedin: "linkedin.com/in/jane_smith~1"\n');
    assert.equal(filled('EMAIL_URL', PERSONA, { header }), 'mailto:jane_smith@example.com');
    assert.equal(filled('LINKEDIN_URL', PERSONA, { header }), 'https://linkedin.com/in/jane_smith\\~1');
    assert.equal(filled('LINKEDIN_DISPLAY', PERSONA, { header }), 'linkedin.com/in/jane\\_smith\\~{}1');
  });

  test('a value holding $\', $& or a placeholder stays as it is: each placeholder is filled once', () => {
    const payload = withChange((cv) => { cv.experience[1].blocks[0].bullets.push('Saved $\' and $& with {{EXPERIENCE}} and {{SKILLS}}'); });
    const tex = render(payload, { header: { ...JANE, title: '{{EXPERIENCE}} $\' $&' } });
    assert.ok(tex.includes('\\cvItem{Saved \\$\' and \\$\\& with \\{\\{EXPERIENCE\\}\\} and \\{\\{SKILLS\\}\\}}\\cvStop{.}'));
    assert.ok(tex.includes('{\\large \\{\\{EXPERIENCE\\}\\} \\$\' \\$\\&}'), tex);
    assert.equal(tex.split('\\begin{document}').length, 2);
    assert.equal(tex.split('\\cvSection{Experience}').length, 2);
    assert.equal(tex.split('\\cvEntryHead{').length, 3);
  });

  test('header facts are escaped as text, with their line breaks gone', () => {
    const header = readCvHeader([
      'candidate:',
      '  full_name: "Jane_Smith & Co"',
      '  title: "R&D 100% #1 $x {y} \\\\z ~w ^v\\nlead"',
      '  location: "San Francisco,\\nCA {HQ}"',
      '  email: "jane_smith@example.com"',
      '  phone: "+1 (555) 0123 #2"',
      '',
    ].join('\n'));
    const values = Object.fromEntries(['NAME', 'FIRST_NAME', 'LAST_NAME', 'TITLE', 'LOCATION', 'EMAIL', 'PHONE'].map((key) => [key, filled(key, PERSONA, { header })]));
    assert.deepEqual(values, {
      NAME: 'Jane\\_Smith \\& Co',
      FIRST_NAME: 'Jane\\_Smith \\&',
      LAST_NAME: 'Co',
      TITLE: 'R\\&D 100\\% \\#1 \\$x \\{y\\} \\textbackslash{}z \\~{}w \\textasciicircum{}v lead',
      LOCATION: 'San Francisco, CA \\{HQ\\}',
      EMAIL: 'jane\\_smith@example.com',
      PHONE: '+1 (555) 0123 \\#2',
    });
    // A header handed over as it is, not read from a profile, is normalised as well.
    assert.equal(filled('NAME', PERSONA, { header: { name: 'Jane\nSmith\u0007\u202E' } }), 'Jane Smith');
    for (const [key, value] of Object.entries(values)) assert.deepEqual(unknownTexCommands(value), [], key);
  });

  test('a profile without title, email, phone, location, LinkedIn or GitHub leaves no bar, comma or empty macro in the header', () => {
    const headerOf = (tex) => tex.slice(tex.indexOf('\\begin{document}'), tex.indexOf('\\cvSection{'));
    const bare = headerOf(render(PERSONA, { header: NAME_ONLY }));
    assert.match(bare, /\{\\LARGE\\bfseries Jane Smith\}/);
    for (const stray of ['\\cvContact', '\\href', ',', '{}', '\\large', '|']) assert.ok(!bare.includes(stray), `${stray} in ${bare}`);
    const full = headerOf(render(PERSONA));
    assert.equal(full.split('\\cvContact{').length - 1, 5);
    const gone = { title: '{\\large ', email: 'mailto:', phone: '555', location: 'San Francisco', linkedin: 'linkedin.com', github: 'github.com' };
    for (const [key, text] of Object.entries(gone)) {
      const header = headerOf(render(PERSONA, { header: janeWithout(key) }));
      assert.ok(full.includes(text) && !header.includes(text), key);
      assert.equal(header.split('\\cvContact{').length - 1, key === 'title' ? 5 : 4, key);
    }
  });

  test('dates=year shows the years of each range, in experience and in education', () => {
    const payload = withChange((cv) => {
      cv.experience[1].dates = 'Mar 2019 \u2013 Present';
      cv.education[0].dates = 'Sep 2014 \u2013 Jan 2017';
    });
    const template = (dates) => `% career-ops-latex: dates=${dates}\n{{EXPERIENCE}}\n{{EDUCATION}}`;
    const years = render(payload, { template: template('year') });
    for (const text of ['{Austin, TX}{2020\u20132024}\\cvStop', '{Remote}{2019\u2013Present}\\cvStop', '{UT Austin}{2014\u20132017}\\cvStop',
      '{UC Berkeley}{2016}\\cvStop']) {
      assert.ok(years.includes(text), text);
    }
    const full = render(payload, { template: template('full') });
    for (const text of ['{Austin, TX}{2020-2024}\\cvStop', '{Remote}{Mar 2019 \u2013 Present}\\cvStop', '{UT Austin}{Sep 2014 \u2013 Jan 2017}\\cvStop']) {
      assert.ok(full.includes(text), text);
    }
  });
});

// ---------------------------------------------------------------------------
// What a rendered CV can run
// ---------------------------------------------------------------------------

describe('the commands a rendered CV holds', () => {
  test('a command the renderer does not write is reported, each one once', () => {
    assert.deepEqual(unknownTexCommands('\\cvItem{a}\\input{x}'), ['input']);
    assert.deepEqual(unknownTexCommands('\\input{a}\\write18{b}\\input{c}'), ['input', 'write']);
    // A line break (which takes an option), the commands the write-backs
    // replace, a link and the bold form the renderer never uses.
    assert.deepEqual(unknownTexCommands('a\\\\b \\textbar{} \\textless{} \\href{x}{y} \\textbf{z}'), ['\\', 'textbar', 'textless', 'href', 'textbf']);
    assert.deepEqual(unknownTexCommands('\\cvItem{\\& \\% \\$ \\# \\_ \\{ \\} \\~{} \\textasciicircum{} \\textbackslash{} {\\bfseries x}}\\cvStop{.}'), []);
    assert.deepEqual(unknownTexCommands(''), []);
  });

  test('every section of the persona CV runs only what the renderer writes', () => {
    const payload = structuredClone(FULL);
    payload.experience[0].blocks[0].bullets.push('[Beta] **Cut** cost by 40% & more: a_b {c} ~d ^e \\f < g > h | i \u2192 j \u00B1 k');
    for (const key of ['SUMMARY', 'EXPERIENCE', 'PROJECTS', 'EDUCATION', 'CERTIFICATIONS', 'SKILLS']) {
      const value = filled(key, payload);
      assert.ok(value.length > 0, key);
      assert.deepEqual(unknownTexCommands(value), [], key);
    }
  });
});

// ---------------------------------------------------------------------------
// The text of a .tex
// ---------------------------------------------------------------------------

describe('the text of a .tex', () => {
  test('contentMacros lists the calls that carry text, in order, from the body', () => {
    const calls = contentMacros(render(FULL));
    assert.deepEqual(calls.map((call) => /^\\(\w+)/.exec(call)[1]), [
      'cvSummary', 'cvEntryHead', 'cvEntryBlock', 'cvItem', 'cvItem', 'cvItem', 'cvItem', 'cvItem', 'cvEntryHead', 'cvEntryBlock', 'cvItem',
      'cvItem', 'cvItem', 'cvProjects', 'cvProject', 'cvProject', 'cvEducation', 'cvEducation', 'cvCert', 'cvSkill', 'cvSkill', 'cvSkill',
      'cvSkill',
    ]);
    // The items of a block or of the projects follow it as calls of their own.
    assert.equal(calls[1], '\\cvEntryHead{Senior ML Engineer / ML Platform Lead}{TechFin Corp}{Austin, TX}{2020-2024}');
    assert.equal(calls[2], '\\cvEntryBlock{ML platform}{A platform team of three.}{\u2026}');
    assert.equal(calls[9], '\\cvEntryBlock{}{}{\u2026}');
    assert.equal(calls[13], '\\cvProjects{Two open-source projects.}{\u2026}');
    assert.ok(calls[14].startsWith('\\cvProject{FraudShield}{https://github.com/alexchen/fraudshield}{(Open Source) Real-time'), calls[14]);
    assert.equal(calls.at(-1), '\\cvSkill{Languages}{Python, Go, TypeScript, SQL}');
  });

  test('a change of layout leaves the list as it is', () => {
    const tex = render(FULL);
    const layout = tex
      .replace('\\cvSection{Experience}\\cvStop{.}\n', '\\cvSection{Experience}\\cvStop{.}\n\\vspace{2pt}\n')
      .replace('\\cvItem{Led ML platform team', '\\vspace{2pt}%\n  \\cvItem{Led ML platform team')
      .replace('{Austin, TX}{2020-2024}', '{Austin,\n  TX} {2020-2024} % the dates')
      .replaceAll('\\cvStop{.}', '')
      .replaceAll('\\cvEntrySep', '\\bigskip');
    assert.notEqual(layout, tex);
    assert.deepEqual(contentMacros(layout), contentMacros(tex));
  });

  test('a change of text is seen', () => {
    const tex = render(FULL);
    const first = '\\cvItem{Implemented search ranking with learning-to-rank models}\\cvStop{.}';
    const second = '\\cvItem{Set up experiment tracking with MLflow and model versioning}\\cvStop{.}';
    assert.ok(tex.includes(first) && tex.includes(second));
    const changed = {
      'a company': tex.replace('{TechFin Corp}', '{Northwind Robotics}'),
      'a bullet': tex.replace('Implemented search ranking', 'Implemented search and ads ranking'),
      'two bullets swapped': tex.replace(first, 'FIRST').replace(second, first).replace('FIRST', second),
      'an added bullet': tex.replace(first, `\\cvItem{Invented a bullet}\\cvStop{.}\n${first}`),
      'a bullet moved to the other job': tex.replace(`${first}\n`, '').replace('\\cvItem{Led ML platform team', `${first}\n\\cvItem{Led ML platform team`),
      'an address': tex.replace('https://github.com/alexchen/fraudshield', 'https://example.com/fraudshield'),
      'a lead': tex.replace('{ML platform}', '{Ads platform}'),
      'a certification taken out': tex.replace('\\cvCert{Certified Kubernetes Administrator}\\cvStop{.}', ''),
    };
    for (const [what, text] of Object.entries(changed)) {
      assert.notEqual(text, tex, what);
      assert.notDeepEqual(contentMacros(text), contentMacros(tex), what);
    }
  });

  test('braces escaped as text are not counted, and a call in a comment, the preamble or after the end is no text', () => {
    const braces = render(withChange((cv) => { cv.summary = ['Parsed {"key": "value"} records } and {']; }));
    assert.equal(contentMacros(braces)[0], '\\cvSummary{Parsed \\{"key": "value"\\} records \\} and \\{}');
    const tex = render(FULL);
    assert.deepEqual(contentMacros(tex.replace('\\end{document}', '% \\cvItem{Invented a bullet}\n\\end{document}')), contentMacros(tex));
    assert.deepEqual(contentMacros(`${tex}\\cvItem{After the end}\n`), contentMacros(tex));
    assert.deepEqual(contentMacros(tex.replace('\\begin{document}', '\\cvSkill{Invented}{Skill}\n\\begin{document}')), contentMacros(tex));
  });
});

// ---------------------------------------------------------------------------
// The default template
// ---------------------------------------------------------------------------

describe('the default template', () => {
  const PLACEHOLDERS = [
    'NAME', 'FIRST_NAME', 'LAST_NAME', 'TITLE', 'LOCATION', 'EMAIL', 'EMAIL_URL', 'PHONE', 'LINKEDIN_URL', 'LINKEDIN_DISPLAY', 'GITHUB_URL',
    'GITHUB_DISPLAY', 'DATE', 'PAPER', 'SUMMARY', 'EXPERIENCE', 'PROJECTS', 'EDUCATION', 'CERTIFICATIONS', 'SKILLS',
  ];
  const MACROS = ['cvSummary', 'cvEntryHead', 'cvEntryBlock', 'cvItem', 'cvEntrySep', 'cvProjects', 'cvProject', 'cvEducation', 'cvCert', 'cvSkill',
    'cvStop'];
  const preamble = DEFAULT_TEMPLATE_TEXT.slice(0, DEFAULT_TEMPLATE_TEXT.indexOf('\\begin{document}'));

  test('it is a plain article for pdflatex, with T1, Latin Modern, geometry, hyperref and enumitem', () => {
    assert.ok(preamble.includes('\\documentclass[{{PAPER}},11pt]{article}'));
    assert.deepEqual(readTemplateOptions(DEFAULT_TEMPLATE_TEXT), { engine: 'pdflatex', dates: 'full' });
    for (const line of ['\\usepackage[T1]{fontenc}', '\\usepackage{lmodern}', '\\usepackage[margin=2cm]{geometry}', '\\usepackage{enumitem}',
      '\\usepackage[hidelinks]{hyperref}']) {
      assert.ok(preamble.includes(line), line);
    }
    assert.match(preamble, /\\ifdefined\\DeclareUnicodeCharacter\s+\\DeclareUnicodeCharacter\{2192\}\{\\ensuremath\{\\rightarrow\}\}\s+\\fi/);
  });

  test('under XeTeX, which xelatex and tectonic run, it loads fontspec in place of T1 and Latin Modern', () => {
    // A copy of this template for xelatex, and the real compile below where
    // tectonic is the engine there is, ran on T1 fonts at first. XeTeX has no
    // glyph past U+00FF in them, so the en dash and the arrow were left out,
    // and it looks a character up at its Unicode code, so ± (U+00B1) printed
    // as the ś that T1 keeps there, with nothing in the log.
    assert.match(preamble,
      /\\ifdefined\\XeTeXrevision\s+\\usepackage\{fontspec\}\s+\\else\s+\\usepackage\[T1\]\{fontenc\}\s+\\usepackage\{lmodern\}\s+\\fi/);
  });

  test('it holds every placeholder and every block once', () => {
    for (const key of PLACEHOLDERS) assert.equal(DEFAULT_TEMPLATE_TEXT.split(`{{${key}}}`).length - 1, 1, key);
    const lines = DEFAULT_TEMPLATE_TEXT.split('\n');
    for (const key of BLOCK_KEYS) {
      assert.equal(lines.filter((line) => line === `%%IF ${key}`).length, 1, key);
      assert.equal(lines.filter((line) => line === `%%END ${key}`).length, 1, key);
    }
  });

  test('it defines each macro the renderer writes, in its preamble, and \\cvStop prints nothing', () => {
    for (const macro of MACROS) assert.match(preamble, new RegExp(`\\\\newcommand\\{\\\\${macro}\\}`), macro);
    assert.ok(preamble.includes('\\newcommand{\\cvStop}[1]{}'));
  });

  test('it prints \\~{} as a tilde, where T1 has an accent', () => {
    // The renderer writes ~ as \~{}, which the final text check reads as a
    // character. Under T1 that is the accent: pdftotext read U+02DC for it in
    // a PDF of this template, U+007E once it was declared \textasciitilde.
    const declaration = /\\ifcsname T1\\string\\~\\endcsname\s+\\makeatletter\s+\\DeclareTextCompositeCommand\{\\~\}\{T1\}\{\\@empty\}\{\\textasciitilde\}\s+\\makeatother\s+\\fi/;
    assert.match(preamble, declaration);
    assert.ok(preamble.search(declaration) > preamble.indexOf('\\usepackage[T1]{fontenc}'));
  });
});

// ---------------------------------------------------------------------------
// The final text check
// ---------------------------------------------------------------------------

describe('the final text check on a rendered CV', () => {
  // A list of tools after "using" is where the check of generate-latex.mjs
  // misread a .tex true to cv.md: a command inside a macro argument read as a
  // tool name, and a list with no full stop after it ran on into the next
  // bullet or heading. No bullet here ends with a full stop, as cv.md writes
  // none, and one holds each character the renderer writes back.
  const BULLETS = [
    'Built the event pipeline using Kafka and Python, cutting latency by 40%',
    'Led a team of 5 engineers',
    'Routed the logs using Grafana \u2192 Loki',
    'Linked the dashboards using Grafana | Loki',
    'Kept reads using Kafka <5ms',
    'Kept writes using Kafka >2ms',
    'Checked drift using Python \u00B1 NumPy',
    'Kept settings using ~home paths',
  ];
  const CV = [
    '# CV -- Alex Chen', '', '## Work Experience', '', '### Northwind Robotics -- Austin, TX', '', '**Data Engineer**', '2020-2024', '',
    ...BULLETS.map((bullet) => `- ${bullet}`), '', '## Certifications', '', '- AWS Solutions Architect Associate', '', '## Skills', '',
    '- **Languages:** Python, Go', '',
  ].join('\n');
  const PAYLOAD = {
    experience: [{
      company: 'Northwind Robotics',
      role: 'Data Engineer',
      location: 'Austin, TX',
      dates: '2020-2024',
      blocks: [{ lead: '', intro: '', bullets: BULLETS.map((bullet, index) => (index === 0 ? bullet.replace('Kafka', '**Kafka**') : bullet)) }],
    }],
    certifications: ['AWS Solutions Architect Associate'],
    skills: [{ category: 'Languages', items: 'Python, Go' }],
  };
  const gate = (tex, root) => runLatexFactGate(tex, { root, label: 'cv.tex' });

  test('a CV true to cv.md passes, its bold word and its written-back characters with it', async () => {
    const { root } = checkRoot(CV);
    const tex = render(PAYLOAD);
    assert.ok(tex.includes('using {\\bfseries Kafka} and Python'), tex);
    const result = await gate(tex, root);
    assert.ok(['pass', 'warn'].includes(result.verdict), JSON.stringify(result, null, 2));
  });

  test('the persona CV in the default template passes', async () => {
    const result = await gate(render(PERSONA), PERSONA_ROOT.root);
    assert.ok(['pass', 'warn'].includes(result.verdict), JSON.stringify(result, null, 2));
  });

  test('written as \\textbf the bold word blocks', async () => {
    const { root } = checkRoot(CV);
    const tex = render(PAYLOAD).replace('{\\bfseries Kafka}', '\\textbf{Kafka}');
    assert.ok(tex.includes('\\textbf{Kafka}'));
    const result = await gate(tex, root);
    assert.equal(result.verdict, 'block', JSON.stringify(result, null, 2));
    assert.match(result.reason, /tool=textbf/);
  });

  test('without \\cvStop{.} the same CV blocks', async () => {
    const { root } = checkRoot(CV);
    const result = await gate(render(PAYLOAD).replaceAll('\\cvStop{.}', ''), root);
    assert.equal(result.verdict, 'block', JSON.stringify(result, null, 2));
    // "using Grafana → Loki" ran on into the next bullet, "Linked the dashboards…".
    assert.match(result.reason, /tool=loki linked/);
  });

  test('the stops the renderer writes after its calls are needed, with the template\'s after its headings in place', async () => {
    // A bullet that ends in its list of tools runs on into the next one, and
    // only the stop the renderer writes after the call ends it: the stops the
    // template writes after its headings stay here, and the CV still blocks.
    const bullets = ['Built the event pipeline using Kafka and Python', 'Led a team of 5 engineers'];
    const { root } = checkRoot([
      '# CV -- Alex Chen', '', '## Work Experience', '', '### Northwind Robotics -- Austin, TX', '', '**Data Engineer**', '2020-2024', '',
      ...bullets.map((bullet) => `- ${bullet}`), '', '## Skills', '', '- **Languages:** Python, Go', '',
    ].join('\n'));
    const tex = render({
      experience: [{ company: 'Northwind Robotics', role: 'Data Engineer', location: 'Austin, TX', dates: '2020-2024', blocks: [{ bullets }] }],
      skills: [{ category: 'Languages', items: 'Python, Go' }],
    });
    const rendered = await gate(tex, root);
    assert.ok(['pass', 'warn'].includes(rendered.verdict), JSON.stringify(rendered, null, 2));
    const headingStopsOnly = tex.split('\n').map((line) => (line.startsWith('\\cvSection{') ? line : line.replaceAll('\\cvStop{.}', ''))).join('\n');
    assert.deepEqual(headingStopsOnly.match(/^.*\\cvStop\{\.\}.*$/gm), ['\\cvSection{Experience}\\cvStop{.}', '\\cvSection{Skills}\\cvStop{.}']);
    const result = await gate(headingStopsOnly, root);
    assert.equal(result.verdict, 'block', JSON.stringify(result, null, 2));
    assert.match(result.reason, /tool=python led/);
  });
});

// ---------------------------------------------------------------------------
// The page count of a compile
// ---------------------------------------------------------------------------

describe('the page count in a log', () => {
  test('the line an engine writes last gives it, as pdflatex, xelatex and tectonic write it', () => {
    assert.equal(pageCountFromLog('This is pdfTeX\nOutput written on cv.pdf (2 pages, 14137 bytes).\nPDF statistics:\n'), 2);
    assert.equal(pageCountFromLog('This is XeTeX\r\nOutput written on cv.pdf (1 page).\r\n'), 1);
    assert.equal(pageCountFromLog('Output written on cv.xdv (3 pages, 9184 bytes).\n'), 3);
  });

  test('a line TeX broke at 79 columns is read whole, in bytes as pdfTeX counts or in characters as XeTeX does', () => {
    // Broken between the two digits of 12: read line by line, the CV would
    // have one page, and pass any limit.
    const first = `Output written on ${'a'.repeat(51)}/cv.pdf (1`;
    assert.equal(first.length, 79);
    assert.equal(pageCountFromLog(`This is pdfTeX\n${first}\n2 pages, 14137 bytes).\n`), 12);
    // 79 bytes in 71 characters: pdfTeX breaks a line with an é in it sooner
    // than its length in characters shows.
    const accented = `Output written on ${'caf\u00e9/'.repeat(8)}aaaacv.pdf (1`;
    assert.deepEqual([Buffer.byteLength(accented), accented.length], [79, 71]);
    assert.equal(pageCountFromLog(`${accented}\n2 pages, 14137 bytes).\n`), 12);
    // 79 characters in 89 bytes: XeTeX counts characters.
    const counted = `Output written on ${'caf\u00e9/'.repeat(10)}aacv.pdf (1`;
    assert.deepEqual([counted.length, Buffer.byteLength(counted)], [79, 89]);
    assert.equal(pageCountFromLog(`${counted}\n2 pages).\n`), 12);
  });

  test('an earlier line that reads like it, from the text of the CV, is not the one read', () => {
    // TeX copies the text of an overfull line into the log. The engine writes
    // its own line after all the text, so the last one is read.
    const log = [
      'Overfull \\hbox (12.0pt too wide) in paragraph at lines 80--80',
      '[]\\T1/lmr/m/n/10.95 Output written on cv.pdf (1 page, 10 bytes).',
      '',
      'Output written on cv.pdf (3 pages, 90210 bytes).',
      'PDF statistics:',
      '',
    ].join('\n');
    assert.equal(pageCountFromLog(log), 3);
  });

  test('without that line the .aux gives it, and without either there is none', () => {
    assert.equal(pageCountFromLog('This is XeTeX\n', '\\relax\n\\gdef \\@abspage@last{2}\n'), 2);
    assert.equal(pageCountFromLog('Output written on cv.pdf (1 page).\n', '\\gdef \\@abspage@last{2}\n'), 1);
    assert.equal(pageCountFromLog('This is XeTeX\n', '\\relax\n'), null);
    assert.equal(pageCountFromLog(''), null);
    assert.equal(pageCountFromLog(undefined, undefined), null);
  });
});

// ---------------------------------------------------------------------------
// Compiling
// ---------------------------------------------------------------------------

/**
 * The files and folders under `dir`, as paths with forward slashes, sorted.
 * Through walkTree() of lib/walk-tree.mjs, the one recursion the repository
 * allows (tests/walk-tree.test.mjs bans `readdirSync` with `recursive`).
 */
function listing(dir) {
  const found = [];
  const add = (abs) => found.push(relative(dir, abs).split(sep).join('/'));
  walkTree(dir, { onFile: add, onDir: add });
  return found.sort();
}

/** A folder of the sandbox to make the build folders of one test in. Its name holds a space, as many a temporary folder's does. */
const buildRoot = () => mkdtempSync(join(SANDBOX, 'build root-'));

/** A pack folder holding `files`, each name a path in it. */
function packFolder(files) {
  const dir = mkdtempSync(join(SANDBOX, 'pack-'));
  for (const [name, text] of Object.entries(files)) {
    mkdirSync(dirname(join(dir, name)), { recursive: true });
    writeFileSync(join(dir, name), text);
  }
  return dir;
}

/** The outcome of a compile, as the runner reads it first. */
const outcome = (result) => [result.ok, result.code];

const STAND_IN_PDF = '%PDF-1.5 stand-in';
const ONE_PAGE_LOG = 'This is pdfTeX, Version 3.141592653-2.6-1.40.26 (TeX Live 2025)\nOutput written on cv.pdf (1 page, 2048 bytes).\n';
const PASS_ARGS = ['-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error', 'cv.tex'];

/** Engines nothing is run for, each with the reason a compile, and the question asked before one, give. */
const ENGINE_REFUSALS = [
  ['lualatex', 'lualatex is not supported: it runs Lua code from the document. Use xelatex.'],
  ['latexmk', '"latexmk" is not an engine a CV compiles with: a template runs on xelatex, pdflatex or tectonic.'],
  [undefined, '"undefined" is not an engine a CV compiles with: a template runs on xelatex, pdflatex or tectonic.'],
  [null, '"null" is not an engine a CV compiles with: a template runs on xelatex, pdflatex or tectonic.'],
  // String() of an object without a prototype throws, so a value that is no
  // string is named by its type.
  [Object.create(null), '"object" is not an engine a CV compiles with: a template runs on xelatex, pdflatex or tectonic.'],
];

/**
 * A stand-in for the engine, given to compileLatex() as its `run`. It records
 * each call with what the build folder held at that moment, answers --version
 * with `version` (null: not installed), and on each pass writes `files` into
 * the build folder and returns `pass(n)` over a status of 0, n counting the
 * passes from 1.
 */
function fakeEngine({
  version = 'pdfTeX 3.141592653-2.6-1.40.26 (TeX Live 2025)',
  files = { 'cv.pdf': STAND_IN_PDF, 'cv.log': ONE_PAGE_LOG },
  pass = () => ({}),
} = {}) {
  const calls = [];
  let passes = 0;
  const run = (bin, args, { cwd, timeout }) => {
    const source = join(cwd, 'cv.tex');
    calls.push({ bin, args: [...args], cwd, timeout, held: listing(cwd), tex: existsSync(source) ? readFileSync(source, 'utf8') : null });
    if (args[0] === '--version') {
      return version === null
        ? { status: null, stdout: '', stderr: '', error: Object.assign(new Error(`spawnSync ${bin} ENOENT`), { code: 'ENOENT' }) }
        : { status: 0, stdout: `${version}\n`, stderr: '', error: undefined };
    }
    passes += 1;
    for (const [name, text] of Object.entries(files)) writeFileSync(join(cwd, name), text);
    return { status: 0, stdout: '', stderr: '', error: undefined, ...pass(passes) };
  };
  return { run, calls };
}

describe('compiling a CV', () => {
  const TEX = render(PERSONA);

  test('pdflatex and xelatex run twice in a build folder of their own, shell escape off, on the .tex as given', async () => {
    for (const engine of ['pdflatex', 'xelatex']) {
      const tmpRoot = buildRoot();
      const { run, calls } = fakeEngine();
      const result = await compileLatex({ tex: TEX, engine, run, tmpRoot, timeoutMs: 30000 });
      assert.equal(result.ok, true, result.reason);
      assert.deepEqual(calls.map(({ bin, args }) => [bin, args]), [[engine, ['--version']], [engine, PASS_ARGS], [engine, PASS_ARGS]]);
      const [version, first, second] = calls;
      assert.deepEqual([dirname(first.cwd), version.cwd, second.cwd], [tmpRoot, first.cwd, first.cwd]);
      assert.match(basename(first.cwd), /^career-ops-latex-[A-Za-z0-9]{6}$/);
      assert.deepEqual(calls.map(({ timeout }) => timeout), [30000, 30000, 30000]);
      // Without a pack the folder holds the .tex alone when the first pass
      // starts, written as it was given.
      assert.deepEqual(first.held, ['cv.tex']);
      assert.equal(first.tex, TEX);
      assert.ok(Buffer.isBuffer(result.pdf));
      assert.deepEqual([result.pdf.toString(), result.pages, result.missingGlyphs, result.engine, result.leftover],
        [STAND_IN_PDF, 1, [], engine, undefined]);
    }
    // Two minutes for each call, unless told otherwise.
    const { run, calls } = fakeEngine();
    await compileLatex({ tex: TEX, engine: 'pdflatex', run, tmpRoot: buildRoot() });
    assert.deepEqual(calls.map(({ timeout }) => timeout), [120000, 120000, 120000]);
  });

  test('the MiKTeX package installer is turned off, and only for MiKTeX', async () => {
    const installerOff = ['-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error', '-disable-installer', 'cv.tex'];
    for (const [engine, version, args] of [
      ['pdflatex', 'MiKTeX-pdfTeX 4.18 (MiKTeX 24.1)', installerOff],
      ['xelatex', 'MiKTeX-XeTeX 4.10 (MiKTeX 24.1)', installerOff],
      ['xelatex', 'XeTeX 3.141592653-2.6-0.999996 (TeX Live 2025)', PASS_ARGS],
    ]) {
      const { run, calls } = fakeEngine({ version });
      const result = await compileLatex({ tex: TEX, engine, run, tmpRoot: buildRoot() });
      assert.equal(result.ok, true, result.reason);
      assert.deepEqual(calls.slice(1).map(({ args: called }) => called), [args, args], version);
    }
  });

  test('tectonic runs once, untrusted, keeping its log and its .aux for the page count', async () => {
    const log = 'Output written on cv.xdv (2 pages, 9184 bytes).\n';
    const { run, calls } = fakeEngine({ version: 'Tectonic 0.15.0', files: { 'cv.pdf': STAND_IN_PDF, 'cv.log': log } });
    const result = await compileLatex({ tex: TEX, engine: 'tectonic', run, tmpRoot: buildRoot() });
    assert.equal(result.ok, true, result.reason);
    // The one install-latex-engine.mjs puts beside the scripts, else the one on PATH.
    const bin = existsSync(localTectonicPath()) ? localTectonicPath() : 'tectonic';
    assert.deepEqual(calls.map(({ bin: called, args }) => [called, args]), [
      [bin, ['--version']],
      [bin, ['--untrusted', '--keep-logs', '--keep-intermediates', '--outdir', '.', 'cv.tex']],
    ]);
    assert.deepEqual([result.pages, result.engine], [2, 'tectonic']);
  });

  test('the pack is copied into the build folder, without a PDF, a log or an .aux of its own', async () => {
    const packDir = packFolder({
      'cv-latex.tex': PLAIN_PACK,
      'plain.cls': '% the class of the pack',
      'fonts/Plain-Regular.ttf': 'a font',
      // Left by a compile run by hand in the pack folder. Read after the
      // engine has run, they would be taken for what it wrote.
      'cv.pdf': '%PDF-1.5 an earlier CV',
      'cv.log': 'Output written on cv.pdf (9 pages).\n',
      'cv.aux': '\\gdef \\@abspage@last{9}\n',
    });
    const { run, calls } = fakeEngine();
    const result = await compileLatex({ tex: TEX, packDir, engine: 'xelatex', run, tmpRoot: buildRoot() });
    assert.equal(result.ok, true, result.reason);
    assert.deepEqual(calls[1].held, ['cv-latex.tex', 'cv.tex', 'fonts', 'fonts/Plain-Regular.ttf', 'plain.cls']);
    assert.equal(calls[1].tex, TEX);
    assert.deepEqual([result.pdf.toString(), result.pages], [STAND_IN_PDF, 1]);
    // The pack itself is left as it was.
    assert.deepEqual(listing(packDir), ['cv-latex.tex', 'cv.aux', 'cv.log', 'cv.pdf', 'fonts', 'fonts/Plain-Regular.ttf', 'plain.cls']);
    // An engine that writes no PDF, or no page count, is not answered by the pack's.
    const silent = fakeEngine({ files: {} });
    assert.deepEqual(outcome(await compileLatex({ tex: TEX, packDir, engine: 'xelatex', run: silent.run, tmpRoot: buildRoot() })),
      [false, 'COMPILE']);
    const pdfOnly = fakeEngine({ files: { 'cv.pdf': STAND_IN_PDF } });
    assert.deepEqual(outcome(await compileLatex({ tex: TEX, packDir, engine: 'xelatex', run: pdfOnly.run, tmpRoot: buildRoot() })),
      [false, 'PAGES']);
  });

  test('a symbolic link in the pack is refused before anything runs, and so is a pack that is one', async (t) => {
    // A link would let the engine read or write outside the build folder
    // through a name inside it. Windows lets an account without the right to
    // make symbolic links make a junction, which Node reports as one.
    const outside = packFolder({ 'notes.txt': 'not part of any CV' });
    const makeLink = (target, path, type) => {
      try {
        symlinkSync(target, path, type);
        return true;
      } catch {
        return false;
      }
    };
    const packDir = packFolder({ 'cv-latex.tex': PLAIN_PACK });
    let entry = null;
    if (makeLink(join(outside, 'notes.txt'), join(packDir, 'notes.txt'), 'file')) entry = 'notes.txt';
    else if (makeLink(outside, join(packDir, 'shared'), 'junction')) entry = 'shared';
    if (entry === null) {
      t.skip('this account can make neither a symbolic link nor a junction');
      return;
    }
    const tmpRoot = buildRoot();
    const { run, calls } = fakeEngine();
    const result = await compileLatex({ tex: TEX, packDir, engine: 'xelatex', run, tmpRoot });
    assert.deepEqual(outcome(result), [false, 'COMPILE']);
    assert.equal(result.reason, `The template pack holds a symbolic link, ${entry}, which is not copied to the build folder. `
      + 'Put the file or folder itself in the pack.');
    assert.deepEqual(calls, []);
    assert.deepEqual(readdirSync(tmpRoot), []);

    const linkedPack = join(mkdtempSync(join(SANDBOX, 'links-')), 'plain');
    if (makeLink(packFolder({ 'cv-latex.tex': PLAIN_PACK }), linkedPack, 'junction')) {
      const linked = await compileLatex({ tex: TEX, packDir: linkedPack, engine: 'xelatex', run, tmpRoot });
      assert.deepEqual(outcome(linked), [false, 'COMPILE']);
      assert.equal(linked.reason, 'The template pack is a symbolic link, which is not copied to the build folder. Put the folder itself in its place.');
      assert.deepEqual(calls, []);
    }
  });

  test('an engine that is not installed is named, with the line that chooses it, and nothing else runs', async () => {
    for (const engine of ['pdflatex', 'xelatex', 'tectonic']) {
      const { run, calls } = fakeEngine({ version: null });
      const result = await compileLatex({ tex: TEX, engine, run, tmpRoot: buildRoot() });
      assert.deepEqual(outcome(result), [false, 'NO_ENGINE']);
      assert.equal(result.reason, `The template's engine, ${engine}, is not installed: ${engine} --version did not run. A template names its `
        + `engine on the line %!TEX TS-program = ${engine} (pdflatex when it has none).`
        + `${engine === 'tectonic' ? ' Install it with: node install-latex-engine.mjs' : ''}`);
      assert.equal(calls.length, 1);
    }
    // An engine that answers --version with an error is not one either.
    const broken = () => ({ status: 1, stdout: '', stderr: 'xelatex: the format file could not be loaded\n', error: undefined });
    assert.deepEqual(outcome(await compileLatex({ tex: TEX, engine: 'xelatex', run: broken, tmpRoot: buildRoot() })), [false, 'NO_ENGINE']);
  });

  test('a program in the build folder is never run for the engine', {
    skip: process.platform !== 'win32' && 'only Windows looks for a program in the folder it is to run in',
  }, async () => {
    // spawnSync() on Windows looks for a program named without a folder in
    // the folder it is to run in before PATH, unless the environment holds
    // NoDefaultCurrentDirectoryInExePath, which a shell does not set. Here
    // that folder is the build folder, where a pack, or the first pass of
    // the engine, could put a pdflatex.exe of its own. cmd.exe, so named,
    // answers --version and each pass with exit code 0.
    const packDir = packFolder({ 'cv-latex.tex': PLAIN_PACK });
    copyFileSync(join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'cmd.exe'), join(packDir, 'pdflatex.exe'));
    const noPrograms = mkdtempSync(join(SANDBOX, 'no programs-'));
    const saved = { path: process.env.PATH, guard: process.env.NoDefaultCurrentDirectoryInExePath };
    delete process.env.NoDefaultCurrentDirectoryInExePath;
    process.env.PATH = noPrograms;
    try {
      const result = await compileLatex({ tex: TEX, packDir, engine: 'pdflatex', tmpRoot: buildRoot() });
      assert.deepEqual(outcome(result), [false, 'NO_ENGINE'], result.reason);
    } finally {
      if (saved.path === undefined) delete process.env.PATH;
      else process.env.PATH = saved.path;
      if (saved.guard !== undefined) process.env.NoDefaultCurrentDirectoryInExePath = saved.guard;
    }
  });

  test('a pass that fails gives the error lines of its log, and a character the engine cannot print a hint', async () => {
    const unicode = [
      'This is pdfTeX, Version 3.141592653-2.6-1.40.25 (MiKTeX 24.1)',
      '! LaTeX Error: Unicode character \u2265 (U+2265)',
      '               not set up for use with LaTeX.',
      '',
      'See the LaTeX manual or LaTeX Companion for explanation.',
      'Type  H <return>  for immediate help.',
      ' ...',
      '',
      'l.97 \\cvItem{Cut latency to \u2265',
      '                         5ms}\\cvStop{.}',
      '!  ==> Fatal error occurred, no output PDF file produced!',
      '',
    ].join('\n');
    const first = fakeEngine({ files: { 'cv.log': unicode }, pass: () => ({ status: 1 }) });
    const result = await compileLatex({ tex: TEX, engine: 'pdflatex', run: first.run, tmpRoot: buildRoot() });
    assert.deepEqual(outcome(result), [false, 'COMPILE']);
    assert.equal(result.reason, 'pdflatex stopped on pass 1 of 2 (exit code 1): ! LaTeX Error: Unicode character \u2265 (U+2265) '
      + 'l.97 \\cvItem{Cut latency to \u2265 The template\'s engine cannot print that character: replace it in the .json, '
      + 'or use a template for xelatex.');
    assert.equal(first.calls.length, 2);

    const undefinedCommand = ['! Undefined control sequence.', 'l.3 \\nosuchcommand', '', '!  ==> Fatal error occurred, no output PDF file produced!'];
    const second = fakeEngine({ files: { 'cv.log': undefinedCommand.join('\n') }, pass: (n) => ({ status: n === 2 ? 1 : 0 }) });
    const failed = await compileLatex({ tex: TEX, engine: 'xelatex', run: second.run, tmpRoot: buildRoot() });
    assert.deepEqual([...outcome(failed), failed.reason],
      [false, 'COMPILE', 'xelatex stopped on pass 2 of 2 (exit code 1): ! Undefined control sequence. l.3 \\nosuchcommand']);

    // No log: what the engine printed. tectonic's error lines begin with error:.
    const stderr = 'note: Running TeX ...\nerror: cv.tex:12: Undefined control sequence\nerror: halted on potentially-recoverable error as specified\n';
    const tectonic = fakeEngine({ version: 'Tectonic 0.15.0', files: {}, pass: () => ({ status: 1, stderr }) });
    const stopped = await compileLatex({ tex: TEX, engine: 'tectonic', run: tectonic.run, tmpRoot: buildRoot() });
    assert.deepEqual([...outcome(stopped), stopped.reason], [false, 'COMPILE', 'tectonic stopped (exit code 1): error: cv.tex:12: Undefined '
      + 'control sequence error: halted on potentially-recoverable error as specified']);

    const mute = fakeEngine({ files: {}, pass: () => ({ status: 3 }) });
    assert.equal((await compileLatex({ tex: TEX, engine: 'pdflatex', run: mute.run, tmpRoot: buildRoot() })).reason,
      'pdflatex stopped on pass 1 of 2 (exit code 3), and gave no error line.');
  });

  test('a call that runs out of time is stopped', async () => {
    const timedOut = () => ({ status: null, error: { code: 'ETIMEDOUT' } });
    const pdflatex = fakeEngine({ pass: timedOut });
    const result = await compileLatex({ tex: TEX, engine: 'pdflatex', run: pdflatex.run, tmpRoot: buildRoot(), timeoutMs: 30000 });
    assert.deepEqual([...outcome(result), result.reason], [false, 'TIMEOUT', 'pdflatex did not finish pass 1 of 2 within 30 s, and was stopped.']);
    const tectonic = fakeEngine({ version: 'Tectonic 0.15.0', pass: timedOut });
    assert.equal((await compileLatex({ tex: TEX, engine: 'tectonic', run: tectonic.run, tmpRoot: buildRoot(), timeoutMs: 30000 })).reason,
      'tectonic did not finish within 30 s, and was stopped.');
    const hung = () => ({ status: null, stdout: '', stderr: '', error: { code: 'ETIMEDOUT' } });
    assert.deepEqual(outcome(await compileLatex({ tex: TEX, engine: 'xelatex', run: hung, tmpRoot: buildRoot() })), [false, 'TIMEOUT']);
  });

  test('a PDF whose page count neither the log nor the .aux gives is refused', async () => {
    const { run } = fakeEngine({ files: { 'cv.pdf': STAND_IN_PDF, 'cv.log': 'This is pdfTeX\n' } });
    const result = await compileLatex({ tex: TEX, engine: 'pdflatex', run, tmpRoot: buildRoot() });
    assert.deepEqual([...outcome(result), result.reason],
      [false, 'PAGES', 'pdflatex wrote a PDF, but neither cv.log nor cv.aux gives its number of pages.']);
    const aux = fakeEngine({ files: { 'cv.pdf': STAND_IN_PDF, 'cv.log': 'This is pdfTeX\n', 'cv.aux': '\\gdef \\@abspage@last{2}\n' } });
    const counted = await compileLatex({ tex: TEX, engine: 'pdflatex', run: aux.run, tmpRoot: buildRoot() });
    assert.deepEqual([counted.ok, counted.pages], [true, 2]);
  });

  test('a compile that writes no PDF fails, and says so when the CV has no pages', async () => {
    const empty = fakeEngine({ files: { 'cv.log': 'No pages of output.\nTranscript written on cv.log.\n' } });
    const result = await compileLatex({ tex: TEX, engine: 'pdflatex', run: empty.run, tmpRoot: buildRoot() });
    assert.deepEqual([...outcome(result), result.reason], [false, 'COMPILE', 'pdflatex finished without writing a PDF: the document has no pages.']);
    const nothing = fakeEngine({ files: {} });
    assert.equal((await compileLatex({ tex: TEX, engine: 'xelatex', run: nothing.run, tmpRoot: buildRoot() })).reason,
      'xelatex finished without writing a PDF.');
  });

  test('the characters the font has no glyph for are named, each once', async () => {
    // The engine leaves them out of the PDF and still exits 0: the log is the
    // only place that says so. Newer engines write the code after the
    // character; one that prints as nothing is named by its code.
    const log = [
      'Missing character: There is no \u2013 in font ec-lmr10!',
      'Missing character: There is no \u2013 in font ec-lmr10!',
      'Missing character: There is no \u2192 (U+2192) in font [lmroman10-regular]:mapping=tex-text;!',
      'Missing character: There is no ^^U in font ec-lmr10!',
      'Missing character: There is no \u200B (U+200B) in font ec-lmr10!',
      'Output written on cv.pdf (1 page).',
    ].join('\n');
    const { run } = fakeEngine({ files: { 'cv.pdf': STAND_IN_PDF, 'cv.log': log } });
    const result = await compileLatex({ tex: TEX, engine: 'xelatex', run, tmpRoot: buildRoot() });
    assert.equal(result.ok, true, result.reason);
    assert.deepEqual(result.missingGlyphs, ['\u2013', '\u2192', '^^U', 'U+200B']);
  });

  test('each compile has a build folder of its own, removed when it ends, whatever the outcome', async () => {
    const tmpRoot = buildRoot();
    const passed = fakeEngine();
    const failed = fakeEngine({ pass: () => ({ status: 1 }) });
    assert.equal((await compileLatex({ tex: TEX, engine: 'pdflatex', run: passed.run, tmpRoot })).ok, true);
    assert.equal((await compileLatex({ tex: TEX, engine: 'pdflatex', run: failed.run, tmpRoot })).ok, false);
    const folders = [passed.calls[0].cwd, failed.calls[0].cwd];
    assert.notEqual(folders[0], folders[1]);
    assert.deepEqual(folders.map((folder) => existsSync(folder)), [false, false]);
    assert.deepEqual(readdirSync(tmpRoot), []);
  });

  test('a build folder that could not be removed is named beside the result, since it holds a copy of the CV', async () => {
    const tmpRoot = buildRoot();
    const kept = fakeEngine();
    const result = await compileLatex({ tex: TEX, engine: 'pdflatex', run: kept.run, tmpRoot, remove: () => {} });
    assert.deepEqual([result.ok, result.leftover], [true, kept.calls[0].cwd]);
    assert.equal(readFileSync(join(result.leftover, 'cv.tex'), 'utf8'), TEX);
    const busy = () => {
      throw Object.assign(new Error('resource busy or locked'), { code: 'EBUSY' });
    };
    const missing = fakeEngine({ version: null });
    const refused = await compileLatex({ tex: TEX, engine: 'pdflatex', run: missing.run, tmpRoot, remove: busy });
    assert.deepEqual([refused.code, refused.leftover], ['NO_ENGINE', missing.calls[0].cwd]);
  });

  test('whatever goes wrong is returned, never thrown', async () => {
    const nowhere = await compileLatex({ tex: TEX, engine: 'pdflatex', run: fakeEngine().run, tmpRoot: join(SANDBOX, 'no such folder', 'below') });
    assert.deepEqual([...outcome(nowhere), nowhere.reason], [false, 'COMPILE', 'No build folder could be made for the compile (ENOENT).']);
    const tmpRoot = buildRoot();
    const crashing = () => {
      throw new Error('the engine could not start');
    };
    const crashed = await compileLatex({ tex: TEX, engine: 'pdflatex', run: crashing, tmpRoot });
    assert.deepEqual([...outcome(crashed), crashed.reason], [false, 'COMPILE', 'The compile stopped (the engine could not start).']);
    // A thrown value that is no error: String() of an object without a
    // prototype would throw in its turn.
    const bare = () => {
      throw Object.create(null);
    };
    const strange = await compileLatex({ tex: TEX, engine: 'pdflatex', run: bare, tmpRoot });
    assert.deepEqual([...outcome(strange), strange.reason], [false, 'COMPILE', 'The compile stopped (no reason given).']);
    assert.deepEqual(readdirSync(tmpRoot), []);
    for (const [engine, reason] of ENGINE_REFUSALS) {
      const { run, calls } = fakeEngine();
      assert.deepEqual(await compileLatex({ tex: TEX, engine, run, tmpRoot }), { ok: false, code: 'COMPILE', reason });
      assert.deepEqual(calls, []);
    }
    assert.deepEqual(await compileLatex({ tex: null, engine: 'pdflatex', run: fakeEngine().run, tmpRoot }),
      { ok: false, code: 'COMPILE', reason: 'There is no LaTeX source to compile.' });
    // No input, or null for it, names no engine.
    for (const input of [undefined, null]) {
      assert.deepEqual(await compileLatex(input), { ok: false, code: 'COMPILE', reason: '"undefined" is not an engine a CV compiles with: '
        + 'a template runs on xelatex, pdflatex or tectonic.' });
    }
    assert.deepEqual(readdirSync(tmpRoot), []);
  });
});

// ---------------------------------------------------------------------------
// Asking for the engine before a compile
// ---------------------------------------------------------------------------

describe('asking for the engine before a compile', () => {
  // The runner asks before it sends the model anything, so that a CV is not
  // tailored for minutes and then refused for an engine that is not there.
  // It asks what compileLatex() asks, of the program compileLatex() runs, so
  // a program in some folder never answers for the engine, and it makes no
  // folder to ask in.
  const TEX = render(PERSONA);
  const ANSWER = { status: 0, stdout: 'pdfTeX 3.141592653-2.6-1.40.26 (TeX Live 2025)\n', stderr: '', error: undefined };

  /** A `run` that gives `answer` to every call, or what `answer()` returns, and records each call. */
  function answering(answer) {
    const calls = [];
    const run = (bin, args, options) => {
      calls.push({ bin, args: [...args], ...options });
      return typeof answer === 'function' ? answer() : answer;
    };
    return { run, calls };
  }

  /** The program compileLatex() runs for `engine`. */
  async function programOf(engine) {
    const { run, calls } = fakeEngine({ version: null });
    await compileLatex({ tex: TEX, engine, run, tmpRoot: buildRoot() });
    return calls[0].bin;
  }

  test('an engine that answers --version is no problem; the question goes to the program a compile runs', async () => {
    for (const engine of LATEX_ENGINES) {
      const { run, calls } = answering(ANSWER);
      assert.equal(await latexEngineProblem(engine, { run }), null, engine);
      assert.deepEqual(calls, [{ bin: await programOf(engine), args: ['--version'], cwd: tmpdir(), timeout: 120000 }], engine);
    }
    const { run, calls } = answering(ANSWER);
    assert.equal(await latexEngineProblem('xelatex', { run, timeoutMs: 30000 }), null);
    assert.equal(calls[0].timeout, 30000);
  });

  test('an engine that does not answer is the failure a compile would give for it', async () => {
    for (const engine of LATEX_ENGINES) {
      const missing = fakeEngine({ version: null });
      const { reason } = await compileLatex({ tex: TEX, engine, run: missing.run, tmpRoot: buildRoot() });
      const notFound = answering({
        status: null, stdout: '', stderr: '', error: Object.assign(new Error(`spawnSync ${engine} ENOENT`), { code: 'ENOENT' }),
      });
      assert.deepEqual(await latexEngineProblem(engine, { run: notFound.run }), { ok: false, code: 'NO_ENGINE', reason });
    }
    // An engine that answers --version with an error is not one either.
    const broken = answering({ status: 1, stdout: '', stderr: 'xelatex: the format file could not be loaded\n', error: undefined });
    assert.equal((await latexEngineProblem('xelatex', { run: broken.run }))?.code, 'NO_ENGINE');
    const hung = answering({ status: null, stdout: '', stderr: '', error: { code: 'ETIMEDOUT' } });
    assert.deepEqual(await latexEngineProblem('xelatex', { run: hung.run, timeoutMs: 30000 }),
      { ok: false, code: 'TIMEOUT', reason: 'xelatex --version did not answer within 30 s.' });
  });

  test('an engine no compile runs on is refused before anything runs, and nothing is thrown', async () => {
    for (const [engine, reason] of ENGINE_REFUSALS) {
      const { run, calls } = answering(ANSWER);
      assert.deepEqual(await latexEngineProblem(engine, { run }), { ok: false, code: 'COMPILE', reason });
      assert.deepEqual(calls, []);
      for (const options of [undefined, null]) {
        assert.deepEqual(await latexEngineProblem(engine, options), { ok: false, code: 'COMPILE', reason });
      }
    }
    // A run that throws asked nothing: that is no sign the engine is missing,
    // and installing it would not help.
    const crashing = answering(() => {
      throw new Error('the engine could not start');
    });
    assert.deepEqual(await latexEngineProblem('pdflatex', { run: crashing.run }),
      { ok: false, code: 'COMPILE', reason: 'The engine check stopped (the engine could not start).' });
    const bare = answering(() => {
      throw Object.create(null);
    });
    assert.deepEqual(await latexEngineProblem('pdflatex', { run: bare.run }),
      { ok: false, code: 'COMPILE', reason: 'The engine check stopped (no reason given).' });
  });
});

// ---------------------------------------------------------------------------
// A real compile
// ---------------------------------------------------------------------------

describe('a real compile', () => {
  // The engine generate-latex.mjs would compile with: a tectonic, else pdflatex.
  const engine = resolveLatexEngine();
  // Each character the escaping touches, a [ that opens the bullet, a LaTeX
  // command as text, and two characters past ASCII.
  const SPECIAL = '[Beta] Cut costs by 30% & $2M for team #1, a_b {c} ~d ^e \\f \\begin{document} at caf\u00e9 \u2013 done';

  test('the default template compiles the persona CV, every special character with it', {
    skip: !engine && !LATEX_REQUIRED && 'no TeX engine installed',
  }, async () => {
    // Where the run requires an engine and there is none, say that, and not
    // what a compile with no engine named says.
    assert.ok(engine, 'no TeX engine was found, and CAREER_OPS_REQUIRE_LATEX=1 requires one');
    const tmpRoot = buildRoot();
    const tex = render(withChange((cv) => { cv.experience[0].blocks[0].bullets.push(SPECIAL); }));
    assert.ok(tex.includes('\\cvItem{{[}Beta] Cut costs by 30\\% \\& \\$2M'), tex);
    const result = await compileLatex({ tex, engine, tmpRoot });
    assert.equal(result.ok, true, result.reason);
    assert.equal(result.pdf.subarray(0, 4).toString('latin1'), '%PDF');
    assert.ok(result.pages >= 1 && result.pages <= 2, `${result.pages} pages`);
    // Every character is printed, by pdflatex and by tectonic alike: the
    // template takes T1 fonts under pdflatex and fontspec under XeTeX.
    assert.deepEqual(result.missingGlyphs, []);
    assert.equal(result.leftover, undefined);
    assert.deepEqual(readdirSync(tmpRoot), []);
  });
});

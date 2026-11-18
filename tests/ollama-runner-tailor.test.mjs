// tests/ollama-runner-tailor.test.mjs — what the local Ollama runner does to
// turn an evaluated posting into a CV: `tailor` and `pdf`.
//
// tests/ollama-runner.test.mjs and tests/ollama-runner-pipeline.test.mjs cover
// the runner up to the saved report. This
// suite starts where they stop, and works the same way: the runner is
// started as a child process, `node ollama-runner.mjs …`, against a stub that
// answers like Ollama, and what is checked is what the run leaves behind. That
// is the requests that reached the server (and the ones that must not), the
// files under output/ and the exit code. tests/ollama-runner-steps.test.mjs
// goes on from here: `apply`, `run` and the --tailor / --pdf steps of
// `pipeline`. Both use tests/helpers/ollama-runner-harness.mjs.
//
// A test that is only about what one request holds, or about what tailorCv()
// returns, saves and says, asks a stand-in client in this process instead
// (tailorHere() and tailorOnce() below). A child process costs a third to
// half a second of one core, and with 73 of them this suite needed 26 to 30
// seconds there, where test-all.mjs gives a suite it runs by itself 30.
// `tailor` and `pdf` keep a test on the command line for each of their
// options, for the exit code, for what is printed and for the PDF step.
//
// The failures that matter here are the quiet ones, and each would end up in
// front of an employer: a CV saved with the model's working notes at the top,
// with a slot of the template still in it, with a section the user locked
// rewritten, with a skill the candidate does not have, or with markup that
// pulls a local file into the PDF; a PDF reported as written when the renderer
// kept only a draft.
//
// No Ollama is needed and none is contacted. No browser either: the PDF step
// starts the real generate-pdf.mjs, with a stand-in for Playwright that hands
// back a few bytes of PDF, so everything generate-pdf.mjs checks before it
// renders is exercised and Chromium is never launched.
//
// Run:  node --test tests/ollama-runner-tailor.test.mjs
import { describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import {
  NO_TOOLS_NOTE, checkReading, pruneRejectedEntries, refusedMarkup, tailorCv, tailoredCvProblem, withoutUnsupportedLinks,
} from '../ollama-runner.mjs';
import { readOllamaConfig } from '../lib/ollama-config.mjs';
import {
  ROOT, TODAY, SANDBOX, JD, tailoredCv, CV_HTML, CSP_META, asSaved, SAVED_CV, makeRoot, writeReport, listing, read,
  asksFor, replies, withStub, cli, assertExit, makeTailoredRoot, pdfIndex, trackerRow, PAGES, evaluationFor,
  makePipelineRoot, fakeClient, quietUi, KEYWORDED_CV, WITHOUT_KEYWORDS, INFLATED_CV,
} from './helpers/ollama-runner-harness.mjs';

// ---------------------------------------------------------------------------
// tailor
// ---------------------------------------------------------------------------

describe('tailor', { concurrency: 6 }, () => {
  const cfg = readOllamaConfig({});
  /**
   * A tailoring run in this process: tailorCv() with a stand-in for the
   * client, which answers every request with `reply` (a text, or a function
   * of the request's number). `client` holds what the run asked the model
   * for, `err` what it said about itself. For what the function itself
   * decides. What the command line adds to it is looked at through a child
   * process.
   */
  const tailorHere = async (root, { ref = '001', reply = CV_HTML, ...deps } = {}) => {
    const { ui, written } = quietUi();
    const client = fakeClient({ reply });
    const result = await tailorCv(ref, { cfg, client, ui, dataRoot: root, ...deps });
    return { result, client, err: written.err };
  };

  test('saves the CV the model returned, cut out of whatever it wrapped it in', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    // What a local model puts around what it was asked for: inline reasoning,
    // a sentence that repeats the prompt's own words, a code fence, a remark.
    const wrapped = [
      '<think>THINK_MARKER the model reasoning in the reply</think>',
      'Here is the tailored CV, starting with <!DOCTYPE html> and ending with </html> as asked:',
      '',
      '```html',
      CV_HTML,
      '```',
      '',
      'The document ends with </html>. Let me know if you want anything changed.',
    ].join('\n');
    await withStub({ reply: () => wrapped }, async (stub) => {
      const run = await cli(['tailor', '001', '--url', stub.url], { root });
      assertExit(run, 0);

      // The model list, the picked model's details, then one generation.
      assert.deepEqual(stub.paths(), ['GET /api/tags', 'POST /api/show', 'POST /api/chat']);
      const sent = stub.requests[2].body;
      assert.equal(sent.model, 'qwen3.8:27b');
      assert.equal(sent.stream, true);
      assert.equal(sent.think, false);
      assert.deepEqual(sent.options, { temperature: 0.2, num_ctx: 65536, num_predict: 16384 });
      const [system, user] = sent.messages.map((message) => message.content);
      assert.deepEqual(sent.messages.map((message) => message.role), ['system', 'user']);

      // The prompt of openai-tailor.mjs: the modes, the template, the CV and the profile.
      assert.ok(system.startsWith('You are career-ops, an AI-powered CV tailoring engine.\n'), system.slice(0, 120));
      for (const part of [
        'SYSTEM CONTEXT (_shared.md)', 'WRITING GUARDRAILS (_writing.md)', 'PDF TAILORING MODE (pdf.md)',
        'HTML TEMPLATE (cv-template.html)', '<div class="section-title">{{SECTION_EXPERIENCE}}</div>',
        '[cv.md]\n# CV -- Alex Chen', '[config/profile.yml]\n# Career-Ops Profile Configuration',
        '1. NEVER invent skills, metrics, or experience the candidate does not have.',
        '4. Replace all {{PLACEHOLDERS}} in the HTML Template exactly as instructed.',
      ]) assert.ok(system.includes(part), `the system prompt lacks: ${part}`);
      assert.ok(system.endsWith('Output the raw HTML starting with <!DOCTYPE html> and ending with </html>.'), system.slice(-160));

      // The report without its archive, then the posting, once.
      // It ends with the sentence that keeps the model from calling a tool it
      // does not have, and then the one that keeps the posting's words and
      // another item's numbers out of a job or a project. Both were measured
      // on the real model in exactly these words, at this place.
      assert.equal(user, `EVALUATION REPORT:\n\n${report.body}\n\nJOB DESCRIPTION:\n\n${JD.trim()}\n\n`
        + 'Now, generate and output the fully filled HTML CV matching the rules above. Output ONLY raw HTML.'
        + `\n\n${NO_TOOLS_NOTE}`
        + '\n\nEvery tool, method, number and scope you write for a job or a project must be one that cv.md states for that '
        + 'same job or project. Do not move a number or a result from one item to another. Use the vocabulary of the job '
        + 'description only to rename what cv.md already states, never to add something it does not state.');
      assert.equal(NO_TOOLS_NOTE, 'You have no tools in this session: do not call Read, Bash or any other function. '
        + 'Every file you need is already in this prompt.');
      assert.equal(user.split('Build the retrieval and agent layer').length - 1, 1, 'the posting was sent twice');

      // Named after the candidate in the profile and after the report.
      assert.deepEqual(listing(join(root, 'output')), [report.html]);
      const saved = read(root, 'output', report.html);
      assert.equal(saved, SAVED_CV);
      assert.ok(!saved.includes('THINK_MARKER') && !saved.includes('Here is the tailored CV') && !saved.includes('```'));

      assert.ok(run.stdout.includes(`Tailored CV saved: output/${report.html}`), run.stdout);
      // A page of HTML and CSS is not streamed to the terminal.
      assert.ok(!run.stdout.includes('<html') && !run.stdout.includes('section-title'), run.stdout);
      assert.match(run.stdout, /pdf payload:\s+\d+\.\dk prompt \/ \d+\.\dk completion/);
      assert.match(run.stdout, /provider=ollama/);
      assert.match(run.stderr, /Tailoring the CV for reports\/001-northwind-robotics-/);
      // The report as it was given: a number is a word to every shell, and stands without quotes.
      assert.match(run.stderr, /^ {4}Next, the PDF: node ollama-runner\.mjs pdf 001$/m);
      assert.doesNotMatch(run.stderr, /❌|⚠️/);
    });
  });

  test('a slot of the template left in the reply is asked for once more, then the run fails', async () => {
    const root = makeRoot();
    writeReport(root);
    const unfilled = CV_HTML.replace('<h1>Alex Chen</h1>', '<h1>{{NAME}}</h1>');
    await withStub({ reply: () => unfilled }, async (stub) => {
      const run = await cli(['tailor', '001', '--url', stub.url], { root });
      assertExit(run, 1);
      const sent = stub.generations();
      assert.equal(sent.length, 2, 'exactly one retry');
      // No seed was set, so none is sent: the server draws a new sample for each request.
      assert.equal('seed' in sent[0].body.options, false);
      assert.equal('seed' in sent[1].body.options, false);
      assert.match(run.stderr, /Attempt 1 was not usable \(the template placeholder \{\{NAME\}\} was left in the HTML\)/);
      assert.match(run.stderr, /Reply rejected after 2 attempts: the template placeholder \{\{NAME\}\} was left in the HTML\./);
      assert.match(run.stderr, /Nothing was saved\./);
      // A CV with "{{NAME}}" for a name must not exist where the PDF step would find it.
      assert.deepEqual(listing(join(root, 'output')), []);
    });
  });

  test('a retry that fills the template is saved like any other reply', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    const unfilled = CV_HTML.replace('Python, Go, TypeScript, SQL', '{{SKILLS}}');
    const { result, client } = await tailorHere(root, { reply: (n) => (n === 0 ? unfilled : CV_HTML) });
    assert.equal(result.success, true, result.reason);
    assert.equal(client.requests.length, 2);
    assert.equal(read(root, 'output', report.html), SAVED_CV);
  });

  test('a slot that is only named in a comment is not a slot left unfilled', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    // The stylesheet of templates/cv-template.html says "{{PHOTO}}" in two of
    // its comments. A model that keeps the stylesheet as it came has still
    // filled in every slot, and so has one that quotes braces in lower case.
    const verbatim = tailoredCv({ extra: '\n  <!-- {{CONSENT}}: the footer slot, left empty -->\n  <p>Templating with {{ user.name }} in Jinja</p>' })
      .replace('<style>', '<style>\n  /* Off by default: with no candidate.photo the {{PHOTO}} slot is empty,\n     no <img> is emitted. */');
    const { result, client } = await tailorHere(root, { reply: verbatim });
    assert.equal(result.success, true, result.reason);
    assert.equal(client.requests.length, 1, 'a complete CV was asked for a second time');
    assert.equal(read(root, 'output', report.html), asSaved(verbatim));
  });

  test('a reply that is not an HTML document, or far too short to be a CV, is not saved', async () => {
    for (const [reply, reason] of [
      ["I'll start by reading the candidate's profile and the template.", 'the reply is not a complete HTML document'],
      // Cut off before the document closed, and reported as finished.
      [CV_HTML.replace('</html>', ''), 'the reply is not a complete HTML document'],
      ['<!DOCTYPE html>\n<html><head><title>CV</title></head><body><p>Alex Chen</p></body></html>', 'the HTML is too short to be a CV'],
    ]) {
      const root = makeRoot();
      writeReport(root);
      const { result, client, err } = await tailorHere(root, { reply });
      assert.deepEqual([result.success, result.code], [false, 'EMPTY'], reason);
      assert.equal(client.requests.length, 2, reason);
      assert.ok(err.includes(`Reply rejected after 2 attempts: ${reason}.`), err);
      assert.deepEqual(listing(join(root, 'output')), []);
    }
  });

  test('a report that archives no posting needs --jd, and says so before anything is sent', async () => {
    const root = makeRoot();
    // An agent-written report can point at a capture under jds/ instead of
    // holding the posting. It is written here as an agent writes one, without
    // the header line of this runner and without its end mark. With the
    // defaults of writeReport() it had both, which no writer produces, and
    // the refusal was shown for that report only.
    const pointer = 'See jds/001-northwind-robotics.md for the full archive (archive-posting.mjs --report=001).';
    writeReport(root, { num: '001', archive: false });
    writeReport(root, { num: '002', archive: pointer, mark: false });
    writeFileSync(join(root, 'empty-jd.txt'), ' \n');
    for (const [deps, code, pattern] of [
      [{ ref: '001' }, 'NO_JD', /❌ {2}The report has no archived job description\. Pass --jd <file>\./],
      [{ ref: '002' }, 'NO_JD', /❌ {2}The report has no archived job description\. Pass --jd <file>\./],
      [{ jdFile: join(root, 'no-such-jd.txt') }, 'BAD_INPUT', /JD file not found: .*no-such-jd\.txt/],
      [{ jdFile: join(root, 'empty-jd.txt') }, 'BAD_INPUT', /The JD file is empty/],
    ]) {
      const { result, client, err } = await tailorHere(root, deps);
      assert.deepEqual([result.success, result.code], [false, code], JSON.stringify(deps));
      assert.match(err, pattern);
      // Not even the model list.
      assert.deepEqual(client.calls, [], JSON.stringify(deps));
    }
    assert.deepEqual(listing(join(root, 'output')), []);
  });

  test('--jd supplies the posting, and the report keeps what follows its archive, but for the Machine Summary', async () => {
    const root = makeRoot();
    const bare = writeReport(root, { num: '001', archive: false });
    const behind = '## Keywords extracted\n\nKEYWORDS_MARKER\n\n## Machine Summary\n\n```yaml\nscore: 4.2\n```';
    // A report that archives one posting, tailored against another file: the file is the posting.
    const archived = writeReport(root, { num: '002', company: 'Globex', after: behind });
    // A report whose archive section points at a capture under jds/, as an
    // agent writes one: without the header line of this runner and its mark.
    const pointer = 'See jds/003-initech.md for the full archive (archive-posting.mjs --report=003).';
    const pointed = writeReport(root, { num: '003', company: 'Initech', archive: pointer, after: behind, mark: false });
    writeFileSync(join(root, 'other-jd.txt'), 'Staff Engineer at Globex.\nOWN_JD_MARKER\n');

    const first = await tailorHere(root, { jdFile: join(root, 'jd.txt') });
    assert.equal(first.result.success, true, first.result.reason);
    const one = first.client.requests[0].user;
    assert.ok(one.startsWith(`EVALUATION REPORT:\n\n${bare.body}\n\nJOB DESCRIPTION:\n\n${JD.trim()}\n\n`), one.slice(0, 200));

    // From the command line, in the other spelling of the option, with a path from the working directory.
    await withStub({}, async (stub) => {
      const second = await cli(['tailor', '002', '--url', stub.url, '--jd=other-jd.txt'], { root });
      assertExit(second, 0);
      const two = stub.generations()[0].body.messages[1].content;
      // The archived posting is cut out, and what the report holds after it
      // stays. The Machine Summary does not: it is the report over again, for
      // scripts, and it is where a wrong statement about the CV reads as evidence.
      assert.ok(two.startsWith(`EVALUATION REPORT:\n\n${archived.body}\n\n## Keywords extracted\n\nKEYWORDS_MARKER\n\n`
        + 'JOB DESCRIPTION:\n\nStaff Engineer at Globex.\nOWN_JD_MARKER\n\n'), two.slice(0, 900));
      assert.ok(!two.includes('Machine Summary') && !two.includes('score: 4.2'), 'the Machine Summary was sent');
      assert.ok(!two.includes('Northwind Robotics'), 'the archived posting was sent beside the one --jd names');
    });

    // A section that only points at the posting is not the posting. It is
    // part of the report and stays in the report part, where it says where
    // the archive is. Nothing held that: read as a posting, the section went
    // out of the request with every suite passing.
    const third = await tailorHere(root, { ref: '003', jdFile: join(root, 'other-jd.txt') });
    assert.equal(third.result.success, true, third.result.reason);
    const three = third.client.requests[0].user;
    assert.ok(three.startsWith(`EVALUATION REPORT:\n\n${pointed.body}\n\n## Job Description (archived verbatim)\n\n${pointer}\n\n`
      + '## Keywords extracted\n\nKEYWORDS_MARKER\n\nJOB DESCRIPTION:\n\nStaff Engineer at Globex.\nOWN_JD_MARKER\n\n'), three.slice(0, 900));
    assert.ok(!three.includes('Machine Summary') && !three.includes('score: 4.2'), 'the Machine Summary was sent');
    assert.deepEqual(listing(join(root, 'output')), [bare.html, archived.html, pointed.html]);
  });

  test('a report that is not there is an error, not a CV tailored to nothing', async () => {
    const root = makeRoot();
    writeReport(root);
    const outside = join(SANDBOX, 'outside-report.md');
    writeFileSync(outside, read(root, 'reports', listing(join(root, 'reports'))[0]));
    for (const [ref, pattern] of [
      ['007', /❌ {2}Report not found: 007\n/],
      ['reports/no-such-report.md', /❌ {2}Report not found: reports\/no-such-report\.md\n/],
      // A report is read from inside the data root only.
      [outside, /Report not found: .*outside-report\.md \(reports are read from inside .+ only\)/],
    ]) {
      const { result, client, err } = await tailorHere(root, { ref });
      assert.deepEqual([result.success, result.code], [false, 'BAD_INPUT'], ref);
      assert.match(err, pattern);
      assert.deepEqual(client.calls, [], ref);
    }
    // What the command line refuses before it looks for a report.
    await withStub({}, async (stub) => {
      for (const [args, pattern] of [
        [[], /tailor needs a report\. Usage: node ollama-runner\.mjs tailor <report>/],
        [['001', '002'], /tailor takes one report at a time, and 2 were given\./],
        [['001', '--pdf', '--no-save'], /--pdf cannot be used with --no-save/],
      ]) {
        const run = await cli(['tailor', '--url', stub.url, ...args], { root });
        assertExit(run, 1, args.join(' '));
        assert.match(run.stderr, pattern);
      }
      assert.deepEqual(stub.requests, []);
    });
    assert.deepEqual(listing(join(root, 'output')), []);
  });

  test('without cv.md or the profile nothing is sent', async () => {
    for (const [file, pattern] of [
      ['cv.md', /cv\.md not found in .+\. Finish onboarding or set CAREER_OPS_ROOT\./],
      [join('config', 'profile.yml'), /Profile not found at .+profile\.yml/],
    ]) {
      const root = makeRoot();
      writeReport(root);
      rmSync(join(root, file));
      const { result, client, err } = await tailorHere(root);
      assert.deepEqual([result.success, result.code], [false, 'SETUP'], file);
      assert.match(err, pattern);
      assert.deepEqual(client.calls, [], file);
    }
  });

  test('checks that cannot be loaded stop the run before the model is asked', async () => {
    const root = makeRoot();
    writeReport(root);
    // generate-pdf.mjs, which holds the locked-sections check, creates output/
    // under the data root while it loads. Here a file is in the way.
    writeFileSync(join(root, 'output'), 'not a folder');
    await withStub({}, async (stub) => {
      const run = await cli(['tailor', '001', '--url', stub.url], { root });
      assertExit(run, 1);
      assert.match(run.stderr, /❌ {2}Could not load the CV checks of generate-pdf\.mjs: /);
      // Said in one line, not as the stack of an error nobody handled.
      assert.doesNotMatch(run.stderr, /^\s+at /m);
      assert.deepEqual(stub.requests, []);
    });
  });

  test('a section the profile locks has to come back as cv.md has it', async () => {
    const locked = 'candidate:\n  full_name: "Alex Chen"\ncv:\n  locked_sections:\n    - "Education"\n';
    const root = makeRoot({ profile: locked });
    const report = writeReport(root);
    const name = `cv-alex-chen-${report.stem}.html`;
    // The same two degrees, with one university swapped for another.
    const rewritten = tailoredCv({ education: '<li>MS Computer Science, MIT (2018)</li><li>BS Computer Science, UC Berkeley (2016)</li>' });
    const changed = await tailorHere(root, { reply: rewritten });
    assert.deepEqual([changed.result.success, changed.result.code], [false, 'LOCKED_SECTIONS']);
    assert.match(changed.err, /❌ {2}Locked section 'education' was modified during tailoring\./);
    assert.match(changed.err, /Expected \(normalized\): .*ut austin/);
    assert.match(changed.err, /Found \(normalized\): .*mit/);
    assert.match(changed.err, /Nothing was saved\./);
    // Not a format problem another sample would fix: the reply is complete, and it is refused.
    assert.equal(changed.client.requests.length, 1);
    assert.deepEqual(listing(join(root, 'output')), []);

    const intact = await tailorHere(root);
    assert.equal(intact.result.success, true, intact.result.reason);
    assert.deepEqual(listing(join(root, 'output')), [name]);
  });

  test('a locked section the profile misnames is refused before the model is asked', async () => {
    for (const [profile, pattern] of [
      // Not a section of cv.md: found before two minutes of generation, not after.
      ['cv:\n  locked_sections:\n    - "Education"\n    - "Publications"\n', /Configured locked_section 'publications' not found in cv\.md\. Available sections: /],
      ['cv:\n  locked_sections: 5\n', /Could not read the locked sections of the CV: Invalid cv\.locked_sections/],
    ]) {
      const root = makeRoot({ profile });
      writeReport(root);
      const { result, client, err } = await tailorHere(root);
      assert.deepEqual([result.success, result.code], [false, 'SETUP'], profile);
      assert.match(err, pattern);
      assert.deepEqual(client.calls, [], profile);
    }
  });

  test('with redact_contact_info the model sees placeholders, and the saved CV carries the real details', async () => {
    const profile = [
      'candidate:', '  full_name: "Alex Chen"', '  email: "alex@example.com"', '  linkedin: "linkedin.com/in/alexchen"',
      'privacy:', '  redact_contact_info: true', '',
    ].join('\n');
    const root = makeRoot({ profile });
    // The report was written by an evaluation that saw the CV as it is, and names the candidate.
    const report = writeReport(root, { blocks: '## Verdict\n\nAlex Chen (alex@example.com) is a strong match for this role.' });
    // What a model returns when "Candidate Name" is all it was shown.
    const anonymous = tailoredCv({ name: 'Candidate Name', email: 'candidate@example.com' });
    const { result, client } = await tailorHere(root, { reply: anonymous });
    assert.equal(result.success, true, result.reason);
    // The whole request as the client is handed it, both messages and every setting.
    const sent = JSON.stringify(client.requests[0]);
    for (const detail of ['Alex Chen', 'alex@example.com', 'linkedin.com/in/alexchen']) {
      assert.ok(!sent.includes(detail), `${detail} was sent to the model`);
    }
    assert.ok(sent.includes('# CV -- Candidate Name') && sent.includes('candidate@example.com'));
    // A CV that ships with "Candidate Name" on it is worse than no CV.
    const saved = read(root, 'output', `cv-alex-chen-${report.stem}.html`);
    assert.equal(saved, SAVED_CV);
  });

  test('with PRIVACY_REDACT=1 a CV that came back with redaction tokens is refused, once', async () => {
    const root = makeRoot();
    writeReport(root);
    writeFileSync(join(root, 'config', 'privacy.yml'), 'protected:\n  full_name: "Alex Chen"\n');
    // The token the request carried in place of the name is all the model can put on the CV.
    const tokens = tailoredCv({ name: '{{PRIVACY_REDACTED_1}}' });
    await withStub({ reply: () => tokens }, async (stub) => {
      const run = await cli(['tailor', '001', '--url', stub.url], { root, env: { PRIVACY_REDACT: '1' } });
      assertExit(run, 1);
      const sent = stub.generations();
      // The next reply would carry the same tokens, so there is no second request.
      assert.equal(sent.length, 1);
      assert.ok(!sent[0].raw.includes('Alex Chen') && sent[0].raw.includes('PRIVACY_REDACTED_'), 'the request did not pass the redacting wrapper');
      assert.match(run.stderr, /The CV came back with \{\{PRIVACY_REDACTED_1\}\} where one of your details belongs\./);
      assert.match(run.stderr, /Tailor the CV without PRIVACY_REDACT\./);
      assert.deepEqual(listing(join(root, 'output')), []);
    });
  });

  test('a privacy config that lists the candidate blocks the request, as it does for an evaluation', async () => {
    const root = makeRoot();
    writeReport(root);
    writeFileSync(join(root, 'config', 'privacy.yml'), 'protected:\n  full_name: "Alex Chen"\n');
    await withStub({}, async (stub) => {
      const run = await cli(['tailor', '001', '--url', stub.url], { root });
      assertExit(run, 1);
      // Refused before it left, and not asked again.
      assert.deepEqual(stub.paths(), ['GET /api/tags', 'POST /api/show']);
      assert.match(run.stderr, /privacy guard checks requests to a local model too/);
      assert.ok(!run.stderr.includes('Alex Chen'), 'the protected value was printed');
      assert.deepEqual(listing(join(root, 'output')), []);
    });
  });

  test('the model and its thinking are chosen the way an evaluation chooses them', async () => {
    const root = makeRoot();
    writeReport(root);
    await withStub({}, async (stub) => {
      const run = await cli(['tailor', '001', '--no-save', '--quiet', '--model', 'gemma4:26b-a4b-it-q4_K_M', '--url', stub.url], { root });
      assertExit(run, 0);
      const sent = stub.generations()[0].body;
      // Thinking is off for a model that can think, so the output cap is the plain one,
      // and how long the model stays loaded is left to the server.
      assert.deepEqual([sent.model, sent.think, sent.options.num_predict], ['gemma4:26b-a4b-it-q4_K_M', false, 16384]);
      assert.equal('keep_alive' in sent, false);
    });
  });

  test('a reply cut at the output cap is not saved and not retried', async () => {
    const root = makeRoot();
    writeReport(root);
    await withStub({ reply: () => ({ content: CV_HTML, doneReason: 'length' }) }, async (stub) => {
      const run = await cli(['tailor', '001', '--url', stub.url], { root });
      assertExit(run, 1);
      assert.equal(stub.generations().length, 1);
      assert.match(run.stderr, /Reply stopped with done_reason "length"/);
      assert.deepEqual(listing(join(root, 'output')), []);
    });
  });

  test('a context window the prompt does not fit is refused before the model is asked', async () => {
    const root = makeRoot();
    writeReport(root);
    await withStub({}, async (stub) => {
      const small = await cli(['tailor', '001', '--url', stub.url, '--num-ctx', '32768'], { root });
      assertExit(small, 1);
      assert.match(small.stderr, /The prompt does not fit the context window: prompt ~\d+ tokens \+ 12000 output needs \d+; num_ctx is 32768/);
      const model = await cli(['tailor', '001', '--url', stub.url, '--model', 'mistral-small'], { root });
      assertExit(model, 1);
      assert.match(model.stderr, /mistral-small:latest holds at most 32768 tokens/);
      assert.deepEqual(stub.generations(), []);
      assert.deepEqual(listing(join(root, 'output')), []);
    });
  });

  test('a CV that cannot be saved fails the run, and is printed so the reply is not lost', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    // A folder where the file belongs.
    mkdirSync(join(root, 'output', report.html), { recursive: true });
    await withStub({}, async (stub) => {
      const run = await cli(['tailor', '001', '--url', stub.url], { root });
      assertExit(run, 1);
      assert.match(run.stderr, /❌ {2}Could not save the tailored CV: /);
      assert.equal(stub.generations().length, 1);
      // Two minutes of generation: shown, since it is in no file.
      assert.ok(run.stdout.startsWith(SAVED_CV), run.stdout.slice(0, 200));
      assert.ok(!run.stdout.includes('Tailored CV saved'));
    });
  });

  test('--no-save prints the CV and writes nothing; --json prints one object', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    await withStub({}, async (stub) => {
      const printed = await cli(['tailor', '001', '--url', stub.url, '--no-save'], { root });
      assertExit(printed, 0);
      assert.deepEqual(listing(join(root, 'output')), []);
      // Once, as the document itself: redirected to a file, stdout opens with the doctype.
      assert.ok(printed.stdout.startsWith(SAVED_CV), printed.stdout.slice(0, 200));
      assert.equal(printed.stdout.split('</html>').length - 1, 1);

      const asJson = await cli(['tailor', '001', '--url', stub.url, '--json'], { root });
      assertExit(asJson, 0);
      assert.equal(asJson.stdout.trim().split('\n').length, 1, asJson.stdout);
      const receipt = JSON.parse(asJson.stdout);
      assert.deepEqual(
        [receipt.command, receipt.success, receipt.saved, receipt.html, receipt.report, receipt.reportNum, receipt.model, receipt.attempts],
        ['tailor', true, true, `output/${report.html}`, report.rel, '001', 'qwen3.8:27b', 1],
      );
      assert.equal('text' in receipt, false, 'the CV is in a file; the receipt points at it');
      assert.equal('session' in receipt, false);
      assert.ok(existsSync(join(root, receipt.html)));

      const failed = await cli(['tailor', '007', '--url', stub.url, '--json'], { root });
      assertExit(failed, 1);
      assert.deepEqual(JSON.parse(failed.stdout), { success: false, command: 'tailor', reason: 'Report not found: 007', code: 'BAD_INPUT', saved: false });
    });
  });
});

// ---------------------------------------------------------------------------
// pdf
// ---------------------------------------------------------------------------

describe('pdf', { concurrency: 6 }, () => {
  test('hands the tailored CV of a report to generate-pdf.mjs, with the report number', async () => {
    const { root, report, pdf } = makeTailoredRoot();
    const run = await cli(['pdf', '001'], { root });
    assertExit(run, 0);
    // generate-pdf.mjs ran, as a process of its own, on these two paths.
    assert.ok(run.stdout.includes(`📄 Input:  ${join(root, 'output', report.html)}`), run.stdout);
    assert.ok(run.stdout.includes(`📁 Output: ${join(root, 'output', pdf)}`), run.stdout);
    // Nothing chose a paper size, so generate-pdf.mjs took the profile's, which is letter when it names none.
    assert.match(run.stdout, /📏 Format: LETTER/);
    // Its checks ran on this CV before anything was rendered.
    assert.ok(run.stdout.includes(`✅ Fact check passed: ${report.html}`), run.stdout);
    assert.ok(run.stdout.includes(`✅  PDF saved: output/${pdf}`), run.stdout);
    // The stand-in's bytes. Another version here means generate-pdf.mjs
    // reached a real browser, and this suite must not start one.
    assert.ok(read(root, 'output', pdf).startsWith('%PDF-1.7'), 'the PDF did not come from the stand-in for Playwright');
    // The report number is what links the PDF to its row in the tracker. It
    // was handed over, not left for generate-pdf.mjs to guess from the file name.
    assert.deepEqual(pdfIndex(root), [['001', `output/${pdf}`, `output/${report.html}`, 'letter', TODAY, 'cv']]);
    assert.doesNotMatch(run.stderr, /No --report given/);
    assert.match(run.stderr, /The next `npm run merge` ticks the PDF column/);
    assert.doesNotMatch(run.stderr, /❌/);
  });

  test('--format chooses the paper, in both spellings, and a size that does not exist is refused', async () => {
    const { root, report, pdf } = makeTailoredRoot();
    for (const option of [['--format', 'a4'], ['--format=A4']]) {
      const run = await cli(['pdf', '001', ...option], { root });
      assertExit(run, 0, option.join(' '));
      assert.match(run.stdout, /📏 Format: A4/);
      assert.deepEqual(pdfIndex(root), [['001', `output/${pdf}`, `output/${report.html}`, 'a4', TODAY, 'cv']]);
    }
    rmSync(join(root, 'output', pdf));
    const refused = await cli(['pdf', '001', '--format', 'tabloid'], { root });
    assertExit(refused, 1);
    assert.match(refused.stderr, /Invalid --format: "tabloid" — must be a4 or letter\./);
    // Refused here, before generate-pdf.mjs was started.
    assert.equal(refused.stdout, '');
    assert.ok(!existsSync(join(root, 'output', pdf)));
  });

  test('an HTML file can be named by its path, from the working directory or from the data root', async () => {
    const { root, report, pdf } = makeTailoredRoot();
    const elsewhere = mkdtempSync(join(SANDBOX, 'cwd-'));
    for (const [ref, cwd] of [
      [join(root, 'output', report.html), elsewhere],
      [`output/${report.html}`, root],
      // Typed from inside output/: the file is there, and not under that name in the data root.
      [report.html, join(root, 'output')],
      // The checkout is the working directory and the data root is somewhere else.
      [`output/${report.html}`, elsewhere],
    ]) {
      rmSync(join(root, 'output', pdf), { force: true });
      const run = await cli(['pdf', ref], { root, cwd });
      assertExit(run, 0, `${ref} from ${cwd}`);
      assert.ok(existsSync(join(root, 'output', pdf)), `${ref} from ${cwd}`);
      // No report was named, so generate-pdf.mjs reads the number off the file name and says so.
      assert.match(run.stderr, /No --report given: using report 001 from the file name/);
    }
    assert.deepEqual(listing(elsewhere), [], 'the PDF belongs in the data root, not where the command was typed');
  });

  test('when two CVs were tailored for one report, the later one is rendered', async () => {
    const { root, report } = makeTailoredRoot();
    // The name in the profile changed between two tailoring runs.
    const earlier = `cv-j-smith-${report.stem}.html`;
    writeFileSync(join(root, 'output', earlier), SAVED_CV);
    const past = new Date(Date.now() - 3_600_000);
    utimesSync(join(root, 'output', earlier), past, past);
    const run = await cli(['pdf', '1'], { root });
    assertExit(run, 0);
    assert.ok(run.stdout.includes(`✅  PDF saved: output/${report.html.replace(/\.html$/, '.pdf')}`), run.stdout);
    assert.ok(!existsSync(join(root, 'output', earlier.replace(/\.html$/, '.pdf'))));
  });

  test('nothing to render is an error, and generate-pdf.mjs is not started', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    for (const [args, pattern] of [
      // The command to run first names the report as it was given: a number stands without quotes.
      [['001'], /❌ {2}No tailored CV for reports\/001-northwind-robotics-\S+\.md in .+output\. Run first: node ollama-runner\.mjs tailor 001\n/],
      [['007'], /❌ {2}Report not found: 007/],
      [['output/no-such-cv.html'], /❌ {2}HTML file not found: output\/no-such-cv\.html/],
      [[], /pdf needs a report\. Usage: node ollama-runner\.mjs pdf <report \| file\.html>/],
      [['001', '--no-save'], /--no-save cannot be used with pdf/],
    ]) {
      const run = await cli(['pdf', ...args], { root });
      assertExit(run, 1, args.join(' '));
      assert.match(run.stderr, pattern);
      assert.equal(run.stdout, '', args.join(' '));
    }
    assert.deepEqual(listing(join(root, 'output')), []);
    // Rendering needs no model, so no command above went looking for a server.
    // This one is given a report the owner renamed, with a blank in its name.
    // The command to run first holds that name in double quotes: without
    // them it read "tailor reports/northwind robotics.md", two reports to a
    // shell, and ended in "tailor takes one report at a time".
    writeFileSync(join(root, 'reports', 'northwind robotics.md'), read(report.path));
    const unreachable = await cli(['pdf', 'reports/northwind robotics.md', '--url', 'http://127.0.0.1:9'], { root });
    assertExit(unreachable, 1);
    assert.doesNotMatch(unreachable.stderr, /not reachable/);
    assert.match(unreachable.stderr, /No tailored CV for reports\/northwind robotics\.md in .+output\. Run first: node ollama-runner\.mjs tailor "reports\/northwind robotics\.md"\n/);
  });

  test('a CV that claims more than cv.md says is stopped by generate-pdf.mjs, and the HTML is kept', async () => {
    // cv.md says the team was three engineers.
    const { root, report, pdf } = makeTailoredRoot({ html: tailoredCv({ team: '30 engineers' }) });
    const run = await cli(['pdf', '001'], { root });
    assertExit(run, 1);
    assert.match(run.stderr, /Fact check failed for cv-jane-smith-001-northwind-robotics-/);
    assert.match(run.stderr, /❌ {2}generate-pdf\.mjs stopped with exit code 1/);
    assert.ok(run.stderr.includes(`The tailored CV is kept: output/${report.html}`), run.stderr);
    assert.deepEqual(listing(join(root, 'output')), [report.html]);
    assert.ok(!run.stdout.includes('PDF saved'));
    assert.deepEqual(pdfIndex(root), []);
    assert.ok(!existsSync(join(root, 'output', pdf)));
  });

  test('a render that generate-pdf.mjs kept as a draft is not reported as a PDF', async () => {
    const { root, pdf } = makeTailoredRoot();
    // Three pages against a budget of two: generate-pdf.mjs warns, keeps the
    // render beside the output path as a draft, and exits 0.
    const run = await cli(['pdf', '001'], { root, env: { STAND_IN_PDF_PAGES: '3' } });
    assertExit(run, 1);
    assert.match(run.stderr, /Overflow draft: /, 'the fixture no longer makes generate-pdf.mjs keep a draft');
    assert.ok(run.stderr.includes(`❌  generate-pdf.mjs finished without writing output/${pdf}.`), run.stderr);
    assert.ok(!run.stdout.includes('PDF saved'));
    assert.ok(!existsSync(join(root, 'output', pdf)));

    // The same when a PDF from an earlier render is already there: it was not replaced.
    writeFileSync(join(root, 'output', pdf), 'an earlier render');
    const again = await cli(['pdf', '001'], { root, env: { STAND_IN_PDF_PAGES: '3' } });
    assertExit(again, 1);
    assert.equal(read(root, 'output', pdf), 'an earlier render');
  });

  test('--json prints one object, although generate-pdf.mjs prints while it works', async () => {
    const { root, report, pdf } = makeTailoredRoot();
    const run = await cli(['pdf', '001', '--json'], { root });
    assertExit(run, 0);
    assert.equal(run.stdout.trim().split('\n').length, 1, run.stdout);
    assert.deepEqual(JSON.parse(run.stdout), {
      command: 'pdf', success: true, html: `output/${report.html}`, report: report.rel, pdf: `output/${pdf}`, factWarnings: 0,
    });
    // What generate-pdf.mjs printed is on stderr, not lost.
    assert.match(run.stderr, /PDF generated: /);

    const failed = await cli(['pdf', '007', '--json'], { root });
    assertExit(failed, 1);
    assert.deepEqual(JSON.parse(failed.stdout), { command: 'pdf', success: false, reason: 'Report not found: 007', code: 'BAD_INPUT' });
  });

  test('tailor --pdf renders the CV it has just saved', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    const pdf = report.html.replace(/\.html$/, '.pdf');
    await withStub({}, async (stub) => {
      const run = await cli(['tailor', '001', '--pdf', '--url', stub.url], { root });
      assertExit(run, 0);
      assert.equal(stub.generations().length, 1);
      assert.deepEqual(listing(join(root, 'output')), [report.html, pdf]);
      assert.deepEqual(pdfIndex(root), [['001', `output/${pdf}`, `output/${report.html}`, 'letter', TODAY, 'cv']]);
      assert.match(run.stdout, /Tailored CV saved: [\s\S]*PDF saved: /);
    });

    // A PDF that could not be rendered fails the command and keeps the CV.
    rmSync(join(root, 'output', pdf));
    await withStub({}, async (stub) => {
      const overflow = await cli(['tailor', '001', '--pdf', '--json', '--url', stub.url], { root, env: { STAND_IN_PDF_PAGES: '3' } });
      assertExit(overflow, 1);
      const receipt = JSON.parse(overflow.stdout);
      assert.deepEqual([receipt.success, receipt.code, receipt.html, 'pdf' in receipt], [false, 'PDF_NOT_WRITTEN', `output/${report.html}`, false]);
      assert.equal(read(root, 'output', report.html), SAVED_CV);
      assert.ok(!existsSync(join(root, 'output', pdf)));
    });
  });
});

// ---------------------------------------------------------------------------
// In-process: the exported functions
// ---------------------------------------------------------------------------

describe('tailorCv', () => {
  const cfg = readOllamaConfig({});

  test('returns where the CV was saved, and stays on a session that is handed in', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    const first = fakeClient();
    const { ui, written } = quietUi();
    const result = await tailorCv('001', { cfg, client: first, ui, dataRoot: root });
    assert.equal(result.success, true, result.reason);
    assert.deepEqual(first.calls, ['listModels', 'showModel qwen3.8:27b', 'chat']);
    assert.equal(result.htmlPath, join(root, 'output', report.html));
    assert.equal(result.relPath, `output/${report.html}`);
    assert.deepEqual([result.report, result.reportNum, result.model, result.saved], [report.rel, '001', 'qwen3.8:27b', true]);
    assert.equal(read(result.htmlPath), SAVED_CV);
    // The estimate the truncation check compares against is the prompt that was sent.
    const sent = first.requests[0];
    const estimate = Math.ceil(sent.system.replace(/\s+/g, ' ').length / 4) + Math.ceil(sent.user.replace(/\s+/g, ' ').length / 4);
    assert.equal(sent.estimatedPromptTokens, estimate);
    // The HTML is not written to the terminal while it arrives, although this ui streams.
    assert.ok(!written.out.includes('<html'), written.out.slice(0, 200));

    // A larger window from earlier in the run is kept, and the model is not looked up again.
    const second = fakeClient();
    const again = await tailorCv(report.rel, {
      cfg, client: second, ui, dataRoot: root, save: false, session: { ...result.session, numCtx: 98304 },
    });
    assert.equal(again.success, true, again.reason);
    assert.deepEqual(second.calls, ['chat']);
    assert.equal(second.requests[0].numCtx, 98304);
    assert.equal(again.session.numCtx, 98304);
    assert.deepEqual([again.saved, 'htmlPath' in again], [false, false]);
    assert.equal(again.html, SAVED_CV);
  });

  test('cuts the document out of the reply by its structure, and keeps or restores the doctype', async () => {
    const root = makeRoot();
    writeReport(root);
    const { ui } = quietUi();
    const tailored = async (reply) => (await tailorCv('001', { cfg, client: fakeClient({ reply }), ui, dataRoot: root, save: false })).html;
    const body = CV_HTML.slice(CV_HTML.indexOf('<html'));

    // Text around the document that names the very tags the prompt names.
    assert.equal(await tailored(`I will output raw <html> starting with <!DOCTYPE html>.\n\n${CV_HTML}\n\nIt ends with </html> now.`), SAVED_CV);
    assert.equal(await tailored(`The <head> keeps the template's styles and the <body> holds the sections.\n${CV_HTML}`), SAVED_CV);
    // A comment between the opening tag and the head is still the start of the document.
    const commented = CV_HTML.replace('<head>', '<!-- tailored for Northwind --> <head>');
    assert.equal(await tailored(`Here: <html>\n${commented}`), asSaved(commented));
    // An opening tag with nothing of a document after it is not one.
    assert.equal(await tailored('<html> is the tag every page starts with, and </html> the one it ends with.'), undefined);
    // The model dropped the template's first line: without it the page is laid out in quirks mode.
    assert.equal(await tailored(body), SAVED_CV);
    assert.equal(await tailored(`\`\`\`html\n${body}\n\`\`\``), SAVED_CV);
    // The doctype is kept as the model wrote it.
    assert.equal(await tailored(CV_HTML.replace('<!DOCTYPE html>', '<!doctype html>')), asSaved(CV_HTML.replace('<!DOCTYPE html>', '<!doctype html>')));
    // Tags are tags in either case.
    const shouting = CV_HTML.replace(/<(\/?)(html|head|body)\b/g, (_tag, slash, name) => `<${slash}${name.toUpperCase()}`);
    assert.equal(await tailored(`Done:\n${shouting}\nEnd.`), asSaved(shouting));
  });

  test('the file is named after the candidate, in letters a file name can carry', async () => {
    const { ui } = quietUi();
    for (const [profile, slug] of [
      ['candidate:\n  full_name: "José Núñez-Ortega"\n', 'jose-nunez-ortega'],
      // Nobody named, and a name without a Latin letter: the file still gets a name.
      ['cv:\n  output_format: html\n', 'candidate'],
      ['candidate:\n  full_name: "山田 太郎"\n', 'candidate'],
      // Where openai-tailor.mjs looks for the name.
      ['name: "Sam O\'Neil"\n', 'sam-o-neil'],
    ]) {
      const root = makeRoot({ profile });
      const report = writeReport(root);
      const result = await tailorCv('001', { cfg, client: fakeClient(), ui, dataRoot: root });
      assert.equal(result.relPath, `output/cv-${slug}-${report.stem}.html`, profile);
    }
  });

  test('a report is found by its number, its name or its path, and never by a look-alike', async () => {
    const root = makeRoot();
    const report = writeReport(root, { num: '003' });
    const thousand = writeReport(root, { num: '1000', company: 'Globex' });
    // What reserve-report-num.mjs leaves while an evaluation is still writing report 004.
    writeFileSync(join(root, 'reports', '004-RESERVED.md'), '{"pid":1}');
    const { ui } = quietUi();
    const found = async (ref) => {
      const result = await tailorCv(ref, { cfg, client: fakeClient(), ui, dataRoot: root, save: false });
      return result.success ? result.report : result.reason;
    };
    // A backslash separates folders on Windows only; elsewhere it is part of a file name.
    const typedOnWindows = process.platform === 'win32' ? [`reports\\${report.stem}.md`] : [];
    for (const ref of ['3', '003', `${report.stem}.md`, report.rel, report.path, ...typedOnWindows]) {
      assert.equal(await found(ref), report.rel, ref);
    }
    assert.equal(await found('1000'), thousand.rel);
    // Matched as a prefix, "100" would be report 1000 and "004" the reservation.
    assert.equal(await found('100'), 'Report not found: 100');
    assert.equal(await found('004'), 'Report not found: 004');
    assert.equal(await found(''), 'No report given.');
  });
});

// ---------------------------------------------------------------------------
// The fact check in tailor
// ---------------------------------------------------------------------------
//
// generate-pdf.mjs refuses a CV that states what cv.md does not. `tailor` used
// to report such a CV as saved, and of 27 real tailored CVs `pdf` then refused
// 18, nearly all for a keyword of the posting in a skills line or a competency
// tag. These tests hold what `tailor` does about it now.

describe('the fact check in tailor', { concurrency: 6 }, () => {
  const cfg = readOllamaConfig({});

  /** How the fact check reads an entry of a CV: its own functions, the ones tailorCv() loads. */
  const realReading = async () => {
    const gate = await import('../verify-cv-facts.mjs');
    const { normalizeTextForATS } = await import('../generate-pdf.mjs');
    return checkReading({ gate, checks: { normalizeTextForATS } });
  };
  /**
   * One tailoring request answered with `reply`, and no second one: what comes
   * back is what the repair made of the reply. `err` is what the run said
   * about it, `sent` what it asked the model for.
   */
  const tailorOnce = async (root, reply, { ref = '001', ...deps } = {}) => {
    const { ui, written } = quietUi();
    const client = fakeClient({ reply });
    const result = await tailorCv(ref, { cfg: { ...cfg, retries: 0 }, client, ui, dataRoot: root, save: false, ...deps });
    return { result, err: written.err, sent: client.requests.map((request) => request.user) };
  };
  // Where the repair works in a CV of tailoredCv(): its competency tags, and the list of its "Infra:" skills line.
  const tagsOf = (html) => [...html.matchAll(/class="competency-tag">([^<]*)</g)].map((tag) => tag[1]);
  const infraOf = (html) => /<span class="skill-category">Infra:<\/span>([^<]*)</.exec(html)?.[1];
  /** A skills line in the markup of the template. */
  const skillLine = (category, list) => `<div class="skill-item"><span class="skill-category">${category}:</span> ${list}</div>`;
  /** A command as a shell hands it over: split at blanks, and what stands in double quotes is one word. */
  const words = (command) => [...command.matchAll(/"([^"]*)"|(\S+)/g)].map(([, quoted, bare]) => quoted ?? bare);
  /**
   * What generate-pdf.mjs lists under "CV fact check warning" for a CV: the
   * check's own verdict on the text that script hands it, printed by the
   * check's own printer, one "  - " line per warning.
   */
  const warningsListed = async (root, html) => {
    const gate = await import('../verify-cv-facts.mjs');
    const { normalizeTextForATS } = await import('../generate-pdf.mjs');
    const lines = [];
    gate.printFactWarnings(gate.verifyFacts(normalizeTextForATS(html).html, {
      sourcePaths: [join(root, 'cv.md'), join(root, 'article-digest.md')], configPath: join(root, 'config', 'cv-facts.json'),
    }), (line) => lines.push(line));
    return lines.filter((line) => /^ {2}- /.test(line));
  };

  test('entries the fact check rejects are removed and named, and the CV that is saved passes `pdf`', async () => {
    const root = makeRoot();
    // Under a name the owner gave the file, with a blank in it, which the CV's file name takes over.
    renameSync(writeReport(root).path, join(root, 'reports', 'northwind robotics.md'));
    const html = 'cv-jane-smith-northwind robotics.html';
    await withStub({ reply: () => KEYWORDED_CV }, async (stub) => {
      const run = await cli(['tailor', 'reports/northwind robotics.md', '--json', '--url', stub.url], { root });
      assertExit(run, 0);
      // Removed, not asked for again: one generation.
      assert.equal(stub.generations().length, 1);
      const receipt = JSON.parse(run.stdout);
      assert.deepEqual(receipt.removed, [{ where: 'competency', text: 'Observability' }, { where: 'skills', text: 'OpenTelemetry' }]);
      assert.deepEqual([receipt.success, receipt.saved, receipt.factCheck.findings], [true, true, []]);
      assert.notEqual(receipt.factCheck.verdict, 'block');
      assert.equal(read(root, 'output', html), asSaved(WITHOUT_KEYWORDS));
      // The owner is told what the posting wants and the CV does not show.
      assert.match(run.stderr, /Removed from the CV, because the fact check found them in neither cv\.md nor article-digest\.md: Observability \(competency\), OpenTelemetry \(skills\)\./);
      assert.match(run.stderr, /If you have this experience, add it to cv\.md and tailor again\./);

      // The run names the command that comes next, with the report in double
      // quotes for the blank in its name. Without them the command read
      // "pdf reports/northwind robotics.md" and ended in "pdf takes one
      // report at a time, and 2 were given".
      const next = /^ {4}Next, the PDF: node ollama-runner\.mjs (pdf .+)$/m.exec(run.stderr)?.[1];
      assert.equal(next, 'pdf "reports/northwind robotics.md"');
      // What tailor called good, generate-pdf.mjs accepts: its own fact check
      // is the judge. It is started by the command the run named.
      const rendered = await cli(words(next), { root });
      assertExit(rendered, 0);
      assert.match(rendered.stdout, /Fact check passed|PDF generated/);
      assert.ok(existsSync(join(root, 'output', html.replace(/\.html$/, '.pdf'))));
    });
  });

  test('a finding in a sentence is asked for once more, with the finding, and the better CV is kept', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    const { ui, written } = quietUi();
    const client = fakeClient({ reply: (n) => (n === 0 ? INFLATED_CV : CV_HTML) });
    const result = await tailorCv('001', { cfg, client, ui, dataRoot: root });
    assert.deepEqual([result.success, result.saved, result.attempts], [true, true, 2]);
    const [first, second] = client.requests.map((request) => request.user);
    assert.equal(client.requests.length, 2);
    // The second request is the first plus what the check rejected, in its own words.
    assert.ok(second.startsWith(first), 'the second request does not repeat the first');
    assert.ok(second.endsWith([
      '',
      '',
      'An earlier draft of this CV was rejected by the fact check, because it named things that cv.md and article-digest.md do not contain:',
      '- metric-like claim absent from sources: 73%',
      '',
      'Write the CV again without them. Do not name them anywhere, not in the skills, the summary or a bullet. '
        + 'A keyword from the job description may only be used where the candidate\'s own files support it.',
    ].join('\n')), second.slice(-600));
    assert.equal(read(root, 'output', report.html), SAVED_CV);
    assert.match(written.err, /The fact check blocks this CV \(1 finding left after removing what could be removed\)\. Asking once more/);
    assert.doesNotMatch(written.err, /❌/);
  });

  test('a CV the check still blocks is kept, reported as FACT_CHECK with exit 1, and gets no PDF', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    await withStub({ reply: () => INFLATED_CV }, async (stub) => {
      // The posting is given in a file, named from the working directory as makeRoot() leaves it there.
      const run = await cli(['tailor', '001', '--pdf', '--json', '--jd', 'jd.txt', '--url', stub.url], { root });
      assertExit(run, 1);
      assert.equal(stub.generations().length, 2, 'one request, and one more with the findings');
      const receipt = JSON.parse(run.stdout);
      assert.deepEqual(
        [receipt.success, receipt.code, receipt.saved, receipt.html, 'pdf' in receipt],
        [false, 'FACT_CHECK', true, `output/${report.html}`, false],
      );
      assert.deepEqual(receipt.factCheck, { verdict: 'block', findings: ['metric-like claim absent from sources: 73%'], warnings: receipt.factCheck.warnings });
      // Kept for the owner to edit, and never handed to generate-pdf.mjs.
      assert.deepEqual(listing(join(root, 'output')), [report.html]);
      assert.equal(read(root, 'output', report.html), asSaved(INFLATED_CV));
      assert.doesNotMatch(run.stderr, /Rendering|generate-pdf\.mjs stopped/);
      assert.match(run.stderr, /❌ {2}The fact check blocks this CV: it states what cv\.md and article-digest\.md do not \(1 finding\)\./);
      assert.match(run.stderr, /- metric-like claim absent from sources: 73%/);
      assert.ok(run.stderr.includes(`The CV is kept for you to work on: output/${report.html}. No PDF was rendered.`), run.stderr);
      // The four ways on. Running it again comes first: on the real model
      // most runs pass, so that is what the owner should try before editing.
      // The command repeats the posting file as it was typed. A plain name
      // like this one stands without quotes, and so do the number and the
      // path of the kept CV.
      assert.match(run.stderr, /Four ways on\. Run `node ollama-runner\.mjs tailor 001 --jd jd\.txt` again: every run writes a different CV, and most pass\. Or edit output\/\S+\.html and run `node ollama-runner\.mjs pdf output\/\S+\.html`; add the evidence to cv\.md or article-digest\.md and tailor again; or, for a claim you have verified, add an exception to config\/cv-facts\.json\./);
      assert.doesNotMatch(run.stdout + run.stderr, /Tailored CV saved/);
    });
  });

  test('the way on names the command that tailors this report, and does not promise another CV from the same seed', async () => {
    // The same function runs inside `run` and `pipeline`. "Run the command
    // again" was the advice there too: in `run` that evaluates the posting a
    // second time and leaves two reports, in `pipeline` the row is ticked and
    // the command finds nothing to do.
    const root = makeRoot();
    writeReport(root);
    const { ui, written } = quietUi();
    const seeded = await tailorCv('001', { cfg: { ...cfg, seed: 42, retries: 0 }, client: fakeClient({ reply: INFLATED_CV }), ui, dataRoot: root, save: false });
    assert.equal(seeded.code, 'FACT_CHECK');
    assert.ok(written.err.includes('Run `node ollama-runner.mjs tailor 001` again with another OLLAMA_SEED or none: with the same seed the '
      + 'model writes the same CV.'), written.err);
    assert.doesNotMatch(written.err, /every run writes a different CV|Run the command again/);
  });

  test('the command the way on names tailors the report the blocked run was about, with the posting that run was given', async () => {
    // hint-2. "tailor 001" was all the way on said. After a run that had its
    // posting from --jd, that command stops for want of a posting or tailors
    // against whatever the report archives. For a report its number does not
    // lead back to, it tailors another report and exits 0.
    const named = (text) => /Run `node ollama-runner\.mjs (tailor [^`]+)` again/.exec(text)?.[1];
    /** Run the command the way on names, typed in the data root and answered with a CV the check passes. */
    const runNamed = (root, command) => {
      const [, ref, flag, jdFile] = words(command);
      assert.ok(flag === undefined || flag === '--jd', command);
      return tailorOnce(root, CV_HTML, { ref, jdFile: jdFile === undefined ? undefined : resolve(root, jdFile) });
    };

    // --jd: the report archives no posting.
    const root = makeRoot();
    writeReport(root, { archive: false });
    writeFileSync(join(root, 'posting.txt'), JD);
    const withJd = await tailorOnce(root, INFLATED_CV, { jdFile: join(root, 'posting.txt') });
    assert.equal(withJd.result.code, 'FACT_CHECK');
    assert.ok(named(withJd.err)?.includes('--jd'), `the named command has no --jd: ${named(withJd.err)}`);
    // In quotes, because the path holds a blank: the folder every data root of this suite lives in has one in its name.
    assert.equal(named(withJd.err), `tailor 001 --jd "${join(root, 'posting.txt')}"`);
    const repeated = await runNamed(root, named(withJd.err));
    assert.equal(repeated.result.success, true, repeated.result.reason);
    assert.deepEqual(repeated.sent, withJd.sent, 'the named command asks for a CV for another posting than the blocked run did');
    // By its number alone, this report cannot be tailored at all.
    assert.equal((await tailorOnce(root, CV_HTML)).result.code, 'NO_JD');

    // A report in a folder under reports/, with another report of the same
    // number in reports/, from the command line, and the posting in a file
    // that is named from the working directory. On Windows that is a file in
    // a folder, elsewhere a file with a backslash in its name.
    const filed = makeRoot();
    const report = writeReport(filed);
    mkdirSync(join(filed, 'reports', 'archive'), { recursive: true });
    renameSync(report.path, join(filed, 'reports', 'archive', `${report.stem}.md`));
    writeReport(filed, { company: 'Globex' });
    mkdirSync(join(filed, 'postings'));
    writeFileSync(join(filed, 'postings\\northwind.txt'), JD);
    await withStub({ reply: () => INFLATED_CV }, async (stub) => {
      const run = await cli(
        ['tailor', `reports/archive/${report.stem}.md`, '--jd', 'postings\\northwind.txt', '--url', stub.url],
        { root: filed, env: { OLLAMA_RETRIES: '0' } },
      );
      assertExit(run, 1);
      const command = named(run.stderr);
      assert.ok(command, run.stderr);
      assert.notEqual(command, 'tailor 001', 'the named command tailors the other report numbered 001');
      assert.ok(command.includes(`archive/${report.stem}.md`) || command.includes(`archive\\${report.stem}.md`), command);
      // The report by its path as the data root names it, without quotes: it
      // is a plain name. The posting as it was typed, in double quotes for
      // its backslash and nothing else, since it holds no blank. Outside
      // quotes Git Bash drops a backslash, and the command names no file.
      assert.equal(command, `tailor reports/archive/${report.stem}.md --jd "postings\\northwind.txt"`);
      assert.equal((await runNamed(filed, command)).result.report, `reports/archive/${report.stem}.md`);
      // By its number it is the other report.
      assert.equal((await tailorOnce(filed, CV_HTML)).result.report, `reports/001-globex-${TODAY}.md`);
    });

    // However the owner files or names a report, the named command leads back to it: by its number where the
    // number finds it, else by its path, in quotes where the path holds a blank.
    const odd = makeRoot();
    const numbered = writeReport(odd);
    const text = read(numbered.path);
    writeFileSync(join(odd, 'reports', `01-northwind-robotics-${TODAY}.md`), text);
    writeFileSync(join(odd, 'reports', 'northwind robotics.md'), text);
    writeFileSync(join(odd, 'reports', 'o\'brien&sons(2026).md'), text);
    const shelved = writeReport(odd, { num: '002', company: 'Globex' });
    mkdirSync(join(odd, 'reports', 'archive'));
    renameSync(shelved.path, join(odd, 'reports', 'archive', `${shelved.stem}.md`));
    for (const [ref, command] of [
      [numbered.rel, 'tailor 001'],
      // A number is read with three digits: "01" is report 001, which is another file.
      [`reports/01-northwind-robotics-${TODAY}.md`, `tailor reports/01-northwind-robotics-${TODAY}.md`],
      ['reports/northwind robotics.md', 'tailor "reports/northwind robotics.md"'],
      // No blank in it, and still no word that every shell reads alike: an
      // apostrophe, an "&" and brackets. Quoted for a blank only, such a
      // name stood bare, and the "&" alone ended the command in PowerShell,
      // in cmd and in Git Bash.
      ['reports/o\'brien&sons(2026).md', 'tailor "reports/o\'brien&sons(2026).md"'],
      // A report is found by its number in reports/ itself, and this one is no longer there.
      [`reports/archive/${shelved.stem}.md`, `tailor reports/archive/${shelved.stem}.md`],
    ]) {
      const blocked = await tailorOnce(odd, INFLATED_CV, { ref });
      assert.equal(blocked.result.code, 'FACT_CHECK', ref);
      assert.equal(named(blocked.err), command, ref);
      const second = await runNamed(odd, command);
      assert.equal(second.result.report, blocked.result.report, `"${command}" tailors another report`);
    }

    // A report kept in the data root itself, outside reports/, is not found
    // by its bare name: the command named that way ended in "Report not
    // found". It is named with "./" in front, which every shell reads alike,
    // where a full Windows path loses its backslashes in Git Bash.
    const loose = join(odd, 'northwind.md');
    writeFileSync(loose, text);
    const outside = await tailorOnce(odd, INFLATED_CV, { ref: loose });
    assert.equal(outside.result.code, 'FACT_CHECK');
    assert.equal(outside.result.report, 'northwind.md');
    assert.equal(named(outside.err), 'tailor ./northwind.md');
    assert.equal((await runNamed(odd, named(outside.err))).result.report, 'northwind.md');
    assert.equal((await tailorOnce(odd, CV_HTML, { ref: 'northwind.md' })).result.reason, 'Report not found: northwind.md');

    // A name is put in double quotes unless it is plain: letters, digits and
    // _ . / : - . So is the file of the kept CV in the second command of the
    // same hint, which a blank in the report's name carries into its name.
    const oddNames = makeRoot();
    writeFileSync(join(oddNames, 'reports', 'acme & sons (2026).md'), read(writeReport(oddNames).path));
    const quoted = await tailorOnce(oddNames, INFLATED_CV, { ref: 'reports/acme & sons (2026).md', save: true });
    assert.equal(named(quoted.err), 'tailor "reports/acme & sons (2026).md"');
    const pdfNamed = /run `node ollama-runner\.mjs (pdf [^`]+)`/.exec(quoted.err)?.[1];
    assert.equal(pdfNamed, `pdf "${quoted.result.relPath}"`);
    assert.match(quoted.result.relPath, /^output\/cv-.* & sons \(2026\)\.html$/);
  });

  test('an error of the fact check after the reply is not reported as a locked section', async () => {
    // config/cv-facts.json is the file the FACT_CHECK hint tells the owner to
    // edit, and a tailoring request lasts a minute. Saved in a broken state
    // meanwhile, it made the check throw on the reply, and the run said that a
    // section "is locked under cv.locked_sections".
    const root = makeRoot();
    writeReport(root);
    const { ui, written } = quietUi();
    const client = fakeClient({
      reply: () => {
        mkdirSync(join(root, 'config'), { recursive: true });
        writeFileSync(join(root, 'config', 'cv-facts.json'), '{ "allow_facts": ["x", }');
        return CV_HTML;
      },
    });
    const result = await tailorCv('001', { cfg, client, ui, dataRoot: root });
    assert.deepEqual([result.success, result.code], [false, 'SETUP'], result.reason);
    assert.match(result.reason, /^The fact check could not run on the reply: /);
    assert.doesNotMatch(written.err, /locked under cv\.locked_sections/);
    assert.deepEqual(listing(join(root, 'output')), []);
  });

  test('with no retry allowed the model is not asked again; --no-save returns the repaired document', async () => {
    const root = makeRoot();
    writeReport(root);
    const { ui } = quietUi();
    const once = fakeClient({ reply: INFLATED_CV });
    const blocked = await tailorCv('001', { cfg: { ...cfg, retries: 0 }, client: once, ui, dataRoot: root, save: false });
    assert.deepEqual([blocked.success, blocked.code, blocked.saved], [false, 'FACT_CHECK', false]);
    assert.equal(once.calls.filter((call) => call === 'chat').length, 1);
    assert.deepEqual(listing(join(root, 'output')), []);

    const repaired = await tailorCv('001', { cfg, client: fakeClient({ reply: KEYWORDED_CV }), ui, dataRoot: root, save: false });
    assert.deepEqual([repaired.success, repaired.saved, repaired.html], [true, false, asSaved(WITHOUT_KEYWORDS)]);
  });

  test('of two documents the one with fewer findings is kept, the first when both have as many, and a second request that fails changes nothing', async () => {
    const root = makeRoot();
    writeReport(root);
    const { ui, written } = quietUi();
    const worse = tailoredCv({ summary: ' Cut inference cost by 73% and grew revenue by $40M.' });
    // The second reply is worse than the first: the first is kept.
    const kept = await tailorCv('001', { cfg, client: fakeClient({ reply: (n) => (n === 0 ? INFLATED_CV : worse) }), ui, dataRoot: root, save: false });
    assert.deepEqual([kept.code, kept.html, kept.factCheck.findings.length], ['FACT_CHECK', asSaved(INFLATED_CV), 1]);
    // The second reply is no better: it has one finding as well, a different
    // one. The first is kept, and the finding the run names is that of the
    // document it returns. No test held this side of the rule.
    const another = tailoredCv({ summary: ' Cut inference cost by 41%.' });
    const twice = fakeClient({ reply: (n) => (n === 0 ? INFLATED_CV : another) });
    const level = await tailorCv('001', { cfg, client: twice, ui, dataRoot: root, save: false });
    assert.equal(twice.requests.length, 2);
    assert.deepEqual(
      [level.code, level.html, level.factCheck.findings],
      ['FACT_CHECK', asSaved(INFLATED_CV), ['metric-like claim absent from sources: 73%']],
    );
    // The second reply is not a document at all: the first is still in hand.
    const client = fakeClient({ reply: (n) => (n === 0 ? INFLATED_CV : 'I will start by reading the files.') });
    const still = await tailorCv('001', { cfg, client, ui, dataRoot: root, save: false });
    assert.deepEqual([still.code, still.html], ['FACT_CHECK', asSaved(INFLATED_CV)]);
    assert.match(written.err, /The second request gave nothing usable \(Reply rejected after 2 attempts: the reply is not a complete HTML document\.\)\. Keeping the first CV\./);
  });

  test('a stock phrase generate-pdf.mjs refuses is found here too, and asked out of the CV', async () => {
    const root = makeRoot();
    writeReport(root);
    const { ui } = quietUi();
    const stock = tailoredCv({ summary: ' A proven track record of shipping.' });
    const client = fakeClient({ reply: (n) => (n === 0 ? stock : CV_HTML) });
    const result = await tailorCv('001', { cfg, client, ui, dataRoot: root, save: false });
    assert.deepEqual([result.success, result.html], [true, SAVED_CV]);
    assert.ok(
      client.requests[1].user.includes('- wording that is not in the source CV: "proven track record" (keep the candidate\'s own words)'),
      client.requests[1].user.slice(-500),
    );
    // generate-pdf.mjs does refuse that CV: the check here mirrors its own.
    const { root: rendered } = makeTailoredRoot({ html: stock });
    const run = await cli(['pdf', '001'], { root: rendered });
    assertExit(run, 1);
    assert.match(run.stderr, /Voice check failed for \S+: "proven track record"/);
  });

  test('a phrase config/cv-facts.json forbids is a finding here too, and asked out of the CV', async () => {
    // unpinned-check-1. config/cv-facts.json is the file the way on sends the
    // owner to, and a phrase it forbids stops a render like a fact cv.md does
    // not have. Right as it is, and no test held it: with the check here
    // listing no forbidden phrase, every suite passed.
    const root = makeRoot();
    writeReport(root);
    writeFileSync(join(root, 'config', 'cv-facts.json'), JSON.stringify({ forbidden_phrases: ['rockstar'] }));
    const forbidden = tailoredCv({ summary: ' A rockstar engineer.' });
    const { result, err } = await tailorOnce(root, forbidden);
    assert.equal(result.code, 'FACT_CHECK');
    assert.deepEqual(result.factCheck, { verdict: 'block', findings: ['forbidden phrase found: rockstar'], warnings: 0 });
    assert.match(err, /❌ {2}The fact check blocks this CV: it states what cv\.md and article-digest\.md do not \(1 finding\)\./);
    assert.match(err, /^ {4}- forbidden phrase found: rockstar$/m);
    // With a second request allowed, that request is told the phrase, and the CV that comes without it passes.
    const { ui } = quietUi();
    const client = fakeClient({ reply: (n) => (n === 0 ? forbidden : CV_HTML) });
    const asked = await tailorCv('001', { cfg, client, ui, dataRoot: root, save: false });
    assert.deepEqual([asked.success, asked.html], [true, SAVED_CV]);
    assert.ok(client.requests[1].user.includes('\n- forbidden phrase found: rockstar\n'), client.requests[1].user.slice(-500));
  });

  test('the number of warnings a run names is the check\'s own count, the facts it only asks to confirm among them', async () => {
    // unpinned-check-1. A CV the check passes with warnings is saved, and the
    // run says how many generate-pdf.mjs will list. Right as it is, and no
    // test read the number: with the facts the check only asks to confirm
    // left out of it, every suite passed.
    const root = makeRoot();
    writeReport(root);
    // A phrase the owner asks to be warned about, which every CV of this suite holds.
    writeFileSync(join(root, 'config', 'cv-facts.json'), JSON.stringify({ warn_phrases: ['Series B'] }));
    // And two Title Case phrases cv.md does not have: the check asks to confirm them, and blocks nothing.
    const reply = tailoredCv({ competencies: ['MLOps', 'Distributed Systems', 'Platform Engineering', 'Kafka'] });
    const listed = await warningsListed(root, reply);
    assert.equal(listed.length, 3, `the fixture no longer draws three warnings: ${listed.join(' | ')}`);
    const { result, err } = await tailorOnce(root, reply);
    assert.equal(result.success, true, result.reason);
    assert.deepEqual(result.factCheck, { verdict: 'warn', findings: [], warnings: listed.length });
    assert.ok(err.includes('The fact check passes this CV with 3 warnings: generate-pdf.mjs lists them when it renders the PDF.'), err);
    // A CV with the phrase alone draws one, and the run says "1 warning".
    assert.equal((await warningsListed(root, CV_HTML)).length, 1);
    const single = await tailorOnce(root, CV_HTML);
    assert.equal(single.result.factCheck.warnings, 1);
    assert.ok(single.err.includes('The fact check passes this CV with 1 warning: generate-pdf.mjs lists them'), single.err);
  });

  test('pruning takes whole entries only, in either markup a model writes, and leaves prose alone', () => {
    const rejected = ['opentelemetry', 'observability', 'ci/cd pipelines'];
    const html = [
      '<div class="competency-tag">Observability</div>',
      '<span class="competency-tag">Observability &amp; Grafana</span>',
      '<span class="skill-item"><span class="skill-category">Infra:</span> Kubernetes, OpenTelemetry, AWS</span>',
      '<div class="skill-item"><span class="skill-category">Tracing:</span> OpenTelemetry</div>',
      '<div class="project-tech">Python, CI/CD Pipelines</div>',
      '<li>Instrumented services with OpenTelemetry</li>',
    ].join('\n');
    const { html: pruned, removed, rewritten, cuts } = pruneRejectedEntries(html, rejected);
    assert.equal(pruned, [
      '<span class="competency-tag">Observability &amp; Grafana</span>',
      '<span class="skill-item"><span class="skill-category">Infra:</span> Kubernetes, AWS</span>',
      // A line that lost every item goes with its category.
      '',
      '<div class="project-tech">Python</div>',
      '<li>Instrumented services with OpenTelemetry</li>',
    ].join('\n'));
    assert.deepEqual(removed.map(({ where, text }) => `${where}: ${text}`), [
      'competency: Observability', 'skills: OpenTelemetry', 'skills: OpenTelemetry', 'project tech: CI/CD Pipelines',
    ]);
    assert.deepEqual(rewritten, []);
    // Each list that lost an item is named as it stood and as it stands now,
    // which is '' for a line that went. A tag is no list, and goes whole.
    assert.deepEqual(cuts, [
      { where: 'skills', from: 'Kubernetes, OpenTelemetry, AWS', to: 'Kubernetes, AWS', removed: [{ where: 'skills', text: 'OpenTelemetry' }] },
      { where: 'skills', from: 'OpenTelemetry', to: '', removed: [{ where: 'skills', text: 'OpenTelemetry' }] },
      { where: 'project tech', from: 'Python, CI/CD Pipelines', to: 'Python', removed: [{ where: 'project tech', text: 'CI/CD Pipelines' }] },
    ]);
    // The items of a cut are its entries in `removed`, the very same ones:
    // that is how repairCvFacts() takes the items of a line that went out of
    // its list, where another line lost an item of the same name.
    assert.deepEqual(cuts.map((cut) => cut.removed.map((entry) => removed.indexOf(entry))), [[1], [2], [3]]);
    assert.deepEqual(pruneRejectedEntries(html, []), { html, removed: [], rewritten: [], cuts: [] });

    // The `;` that ends an entity separates nothing: "AWS &amp; Terraform" is
    // one item. Split there, the line came out as "AWS &amp, Kafka".
    const entity = '<div class="skill-item"><span class="skill-category">Cloud:</span> AWS &amp; Terraform, Kafka; Redis</div>';
    assert.deepEqual(pruneRejectedEntries(entity, ['terraform', 'redis']), {
      html: '<div class="skill-item"><span class="skill-category">Cloud:</span> AWS &amp; Terraform, Kafka</div>',
      removed: [{ where: 'skills', text: 'Redis' }],
      rewritten: [],
      cuts: [{ where: 'skills', from: 'AWS & Terraform, Kafka; Redis', to: 'AWS & Terraform, Kafka', removed: [{ where: 'skills', text: 'Redis' }] }],
    });
    // An item cut out from behind a "·" leaves no blank in front of the comma, however many blanks stood there
    // and however many items went.
    assert.equal(pruneRejectedEntries(skillLine('Infra', 'Kafka  · OpenTelemetry, Redis'), ['opentelemetry']).html, skillLine('Infra', 'Kafka, Redis'));
    assert.equal(
      pruneRejectedEntries(skillLine('Infra', 'Kafka · OpenTelemetry, Redis · Jaeger, AWS'), ['opentelemetry', 'jaeger']).html,
      skillLine('Infra', 'Kafka, Redis, AWS'),
    );
  });

  test('a list in brackets that the check rejected a piece of is written without the brackets, and loses nothing; a project\'s tech line is never rewritten', async () => {
    // The check reads "Cloud (AWS, GCP, Azure)" item by item and rejects
    // "Azure)". Cut out by itself, that left "Cloud (AWS, GCP, Kubernetes":
    // a bracket nothing closes, in a CV both `tailor` and `pdf` called good.
    const html = [
      '<div class="skill-item"><span class="skill-category">Platforms:</span> Cloud (AWS, GCP, Azure), Kubernetes</div>',
      // The check does not read a project's tech line. A piece it rejected was
      // rejected in a tag or on a skills line, so the brackets of the tech
      // line were never misread, and the line stays as it is written.
      '<div class="project-tech">Kafka Streams, model serving (SageMaker, MLflow), Redis</div>',
      // Brackets the check did not stumble over stay as they are.
      '<div class="skill-item"><span class="skill-category">Data:</span> SQL (PostgreSQL), Spark</div>',
    ].join('\n');
    const repaired = pruneRejectedEntries(html, ['azure)', 'mlflow)']);
    assert.equal(repaired.html, [
      '<div class="skill-item"><span class="skill-category">Platforms:</span> Cloud, AWS, GCP, Azure, Kubernetes</div>',
      '<div class="project-tech">Kafka Streams, model serving (SageMaker, MLflow), Redis</div>',
      '<div class="skill-item"><span class="skill-category">Data:</span> SQL (PostgreSQL), Spark</div>',
    ].join('\n'));
    assert.deepEqual(repaired.removed, []);
    assert.deepEqual(repaired.rewritten, [
      { where: 'skills', from: 'Cloud (AWS, GCP, Azure), Kubernetes', to: 'Cloud, AWS, GCP, Azure, Kubernetes', item: 'azure)' },
    ]);
    assert.deepEqual(repaired.cuts, []);
    // Read again without the brackets, a tool cv.md does not have is an item like any other, and goes alone.
    const next = pruneRejectedEntries(repaired.html, ['azure']);
    assert.ok(next.html.startsWith('<div class="skill-item"><span class="skill-category">Platforms:</span> Cloud, AWS, GCP, Kubernetes</div>'), next.html);
    assert.deepEqual([next.removed, next.rewritten], [[{ where: 'skills', text: 'Azure' }], []]);
    // The cut names the list as the first round left it. By that name the
    // entry reported as rewritten is brought up to what the CV now holds.
    assert.deepEqual(next.cuts, [
      { where: 'skills', from: 'Cloud, AWS, GCP, Azure, Kubernetes', to: 'Cloud, AWS, GCP, Kubernetes', removed: [{ where: 'skills', text: 'Azure' }] },
    ]);

    // With every bracket left where it is, which is how the repair is tried
    // again when taking them off left the CV no better: nothing is rewritten, a
    // piece of a list in brackets is not cut out of it, and an item that
    // stands by itself still goes.
    assert.deepEqual(pruneRejectedEntries(html, ['azure)', 'mlflow)'], { keepBrackets: true }), { html, removed: [], rewritten: [], cuts: [] });
    const kept = pruneRejectedEntries(html, ['azure)', 'kubernetes'], { keepBrackets: true });
    assert.ok(kept.html.startsWith('<div class="skill-item"><span class="skill-category">Platforms:</span> Cloud (AWS, GCP, Azure)</div>'), kept.html);
    assert.deepEqual([kept.removed, kept.rewritten], [[{ where: 'skills', text: 'Kubernetes' }], []]);

    // What a bracket holds decides whether it goes. With a separator in it, or
    // an "and" or an "or", it is a list the check split. Anything else is a
    // level, a version or a remark, and stays as it is written: taken apart, a
    // CV that then passed listed "expert" and "3.11" as skills. "X)" stands for
    // the piece the check rejected.
    for (const [text, flat] of [
      ['(a) b, c', '(a) b, c, X'],
      ['a (b) c', 'a (b) c, X'],
      ['A ()', 'A (), X'],
      ['Python (3.11), Go (basic)', 'Python (3.11), Go (basic), X'],
      // The ";" that ends an entity separates nothing: such a group holds no list either.
      ['Sandbox (R&amp;D), Tools (x&nbsp;y)', 'Sandbox (R&amp;D), Tools (x&nbsp;y), X'],
      // A ";" that stands for itself does.
      ['A (b &amp; c; d)', 'A, b &amp; c; d, X'],
      ['A ( b , c ) , d', 'A, b, c, d, X'],
      // The space the entry began with stays.
      [' A (b, c) ', ' A, b, c, X'],
      ['Cloud (Azure and AWS)', 'Cloud, Azure and AWS, X'],
      ['Cloud (AWS or GCP)', 'Cloud, AWS or GCP, X'],
      // The list goes, and the remark inside it stays.
      ['Platform (AWS (SageMaker), Kubernetes)', 'Platform, AWS (SageMaker), Kubernetes, X'],
      // Square, curly and full-width brackets are read like round ones.
      ['SQL [PostgreSQL], Data [SQL, Spark]', 'SQL [PostgreSQL], Data, SQL, Spark, X'],
      ['SQL {PostgreSQL}, Data {SQL, Spark}', 'SQL {PostgreSQL}, Data, SQL, Spark, X'],
      ['SQL（PostgreSQL）, Data（SQL，Spark）', 'SQL（PostgreSQL）, Data, SQL，Spark, X'],
    ]) {
      const line = pruneRejectedEntries(`<div class="skill-item">${text}, X)</div>`, ['x)']).html;
      assert.equal(line, `<div class="skill-item">${flat}</div>`);
      // No comma twice, and none at either end.
      assert.doesNotMatch(line, /,\s*,|,\s*<|>\s*,/);
      // On a project's tech line none of these changes.
      const tech = `<div class="project-tech">${text}, X)</div>`;
      assert.deepEqual(pruneRejectedEntries(tech, ['x)']), { html: tech, removed: [], rewritten: [], cuts: [] });
    }

    // Punctuation beside a bracket keeps its place, and no comma is put where
    // a separator already stands. Read as the check reads the line, which is
    // what takes the full stop off the piece it rejected.
    const reading = await realReading();
    for (const [list, flat] of [
      ['Kubernetes, Model serving (SageMaker, MLflow).', 'Kubernetes, Model serving, SageMaker, MLflow.'],
      ['Kubernetes; (SageMaker, MLflow); Redis', 'Kubernetes; SageMaker, MLflow; Redis'],
      ['Model serving&nbsp;(SageMaker, MLflow), Redis', 'Model serving, SageMaker, MLflow, Redis'],
    ]) {
      assert.equal(pruneRejectedEntries(skillLine('Platforms', list), ['mlflow)'], reading).html, skillLine('Platforms', flat));
    }

    // Whether a bracket holds a list is asked of its text as the check reads
    // it, which decodes entities before it splits. An entity that stands for
    // a letter or a blank separates nothing, in any spelling, and one that
    // spells a separator makes a list. Read by the ";" alone, the first kind
    // was taken apart, and then the second was left for the tag or the line
    // to be blocked for "mlflow)".
    //
    // A "&name;" the check does not decode is no separator either, although
    // the check splits at its ";". Taken apart, "(Montr&eacute;al)" gave the
    // check the piece "montr&eacute", a finding that nothing can remove, and a
    // line that passes with the group left alone was blocked. So a list that
    // only such an entity separates stays in its brackets, and its finding
    // goes to the second request.
    for (const [list, flat] of [
      ['Kubernetes, Lab (R&#38;D), Unit (R&AMP;D), Desk (x&#160;y), Model serving (SageMaker, MLflow)',
        'Kubernetes, Lab (R&#38;D), Unit (R&AMP;D), Desk (x&#160;y), Model serving, SageMaker, MLflow'],
      ['Kubernetes, Model serving (SageMaker &middot; MLflow)', 'Kubernetes, Model serving, SageMaker &middot; MLflow'],
      ['Kubernetes, Model serving (SageMaker &bull; MLflow)', 'Kubernetes, Model serving, SageMaker &bull; MLflow'],
      ['Kubernetes, Model serving (SageMaker&#44; MLflow)', 'Kubernetes, Model serving, SageMaker&#44; MLflow'],
      ['Kubernetes, Model serving (SageMaker &#x7C; MLflow)', 'Kubernetes, Model serving, SageMaker &#x7C; MLflow'],
      ['Kubernetes, Kafka (Montr&eacute;al), Model serving (SageMaker, MLflow)', 'Kubernetes, Kafka (Montr&eacute;al), Model serving, SageMaker, MLflow'],
      ['Kubernetes, Python (C&plus;&plus;), Model serving (SageMaker, MLflow)', 'Kubernetes, Python (C&plus;&plus;), Model serving, SageMaker, MLflow'],
      ['Kubernetes, Kafka (&Eacute;cole), Model serving (SageMaker, MLflow)', 'Kubernetes, Kafka (&Eacute;cole), Model serving, SageMaker, MLflow'],
      ['Kubernetes, Kafka (A &rarr; B), Model serving (SageMaker &middot; MLflow)', 'Kubernetes, Kafka (A &rarr; B), Model serving, SageMaker &middot; MLflow'],
      ['Kubernetes, Model serving (SageMaker&comma; MLflow)', 'Kubernetes, Model serving (SageMaker&comma; MLflow)'],
    ]) {
      assert.equal(pruneRejectedEntries(skillLine('Platforms', list), ['mlflow)'], reading).html, skillLine('Platforms', flat), list);
    }
    // With `entityLists`, a group that only such an entity separates is the
    // list the check made of it where the check reads a tool in its last
    // piece and in no other: "(A &rarr; B)" gives it "A &rarr", no tool, and
    // "B)". Any other such group stays. A tool in a piece that ends in the
    // entity would be a finding once the group is apart, and with no tool in
    // the last piece nothing in the group is misread.
    for (const [list, flat] of [
      ['Kubernetes, Kafka (A &rarr; B), Model serving (SageMaker, MLflow)', 'Kubernetes, Kafka, A &rarr; B, Model serving, SageMaker, MLflow'],
      ['Kubernetes, Kafka (queue &rarr; stream), Model serving (SageMaker, MLflow)', 'Kubernetes, Kafka (queue &rarr; stream), Model serving, SageMaker, MLflow'],
      ['Kubernetes, Kafka (Montr&eacute;al), Python (C&plus;&plus;), Model serving (SageMaker, MLflow)',
        'Kubernetes, Kafka (Montr&eacute;al), Python (C&plus;&plus;), Model serving, SageMaker, MLflow'],
      ['Kubernetes, Kafka {5 &times; 3}, Kafka (x&sup2;), Kafka (&alpha;), Model serving (SageMaker, MLflow)',
        'Kubernetes, Kafka {5 &times; 3}, Kafka (x&sup2;), Kafka (&alpha;), Model serving, SageMaker, MLflow'],
      ['Kubernetes, Model serving (SageMaker&comma; MLflow)', 'Kubernetes, Model serving (SageMaker&comma; MLflow)'],
    ]) {
      assert.equal(pruneRejectedEntries(skillLine('Platforms', list), ['mlflow)'], { ...reading, entityLists: true }).html, skillLine('Platforms', flat), list);
    }
    // Without the check's reading no entity is decoded, and none in a group separates anything.
    assert.equal(
      pruneRejectedEntries(skillLine('Platforms', 'Lab (R&#38;D), Span (10&ndash;20 nodes), Model serving (SageMaker, MLflow)'), ['mlflow)']).html,
      skillLine('Platforms', 'Lab (R&#38;D), Span (10&ndash;20 nodes), Model serving, SageMaker, MLflow'),
    );
    // The same for a tag, through a tailoring run: its tools are ones cv.md names, so it stays, without its brackets.
    const root = makeRoot();
    writeReport(root);
    const dotted = await tailorOnce(root, tailoredCv({ competencies: ['MLOps', 'MLOps Platform (SageMaker &middot; Kubeflow &middot; MLflow)', 'Kafka'] }));
    assert.equal(dotted.result.success, true, dotted.result.reason);
    assert.deepEqual(tagsOf(dotted.result.html), ['MLOps', 'MLOps Platform, SageMaker &middot; Kubeflow &middot; MLflow', 'Kafka']);
    assert.deepEqual(dotted.result.removed, []);
    assert.doesNotMatch(dotted.err, /your CV does not show/);
    // And a skills line with both kinds of group passes a run: the list loses
    // its brackets, the group with the entity the check does not decode stays.
    for (const [written, saved] of [
      ['Kubernetes, Kafka (Montr&eacute;al), Model serving (SageMaker, MLflow), Redis', ' Kubernetes, Kafka (Montr&eacute;al), Model serving, SageMaker, MLflow, Redis'],
      ['Kubernetes, Python (C&plus;&plus;), Model serving (SageMaker, MLflow), Redis', ' Kubernetes, Python (C&plus;&plus;), Model serving, SageMaker, MLflow, Redis'],
    ]) {
      const accented = await tailorOnce(root, tailoredCv({ infra: written }));
      assert.equal(accented.result.success, true, `${written}: ${accented.result.reason}`);
      assert.equal(infraOf(accented.result.html), saved);
      assert.deepEqual(accented.result.removed, [], written);
    }
    // A group that only an arrow separates, of two tools cv.md names: the
    // check reads "Redis)", and nothing but taking the brackets off repairs
    // that. Left in its brackets, the line was blocked with FACT_CHECK where
    // the runner before had repaired it, and such a tag was removed as
    // something the CV does not show.
    const arrowed = await tailorOnce(root, tailoredCv({ infra: 'Kubernetes, MLOps (Kafka &rarr; Redis), AWS' }));
    assert.equal(arrowed.result.success, true, arrowed.result.reason);
    assert.equal(infraOf(arrowed.result.html), ' Kubernetes, MLOps, Kafka &rarr; Redis, AWS');
    assert.deepEqual([arrowed.result.removed, arrowed.result.rewritten.map(({ item }) => item)], [[], ['redis)']]);
    const arrowedTag = await tailorOnce(root, tailoredCv({ competencies: ['MLOps', 'MLOps (Kafka &rarr; Redis)', 'Kafka'] }));
    assert.equal(arrowedTag.result.success, true, arrowedTag.result.reason);
    assert.deepEqual(tagsOf(arrowedTag.result.html), ['MLOps', 'MLOps, Kafka &rarr; Redis', 'Kafka']);
    assert.deepEqual(arrowedTag.result.removed, []);
    assert.doesNotMatch(arrowedTag.err, /your CV does not show/);
    // Beside a group nothing repairs, the CV is still better for it by one
    // finding, with a tag that goes either way and without one.
    for (const competencies of [['MLOps', 'Kafka'], ['MLOps', 'Observability', 'Kafka']]) {
      const beside = await tailorOnce(root, tailoredCv({ competencies, infra: 'Kubernetes {5 &times; 3}, MLOps (Kafka &rarr; Redis), AWS' }));
      assert.equal(beside.result.code, 'FACT_CHECK', beside.result.reason);
      assert.equal(infraOf(beside.result.html), ' Kubernetes {5 &times; 3}, MLOps, Kafka &rarr; Redis, AWS');
      assert.equal(beside.result.factCheck.findings.length, 2);
      assert.deepEqual(tagsOf(beside.result.html), ['MLOps', 'Kafka']);
    }
    // Where it leaves as many findings as the repair without it, the group
    // stays as the model wrote it: the arrow leads to a tool cv.md lacks, and
    // "Azure" cannot be cut out of that item any more than "Azure)" can.
    const lacking = await tailorOnce(root, tailoredCv({ competencies: ['MLOps', 'Observability', 'Kafka'], infra: 'Kubernetes, MLOps (Kafka &rarr; Azure), AWS' }));
    assert.equal(lacking.result.code, 'FACT_CHECK', lacking.result.reason);
    assert.equal(infraOf(lacking.result.html), ' Kubernetes, MLOps (Kafka &rarr; Azure), AWS');
    assert.deepEqual([lacking.result.removed, lacking.result.rewritten], [[{ where: 'competency', text: 'Observability' }], []]);
    // Taken apart, such a group can show the check a tool its brackets hid,
    // one that cannot be cut out of its list. Then the repair that leaves the
    // group alone stands: this line passes with the outer brackets gone.
    const hiding = await tailorOnce(root, tailoredCv({ infra: 'Kubernetes, MLOps (TypeScript &vert; Jaeger (Kafka &rarr; Redis) &vert; SQL)' }));
    assert.equal(hiding.result.success, true, hiding.result.reason);
    assert.equal(infraOf(hiding.result.html), ' Kubernetes, MLOps, TypeScript &vert; Jaeger (Kafka &rarr; Redis) &vert; SQL');
  });

  test('a skills line whose list in brackets an entity separates loses the brackets in a run, passes, and is reported once', async () => {
    // entity-separator-1, for a skills line. The ";" of "&middot;" was taken
    // for the end of an entity and nothing more, the group for one that
    // holds no list, and the line was left as it was written: blocked for
    // "mlflow)" although cv.md names both tools, with FACT_CHECK and a
    // second request where the runner before had repaired it.
    const root = makeRoot();
    writeReport(root);
    const { result, err } = await tailorOnce(root, tailoredCv({ infra: 'Kubernetes, Model serving (SageMaker &middot; MLflow), Redis' }));
    assert.equal(result.success, true, result.reason);
    assert.equal(infraOf(result.html), ' Kubernetes, Model serving, SageMaker &middot; MLflow, Redis');
    assert.deepEqual([result.removed, result.rewritten], [[], [{
      where: 'skills', from: 'Kubernetes, Model serving (SageMaker &middot; MLflow), Redis',
      to: 'Kubernetes, Model serving, SageMaker &middot; MLflow, Redis', item: 'mlflow)',
    }]]);
    assert.doesNotMatch(err, /Removed from the CV|your CV does not show/);

    // A line with two lists in brackets, one separated by the sign and one by
    // its entity. Both are taken apart in one go, and the line is reported
    // once, as the CV holds it. With the second list left in its brackets
    // the line was reported twice: first with a text the saved CV did not
    // hold, then for "go)" with a blank taken off and no bracket. The CV
    // stayed blocked for "go)".
    const twoLists = 'MLOps Platform (MLflow · Snowflake), Databases (PostgreSQL &middot; Go), SageMaker and Grafana';
    const chained = await tailorOnce(root, tailoredCv({ infra: twoLists }));
    assert.equal(chained.result.success, true, chained.result.reason);
    const saved = infraOf(chained.result.html);
    assert.doesNotMatch(saved, /[()]/);
    assert.deepEqual(chained.result.rewritten.map(({ from, to }) => [from, to]), [[twoLists, saved.trim()]]);
    // What goes are the two words cv.md does not have, each by itself.
    assert.deepEqual(chained.result.removed, [{ where: 'skills', text: 'Snowflake' }, { where: 'skills', text: 'Databases' }]);
    // The item cut out from behind the "·" does not leave its blank in front of the comma.
    assert.doesNotMatch(saved, /\s,/);

    // A group the check splits and the runner leaves alone: "{5 &times; 3}"
    // gives the check the piece "3}", which nothing can remove. The line is
    // reported once, as the CV holds it, and the finding goes to the second
    // request. Asked to lose its brackets in every round, the line came back
    // with a blank less and nothing else changed, and it was reported as
    // rewritten a second time, from a text the reply did not hold.
    const written = 'Databases (Kubeflow · Redis · Terraform), Kubernetes {5 &times; 3}, AWS, SageMaker';
    const partly = await tailorOnce(root, tailoredCv({ infra: written }));
    assert.equal(partly.result.code, 'FACT_CHECK', partly.result.reason);
    const held = infraOf(partly.result.html);
    assert.equal(held, ' Kubeflow · Redis, Kubernetes {5 &times; 3}, AWS, SageMaker');
    assert.deepEqual(partly.result.rewritten.map(({ from, to }) => [from, to]), [[written, held.trim()]]);
    assert.deepEqual(partly.result.removed, [{ where: 'skills', text: 'Databases' }, { where: 'skills', text: 'Terraform' }]);

    // A line with no bracket to take off comes back as the model wrote it,
    // its blanks included. Here the model put two blanks behind a comma, and
    // nothing is cut from the line. Tidied all the same, the line lost a
    // blank and was reported as written without brackets, from and to the
    // same text, while every bracket stood where it was.
    const spaced = 'Kubernetes {5 &times; 3},  AWS, SageMaker';
    const untouched = await tailorOnce(root, tailoredCv({ competencies: ['MLOps', 'Observability', 'Kafka'], infra: spaced }));
    assert.equal(untouched.result.code, 'FACT_CHECK', untouched.result.reason);
    assert.equal(infraOf(untouched.result.html), ` ${spaced}`);
    assert.deepEqual([untouched.result.removed, untouched.result.rewritten], [[{ where: 'competency', text: 'Observability' }], []]);
    assert.doesNotMatch(untouched.err, /Written without brackets/);
    // Two items cut out from behind a "·" on one line: neither leaves its blank.
    const twice = await tailorOnce(root, tailoredCv({ infra: 'Kubeflow · Terraform, Redis · Docker, AWS' }));
    assert.equal(twice.result.success, true, twice.result.reason);
    assert.equal(infraOf(twice.result.html), ' Kubeflow, Redis, AWS');
  });

  test('a competency tag is read as the fact check reads it: its brackets go, or the tag goes as a whole', async () => {
    // The check's own functions say which tools it reads in a tag. A tag with
    // a list in brackets gives it "MLflow)": a tool cv.md names, spelled as no
    // CV has it. Of 18 real CVs, 5 were blocked for that and nothing else, and
    // while such a tag was removed the run said the CV did not show it.
    const reading = await realReading();
    const tag = (text) => `<span class="competency-tag">${text}</span>`;
    const html = [
      tag('MLOps Platform (SageMaker, Kubeflow, MLflow)'),
      tag('Kubernetes and OpenTelemetry'),
      // `&` is no separator to the check, however the entity is spelled, so
      // neither of these names "feature store" or "opentelemetry" by itself.
      tag('Model Registry &amp; Feature Store'),
      tag('Grafana &#38; OpenTelemetry'),
      tag('Streaming Systems (Kafka)'),
    ].join('\n');
    const repaired = pruneRejectedEntries(html, ['mlflow)', 'opentelemetry', 'feature store'], reading);
    assert.equal(repaired.html, [
      tag('MLOps Platform, SageMaker, Kubeflow, MLflow'),
      tag('Model Registry &amp; Feature Store'),
      tag('Grafana &#38; OpenTelemetry'),
      tag('Streaming Systems (Kafka)'),
    ].join('\n'));
    assert.deepEqual(repaired.removed, [{ where: 'competency', text: 'Kubernetes and OpenTelemetry', item: 'opentelemetry' }]);
    assert.deepEqual(repaired.rewritten, [{
      where: 'competency', from: 'MLOps Platform (SageMaker, Kubeflow, MLflow)', to: 'MLOps Platform, SageMaker, Kubeflow, MLflow', item: 'mlflow)',
    }]);

    // Every way the check separates the tools of a tag, also one an entity spells.
    for (const text of [
      'Kafka, OpenTelemetry', 'Kafka; OpenTelemetry', 'Kafka | OpenTelemetry', 'Kafka • OpenTelemetry', 'Kafka · OpenTelemetry',
      'Kafka or OpenTelemetry', 'Kafka,and OpenTelemetry', 'OpenTelemetry &middot; Kafka', 'OpenTelemetry&#44; Kafka',
    ]) {
      const result = pruneRejectedEntries(`${tag('MLOps')}\n${tag(text)}`, ['opentelemetry'], reading);
      assert.equal(result.html, `${tag('MLOps')}\n`, `the check reads OpenTelemetry in "${text}", and the tag stayed`);
      assert.equal(result.removed[0].item, 'opentelemetry', text);
    }

    // Through the real check: such a CV is repaired with one request, keeps the tag, and is told why.
    const root = makeRoot();
    writeReport(root);
    const { ui, written } = quietUi();
    const bracketed = tailoredCv({ competencies: ['MLOps', 'MLOps Platform (SageMaker, Kubeflow, MLflow)', 'Kafka'] });
    const client = fakeClient({ reply: bracketed });
    const result = await tailorCv('001', { cfg, client, ui, dataRoot: root, save: false });
    assert.equal(result.success, true, result.reason);
    assert.equal(client.calls.filter((call) => call === 'chat').length, 1, 'the model was asked again for a tag the runner can repair');
    assert.deepEqual(result.removed, []);
    assert.deepEqual(result.rewritten, [{
      where: 'competency', from: 'MLOps Platform (SageMaker, Kubeflow, MLflow)', to: 'MLOps Platform, SageMaker, Kubeflow, MLflow', item: 'mlflow)',
    }]);
    assert.equal(result.html, asSaved(tailoredCv({ competencies: ['MLOps', 'MLOps Platform, SageMaker, Kubeflow, MLflow', 'Kafka'] })));
    assert.ok(written.err.includes('Written without brackets, because the fact check reads a list in brackets item by item and rejected a piece '
      + 'of it: "MLOps Platform (SageMaker, Kubeflow, MLflow)" is now "MLOps Platform, SageMaker, Kubeflow, MLflow" (competency, for "mlflow)").'), written.err);
    // cv.md names all three tools: nothing here is something the CV does not show.
    assert.doesNotMatch(written.err, /Removed from the CV|your CV does not show/);
  });

  test('a list in brackets with a tool cv.md does not have loses its brackets, then the tag, and is named as it was written', async () => {
    const root = makeRoot();
    writeReport(root);
    const { ui, written } = quietUi();
    const client = fakeClient({
      reply: tailoredCv({
        competencies: ['MLOps', 'Monitoring (Grafana, OpenTelemetry)', 'Kafka'],
        infra: 'Kubernetes, Kafka, Model serving (SageMaker, Kubeflow, MLflow), Redis',
      }),
    });
    const result = await tailorCv('001', { cfg, client, ui, dataRoot: root, save: false });
    assert.equal(result.success, true, result.reason);
    assert.equal(client.calls.filter((call) => call === 'chat').length, 1);
    // The tag: without its brackets the check reads OpenTelemetry, which cv.md does not name.
    assert.deepEqual(result.removed, [{ where: 'competency', text: 'Monitoring (Grafana, OpenTelemetry)', item: 'opentelemetry' }]);
    // The skills line: cv.md names all three tools, so the line only loses its brackets.
    assert.deepEqual(result.rewritten, [{
      where: 'skills', from: 'Kubernetes, Kafka, Model serving (SageMaker, Kubeflow, MLflow), Redis',
      to: 'Kubernetes, Kafka, Model serving, SageMaker, Kubeflow, MLflow, Redis', item: 'mlflow)',
    }]);
    assert.equal(result.html, asSaved(tailoredCv({
      competencies: ['MLOps', 'Kafka'], infra: 'Kubernetes, Kafka, Model serving, SageMaker, Kubeflow, MLflow, Redis',
    })));
    assert.ok(written.err.includes('Monitoring (Grafana, OpenTelemetry) (competency, for "opentelemetry")'), written.err);
    assert.match(written.err, /The check looks for the exact words\./);
  });

  test('a bracket that holds no list stays as it is written when a list on the same line loses its brackets', async () => {
    // flat-1. Every bracket of a rewritten line used to go, and the CV the
    // check then passed listed "expert", "basic" and "3.11" among the skills.
    const root = makeRoot();
    writeReport(root);
    const { result } = await tailorOnce(root, tailoredCv({ infra: 'Python (expert), Go (basic), Model serving (SageMaker, MLflow)' }));
    assert.equal(result.success, true, result.reason);
    assert.equal(infraOf(result.html), ' Python (expert), Go (basic), Model serving, SageMaker, MLflow');

    // A cv.md that has a version in brackets beside a group; the reply gives the group in another order.
    const grouped = makeRoot();
    writeReport(grouped);
    writeFileSync(join(grouped, 'cv.md'), read(grouped, 'cv.md')
      .replace('- **Infra:** Kubernetes, Kafka, Redis, PostgreSQL, AWS', '- **Infra:** Cloud (AWS, GCP), Python (3.11), Kubernetes, Kafka, Redis, PostgreSQL'));
    assert.ok(read(grouped, 'cv.md').includes('Python (3.11)'), 'the example CV no longer has the skills line this test rewrites');
    const reordered = await tailorOnce(grouped, tailoredCv({ infra: 'Cloud (GCP, AWS), Python (3.11), Kubernetes, Kafka' }));
    assert.equal(reordered.result.success, true, reordered.result.reason);
    assert.equal(infraOf(reordered.result.html), ' Cloud, GCP, AWS, Python (3.11), Kubernetes, Kafka');

    // The other side of it: a version cv.md does not state is not split off
    // its tool. The item goes as one, with a list in brackets beside it as
    // without one, where taking the line apart had kept "Python" and removed
    // "3.11".
    for (const [infra, left] of [
      ['Python (3.11), Kubernetes, Kafka', ' Kubernetes, Kafka'],
      ['Python (3.11), Cloud (AWS, Azure), Kafka', ' AWS, Kafka'],
    ]) {
      const versioned = await tailorOnce(root, tailoredCv({ infra }));
      assert.equal(versioned.result.success, true, versioned.result.reason);
      assert.equal(infraOf(versioned.result.html), left);
      assert.ok(
        versioned.result.removed.some(({ where, text }) => where === 'skills' && text === 'Python (3.11)'),
        JSON.stringify(versioned.result.removed),
      );
    }
  });

  test('an item is not cut out of a list in square or curly brackets when that would leave the bracket open', async () => {
    // square-1. Only round brackets were counted: "MLflow]" went, and the CV
    // was saved as passing with "Model serving [SageMaker, Kubeflow, Redis".
    const reading = await realReading();
    for (const [open, close] of [['[', ']'], ['{', '}']]) {
      const html = skillLine('Infra', `Kubernetes, Model serving ${open}SageMaker, Kubeflow, MLflow${close}, Redis`);
      const repaired = pruneRejectedEntries(html, [`mlflow${close}`], reading);
      const line = />([^<]*)<\/div>/.exec(repaired.html)[1];
      assert.equal(line.split(open).length, line.split(close).length, `the bracket is left open: ${line}`);
      // Read like a list in round brackets: the brackets go, and every item stays.
      assert.equal(line, ' Kubernetes, Model serving, SageMaker, Kubeflow, MLflow, Redis');
      assert.deepEqual(repaired.removed, []);
      // With the brackets left where they are, the piece stays in its list.
      assert.deepEqual(
        pruneRejectedEntries(html, [`mlflow${close}`], { ...reading, keepBrackets: true }),
        { html, removed: [], rewritten: [], cuts: [] },
      );
    }
  });

  test('a tag with a list in square or curly brackets whose tools cv.md names loses the brackets, and is not removed as something the CV does not show', async () => {
    // square-1, for a tag. It was removed as a whole, and the run said of
    // three tools cv.md names that the CV does not show them.
    const root = makeRoot();
    writeReport(root);
    for (const [open, close] of [['[', ']'], ['{', '}']]) {
      const bracketed = tailoredCv({ competencies: ['MLOps', `MLOps Platform ${open}SageMaker, Kubeflow, MLflow${close}`, 'Kafka'] });
      const { result, err } = await tailorOnce(root, bracketed);
      assert.equal(result.success, true, result.reason);
      assert.doesNotMatch(err, /your CV does not show/);
      assert.deepEqual(result.removed, []);
      assert.deepEqual(tagsOf(result.html), ['MLOps', 'MLOps Platform, SageMaker, Kubeflow, MLflow', 'Kafka']);
    }
  });

  test('a list in the full-width brackets of Chinese and Japanese text is repaired like one in round brackets', async () => {
    // What square-1 is for square and curly brackets, for the ones a CV in
    // Chinese or Japanese groups its tools with. The runner looked for ASCII
    // brackets in an entry as it is written.
    const root = makeRoot();
    writeReport(root);
    // cv.md names SageMaker, Kubeflow and MLflow. The check folds the brackets to ASCII ones and rejects "mlflow)".
    const tag = await tailorOnce(root, tailoredCv({ competencies: ['MLOps', '机器学习平台（SageMaker，Kubeflow，MLflow）', 'Kafka'] }));
    assert.equal(tag.result.success, true, tag.result.reason);
    assert.deepEqual(tag.result.removed, [], 'the tag was removed although cv.md names every tool in it');
    assert.doesNotMatch(tag.err, /your CV does not show/);
    assert.deepEqual(tagsOf(tag.result.html), ['MLOps', '机器学习平台, SageMaker，Kubeflow，MLflow', 'Kafka']);

    const list = 'Kubernetes, Model serving（SageMaker, Kubeflow, MLflow）, Redis';
    const line = await tailorOnce(root, tailoredCv({ infra: list }));
    const saved = infraOf(line.result.html);
    assert.equal(saved.split('（').length, saved.split('）').length, `the bracket is left open: ${saved}`);
    assert.equal(saved, ' Kubernetes, Model serving, SageMaker, Kubeflow, MLflow, Redis');
    assert.deepEqual([line.result.success, line.result.removed], [true, []]);
    // With the brackets left where they are, "MLflow）" is not cut out. As it
    // is written its brackets balance, for it holds no ASCII one: what counts
    // as well is how the check reads it.
    const html = skillLine('Infra', list);
    assert.deepEqual(
      pruneRejectedEntries(html, ['mlflow)'], { ...await realReading(), keepBrackets: true }),
      { html, removed: [], rewritten: [], cuts: [] },
    );
  });

  test('a skills line with markup or a bare "<" in its list is left as it is, and does not go with the line after it', async () => {
    // span-1. The category of such a line ran on into the next line, the two
    // were read as one, and both went when the second lost its only item: a
    // line the check had nothing against left a CV that then passed.
    const reading = await realReading();
    const first = skillLine('Infra', 'Kubernetes, <strong>Kafka</strong>, Redis');
    const languages = skillLine('Languages', 'Python, Go');
    const html = [first, skillLine('Tracing', 'OpenTelemetry'), languages].join('\n');
    const repaired = pruneRejectedEntries(html, ['opentelemetry'], reading);
    assert.ok(repaired.html.includes(first), `the Infra line is gone: ${repaired.html}`);
    assert.ok(!repaired.html.includes('Tracing'), repaired.html);
    assert.deepEqual(repaired.removed, [{ where: 'skills', text: 'OpenTelemetry' }]);
    assert.equal(repaired.html, [first, '', languages].join('\n'));

    // No element is needed: a bare "<" in an item did the same.
    const stats = skillLine('Stats', 'A/B tests (p<0.05), SQL');
    const bare = pruneRejectedEntries([stats, skillLine('Tracing', 'OpenTelemetry')].join('\n'), ['opentelemetry'], reading);
    assert.equal(bare.html, `${stats}\n`);
    assert.deepEqual(bare.removed, [{ where: 'skills', text: 'OpenTelemetry' }]);

    // With markup in two lines running, a third that lost every item took all three.
    const linked = skillLine('Infra', 'Kubernetes, <a href="https://kafka.apache.org">Kafka</a>, Redis');
    const stressed = skillLine('Data', 'SQL, <em>Spark</em>');
    const three = pruneRejectedEntries([linked, stressed, skillLine('Tracing', 'OpenTelemetry, Jaeger')].join('\n'), ['opentelemetry', 'jaeger'], reading);
    assert.equal(three.html, `${linked}\n${stressed}\n`);
    assert.deepEqual(three.removed, [{ where: 'skills', text: 'OpenTelemetry' }, { where: 'skills', text: 'Jaeger' }]);

    // Markup in the category is not markup in the list: such a line is still repaired.
    const bold = '<div class="skill-item"><span class="skill-category"><strong>Infra:</strong></span> Kubernetes, OpenTelemetry</div>';
    assert.equal(pruneRejectedEntries(bold, ['opentelemetry'], reading).html, bold.replace(', OpenTelemetry', ''));
    // A line that is left alone keeps what the check rejected in it: that finding is for the second request.
    const own = skillLine('Infra', 'Kubernetes, <strong>OpenTelemetry</strong>, Redis');
    assert.deepEqual(pruneRejectedEntries(own, ['opentelemetry'], reading), { html: own, removed: [], rewritten: [], cuts: [] });

    // In a run, through the real check: the line with markup is in the CV that comes back, and the CV passes.
    const root = makeRoot();
    writeReport(root);
    const last = skillLine('Languages', 'Python, Go, TypeScript, SQL');
    const reply = tailoredCv({ infra: 'Kubernetes, <strong>Kafka</strong>, Redis' }).replace(last, `${skillLine('Tracing', 'OpenTelemetry')}\n    ${last}`);
    assert.ok(reply.includes('Tracing'), 'tailoredCv() no longer has the skills line this test puts another in front of');
    const { result } = await tailorOnce(root, reply);
    assert.equal(result.success, true, result.reason);
    assert.ok(result.html.includes(first), 'the Infra line is gone from the CV');
    assert.ok(!result.html.includes('Tracing'), 'the line that lost its only item is still in the CV');
    assert.deepEqual(result.removed, [{ where: 'skills', text: 'OpenTelemetry' }]);
  });

  test('what the runner can remove stays removed when a tool the brackets hid cannot be cut out of its list', async () => {
    // discard-1. Without its brackets "Cloud (AWS, GCP, and Azure)" shows the
    // check "Azure", which cannot be cut out beside its "and". The repair
    // counted that as a finding the reply did not have, gave itself up, and
    // handed the reply back as it came: with the tag and the tools it could
    // have removed, and "4 findings left after removing what could be removed".
    const root = makeRoot();
    writeReport(root);
    const oxford = tailoredCv({ competencies: ['MLOps', 'Observability', 'Kafka'], infra: 'Kubernetes, OpenTelemetry, Cloud (AWS, GCP, and Azure), Kafka' });
    const { result, err } = await tailorOnce(root, oxford);
    assert.equal(result.code, 'FACT_CHECK');
    assert.deepEqual(tagsOf(result.html), ['MLOps', 'Kafka'], 'the tag cv.md does not support is back in the CV');
    assert.equal(result.factCheck.findings.length, 1, JSON.stringify(result.factCheck.findings));
    assert.ok(
      result.removed.some(({ text }) => text === 'Observability') && result.removed.some(({ text }) => text === 'OpenTelemetry'),
      JSON.stringify(result.removed),
    );
    assert.match(err, /\(1 finding\)/);
    // "azure)" in the reply and "azure" in the repaired CV are one finding, so
    // the line keeps its repair, and the finding is named as the tool it is.
    assert.equal(infraOf(result.html), ' Kubernetes, AWS, and Azure, Kafka');
    assert.deepEqual(result.factCheck.findings, ['non-metric fact absent from sources: tool=azure']);
    // The second request is told what is left, and nothing the runner had already removed.
    const { ui, written } = quietUi();
    const client = fakeClient({ reply: (n) => (n === 0 ? oxford : CV_HTML) });
    const asked = await tailorCv('001', { cfg, client, ui, dataRoot: root, save: false });
    assert.equal(asked.success, true, asked.reason);
    assert.match(written.err, /The fact check blocks this CV \(1 finding left after removing what could be removed\)/);
    const [firstRequest, secondRequest] = client.requests.map((request) => request.user);
    const feedback = secondRequest.slice(firstRequest.length);
    assert.ok(feedback.includes('\n- non-metric fact absent from sources: tool=azure\n'), feedback);
    assert.doesNotMatch(feedback, /observability|opentelemetry|gcp/i);
    // Whichever kind of bracket the list was in: "azure]" and "azure}" in the
    // reply are that same finding too. Compared with round brackets only, the
    // repair of such a line was given up for the one that keeps the brackets.
    for (const [open, close] of [['[', ']'], ['{', '}'], ['（', '）']]) {
      const other = await tailorOnce(root, tailoredCv({
        competencies: ['MLOps', 'Observability', 'Kafka'], infra: `Kubernetes, OpenTelemetry, Cloud ${open}AWS, GCP, and Azure${close}, Kafka`,
      }));
      assert.equal(infraOf(other.result.html), ' Kubernetes, AWS, and Azure, Kafka', open);
      assert.deepEqual(other.result.factCheck.findings, ['non-metric fact absent from sources: tool=azure'], open);
      assert.deepEqual(tagsOf(other.result.html), ['MLOps', 'Kafka'], open);
    }

    // The tool cv.md lacks stands before "and", where the check did not read it in the reply at all.
    const hidden = tailoredCv({ competencies: ['MLOps', 'Observability', 'Kafka'], infra: 'Kubernetes, Cloud (Azure and AWS), Kafka' });
    const second = await tailorOnce(root, hidden);
    assert.deepEqual(tagsOf(second.result.html), ['MLOps', 'Kafka'], 'hidden tool: the tag cv.md does not support is back in the CV');
    assert.equal(second.result.factCheck.findings.length, 1, JSON.stringify(second.result.factCheck.findings));
    // Kept is the repair that takes no bracket off: the line is as the reply had it, and the tag is gone.
    assert.equal(infraOf(second.result.html), ' Kubernetes, Cloud (Azure and AWS), Kafka');
    assert.deepEqual([second.result.removed, second.result.rewritten], [[{ where: 'competency', text: 'Observability' }], []]);

    // That repair leaves the brackets of a skills line alone, not those of a
    // tag: a tag can always go as a whole in the next round. Left with its
    // brackets, a tag whose tools cv.md names was removed for "mlflow)", and
    // the run said the CV does not show what the posting asks for.
    const both = await tailorOnce(root, tailoredCv({
      competencies: ['MLOps', 'MLOps Platform (SageMaker, Kubeflow, MLflow)', 'Kafka'], infra: 'Kubernetes, Cloud (Azure and AWS), Kafka',
    }));
    assert.deepEqual(tagsOf(both.result.html), ['MLOps', 'MLOps Platform, SageMaker, Kubeflow, MLflow', 'Kafka']);
    assert.equal(infraOf(both.result.html), ' Kubernetes, Cloud (Azure and AWS), Kafka');
    assert.deepEqual(both.result.removed, []);
    assert.deepEqual(both.result.rewritten.map(({ where, to }) => [where, to]), [['competency', 'MLOps Platform, SageMaker, Kubeflow, MLflow']]);
    assert.doesNotMatch(both.err, /your CV does not show/);
    assert.deepEqual(both.result.factCheck.findings, ['non-metric fact absent from sources: tool=aws)']);
    // A tag with a tool cv.md does not name still goes, in that repair too, and is named for the tool.
    const lacking = await tailorOnce(root, tailoredCv({
      competencies: ['MLOps', 'Monitoring (Grafana, OpenTelemetry)', 'Kafka'], infra: 'Kubernetes, Cloud (Azure and AWS), Kafka',
    }));
    assert.deepEqual(tagsOf(lacking.result.html), ['MLOps', 'Kafka']);
    assert.deepEqual(lacking.result.removed.map(({ where, text }) => [where, text]), [['competency', 'Monitoring (Grafana, OpenTelemetry)']]);
  });

  test('a repair that leaves as many findings as the reply had is not kept, and nothing is reported as removed', async () => {
    // unpinned-repair-1. The tool cv.md lacks stands twice on the line: by
    // itself, where it can be cut out, and behind an "and", where it cannot.
    // Cut out once, it is still on the line and still the one finding. The
    // rule keeps a repair only when it leaves fewer findings, and no test
    // held it: kept for as many, the run said the tool was removed from a CV
    // that still named it, and every suite passed.
    const root = makeRoot();
    writeReport(root);
    const infra = 'SageMaker, OpenTelemetry, SageMaker and OpenTelemetry, Go, SageMaker';
    const reply = tailoredCv({ infra });
    const { result, err } = await tailorOnce(root, reply);
    assert.equal(result.code, 'FACT_CHECK');
    assert.deepEqual(result.factCheck.findings, ['non-metric fact absent from sources: tool=opentelemetry']);
    // The reply comes back as it was, and the run says of nothing that it was removed.
    assert.deepEqual([result.removed, result.rewritten], [[], []]);
    assert.equal(infraOf(result.html), ` ${infra}`);
    assert.equal(result.html, asSaved(reply));
    assert.doesNotMatch(err, /Removed from the CV/);
  });

  test('a rewritten entry is reported as the CV holds it, and a line that lost every item is reported as removed', async () => {
    // to-1. The receipt and the note gave a rewritten line as the first round
    // left it: "is now Kubernetes, Cloud, AWS, Azure, Kafka" for a line the CV
    // holds as "Kubernetes, AWS, Kafka", and "is now Cloud, GCP, Azure" for a
    // line that is not in the CV at all.
    const root = makeRoot();
    writeReport(root);
    const { result, err } = await tailorOnce(root, tailoredCv({ infra: 'Kubernetes, Cloud (AWS, Azure), Kafka' }));
    assert.equal(result.success, true, result.reason);
    assert.equal(infraOf(result.html), ' Kubernetes, AWS, Kafka');
    for (const { to } of result.rewritten) {
      assert.ok(result.html.includes(`>${to}<`) || result.html.includes(`> ${to}<`), `"${to}" is not in the CV: ${infraOf(result.html)}`);
    }
    assert.doesNotMatch(err, /is now "Kubernetes, Cloud, AWS, Azure, Kafka"/);
    assert.deepEqual(result.rewritten, [{ where: 'skills', from: 'Kubernetes, Cloud (AWS, Azure), Kafka', to: 'Kubernetes, AWS, Kafka', item: 'azure)' }]);
    assert.ok(err.includes('"Kubernetes, Cloud (AWS, Azure), Kafka" is now "Kubernetes, AWS, Kafka" (skills, for "azure)").'), err);

    // Every item goes: the line was removed, not written without brackets.
    const emptied = tailoredCv({ infra: 'Cloud (GCP, Azure)' });
    const gone = await tailorOnce(root, emptied);
    assert.equal(infraOf(gone.result.html), undefined);
    assert.deepEqual(gone.result.rewritten, [], 'a line that is no longer in the CV is reported as written without brackets');
    assert.doesNotMatch(gone.err, /Written without brackets/);
    // Once, as it was written, and not item by item.
    assert.deepEqual(gone.result.removed, [{ where: 'skills', text: 'Cloud (GCP, Azure)' }]);
    assert.match(gone.err, /Removed from the CV, because the fact check found it in neither cv\.md nor article-digest\.md: Cloud \(GCP, Azure\) \(skills\)\./);

    // The summary of a run counts what the receipt lists: it said "3 entries
    // removed by the fact check, 1 written without brackets" for this line.
    const fresh = makeRoot();
    await withStub({ reply: replies({ tailor: emptied }) }, async (stub) => {
      const run = await cli(['run', '--file', join(fresh, 'jd.txt'), '--quiet', '--no-merge', '--url', stub.url], { root: fresh });
      assertExit(run, 0);
      assert.match(run.stdout, /^\s+tailor\s+done\s+output\/\S+\.html \(1 entry removed by the fact check\)$/m);
      assert.doesNotMatch(run.stdout, /written without brackets/);
    });
  });

  test('a link to an address none of the candidate\'s files holds is taken off, and its text stays', async () => {
    // cv.md names two projects and no address for them. 4 of the first 7 real
    // CVs linked github.com/<user>/<project>, made up from the profile's
    // GitHub address and the project's name. No check of generate-pdf.mjs
    // reads an address.
    const own = [
      'GitHub: github.com/alexchen', 'LinkedIn: linkedin.com/in/alexchen', 'Portfolio: https://www.alexchen.dev/',
      'Paper: https://example.org/papers/Evals-2024.pdf', 'alex@example.com',
    ].join('\n');
    const html = [
      '<a href="https://linkedin.com/in/alexchen">linkedin.com/in/alexchen</a>',
      '<a class="project-title" href="https://github.com/alexchen/llm-eval-toolkit">LLM Eval Toolkit</a>',
      // A shorter address is not supported by a longer one in the files.
      '<a href="https://github.com/alex">short</a>',
      // Scheme, "www.", upper case, a trailing slash and a fragment make no difference.
      '<a href="HTTPS://WWW.AlexChen.dev">site</a> <a href="https://example.org/papers/evals-2024.pdf#page=2">paper</a>',
      // Only web addresses are judged here.
      '<a href="mailto:someone@else.example">mail</a> <a href="tel:+1-555-0123">phone</a> <a href="#top">top</a>',
      '<a href=\'https://lumenfield.example/careers\'><strong>Lumenfield</strong></a>',
      '<a href="https://github.com/alexchen/llm-eval-toolkit">again</a>',
    ].join('\n');
    assert.deepEqual(withoutUnsupportedLinks(html, own), {
      html: [
        '<a href="https://linkedin.com/in/alexchen">linkedin.com/in/alexchen</a>',
        'LLM Eval Toolkit',
        'short',
        '<a href="HTTPS://WWW.AlexChen.dev">site</a> <a href="https://example.org/papers/evals-2024.pdf#page=2">paper</a>',
        '<a href="mailto:someone@else.example">mail</a> <a href="tel:+1-555-0123">phone</a> <a href="#top">top</a>',
        '<strong>Lumenfield</strong>',
        'again',
      ].join('\n'),
      // Each address once, in the order it was met.
      unlinked: ['https://github.com/alexchen/llm-eval-toolkit', 'https://github.com/alex', 'https://lumenfield.example/careers'],
    });
    assert.deepEqual(withoutUnsupportedLinks('<p>no links</p>', own), { html: '<p>no links</p>', unlinked: [] });

    // In a run: the example CV has github.com/alexchen and no project address.
    const root = makeRoot();
    const report = writeReport(root);
    const linked = tailoredCv({
      extra: '\n  <div class="section"><div class="section-title">Projects</div>'
        + '<a href="https://github.com/alexchen/llm-eval-toolkit">LLM Eval Toolkit</a> and <a href="https://github.com/alexchen">my GitHub</a></div>',
    });
    const { ui, written } = quietUi();
    const client = fakeClient({ reply: linked });
    const result = await tailorCv('001', { cfg, client, ui, dataRoot: root });
    assert.equal(result.success, true, result.reason);
    assert.deepEqual(result.unlinked, ['https://github.com/alexchen/llm-eval-toolkit']);
    const saved = read(root, 'output', report.html);
    assert.ok(saved.includes('<div class="section-title">Projects</div>LLM Eval Toolkit and <a href="https://github.com/alexchen">my GitHub</a></div>'), saved.slice(-700));
    assert.match(written.err, /1 link taken off the CV, because its address is in none of your files \(cv\.md, config\/profile\.yml, article-digest\.md\): https:\/\/github\.com\/alexchen\/llm-eval-toolkit\./);
    assert.equal(client.requests.length, 1, 'a link the runner can take off is no reason to ask again');
  });

  test('in `run` a blocked CV gets no PDF step, the answers are still drafted, and the run exits 1', async () => {
    const root = makeRoot();
    const stem = `001-northwind-robotics-${TODAY}`;
    await withStub({ reply: replies({ tailor: INFLATED_CV }) }, async (stub) => {
      const run = await cli(['run', '--file', join(root, 'jd.txt'), '--quiet', '--url', stub.url], { root });
      assertExit(run, 1);
      assert.deepEqual(stub.generations().map(asksFor), ['evaluate', 'tailor', 'tailor', 'apply']);
      // The CV is kept and the answers are drafted; generate-pdf.mjs was never started.
      assert.deepEqual(listing(join(root, 'output')), [`apply-${stem}.md`, `cv-jane-smith-${stem}.html`]);
      assert.doesNotMatch(run.stderr, /Rendering/);
      // The way on names the command for this report. "The command again"
      // would be `run`, which evaluates the posting a second time.
      assert.match(run.stderr, /Run `node ollama-runner\.mjs tailor 001` again/);
      assert.doesNotMatch(run.stderr, /Run the command again/);
      assert.equal(trackerRow(root, 'Northwind Robotics')[6], '❌');
      // Said in the summary, not left out of it: a summary without a pdf row read as if none had been asked for.
      assert.match(run.stdout, /^\s+tailor\s+failed\s+The fact check blocks this CV/m);
      assert.match(run.stdout, /^\s+pdf\s+not run\s+there is no CV that passed its checks$/m);
      assert.match(run.stdout, /^\s+apply\s+done\s+output\/apply-001-/m);
    });
  });

  test('in `pipeline --pdf` a blocked CV gets no PDF step, and the others are rendered', async () => {
    const root = makePipelineRoot();
    const tailor = (seen) => (seen.body.messages[1].content.includes('Acme is hiring') ? INFLATED_CV : KEYWORDED_CV);
    await withStub({ reply: replies({ evaluate: (seen) => evaluationFor(seen), tailor }) }, async (stub) => {
      const run = await cli(['pipeline', '--pdf', '--json', '--no-merge', '--url', stub.url], { root, pages: PAGES });
      assertExit(run, 1);
      const steps = Object.fromEntries(JSON.parse(run.stdout).done.map((entry) => [entry.company, entry.steps]));
      assert.deepEqual(steps.Acme.map((step) => [step.step, step.success, step.code ?? null]), [['tailor', false, 'FACT_CHECK']]);
      // The failed step still says where the kept file is, and what the check found.
      assert.equal(steps.Acme[0].path, `output/cv-jane-smith-001-acme-${TODAY}.html`);
      assert.deepEqual(steps.Acme[0].factCheck.findings, ['metric-like claim absent from sources: 73%']);
      assert.deepEqual(steps.Initech.map((step) => [step.step, step.success]), [['tailor', true], ['pdf', true]]);
      assert.deepEqual(steps.Initech[0].removed.map(({ text }) => text), ['Observability', 'OpenTelemetry']);
      assert.deepEqual(listing(join(root, 'output')), [
        `cv-jane-smith-001-acme-${TODAY}.html`, `cv-jane-smith-003-initech-${TODAY}.html`, `cv-jane-smith-003-initech-${TODAY}.pdf`,
      ]);
    });
  });
});

// ---------------------------------------------------------------------------
// What a tailored CV may hold
// ---------------------------------------------------------------------------

describe('what a tailored CV may hold', { concurrency: 6 }, () => {
  const cfg = readOllamaConfig({});

  test('markup that loads, embeds, runs or sends something is refused, wherever it stands', () => {
    for (const [markup, named] of [
      // Puts the text of a local file into the PDF, invisible on the page.
      ['<iframe src="../.env" style="position:absolute;transform:scale(0.01);opacity:0.05"></iframe>', 'a <iframe> element'],
      ['<object data="../data/applications.md"></object>', 'a <object> element'],
      // On Windows, a file: address with a host is a share on that host.
      ['<img src="file://HOST/share/x.png" width="1" height="1">', 'a src attribute on <img>'],
      ['<div style="background:url(file://HOST/share/x.png)">x</div>', 'a url(file://HOST/share/x.png) value in its styles'],
      ['<div style="background:&#117;rl(file://HOST/x.png)">x</div>', 'a character reference in a style attribute on <div>'],
      // What a browser that opens the saved file would run or fetch.
      ['<script>fetch("https://collector.invalid/")</script>', 'a <script> element'],
      ['<link rel="stylesheet" href="https://fonts.collector.invalid/css">', 'a <link> element'],
      ['<img src="https://collector.invalid/p.gif?n=Alex%20Chen" width="1" height="1">', 'a src attribute on <img>'],
      ['<IMG SRCSET="https://collector.invalid/p.gif 1x">', 'a srcset attribute on <img>'],
      ['<p onclick="x()">x</p>', 'an onclick handler on <p>'],
      ['<meta http-equiv="refresh" content="0;url=../.env">', 'a <meta http-equiv> element'],
      ['<form action="https://collector.invalid/"><input name="q"></form>', 'a <form> element'],
      ['<a href="javascript:alert(1)">x</a>', 'an href that is not a web, mail or phone link ("javascript:alert(1)")'],
      ['<a href="file://HOST/share/cv.pdf">x</a>', 'an href that is not a web, mail or phone link ("file://HOST/share/cv.pdf")'],
      // A comment does not hide it: where a comment ends is the browser's call.
      ['<!--><img src="file://HOST/share/x.png"><!-- -->', 'a src attribute on <img>'],
    ]) {
      const html = tailoredCv({ extra: `\n  ${markup}` });
      assert.equal(refusedMarkup(html), named, markup);
      assert.equal(tailoredCvProblem(html), `the HTML holds ${named}, which the CV template does not have`, markup);
    }
    for (const [css, named] of [
      ['@import url("https://collector.invalid/a.css");', 'an @import rule in its styles'],
      ['body { background: image-set("file://HOST/x.png" 1x); }', 'an image-set() value in its styles'],
      // An escape spells the same function.
      ['body { background: \\75 rl(file://HOST/x.png); }', 'a url(file://HOST/x.png) value in its styles'],
    ]) {
      assert.equal(refusedMarkup(CV_HTML.replace('<style>', `<style>\n  ${css}`)), named, css);
    }
    // A stylesheet that is never closed runs to the end of the document.
    assert.equal(
      refusedMarkup(CV_HTML.replace('</body>', '<style>body { background: url(file://HOST/x.png) }</body>')),
      'a url(file://HOST/x.png) value in its styles',
    );
  });

  test('what the template holds is not refused: its own markup with every slot filled, links, an inline image', () => {
    const template = readFileSync(join(ROOT, 'templates', 'cv-template.html'), 'utf-8');
    const filled = template.replace(/\{\{([A-Z_]+)\}\}/g, (_slot, name) => {
      if (name.endsWith('_URL')) return 'https://www.linkedin.com/in/alexchen';
      return { PHOTO: '', LANG: 'en', PAGE_WIDTH: '8.5in', EMAIL: 'alex@example.com', PHONE: '+1-555-0123' }[name] ?? 'Alex Chen';
    });
    assert.doesNotMatch(filled, /\{\{[A-Z_]+\}\}/);
    assert.equal(refusedMarkup(filled), null);
    assert.equal(refusedMarkup(CV_HTML), null);
    assert.equal(refusedMarkup(tailoredCv({
      extra: '<a href="https://github.com/alexchen">g</a> <a href="mailto:alex@example.com">m</a> <a href="tel:+1-555-0123">t</a> '
        + '<a href="#skills">s</a> <img src="data:image/png;base64,AAAA" alt=""> <div data-section="skills" style="color: #1a1a2e">x</div>',
    })), null);
    // The photo the profile names is the owner's own choice; any other file is not.
    const photo = tailoredCv({ extra: '<img class="cv-photo" src="photos/me.jpg" alt="">' });
    assert.equal(refusedMarkup(photo, { photo: 'photos/me.jpg' }), null);
    assert.equal(refusedMarkup(photo, { photo: 'photos/other.jpg' }), 'a src attribute on <img>');
    assert.equal(refusedMarkup(photo), 'a src attribute on <img>');
  });

  test('a reply that pulls a local file into the CV is asked for again and never saved', async () => {
    const root = makeRoot();
    writeReport(root);
    writeFileSync(join(root, '.env'), 'OPENROUTER_API_KEY=sk-or-v1-FAKE-SECRET-FOR-A-TEST\n');
    const reply = tailoredCv({ extra: '\n  <iframe src="../.env" style="transform:scale(0.01);opacity:0.05"></iframe>' });
    const { ui } = quietUi();
    const client = fakeClient({ reply });
    const result = await tailorCv('001', { cfg, client, ui, dataRoot: root });
    assert.deepEqual([result.success, result.code, result.saved], [false, 'EMPTY', undefined]);
    assert.equal(result.reason, 'Reply rejected after 2 attempts: the HTML holds a <iframe> element, which the CV template does not have.');
    assert.equal(client.requests.length, 2);
    // No file for generate-pdf.mjs to render, and none for the owner to open.
    assert.deepEqual(listing(join(root, 'output')), []);
  });

  test('a saved CV carries a content policy as the first thing in its head', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    const { ui } = quietUi();
    const result = await tailorCv('001', { cfg, client: fakeClient(), ui, dataRoot: root });
    const saved = read(root, 'output', report.html);
    assert.equal(result.html, saved);
    assert.ok(saved.includes(`<head>\n${CSP_META}\n<meta charset="UTF-8">`), saved.slice(0, 400));
    assert.match(CSP_META, /default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:; base-uri 'none'; form-action 'none'/);

    // A photo the profile names by a local path stays loadable; nothing else from a file does.
    const withPhoto = makeRoot({ profile: 'candidate:\n  full_name: "Alex Chen"\n  photo: "photos/me.jpg"\n' });
    writeReport(withPhoto);
    const reply = tailoredCv({ extra: '\n  <img class="cv-photo" src="photos/me.jpg" alt="">' });
    const pictured = await tailorCv('001', { cfg, client: fakeClient({ reply }), ui, dataRoot: withPhoto, save: false });
    assert.equal(pictured.success, true, pictured.reason);
    assert.match(pictured.html, /img-src data: file:;/);
  });

  test('two documents in one reply, or a quoted opening and notes ahead of the document, are not cut into a CV', async () => {
    const quoted = [
      'The document has to open like this:', '```', '<!DOCTYPE html>', '<html lang="en">', '<head>', '```',
      'NOTES_MARKER: I will move the fraud-detection work to the top.', '', 'Here is the full document:', '', CV_HTML,
    ].join('\n');
    const twice = `${CV_HTML.replace('Full-stack AI engineer', 'DRAFT_MARKER engineer')}\n\nCorrected version:\n\n${CV_HTML}`;
    const root = makeRoot();
    writeReport(root);
    const { ui } = quietUi();
    for (const reply of [quoted, twice]) {
      assert.equal(tailoredCvProblem(reply), 'the reply holds more than one HTML document');
      const client = fakeClient({ reply });
      const result = await tailorCv('001', { cfg, client, ui, dataRoot: root });
      assert.deepEqual([result.success, result.code], [false, 'EMPTY']);
      assert.match(result.reason, /Reply rejected after 2 attempts: the reply holds more than one HTML document\./);
      assert.equal(client.calls.filter((call) => call === 'chat').length, 2);
      assert.deepEqual(listing(join(root, 'output')), []);
    }
    // A sentence that only names the tags is still not an opening.
    assert.equal(tailoredCvProblem(`I will output <html> and then the <head>.\n${CV_HTML}`), null);
  });

  test('a shell is not a CV: an empty body, a header alone, or the stylesheet swapped for a comment', async () => {
    const root = makeRoot();
    writeReport(root);
    const { ui } = quietUi();
    const reasonFor = async (reply) => (await tailorCv('001', { cfg, client: fakeClient({ reply }), ui, dataRoot: root, save: false })).reason;
    // Long enough to be a document, with nothing in it.
    const padded = CV_HTML.replace('<style>', `<style>\n  /* ${'styles '.repeat(200)} */`);
    const emptyBody = padded.replace(/<body>[\s\S]*<\/body>/, '<body>\n<div class="page"></div>\n</body>');
    assert.match(await reasonFor(emptyBody), /the CV holds 0 characters of text where cv\.md calls for at least \d+\./);
    const headerOnly = padded.replace(/<div class="section">[\s\S]*<\/body>/, '</div>\n</body>');
    assert.match(await reasonFor(headerOnly), /the CV holds \d+ characters of text where cv\.md calls for at least \d+\./);
    const noStyles = CV_HTML.replace(/<style>[\s\S]*<\/style>/, '<style>/* styles unchanged from the template */</style>');
    assert.match(await reasonFor(noStyles), /the stylesheet of the template is gone from the HTML \(no \.page rule\)\./);
    // A stylesheet cut down to the rules the document uses is a whole CV: the fixture has three rules.
    assert.equal(await reasonFor(CV_HTML), undefined);
  });

  test('only the template\'s own slots count as left unfilled, and a token cv.md itself contains is not one', async () => {
    const root = makeRoot();
    writeReport(root);
    writeFileSync(join(root, 'cv.md'), `${read(root, 'cv.md')}\n- Built a Helm chart library using {{VALUES}} templating\n`);
    const { ui } = quietUi();
    const reply = tailoredCv({ extra: '\n  <p>Built a Helm chart library using {{VALUES}} templating</p>' });
    const result = await tailorCv('001', { cfg, client: fakeClient({ reply }), ui, dataRoot: root, save: false });
    assert.equal(result.success, true, result.reason);
    assert.ok(result.html.includes('{{VALUES}}'));
    // A slot of the template is still a slot, in a CV whose cv.md does not spell it.
    assert.equal(
      tailoredCvProblem(CV_HTML.replace('<h1>Alex Chen</h1>', '<h1>{{NAME}}</h1>'), { slots: new Set(['NAME']), cv: 'no token here' }),
      'the template placeholder {{NAME}} was left in the HTML',
    );
    assert.equal(tailoredCvProblem(reply, { slots: new Set(['NAME']), cv: '' }), null);
    assert.equal(tailoredCvProblem(CV_HTML.replace('<h1>Alex Chen</h1>', '<h1>{{NAME}}</h1>'), { slots: new Set(['NAME']), cv: 'uses {{NAME}}' }), null);
  });

  test('tailoring a report again keeps the CV it replaces as .bak, and says so', async () => {
    const root = makeRoot();
    const report = writeReport(root);
    const { ui, written } = quietUi();
    const first = await tailorCv('001', { cfg, client: fakeClient(), ui, dataRoot: root });
    assert.equal('backup' in first, false);
    // The same CV again replaces nothing.
    const same = await tailorCv('001', { cfg, client: fakeClient(), ui, dataRoot: root });
    assert.equal('backup' in same, false);
    assert.deepEqual(listing(join(root, 'output')), [report.html]);

    // The owner corrects a sentence by hand, then tailors again where `pdf` was meant.
    writeFileSync(join(root, 'output', report.html), SAVED_CV.replace('Full-stack AI engineer', 'MY HAND EDIT'));
    const again = await tailorCv('001', { cfg, client: fakeClient(), ui, dataRoot: root });
    assert.equal(again.backup, `output/${report.html}.bak`);
    assert.ok(read(root, 'output', `${report.html}.bak`).includes('MY HAND EDIT'));
    assert.equal(read(root, 'output', report.html), SAVED_CV);
    assert.ok(written.err.includes(`An earlier CV for this report was replaced. It is kept as output/${report.html}.bak.`), written.err);
    // Its name does not end in .html, so `pdf` cannot take the copy for the report's CV.
    assert.deepEqual(listing(join(root, 'output')), [report.html, `${report.html}.bak`]);
  });
});

// ---------------------------------------------------------------------------
// The PDF step: what generate-pdf.mjs said
// ---------------------------------------------------------------------------

describe('pdf: what generate-pdf.mjs said', { concurrency: 6 }, () => {
  test('--json carries the reason when the fact check blocks, and the warning count when it warns', async () => {
    const blocked = makeTailoredRoot({ html: tailoredCv({ team: '30 engineers' }) });
    const refused = await cli(['pdf', '001', '--json'], { root: blocked.root });
    assertExit(refused, 1);
    const failure = JSON.parse(refused.stdout);
    assert.equal(failure.code, 'PDF');
    // The script's own last line, so a caller can tell a blocked fact check from a missing browser.
    assert.match(failure.reason, /^generate-pdf\.mjs stopped with exit code 1: Fact check failed for cv-jane-smith-001-\S+\.html: .*\. No PDF was written\.$/);
    assert.match(refused.stderr, /PDF generation failed: Fact check failed/, 'what the script printed is still on stderr');

    // A Title Case phrase cv.md does not have is a warning, not a block.
    const warned = makeTailoredRoot({ html: tailoredCv({ competencies: ['MLOps', 'Distributed Systems'] }) });
    const run = await cli(['pdf', '001', '--json'], { root: warned.root });
    assertExit(run, 0);
    const receipt = JSON.parse(run.stdout);
    const listed = run.stderr.split('\n').filter((line) => /^ {2}- /.test(line)).length;
    assert.match(run.stderr, /CV fact check warning: /, 'the fixture no longer draws a warning');
    assert.ok(listed >= 1);
    assert.deepEqual([receipt.success, receipt.factWarnings], [true, listed]);
    assert.ok(run.stderr.includes(`The fact check passed it with ${listed} warning${listed === 1 ? '' : 's'}, listed above`), run.stderr);
  });

  test('a setting from a .env in the working directory does not reach generate-pdf.mjs; the shell\'s does', async () => {
    // Blocked by the fact check, so the script stops before it would start a browser.
    const { root } = makeTailoredRoot({ html: tailoredCv({ team: '30 engineers' }) });
    const elsewhere = mkdtempSync(join(SANDBOX, 'cwd-'));
    const marker = join(elsewhere, 'marker.txt');
    // Every process that loads the module leaves its own id and its parent's.
    // Not the script it runs: tests/main-guard-convention.test.mjs allows no
    // tracked file to read the process entry path, in a string or out of one.
    writeFileSync(join(elsewhere, 'marker.cjs'), `require('node:fs').appendFileSync(${JSON.stringify(marker)}, process.pid + ' ' + process.ppid + '\\n');\n`);
    // The path in quotes: Node splits NODE_OPTIONS at a space, and a temp folder can have one in its path.
    const option = `--require="${join(elsewhere, 'marker.cjs').replace(/\\/g, '/')}"`;
    writeFileSync(join(elsewhere, '.env'), `NODE_OPTIONS=${option}\n`);
    const fromFile = await cli(['pdf', '001'], { root, cwd: elsewhere, env: { NODE_OPTIONS: undefined } });
    assertExit(fromFile, 1);
    assert.match(fromFile.stderr, /Fact check failed/);
    assert.equal(existsSync(marker), false, 'a module named in a .env was loaded by a child process');

    const fromShell = await cli(['pdf', '001'], { root, env: { NODE_OPTIONS: option } });
    assertExit(fromShell, 1);
    // The runner loaded it, and so did a process the runner started: the only
    // one this command starts is generate-pdf.mjs.
    const loaders = read(marker).trim().split('\n').map((line) => line.split(' '));
    assert.ok(
      loaders.some(([, parent]) => loaders.some(([pid]) => pid === parent)),
      `no process the runner started loaded the module: ${JSON.stringify(loaders)}`,
    );
  });

  test('a request that follows a slow PDF step is answered, and the summary of the run carries the warnings', async () => {
    const root = makeRoot();
    // The server drops a connection that was idle for 300 ms; the PDF step
    // takes over three times that, and the runner cannot see the close
    // meanwhile. The request for the answers used to go to the dead connection.
    const warned = tailoredCv({ competencies: ['MLOps', 'Distributed Systems'] });
    await withStub({ idleMs: 300, reply: replies({ tailor: warned }) }, async (stub) => {
      const run = await cli(['run', '--file', join(root, 'jd.txt'), '--quiet', '--no-merge', '--url', stub.url], {
        root, env: { STAND_IN_PDF_DELAY_MS: '1000' },
      });
      assertExit(run, 0);
      assert.deepEqual(stub.generations().map(asksFor), ['evaluate', 'tailor', 'apply']);
      assert.match(run.stdout, /^\s+tailor\s+done\s+output\/\S+\.html$/m);
      // In a run over many postings the warnings scroll past; the summary still shows them.
      assert.match(run.stdout, /^\s+pdf\s+done\s+output\/\S+\.pdf \(\d+ fact-check warnings?\)$/m);
      assert.match(run.stdout, /^\s+apply\s+done\s+output\/apply-001-/m);
    });
  });
});

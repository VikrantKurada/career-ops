# Running career-ops on a local Ollama server

This fork (branch `ollama-local`) adds `ollama-runner.mjs`. It runs the
career-ops pipeline against an Ollama server on your own machine: scan,
evaluate, work through the pipeline, tailor the CV, render the PDF and draft
application answers. It needs no API key and no agent CLI.

Model calls go to the Ollama address in use, `http://localhost:11434` unless
you change it, and nowhere else. An answer that redirects to another address
is refused, never followed. The other network traffic is what a job search
needs anyway: reading job boards and the postings you evaluate.

## What was tested, and what not to trust

Tested on this machine against the real server (Ollama 0.40.16, RTX 5090,
Windows 11, PowerShell 5.1), with the example CV and profile in a scratch data
root. There were six rounds. The first, on 2026-11-17, ran the runner as
first built: scanning, evaluating and rendering worked, tailoring and
application answers did not. The second, on 2026-11-18, ran every command
while fixes were still going in, and a review of the code and of this page
then found more. The third, later that day, ran every command again in a new
data root, and a review then found faults in cases no live run had hit. The
fourth and the fifth each ran every command in a new data root without a
failure, and the reviews of the code that followed them found further faults
of that kind, one of them made by a fix for another. The sixth ran every
command once more in a new data root, on the code as it is here. Its results
are the table below.

**The sixth round: every command.**

`jd.txt` is a posting in the United States, where the candidate of the
example profile may work. `other.txt` is a second posting, remote in European
time zones, from an employer that sponsors no visas. `questions.txt` holds
seven questions pasted from a form. Each command was typed in Windows
PowerShell 5.1 as it is printed here, with the three files given by their
full path.

| Command | Result |
|---|---|
| `npm run ollama:doctor` | Exit 0: the server answers, the model is installed, the window fits, thinking is off. |
| `npm run ollama:models` | Exit 0: 15 chat models, the default one marked. |
| `npm run ollama:eval -- --file jd.txt` | Exit 0 in 84 seconds, loading the model included. Report 001 scored 4.7/5, with every block, the Machine Summary, and the posting archived word for word with the end mark under it. |
| `npm run ollama:tailor -- 001`, eight times | Eight CVs, all passing the fact check with one request each, in 34 to 53 seconds. Four entries were written without their brackets and none was removed. In one CV two links to project pages that do not exist were taken off. |
| `npm run ollama:pdf -- 001` | Exit 0: a two-page PDF, with the fact check's three warnings listed. |
| `npm run ollama:apply -- 001` | Exit 0 in 26 seconds, with an answer to each of the eight usual questions. With `--questions questions.txt`: 18 seconds, with an answer to each of the seven pasted questions. Salary, work authorisation and the like came back as `Ask candidate`. |
| `npm run ollama:scan -- --company Cohere` | Exit 0 in 1 second: 31 live postings added to `data/pipeline.md`. |
| `npm run ollama:pipeline -- --limit 1 --pdf --min-score 1` | Exit 0 in 110 seconds. A live posting was read and scored 3.4/5, its row ticked, the CV tailored with one request, the PDF rendered, the missing tracker created and two rows merged. |
| `npm run ollama:run -- --file jd.txt --questions questions.txt` | Exit 0 in 141 seconds: evaluation (4.5/5), CV, PDF, answers and tracker. |
| `npm run ollama:run -- --file other.txt --min-score 3` | Exit 0 in 67 seconds. The posting scored 4.0/5 and the run stopped after the evaluation, because the report advises against applying: no visa sponsorship. The model had listed that hard stop and written Apply all the same. The report was saved with the decision Skip. |
| `npm run ollama:eval -- --file jd.txt --model gemma4:26b-a4b-it-q4_K_M --no-save` | Exit 0 in 38 seconds, 4.8/5. |
| Six commands the runner has to refuse | All six ended with exit 1 and no request for a reply: a posting typed as one line, a window set too small by hand, a model that is not installed, `pipeline --no-save`, `apply` for a report that scored 3.4, and `apply` for the report that advises against applying. |
| `node check-jd-archive.mjs` and `node verify-pipeline.mjs` on the result | Every report has its posting archived. No errors, and two warnings: the one the [`ollama:eval`](#ollamaeval) section describes for a posting evaluated a second time with a lower score, and one about an entry of the example `portals.yml`. |

No code changed during this round.

The model wrote more than 120 replies in the second and third round and 18
each in the fourth, the fifth and the sixth. None was a tool call.

**What the first five rounds found, and what was changed.**

| What went wrong | Measured | What was changed |
|---|---|---|
| The model called a tool instead of writing the CV. The first real `ollama:tailor` run failed that way after 17 seconds. | Six tries of the same request: 2 complete CVs, 3 tool calls, 1 cut-off CV. With one sentence added to the end of the request: 12 complete CVs of 12. | That sentence is now in every request the runner sends, and a tool call is asked for again. |
| The fact check of the PDF step refused the tailored CV. The model had added keywords of the posting to the skills: OpenTelemetry, Observability, CI/CD Pipelines. | 18 of 27 complete CVs were blocked. Naming the rejected keyword to the model did not help: 6 of 6 came back blocked for another one. Removing the rejected entries did: 2 of the 27 stayed blocked. | `ollama:tailor` runs the fact check itself, removes what it rejects and tells you what it removed. |
| The fact check refused a CV for a tag with a list in brackets, such as `MLOps Platform (SageMaker, Kubeflow, MLflow)`. It reads `MLflow)` as a tool the CV does not name. | 5 of 18 CVs were blocked for that alone: 4 for `MLflow)` and 1 for `Feature Store)`. Removing such a tag unblocked the CV and lost tools `cv.md` names: all 5 tags removed in the second round's eight runs were of this kind. | An entry with a rejected piece of a list in brackets is written without the brackets and checked again. In the third round's eight runs that kept 8 entries, and 1 other entry was removed. In the eight runs of the fourth round it kept 3, in those of the fifth 5 and in those of the sixth 4, and none was removed. |
| `ollama:apply` saved a refusal as the answers: the model asked to be sent the form. | 2 of 7 replies held answers. With a list of eight usual questions in the request: 4 of 4. With questions pasted from a form: 3 of 3. | The request carries questions, yours or the eight usual ones. A reply that is not a set of answers is never saved. |
| With thinking on, `gemma4` spent its whole output budget thinking. That was the automatic setting for it. | The live run failed after 189 seconds. In 2 of 3 more runs it wrote 25,520 tokens of thinking and no report. With thinking off: 3 complete reports of 3, in 19 to 26 seconds each. | Thinking is off for every model unless you force it on. |
| The model copied the whole posting into each report. Most replies that came back incomplete had stopped inside that copy. | 5 of 6 replies were complete. With one sentence telling the model not to copy the posting: 8 of 8, at 7,000 to 7,800 output tokens against 7,700 to 8,700. | The request says so. The runner archives the posting itself, as it always did. |
| Tailored CVs stated things `cv.md` does not. The evaluation's proposed CV changes were written in the posting's words, and the tailoring step applied them as facts. | Of 7 CVs, 6 gave a project "golden-set management" and "LLM-as-judge pipelines", 6 called a fraud pipeline a "gateway", 5 gave one project the star count of the other, and 4 linked project pages that do not exist. After the change: 0 of 8 for each of the four in the second round. In the third round 0 of 8 for three of them, and 1 CV of 8 said the candidate is "now focused on" an "LLM gateway" and more of the posting's list, which `cv.md` does not say. In one more run of the third round the model linked the two made-up project pages again, and the runner took both links off. In the fourth and in the fifth round: 0 of 8 for each of the four, and no link to take off. In the sixth round 1 CV of 8 ended a project's description with "the same pattern required for golden sets, regression gates, and LLM-as-judge pipelines", words of the posting that `cv.md` does not have, and in another the model linked the two made-up project pages, which the runner took off. | The tailoring step is no longer sent the report's proposed changes, interview stories, draft answers, Machine Summary and cover letter draft. Its request ends with a rule. A web link whose address is in none of your files is taken off. |
| Application answers repeated the same claims. | 4 of 5 sets of answers to the eight usual questions. After the change: 1 of 10, which gave one project the star count of the other. One more of the ten called the fraud pipeline "a production inference gateway". | The answers step is sent the report without the same five parts, and its request ends with a rule. |
| The model stated answer lengths it had not counted. | `998/1000 characters` under an answer of 773 characters, and `1498/1500` under one of 1,270. | The runner recounts each `Length: n/limit` line the model writes and marks an answer that is over its limit. |
| A report scored 4.2/5 while it called the posting a hard blocker: the employer sponsors no visas, and the role is outside the countries the candidate may work in. The run compared the score alone and went on to a CV, a PDF and answers. | Four live evaluations of that posting: 2 listed the hard stop, and 2 recorded the blocker only as the work-authorisation tier and listed none. One of those two said Apply. So did the one evaluation of the fourth round: the tier, no hard stop, and Apply. The ones of the fifth and the sixth round listed the hard stop and said Apply all the same. | The run reads the report's hard stops and that tier, and stops before the CV. A report with the tier and no hard stop is saved with the hard stop and the decision Skip. `ollama:apply` asks for `--force`. |
| A report can carry a cover letter draft, written in the same pass as its proposed CV changes and with the same inventions. | Found by review: 3 of the 12 evaluations recorded before the model was told not to copy the posting had one, and 2 of those gave a project the other's star count. None of the 32 saved since has one. | The draft is left out of both requests like the other four parts. |
| What was added to a report after its archived posting was read as part of the posting. career-ops' apply mode appends its answers there. | Found by review, with career-ops' own writer on a real report: the tailoring request then carried the candidate's answers as job description. | The runner writes an end mark under the posting it archives. |

The reviews also found faults in cases no live run had hit, such as a report
saved with Windows line ends, a skills line with a list in brackets, a fact
check that fails after the reply, a hard stop listed without indentation or
written in braces, and a remark behind a hyphen after the work-authorisation
tier. Each is fixed and has a test.

**What not to trust.**

- **Read every CV and every set of answers against your own CV before you
  send it.** The numbers above come from one example CV and five postings,
  and from claims that were looked for by name. They are not a guarantee.
  Answers to questions pasted from a form showed it. Asked to "describe your
  experience building evaluation systems for LLM features", the model claimed
  more than `cv.md` says in every run that was read: that an LLM-as-judge
  pattern was "something I've prototyped" in a project whose line in `cv.md`
  does not mention it, that golden sets were "something I've already built and
  open-sourced", and in the third round that a recommendation engine "required
  evaluating LLM output quality in a production setting and catching
  regressions before they shipped". In the fourth round it was "practical
  experience catching regressions before they ship". In the fifth the
  candidate's toolkit "lets teams define golden sets", and the answer closed
  with "human calibration for LLM-as-judge pipelines". In the sixth the
  recommendation engine "required evaluating LLM output quality against a
  baseline and catching regressions before shipping".
- **A report's score and its advice can disagree.** The evaluation keeps them
  apart on purpose: a hard stop sets the decision to Skip and leaves the
  score where it was. The run stops before the CV when the report lists a
  hard stop, gives the work-authorisation tier No sponsorship, or rates the
  posting Suspicious. Nothing else is read. A reason against applying that
  stands only in the report's text does not stop a run, so read the first
  lines of a report before you act on its score, and go by its Decision line.
  The sentence under the header fields can say otherwise. Like the sentence
  in the report's Verdict (lead) block, it opens with the call, and the
  runner corrects only that word, to Skip for example. Its other words stay
  as the model wrote them: in the report with the visa blocker of each of the
  last three rounds the sentence reads "**Skip.** Strong match on the role's
  core axis", and goes on "with no hard stops". The three reports
  of the second and third round that were saved with a hard stop said "with
  no hard stops" as well.
- **An answer can be longer than the form allows.** In the last of the second
  round's three runs on a form's questions, both answers with a limit were
  over it: 1,339 characters against 1,000 and 1,795 against 1,500. In the
  third round one of four was: 1,018 against 1,000. In the fourth round none
  of four was, in the fifth one of four: 1,072 against 1,000, and in the
  sixth none. The saved file marks an answer that is over. Shortening it is
  yours to do. The model's own closing notes can say
  otherwise: one called that answer "exactly at the 1000-character limit".
- **A CV that passes the fact check can still overstate.** The check blocks
  some things, such as an unknown tool on a skills line or a number `cv.md`
  does not have. It only warns about others, and some it does not see at all.
  One of the sixth round's eight CVs passed with a project described as "the
  same pattern required for golden sets, regression gates, and LLM-as-judge
  pipelines", words of the posting that `cv.md` does not have.
  `ollama:pdf` lists the warnings: each is a sentence to confirm against
  `cv.md`.
- **The interview stories in a report (Block F) can contain invented
  results.** In the first round report 001 held results the example CV does
  not have: "caught 3 regressions pre-ship", "Reduced MTTR from hours to
  minutes", "Passed internal audit". A rule added to the evaluation request
  reduced this and did not end it. In the second round the report still held
  "Reduced undetected model degradation" and "reduced onboarding time for new
  models" in Block F. In the third round it gave one project "500+ GitHub
  stars; used by external teams", where the stars are another project's, and
  a fraud pipeline that "passed regulatory audits". It wrote "(candidate to
  add)" for one story the CV has nothing for. Read Block F as interview
  preparation to check against your memory, never as a record of what you did.
- **The proposed CV changes (Block E), the draft answers (Block H) and a
  cover letter draft in a report are suggestions in the posting's words**,
  some of them marked "(if accurate)". Do not copy from them. `ollama:tailor`
  and `ollama:apply` do not read them.
- **A tailored CV can lose keywords.** The run lists what it removed. Read
  that list: an entry is a word the posting wants and the check did not find
  in `cv.md` in that spelling. In the third round's eight runs one entry was
  removed, `CI/CD Pipelines`, where `cv.md` says "CI/CD pipeline". In the
  eight runs of each of the last three rounds none was.
- **The second model writes a shorter report, and reads work authorisation
  wrongly.** In the second and third round `gemma4` wrote Blocks A to G in
  2,300 tokens, where the default model wrote 5,500 to 8,800, and left out the
  Machine Summary and the draft answers. In the fourth round it wrote 3,600
  tokens: no draft answers again, and a short Machine Summary without the
  work-authorisation tier. In the fifth and in the sixth it wrote 3,800
  tokens of the same shape. In the first round it left out the score summary
  block in all 4 of its complete reports, and `qwen3.8:27b` in 1 of 6. The
  runner reads the company, the role and the score from two other places in
  the report, so such a report is saved. In its one run in each round from
  the second to the sixth, `gemma4` gave the example posting, a role
  in the country the candidate may work in, the tier No sponsorship. Its four
  complete reports of the first round, written from an earlier wording of the
  request, gave Not needed, and so did every report of the default model for
  that posting, more than twenty. With `gemma4` a run
  stops at that posting for a reason that does not hold. The default model
  once wrote No sponsorship into the header of a report whose text and
  Machine Summary said Unstated. The summary decides now, so such a report
  does not stop a run. In the earlier probe `gemma4`'s one run without
  thinking contradicted itself: see [Models](#models).
- **Scores move between runs and between models.** For one posting the
  default model gave 4.5 to 4.8 over more than twenty runs, and `gemma4` gave
  4.8 in all nine of its complete reports. A second posting scored 2.5 to
  3.5 in the earlier probe, 4.2 in the second round, 4.2, 3.8 and 4.2 in the
  third, 4.2 in the fourth and in the fifth, and 4.0 in the sixth.

The runner's automated tests use a stub server, never a real model. The
numbers under [Models](#models) come from an earlier manual probe of the
evaluation prompt against the Ollama API, done before the runner was written,
except where that section says "live test", gives the date 2026-11-18 or
names a live round.

## Quick start

You need Ollama running and one usable model installed. On this machine both
are already true: Ollama 0.40.16 answers on `http://localhost:11434` and
`qwen3.8:27b` is installed. No environment variable and no `.env` file is
needed.

The commands are written for Windows PowerShell 5.1 and run from the
checkout. Every `npm run` line works unchanged in Git Bash.

Two rules for every command on this page:

- **The runner's options go after `--`**, as in
  `npm run ollama:pipeline -- --limit 5`. Without it, npm keeps the option.
- **Never pipe into `npm run`.** In PowerShell a pipe makes npm take every
  option, `--` or not, and the piped text loses its non-ASCII characters.
  Give a posting with `--file`.

[Options for every subcommand](#options-for-every-subcommand) has the details.

### 1. Check the server

```powershell
npm run ollama:doctor
npm run ollama:models
```

`ollama:doctor` checks, in order: the server answers, a model can be picked,
the context window holds the evaluation prompt, how thinking is set, and
whether the model is already loaded in a way that will cost you (a different
context size, or partly on the CPU). It exits 1 if a check fails.

`ollama:models` lists the installed chat models, says whether each one can
hold the evaluation prompt, and marks the one the runner picks by default with
`*`.

### 2. Try one evaluation with example data

The runner refuses to evaluate without a CV and a profile, so that it never
scores a job against nothing. To try it before you set up your own files,
point it at a scratch folder that holds the example CV and the example
profile. Nothing is written to the checkout.

```powershell
$S = Join-Path $env:TEMP 'career-ops-trial'
New-Item -ItemType Directory -Force "$S\config", "$S\modes" | Out-Null
Copy-Item examples\cv-example.md "$S\cv.md"
Copy-Item config\profile.example.yml "$S\config\profile.yml"
Copy-Item modes\_profile.template.md "$S\modes\_profile.md"
node -e "require('fs').writeFileSync(process.argv[1], require('./evals/golden/llmops-long-realistic.json').jd)" "$S\jd.txt"

$env:CAREER_OPS_ROOT = $S
npm run ollama:eval -- --file "$S\jd.txt"
```

`CAREER_OPS_ROOT` tells every career-ops script where your personal files
live. The job description is a synthetic posting from the project's own
evaluation set.

Expect one to two minutes with `qwen3.8:27b`: the model has to load and read
a prompt of roughly 40,000 tokens before it writes. It took 84 seconds in the
last live round. The report streams to the terminal while it is written. When
the command finishes there are two new files under `$S`:

- `reports\001-<company>-<date>.md`, the evaluation report
- `batch\tracker-additions\001-<company>.tsv`, its row for the tracker

The run prints a note that `modes/_profile.md` is still the shipped template.
In this trial that is expected: the template was copied in for the trial
only. Step 3 is where it matters.

Clean up afterwards, so that later commands read your real data again:

```powershell
Remove-Item Env:CAREER_OPS_ROOT
Remove-Item -Recurse -Force $S
```

<details>
<summary>The same trial in Git Bash</summary>

```bash
S="$(cygpath -m "$(mktemp -d)")"
mkdir -p "$S/config" "$S/modes"
cp examples/cv-example.md "$S/cv.md"
cp config/profile.example.yml "$S/config/profile.yml"
cp modes/_profile.template.md "$S/modes/_profile.md"
node -e "require('fs').writeFileSync(process.argv[1], require('./evals/golden/llmops-long-realistic.json').jd)" "$S/jd.txt"

CAREER_OPS_ROOT="$S" npm run ollama:eval -- --file "$S/jd.txt"
rm -rf "$S"
```

</details>

### 3. Use your own data

By default your personal files live in the checkout. An evaluation needs
`cv.md` and `config/profile.yml`, and its score is measured against
`modes/_profile.md`. A scan needs `portals.yml`.

```powershell
npm run doctor
```

lists what is missing and how to create each file. In short, four files:

1. `cv.md`: your CV as markdown. `examples/cv-example.md` shows the format.
2. `config/profile.yml`: copy `config/profile.example.yml` and edit the copy.
3. `portals.yml`: copy `templates/portals.example.yml` and edit the copy.
4. `modes/_profile.md`: your target roles and how you want to be presented.
   **Edit this one by hand before you trust a score.**

All four are gitignored.

About the fourth file. Until you edit it, your copy is the shipped template,
`modes/_profile.template.md`, unchanged. It describes the template author's
six AI roles, not yours, and a posting outside the roles listed there gets
the lowest mark for fit with your targets. Edit at least these sections: Your
Target Roles, Your Adaptive Framing, Your Comp Targets and Your Location
Policy. If you have no `modes/_profile.md`, copy the template to that name
first.

Two things remind you while the file is still the template. `npm run doctor`
prints `Personalization incomplete` and tells you to ask your agent. On this
route there is no agent, so that line means: edit the file yourself. And every
run of the runner that reads the file prints one note that it
`is still the shipped template`.

You do not have to create the tracker, `data/applications.md`. The first run
that merges creates it. If you already have one, see
[The tracker](#the-tracker): one with nine columns needs one command.

After that the loop is:

```powershell
npm run ollama:scan                      # find new postings (no model involved)
npm run ollama:pipeline -- --limit 5     # evaluate five pending postings
npm run ollama:tailor -- 003 --pdf       # tailored CV and PDF for report 003
npm run ollama:apply -- 003              # draft answers for the application form
```

Or, for one posting from start to finish, with the posting's own address in
place of this example:

```powershell
npm run ollama:run -- https://example.com/jobs/123
```

Put the address in double quotes when it contains `&`, which PowerShell
otherwise reads as its own operator.

## Commands

| Command | What it does | Calls the model |
|---|---|---|
| `npm run ollama:doctor` | Checks the server, the model, the context size, thinking and what is loaded. Exits 1 if a check fails. | no |
| `npm run ollama:models` | Lists the installed chat models: size, maximum context, capabilities, whether the evaluation prompt fits, and `*` on the default pick. | no |
| `npm run ollama:scan` | Runs the portal scanner (`scan.mjs`) with the arguments you pass. New postings land in `data/pipeline.md`. | no |
| `npm run ollama:eval -- <input>` | Evaluates one posting. Writes the report and a tracker row. | yes |
| `npm run ollama:pipeline` | Evaluates the pending rows of `data/pipeline.md` one by one, ticks each row that succeeds, then merges the tracker. | yes |
| `npm run ollama:tailor -- <report>` | Writes a tailored CV for one report, checked against `cv.md`: as HTML, or with `--latex` as a `.json` of its text and a `.tex` made from it. | yes |
| `npm run ollama:pdf -- <report>` | Renders that CV to PDF: an HTML CV with `generate-pdf.mjs`, a LaTeX CV with the TeX engine its template names. | no |
| `npm run ollama:apply -- <report>` | Drafts application answers for one report. Never submits anything. | yes |
| `npm run ollama:run -- <input>` | One posting end to end: evaluate, then tailor, PDF and answers if the score is high enough and the report does not advise against applying, then merge the tracker. | yes |
| `npm run ollama -- <subcommand>` | The runner itself. `npm run ollama -- --help` prints the usage. | depends |

`npm run ollama:eval` used to run the upstream script `ollama-eval.mjs`. In
this fork it runs the new runner. The old script is still there: see
[The legacy script](#the-legacy-script).

Everything is written under your data root, in folders git ignores:

| What | Where |
|---|---|
| Postings waiting to be evaluated | `data/pipeline.md` |
| Evaluation reports | `reports/` |
| Tracker rows waiting to be merged | `batch/tracker-additions/` |
| The tracker | `data/applications.md` |
| Tailored CVs, PDFs and application answers | `output/` |

### The tracker

`ollama:pipeline` and `ollama:run` finish by merging their new rows into
`data/applications.md`. `ollama:eval` leaves its row in
`batch/tracker-additions/`, and so do the other two with `--no-merge`.
`npm run merge` applies whatever is waiting there. `ollama:pipeline` and
`ollama:run` also merge rows an earlier run left behind: the pipeline even
when nothing is pending, a run even when its own evaluation failed.

**You do not have to create the tracker.** When there is none,
`ollama:pipeline` and `ollama:run` create it before they merge, and say so:

```text
No tracker yet: created data/applications.md (ten columns, the last one the posting URL).
```

This is the file they create:

```markdown
# Applications Tracker

| # | Date | Company | Role | Score | Status | PDF | Report | Notes | URL |
|---|------|---------|------|-------|--------|-----|--------|-------|-----|
```

The tenth column holds the posting's address. The merge matches rows on it
before anything else, so a posting that is evaluated twice updates its row.

**If your tracker has nine columns**, add the tenth once. Nine columns is
what career-ops' own onboarding writes. With nine, every merge of a posting
that was read from a URL prints `URL dedup is INACTIVE`. This adds the column
and changes nothing else:

```powershell
node merge-tracker.mjs --backfill-urls
```

The runner never restructures a tracker that exists.

`npm run merge` on its own does not create a tracker. With none it prints
`No applications.md found` and leaves the rows waiting. If you only ever use
`ollama:eval`, save the four lines above as `data/applications.md` under your
data root, then run `npm run merge`.

### Options for every subcommand

| Option | Meaning |
|---|---|
| `--model <name>` | Model for this run. Wins over `OLLAMA_MODEL`. |
| `--url <address>` | Ollama address for this run. Wins over `OLLAMA_BASE_URL`. |
| `--num-ctx <n>` | Context window for this run. Counts as an explicit value, like `OLLAMA_NUM_CTX`. |
| `--think=on`, `--think=off`, `--think=auto` | Thinking for this run. `auto` is off. |
| `--no-save` | Print the result and write nothing. For `ollama:eval`, `ollama:tailor` and `ollama:apply`. |
| `--quiet` | Do not stream the reply to the terminal. |
| `--json` | Print one JSON receipt on standard output. Implies `--quiet`. Under `npm run`, add `--silent`, or npm prints its own banner first. |
| `--help`, `-h` | Print the usage. |

A value may follow its option after a space or after `=`. A misspelled option
stops the run with an error. It is not ignored.

`ollama:pipeline`, `ollama:pdf` and `ollama:run` refuse `--no-save`, and so
does `ollama:tailor --pdf`: each of them works from a file an earlier step
saved. The options of a single subcommand are listed under it below, and
`npm run ollama -- --help` prints all of them.

**Options go after `--`.** Without it npm keeps them for itself. The runner
notices that for most of its options and refuses to start, naming the option:

```text
npm kept --limit for itself, so it never reached the runner.
```

It cannot notice `--json`, `--quiet`, `--force`, `--pdf` and `--help`. Those
are npm's own options too, and without the `--` they are lost without a word.
`ollama:pipeline`, `ollama:models` and `ollama:doctor` take no argument, so
`npm run ollama:pipeline -- 5` is refused as well.

**Do not pipe into `npm run` in PowerShell.** A pipe makes PowerShell drop the
`--`, so npm takes every option after it. It also sends the piped text in
ASCII: a typographic dash, an accented letter or a euro sign arrives as `?`.
Use `--file`.

**A receipt you can parse.** `--json` prints one JSON object. To keep npm's
banner out of it:

```powershell
npm run --silent ollama:models -- --json
```

The receipt is written in ASCII, with every other character as a `\u` escape,
so PowerShell 5.1 reads it correctly. A run that was refused before it started
leaves a receipt too, with the reason in it.

### `ollama:doctor`

`npm run ollama:doctor` runs only the Ollama checks. The general setup check
can run them as well: `npm run doctor -- --ollama`.

Without that flag, `npm run doctor` contacts Ollama only when you have set one
of the runner's own variables: `OLLAMA_BASE_URL`, `OLLAMA_ALLOW_REMOTE`,
`OLLAMA_MODEL`, `OLLAMA_MODEL_PREFERENCE`, `OLLAMA_NUM_CTX`,
`OLLAMA_NUM_PREDICT`, `OLLAMA_THINK`, `OLLAMA_TEMPERATURE`,
`OLLAMA_TIMEOUT_MS`, `OLLAMA_RETRIES` or `OLLAMA_SEED`. It looks in the shell,
in the data root's `.env` and in the `.env` of the folder you run from.

The Ollama server's own variables do not count: `OLLAMA_HOST`,
`OLLAMA_KEEP_ALIVE` and the rest. Neither does `--strict`. While this is
written, this machine sets only server variables, `OLLAMA_HOST` among them,
so a plain `npm run doctor` here makes no request to Ollama. With `--target`,
only the `.env` of the checkout you name is read.

In a run that was not started with `--ollama`, what depends on the server is
a warning: it is off, it lacks the model, its model data cannot be read. A
setting the runner would refuse and a context window too small for your
prompt are failures in every run. With `--ollama`, and in
`npm run ollama:doctor`, each of these is a failure.

Both checks size the context window for your own CV and profile, and the
runner sizes a run the same way: the files the prompt is built from, the
prompt's own wording, and the longest posting the page reader returns. So the
three name the same number. If you pin `OLLAMA_NUM_CTX` by hand, use the
number they name.

One posting can need less or more than that. Every command checks the window
again against the prompt it is about to send, and a posting given with
`--file` has no length limit.

### `ollama:eval`

The input is one of:

- a URL: `npm run ollama:eval -- https://example.com/jobs/123`. The posting is
  read through the job board's API when there is one, then with a headless
  browser, then with a plain request. At most the first 16,000 characters of
  the posting are used. The URL is recorded in the report.
- a file: `npm run ollama:eval -- --file jd.txt`. Add `--posting-url <url>` to
  record where the posting lives. This is the way to evaluate text you
  copied.
- nothing: the runner asks you to paste the job description. End the input
  with Ctrl+Z and then Enter on Windows, Ctrl+D elsewhere.

To save a posting you copied into a file, in PowerShell:

```powershell
Get-Clipboard | Set-Content -Encoding UTF8 jd.txt
npm run ollama:eval -- --file jd.txt
```

The file may be UTF-8 or UTF-16, which is what `>` writes in Windows
PowerShell. A file that is neither, such as a PDF or a Word file, is refused.

Do not type the posting as text in quotes. On Windows npm passes on only the
first line of quoted text, so a pasted posting arrives as its title. The
runner refuses one line of fewer than 200 characters given this way, and says
that it `is not a job description`. It also refuses the name of a file given
without `--file`, and a URL given beside other input.

Before it asks the model, the run prints how long the posting is, in
characters and lines. A posting that arrived cut short shows there, before
two minutes are spent on it.

It writes the report to `reports/{NNN}-{company}-{date}.md`, with the job
description archived at the end, and one row to
`batch/tracker-additions/{NNN}-{company}.tsv`. Run `npm run merge` to move
that row into the tracker. `ollama:pipeline` and `ollama:run` merge for you.
The archived job description is the text the runner read, written by the
runner. It is not the model's copy: the request tells the model not to copy
the posting into its reply, and a copy it makes anyway is replaced. So in a
report from this runner the archive follows the Machine Summary, where an
agent's report has it the other way round. career-ops' own tools read both
orders.

Under the posting the runner writes one comment line, the end mark:
`<!-- end of the archived job description -->`. It marks where the posting
ends, so that a section added to the report later is not read as part of the
posting: career-ops' apply mode appends `## Application Answers` there, and
you may add notes of your own. What stands behind the mark is part of the
report, and is sent to `ollama:tailor` and `ollama:apply` with it. The
dashboard (`npm run ui`) shows the mark as a line of text under the posting.

The mark counts only in a report this runner saved, which it knows by the
header line that starts with `**Tool:** Ollama runner (`. No other script
writes that line: career-ops' own `ollama-eval.mjs` heads its reports
`**Tool:** Ollama (` and writes no mark. In any other report the mark counts
for nothing, also when the posting itself holds such a line. There the
posting ends at the next section of the report, where career-ops' own
`check-jd-archive.mjs` ends it. In the order an agent writes a report that
section is the Machine Summary, and what is added behind it under a heading
of its own (`## My notes`) is sent as part of the report too. A line typed
straight under the summary, without a heading, goes with the summary and is
not sent.

One layout is left in which added text is read as posting: a report without
the mark that ends with its archive, as this runner saved them before the
mark existed. `ollama:apply` sends such text all the same, inside the
archive. For `ollama:tailor`, pass `--jd <file>` for such a report: the
posting then comes from the file, and what was added behind the archive is
not sent.

The run ends with the score, the archetype and the legitimacy rating. When
the report itself advises against applying, a line under them says so, for
example `The report advises against applying: hard stop: ...`. That is read
from three places: the hard stops in the report's Machine Summary, the
work-authorisation tier No sponsorship, and a legitimacy rating of
Suspicious. The score shows none of them. In the evaluation mode a hard stop
sets the decision to Skip and leaves the number where it was, and one live
report scored 4.2/5 with a hard stop, no visa sponsorship. `ollama:run`,
`ollama:pipeline` and `ollama:apply` act on that line as described in their
sections.

All three are read from the report's own text, never from the posting
archived in it. A `hard_stops` list that only says there is none, such as
`- "None"`, counts as empty: the report is saved with an empty list and keeps
the decision its score gives. The list is read from the report's Machine
Summary. Only where no summary can be read, as behind a code fence that the
report's own text leaves open, is a `hard_stops` list elsewhere in the
report's own text read in its place.

A rating of Suspicious sets the decision to Research first, as the evaluation
mode asks, also when the reply states the rating only in its Machine Summary.

The tier No sponsorship means that the posting refuses visa sponsorship and
the role is outside the countries your profile lists under
`location.authorized_in`. The evaluation mode calls that a hard blocker, and
the model does not always list it as a hard stop. So a report whose Machine
Summary gives that tier and lists no hard stop is saved with the hard stop
added and the decision set to Skip. The added entry says who listed it. It
starts with
`No visa sponsorship (listed by the runner from the tier work_auth: no_sponsorship)`.
The score stays as the model wrote it. One summary is left as it is: one
whose `hard_stops` key holds a value that is no list, such as
`hard_stops: none`. Its report keeps the decision of its score, and the run
stops for the tier as described below.

The tier in the Machine Summary decides. It is read in the spellings a model
writes it in: `no_sponsorship`, `No_sponsorship`, or `No sponsorship` as the
header has it, with or without the sign in front. A remark behind the tier
leaves it standing when a sign comes first: a dash, a hyphen with a blank
beside it, a colon, a bracket, as in `No sponsorship - hard blocker`. A
hyphen straight between two words joins them. A value that goes on in words,
such as `No sponsorship mentioned` or `No sponsorship-related`, names no
tier. The `Work Auth`
line of the header is read only when the summary names no tier: in a report
without a Machine Summary, with one that has no `work_auth` line, as `gemma4`
writes it, or with a value there that is no tier. Such a report gets no hard
stop added, so its Decision line follows its score like that of any report
without a hard stop: it can say Apply while the run stops. For it the run
prints the tier in place of a hard stop:
`The report advises against applying: work authorisation: the report gives the tier No sponsorship`,
followed by what that tier stands for.

Evaluating a posting a second time, for example with another model, writes a
second report. The merge then moves the posting's tracker row to the new
report and its score, and `node verify-pipeline.mjs` warns about the older
file as a report no row links to. When the new score is lower, the row's
notes name the report it replaced, and that is the only warning. Otherwise
the two files are reported as duplicates as well. To compare models without
touching the tracker, use `--no-save`.

The model is asked to end with a score summary block. When the block is
missing or malformed, the runner reads the company, the role and the score
from the report's Machine Summary, then from its title and header. A reply is
asked for once more only when none of the three gives a company, a role and a
score, or when it is very short. If the second reply fails too, the run fails
and nothing is saved.

The evaluation mode has gates at which an agent would stop and ask you: a
blacklisted company, one of your dealbreakers, an agency posting, a closed
posting. The runner tells the model to report such a hit as a hard stop
inside the full report. If the model stops anyway, the run ends with
`Not evaluated. The model stopped at a gate` and the model's own line. It is
not asked again, nothing is saved, and the exit code is 1.

### `ollama:pipeline`

It reads `data/pipeline.md` and evaluates the pending rows
(`- [ ] url | company | role`) in file order. Run one pipeline at a time on a
data root.

| Option | Meaning |
|---|---|
| `--one`, `-1` | Stop after one posting was evaluated. |
| `--limit N` | Stop after N postings were evaluated. A row whose page cannot be read does not count. |
| `--tailor` | Also tailor the CV for each posting that scores at least `--min-score`, unless its report advises against applying. The run names each posting it left out for that, with the two commands that make its CV and answers anyway. |
| `--pdf` | Also render the PDF of that tailored CV. Implies `--tailor`. |
| `--min-score X` | Threshold for `--tailor` and `--pdf`. Default 4.0. |
| `--no-merge` | Leave the tracker rows in `batch/tracker-additions/`. |
| `--latex`, `--html`, `--template <name>` | The format of the tailored CV, and the template of a CV in LaTeX: see [A CV in LaTeX](#a-cv-in-latex). Only with `--tailor` or `--pdf`. |

What happens to a row:

| The row | What the run does with it |
|---|---|
| was evaluated | Ticks it (`- [x]`). |
| is a posting that has closed (HTTP 404 or 410, or the page says so) | Marks it `- [!]` with the reason and the date. Later runs skip it. It does not count toward `--limit`. |
| is not a URL | Marks it `- [!]`. `local:` rows are the exception: see below. |
| could not be read this time (a bot wall, a page too short to be a posting, no network) | Leaves it pending. It does not count toward `--limit`. |
| was read, and the model's reply failed | Leaves it pending. It counts toward `--limit`. |
| is no longer pending when its turn comes | Skips it. Another program ticked or removed it meanwhile. |
| got a report, and its tracker row could not be written | Marks it `- [!]`, so that the posting is not evaluated a second time. |

If the model's reply names no company or no role, the row's own are used.

The run says in two places when a report advises against applying, with or
without `--tailor`. Under the posting it prints
`The report advises against applying: ...`, and the posting's row in the
summary ends with `the report advises against applying`. With `--tailor` or
`--pdf`, a posting that reached `--min-score` and gets no CV for that reason
also gets the line `No CV for this one, for that reason.` and the two
commands, and its row says `no CV: the report advises against applying`. With
`--json`, the posting's entry in the receipt carries the reasons under
`stops`, and with `--tailor` or `--pdf` the receipt carries the threshold
under `minScore`.

The run stops early after two failures in a row that every later row would
share: the server is not reachable, the model is missing, a request timed
out, or the privacy guard refused the request. It also stops once 20 pages
could not be read, which usually means the machine is offline. The summary
says why, on a line that starts with `Stopped early:`.

The exit code is 1 if any posting failed, if a `--tailor` or `--pdf` step
failed, or if the tracker merge failed.

The model is loaded once, after the first posting has been read. A run in
which no page can be read loads nothing.

Only a row whose URL starts with `http://` or `https://` can be evaluated.
The runner does not read `local:` rows, which point at a saved job
description. Such a row stays pending and is reported as failed on every run,
so the exit code is 1 while one is pending. Evaluate the file with
`npm run ollama:eval -- --file <path>`, then tick the row by hand.

### `ollama:tailor` and `ollama:pdf`

`<report>` is a report number (`003`, or `3`) or the path of a report file.

`ollama:tailor` needs the job description. It uses the copy archived in the
report, or the file you pass with `--jd <file>`. It writes
`output/cv-<your-name>-<report-name>.html`. With `--pdf` it renders the PDF
straight after. The HTML is not streamed to the terminal: a token counter
shows that the model is writing, and the result is the file. The template of
an HTML CV is always `templates/cv-template.html`.

With `--latex` it makes the CV in LaTeX instead, in a template you can
change. What follows describes the HTML CV, the default, as far as
[A CV in LaTeX](#a-cv-in-latex), which says how a CV in LaTeX is made and
checked, and where that differs.

The model reads `cv.md`, the profile, the posting and the report. The report
goes without five of its parts: the proposed CV changes (Block E), the
interview stories (Block F), the draft answers (Block H), the Machine Summary
and a cover letter draft, when the report has one. In those the evaluation
writes things about you that `cv.md` does not say, and a tailoring request
that was sent them wrote them into the CV. The request also ends with a rule:
give a job or a project only what `cv.md` gives it, and move no number from
one item to another.

The parts are found by the headings career-ops gives them, `## E)` or
`## Block E`. A report that heads them another way is sent with them. The one
case known is a report an agent wrote from a translated mode set before
2026-10-25, which has its draft answers under `## G)`.

A reply is held to four checks. The first three decide whether anything is
saved. A CV that fails only the fourth is saved for you to edit, and the run
fails:

- It is one complete HTML document and a whole CV: text in its body, section
  titles, the template's stylesheet. No slot of the template is left
  unfilled.
- It holds nothing that runs, embeds or loads something: no script, no frame,
  no image or style taken from a file or from another address. Links to the
  web, to mail and to a phone number are fine.
- The CV sections you listed under `cv.locked_sections` in
  `config/profile.yml` are unchanged.
- Its facts are in `cv.md` or `article-digest.md`. This is the fact check
  `generate-pdf.mjs` runs before it renders.

A reply that fails one of the first two is asked for once more. If that reply
fails too, the command exits 1 and nothing is saved. If the locked sections
changed, nothing is saved and the command exits 1 with the reason.

The fact check is where a local model fails most often, because it adds
keywords from the posting to your skills. So the runner repairs the CV. It
removes each entry the check rejected: a competency tag, one item of a skills
list, one item of a project's technology list. It checks again. If findings
remain it asks the model once more, repairs that reply the same way, and
keeps the better of the two. With `OLLAMA_RETRIES=0` it does not ask again.
Then it tells you what it removed, by name.

Read that list. An entry on it is a word the posting asks for and the check
did not find in `cv.md`. The check looks for the exact words, so an entry can
also be something `cv.md` has in another spelling: one live run lost
`CI/CD Pipelines` where `cv.md` says "CI/CD pipeline". If you do have the
experience, add it to `cv.md` and tailor again. If `cv.md` has it in another
spelling, write it into the HTML in that spelling and run `ollama:pdf`.

A competency tag is removed as a whole, also when the check rejected only one
tool in it. The run then names that tool, as in
`Kubernetes and OpenTelemetry (competency, for "opentelemetry")`.

One kind of entry is not removed. The check reads a list in brackets item by
item, so in `MLOps Platform (SageMaker, Kubeflow, MLflow)` it sees `MLflow)`,
which is in no CV, and rejects it although `cv.md` names MLflow. The runner
writes such a tag or skills line without the brackets around its lists,
`MLOps Platform, SageMaker, Kubeflow, MLflow`, and lets the check read it
again. No word is lost in that step, and the run says
`Written without brackets` with the entry before and after. Round, square,
curly and full-width brackets are handled alike. A bracket that holds no
list, as in `Python (3.11)` or `Go (basic)`, is not taken apart: the item
stays one item. A project's technology line is not rewritten: the check does
not read that line, so it never misread its brackets.

Without the brackets the check reads every item by itself, the name in front
of the list included. One that `cv.md` does not have is then removed like any
other entry: the item from a skills line, a tag as a whole. The run names a
tag that went this way as the model wrote it, with its brackets. What the run
prints after `is now`, and what the receipt holds as `to`, is the entry as it
was saved. A skills line that lost every item is reported as removed, under
the text the model wrote. When writing the skills lines without their
brackets leaves the CV no better by the check's own count, every skills line
keeps its brackets, also one the runner could have rewritten by itself, and
the runner keeps the repair it can make with them in place. A tag loses its
brackets either way. If you write such an entry
into the HTML yourself, use commas or a dash. With a colon after the name the
check rejects it.

Web links are checked by the runner itself, because the fact check does not
read addresses. A link to an `http://` or `https://` address that is in none
of your files (`cv.md`, `config/profile.yml`, `article-digest.md`) is taken
off the CV. The text it was on stays, and the run names each address. The
first CVs linked project pages that do not exist, made up from the GitHub
profile and the project's name. If an address is right, add it to `cv.md` and
tailor again. Mail and phone links are not checked: read them.

The same step runs the wording check of `generate-pdf.mjs`: a stock phrase
such as "proven track record" that `cv.md` does not use blocks the CV like an
unsupported fact. What the fact check only warns about does not block: the run
says how many warnings there are, and `generate-pdf.mjs` lists them when it
renders. Two checks are still made only at the PDF step: the order of the
sections and the order of the jobs.

If the CV is still blocked after that, the HTML is saved for you to edit, the
command exits 1, and no PDF is started. There are four ways on:

- Run `ollama:tailor` again. Every run writes a different CV, and most pass:
  22 of 23 runs for two reports of one posting did here. This is the one to
  try first. The message names the command. It names the report by its
  number, or by its path when the number would not lead back to it, and
  repeats `--jd <file>` when the run had it. A name with a blank or another
  character a shell reads in its own way is in double quotes. After
  `ollama:run` or
  `ollama:pipeline` it is still `ollama:tailor` for that report: `ollama:run`
  again would evaluate the posting a second time. With `OLLAMA_SEED` set, the
  same request gives the same CV, so change the seed or unset it first.
- Edit the HTML and run `ollama:pdf` on it.
- Add the evidence to `cv.md` or `article-digest.md` and tailor again.
- For a claim you have verified, add an exception to `config/cv-facts.json`.

The one blocked run is worth knowing. The model had joined two
lines of the CV into one sentence that neither line supports: a pipeline
"serving as the production gateway for 4 engineering teams", built from a
fraud-detection pipeline and a platform used by four teams. The check stopped
that CV for its verb, "Designed and operated" where `cv.md` says "Designed".
It let the same joined claim through, as a warning, in four of the six CVs
that passed beside it. A claim like this is yours to find.

With `--json`, the receipt carries what was removed under `removed`, what was
written without brackets under `rewritten` (each entry as the model wrote it
under `from` and as it was saved under `to`), the addresses of the links
taken off under `unlinked`, and the check's verdict and findings under
`factCheck`.

Tailoring a report again replaces its HTML file. When the file that is there
differs from the new one, for example because you edited it, it is kept as a
`.bak` copy and the run says so. `ollama:apply` does the same with its answers
file.

For an HTML CV, `ollama:pdf` hands the HTML to `generate-pdf.mjs` and writes
the PDF to `output/`. Besides a report it takes the path of an HTML file:
`npm run ollama:pdf -- output/cv-<your-name>-<report-name>.html`. The paper
size comes from your profile unless you pass `--format=letter` or
`--format=a4`; for the other options of `generate-pdf.mjs`, run that script
yourself. `generate-pdf.mjs` compares the CV's facts with `cv.md` again
before it renders. If it stops there, the command exits with its exit code
and the HTML stays in `output/` for you to read. The failure names the line
`generate-pdf.mjs` stopped with, also in the `--json` receipt. A CV that runs
over its page budget is kept as an `.overflow.pdf` draft beside the output
path: no PDF is delivered and the command exits 1. Rendering needs the
Playwright browser: `npx playwright install chromium`.

When the fact check passes a CV with warnings, the render goes ahead and the
command says how many there were. They are claims to confirm against `cv.md`
before you send the CV. The count is in the receipt as `factWarnings`, and in
the summaries of `ollama:run` and `ollama:pipeline`, next to the number of
entries the tailoring step removed.

The tracker's PDF column is ticked by the merge, not by the render:
`generate-pdf.mjs` records each PDF it writes for a report, and
`merge-tracker.mjs` reads that record. So after `ollama:pdf` or
`ollama:tailor --pdf`, run `npm run merge`. `ollama:pipeline` and
`ollama:run` merge after the PDF, and their rows arrive ticked.

#### A CV in LaTeX

`npm run ollama:tailor -- 003 --latex` makes the CV for report 003 in LaTeX,
in a template whose look you decide. The model is asked for the text of the
CV as one JSON object, never for a line of LaTeX. One `\input` in a bullet
would read a file into the PDF, and turning shell escape off does not stop
that, so the runner writes every command around the text itself.

This path is newer than the live rounds at the top of this page and was not
part of them. Its tests use the stub server and a stand-in for the TeX
engine, and one of them compiles the default template with the example CV
in the real engine where one is installed.

**Choosing the format and the template.** `--latex` or `--html` chooses the
format. Without either, `cv.output_format` in `config/profile.yml` decides:
`latex` makes LaTeX, anything else HTML. The runner makes no Markdown CV, so
`text` makes HTML here. `--template <name>` chooses the template, else
`cv.latex_template` in the profile, else `default`, which is
`lib/ollama-cv-latex-default.tex`. Any other name is a template pack of your
own: see [Writing a template pack](#writing-a-template-pack).

```yaml
cv:
  output_format: latex   # optional: --latex does the same for one run
  latex_template: mine   # optional: the pack in data/templates/mine/
```

With `cv.output_format: latex`, a CV made with `--html` is named by its file
in the step after it,
`Next, the PDF: node ollama-runner.mjs pdf output/cv-<your-name>-<report-name>.html`,
and each `tailor` command that a run in HTML prints as a way on carries
`--html`, so that it makes HTML again: under that profile, `pdf <report>`
takes the report's LaTeX CV, and `tailor` alone makes one. In the same way,
a `tailor` command that a run in LaTeX prints as a way on carries
`--template <name>` when the run took a template the profile does not
choose.

`ollama:tailor`, `ollama:pdf` and `ollama:run` take the three options.
`ollama:pipeline` takes them with `--tailor` or `--pdf`, and refuses them
without either:
`--latex needs --tailor or --pdf: it chooses the format of the tailored CV.`
Two more refusals come before anything is sent:
`--latex cannot be used with --html: a CV is made in one format.` and, for a
CV that is not made in LaTeX,
`--template is for LaTeX CVs: add --latex, or set cv.output_format to latex in config/profile.yml.`

`cv.output_format` is career-ops' own key, and the agent modes read it too.
Set to `latex`, it also sends an agent's `auto-pipeline` and `pipeline` to
career-ops' own LaTeX path, `modes/latex.md`, where `build-cv-latex.mjs`
fills `templates/cv-template.tex`. That path reads neither
`cv.latex_template` nor your packs. To make LaTeX CVs with the runner alone,
leave the key as it is and pass `--latex`. `cv.template` names an HTML
template for career-ops' own tools, and the runner does not read it.

**The header is the profile's.** The name, the title and the contact
details come from `candidate` in `config/profile.yml`: `full_name`, `title`,
`location`, `email`, `phone`, `linkedin` and `github`. The model is never
asked for them. Without `full_name` the run stops before anything is sent:
`config/profile.yml has no candidate.full_name: the CV header needs it.`
A fact the profile leaves out is left out of the header.

**Before the model is asked.** The template is found, checked and written
once with no text, so a template the runner cannot use stops the run here.
A CV in LaTeX has six sections: the summary, experience, projects,
education, certifications and skills. Any other section under
`cv.locked_sections` stops the run too, as in
`cv.locked_sections names 'competencies', which a LaTeX CV has no section for. Remove it from the list, or tailor with --html.`
With `--pdf`, the template's engine is asked for its version as well. Each
of these ends the run with exit 1, before anything is sent or written.

**The request and the reply.** The model reads what it reads for an HTML
CV, with the same five parts of the report left out, and the request ends
with the same rule. In place of `modes/pdf.md` and the HTML template it
gets tailoring rules written for the JSON object, the object's schema, and
the page budget, `cv.max_pages` (2 when unset). A reply in a code fence, or
with a sentence before or after the object, is read all the same. A reply
that is no such object is asked for once more: one cut off before its end,
or one with a field of the wrong type, a job without a company, a role, or
an intro or a bullet, a project without a name, a project address that is
not a plain `http://` or `https://` link, an entry of the education without
a degree or an institution, fewer than two sections with content, or too
little of `cv.md`'s text. In every string, line breaks and tabs become
spaces and invisible characters are removed (control characters, zero-width
spaces, soft hyphens, bidi marks), so no spacing and no character the page
does not show sets the text the checks read apart from the text that
prints. A field that prints no bold (a role, a company, a location, dates, a
project's name, a degree, an institution, a skills category and its items)
also loses the `**` around a word, which `cv.md` writes around its titles and
a model copies. No field prints italics, so every string but a project's
address loses the underscores that mark them, as in `_LLM Eval Toolkit_`.
An underscore inside a word, as in `feature_store`, beside a slash or a
backslash, as in `docs/_draft_/cv.md`, or in an email address stays, as it
does in an HTML CV. The `.json`, the checks and the `.tex` then hold the
same words, except for a pair of underscores beside a backslash: it stays
in the `.json` and the `.tex`, and the checks read it as italics. TeX itself
still prints a few characters as others, which no check sees: see
Characters, under Writing a template pack.

**The checks.** The CV is also written as the plain HTML the checks of an
HTML CV read, and goes through them, with the same functions: the locked
sections, the link rule, and the fact check with its repair and its one
more request. Its dates are the ones the template prints: for a template
with `dates=year`, the years alone (see `dates=year` under Writing a
template pack). One check comes earlier than for an HTML CV: the order of
the jobs, newest first by start date. A reply out of that order is asked
for again with that finding. The run lists the fact check's warnings
itself, after `The fact check passes this CV with`, since no
`generate-pdf.mjs` renders this CV to list them. The order of the sections
is the template's.

Then the runner writes the `.tex`, and for a CV those checks pass, the final
text check of `generate-latex.mjs` reads it as text. It does not stand in
for the checks above: in the macros the text is written in, it sees no
employer, date, degree or skill. When it finds no `cv.md` it does not run,
and says
`The final text check did not run: cv.md was not found.`

**What is written.** `output/cv-<your-name>-<report-name>.json` holds the
text of the CV and the name of its template, and
`output/cv-<your-name>-<report-name>.tex` is made from it. A file that
differs from the one it replaces keeps the old one as a `.bak` copy. The run
ends with
`Next, the PDF: node ollama-runner.mjs pdf output/cv-<your-name>-<report-name>.tex`.
With `--no-save` the `.tex` is printed and nothing is written. With `--pdf`
as well, `--no-save` is refused:
`--pdf cannot be used with --no-save: the PDF is compiled from the saved .tex.`

A CV that the fact check, the order check or the final text check blocks
is saved as its `.json` alone, and the run exits 1 with code `FACT_CHECK`:
`The CV is kept for you to work on: output/cv-<your-name>-<report-name>.json. No .tex was written, and no PDF.`
A `.tex` of an earlier run is left as it was. The ways on are those of a
blocked HTML CV, with the `.json` in place of the HTML file. With
`--no-save` no `.json` is written, so the way that edits it is left out, and
the run says
`Run without --no-save to keep a blocked CV as its .json, for you to edit.`
When the final text check stops at a value with a backslash in it, the run
adds
`The check misread an escaped character: reword that line in the .json.`
With `--no-save`, which writes no `.json`, it adds
`The check misread an escaped character: reword that line.`

A changed locked section saves nothing, not even the `.json`, asks the
model no second time, and exits 1 with code `LOCKED_SECTIONS`, as for an
HTML CV.

With `--json` the receipt carries `"format": "latex"`, the two files under
`tex` and `payload`, and `removed`, `rewritten`, `unlinked` and `factCheck`
as for an HTML CV. The receipt of a blocked CV carries `payload` alone,
since no `.tex` was written, and with `--no-save` the receipt carries the
`.tex` under `text` and no file.

**The PDF.** `--pdf` compiles the CV once it is saved, and `ollama:pdf`
compiles it later. For a report, `ollama:pdf` takes the CV in the format the
options and the profile choose, so with a profile that does not choose
LaTeX it needs `--latex`. It also takes the path of the `.json` or of the
`.tex`:

```powershell
npm run ollama:pdf -- 003 --latex
npm run ollama:pdf -- output/cv-<your-name>-<report-name>.json
npm run ollama:pdf -- output/cv-<your-name>-<report-name>.tex
```

For a report it compiles the `.tex` unless the `.json` is the newer of the
two files, so right after `tailor`, which writes the `.tex` straight after
the `.json`, it compiles the `.tex`, even when the file system gave both the
same time. Once the `.json` is the newer, after you saved it, it works from
the `.json`. With no CV in LaTeX for the report it says
`No tailored LaTeX CV for` and the report, and names the `tailor` command to
run first. A `.json` is written into its template again (`--template`, else
the one the `.json` names, else the profile's), and the `.tex` it makes is
saved in `output/` under the name of the `.json`, the old one kept as
`.bak`. A `.tex` is compiled as it is: see Editing by hand, below. A path
names its own format, so `--latex` and `--html` are refused with one
(`--latex cannot be used with a file: its extension names the format.`),
and with a `.tex` so are the two options it answers itself:
`--template cannot be used with a .tex file: its first line names the template.`
and
`--format cannot be used with a .tex file: the paper size is written in it.`
With a report, they are refused too when it is its `.tex` that is compiled,
as right after `tailor`. To write that CV into another template, or on
other paper, name its `.json` by its path, as the second command above
does, and add the option. A path that leads to no file:
`LaTeX file not found:` and the path.

Whatever the route, the `.json` is checked again first, with its dates as
the template prints them, and nothing in it is repaired or written back. A
changed locked section, jobs out of order, a link to an address none of
your files holds, or any finding of the fact check stops the command with
exit 1 and no PDF. So nothing compiles a CV the checks block. Each of these
failures, and each failure of the compile below, names the file that is
kept, after `The CV is kept:`, and the command to run once it is mended.
For a `.json` outside `output/`, that command names the `.json`: its `.tex`
is written to `output/`, where no `.json` is beside it for `ollama:pdf` of
that `.tex` to read.

`ollama:run`, and `ollama:pipeline` with `--tailor` or `--pdf`, make each CV
in the format the options and the profile choose, and compile its PDF the
same way. With a PDF step, the engine is asked before each CV is asked for.

**The compile.** The `.tex` is compiled in a folder of its own under the
system's temporary folder, and the folder is removed afterwards. A template
pack is copied in first. The engine is the one the template names, and
`pdflatex` when it names none:

| Engine | How it runs |
|---|---|
| `pdflatex`, `xelatex` | Found on PATH. Two passes, because the second prints what the first measured, each with shell escape off, `-interaction=nonstopmode` and `-halt-on-error`, and under MiKTeX with `-disable-installer`. |
| `tectonic` | One call, with `--untrusted`: the copy `node install-latex-engine.mjs` puts beside the scripts when it is there, else the one on PATH. |
| `lualatex` | Refused: `lualatex is not supported: it runs Lua code from the document. Use xelatex.` |

Each call of the engine may take two minutes. **On MiKTeX**, which this
machine has, a compile can install a package the document asks for and
MiKTeX lacks, and can stop to ask first. The runner turns that off, so such
a package stops the compile with the log's error line instead. Install it
with the MiKTeX Console, then run `ollama:pdf` again. The packages of the
default template are installed here.

With `--pdf`, an engine that is not installed is found before the model is
asked; `ollama:pdf` finds it when it compiles. The code is `NO_ENGINE`, and
the reason names the engine and the line that chose it:
`The template's engine, xelatex, is not installed: xelatex --version did not run. A template names its engine on the line %!TEX TS-program = xelatex (pdflatex when it has none).`
A compile that stops has code `LATEX` and quotes the error lines of the
log. The `.json` and the `.tex` stay, and the message ends with
`Fix the template or the .tex` and the command to run again. For a `.json`
outside `output/`, whose `.tex` the next `ollama:pdf` of that `.json` writes
again, it ends with `Fix the template` and the command for the `.json`.
When pdflatex has no way to print a character of the text, the reason adds
`The template's engine cannot print that character: replace it in the .json, or use a template for xelatex.`

**The page limit** is `cv.max_pages`, 2 when unset, as for an HTML CV. A
CV that comes out longer is no PDF: the command exits 1 with code
`PDF_NOT_WRITTEN` and a reason such as
`The CV is 3 pages; the allowed maximum is 2.` Its pages are kept as a
draft beside the PDF, `<name>.overflow.pdf`, then `<name>.overflow-2.pdf`
and on, never over an earlier draft, and the message names the draft after
`The draft is kept:`. A PDF made before stays as it was.

A PDF that is delivered gets its row in `data/pdf-index.tsv`, where
`generate-pdf.mjs` records its own, so the next `npm run merge` ticks the
PDF column; `ollama:run` and `ollama:pipeline` merge after it. A PDF made
from a file named by its path is recorded under the number of the report
whose CV the file is by the name `tailor` gave it, however many digits the
number has, and else under three digits in the file's name, as
`generate-pdf.mjs` reads one. pdflatex and xelatex leave out a character
the font has no glyph for, and exit 0 all the same. The run then says so, with
`The PDF lacks these characters, which the template's font does not have:`
and the characters. When the build folder cannot be removed, the run names
it after `The build folder could not be removed:`. It holds a copy of the
CV, so delete it yourself. With `--json`, the receipt of `ollama:pdf`
carries `"format": "latex"`, `tex`, `payload`, `pdf`, `pages` and
`factWarnings`.

**Editing by hand.** The `.json` is the CV, and the place to change its
text. Edit it and run `npm run ollama:pdf -- output/<name>.json`: it is read
as a reply is read, checked as above, and never rewritten.

A `.tex` may change the layout of its CV and never its text: a space, a
font, a page break. While it is not older than its `.json`, it is the file
`npm run ollama:pdf -- 003 --latex` compiles, and you can name it by its
path as well. The runner writes the `.json` into the template the `.tex`
names and compares the text of the two, call by call of the macros that
carry text. A difference stops it with code `FACT_CHECK`:
`The text of <name>.tex is not the text of <name>.json. Change the text in the .json and run pdf on it: a .tex may change layout only.`
A `.tex` without its `.json` beside it is refused with code `BAD_INPUT`:
`pdf needs the payload of this .tex: <name>.json was not found beside it.`
So is one whose first line no longer names its template:
`This .tex does not say which template it was made from: its first line must read % career-ops-latex: template=<name>.`
The final text check reads the `.tex` as it is, and the engine is the one
its header lines name.

The comparison has a limit: it reads what the calls hold, not what the page
prints. A `.tex` that redefines a macro in its preamble, or in its body
under another spelling such as `\csname`, that puts a call inside
`\iffalse ... \fi` with other text after it, or that adds a line of prose
between two calls, compares as unchanged, and the final text check does not
notice a changed employer, date, degree or skill. Change the text in the
`.json` only, and read a `.tex` you edited before you send its PDF.

For a report, `ollama:pdf` works from the `.tex` unless the `.json` is
newer. `tailor` writes the `.tex` last, so it is the `.tex` until you save
the `.json`: a `.json` saved after its `.tex` is written into the template
again, over the `.tex`, which is kept as `.bak`. Tailoring the report again
replaces both, keeping each as `.bak`. A layout you want in every CV belongs
in a template pack.

#### Writing a template pack

A template decides how a CV in LaTeX looks: its fonts, its headings and
their order, its spacing. The runner fills in the text, and writes no
heading and no layout of its own. The built-in template,
`lib/ollama-cv-latex-default.tex`, is a plain `article` for `pdflatex`. A
pack is a template of your own, in a folder under your data root:

```text
data/templates/<name>/
    cv-latex.tex      the template
    ...               what it uses: a class, fonts, images
```

`<name>` is lower-case letters, digits and hyphens, and begins with a letter
or a digit. `default` is the built-in template and has no folder. Choose a
pack with `--template <name>`, or with `cv.latex_template: <name>` in the
profile. `data/` is ignored by git, so a pack stays on your machine. The
file is `cv-latex.tex`, not `cv-template.<name>.tex`: that is how
career-ops' own CV templates are named, and `build-cv-latex.mjs` would try
to fill it.

Start from a copy of the default template, whose comments explain each part.
With the data root in the checkout:

```powershell
New-Item -ItemType Directory -Force data\templates\mine | Out-Null
Copy-Item lib\ollama-cv-latex-default.tex data\templates\mine\cv-latex.tex
```

With a `.json` from an earlier run, this compiles that CV in the pack
without asking the model. It replaces that CV's `.tex`, kept as `.bak`, and
its PDF:

```powershell
npm run ollama:pdf -- output/cv-<your-name>-<report-name>.json --template mine
```

For each compile the whole folder is copied beside the `.tex`, so
`\documentclass{mycv}` finds `mycv.cls`, and a font or an image is found by
its path inside the folder. A symbolic link in the pack is refused, and a
`cv.pdf`, `cv.log` or `cv.aux` in it is left out of the copy. A template the
runner cannot use is refused before the model is asked, with its file, its
line and what is wrong. The reason ends with the templates there are, after
`Templates here:`.

**Header lines.** The comment lines before `\documentclass` set two things,
each on a line of its own:

- `%!TEX TS-program = xelatex` names the engine: `xelatex`, `pdflatex` or
  `tectonic`, and `pdflatex` when no line names one. `% !TeX program =` is
  read as well. `lualatex` is refused.
- `% career-ops-latex: dates=year` shows only the years of each date, as
  described below. `dates=full`, the default, shows dates as written.

An option or a value the runner does not know, two options on one line and
an option set twice are refused. Line 1 of every `.tex` the runner writes is
`% career-ops-latex: template=<name>`, which `ollama:pdf` reads. A template
does not carry that line.

**Placeholders.** Each is filled in once, in a single pass, so a value that
holds `{{` and `}}` stays text:

| Placeholder | What it holds |
|---|---|
| `{{NAME}}` | `candidate.full_name`. |
| `{{FIRST_NAME}}`, `{{LAST_NAME}}` | The name without its last word, and its last word. A name of one word is all `{{FIRST_NAME}}`, and `{{LAST_NAME}}` is empty. |
| `{{TITLE}}`, `{{LOCATION}}`, `{{PHONE}}` | `candidate.title`, `candidate.location` and `candidate.phone`. |
| `{{EMAIL}}`, `{{EMAIL_URL}}` | `candidate.email` as text, and as a `mailto:` address for `\href`. |
| `{{LINKEDIN_DISPLAY}}`, `{{LINKEDIN_URL}}` | `candidate.linkedin` as text, without its scheme or a final slash, and as an address for `\href`. |
| `{{GITHUB_DISPLAY}}`, `{{GITHUB_URL}}` | The same for `candidate.github`. |
| `{{DATE}}` | The day the `.tex` is written, as `21 Nov 2026`. |
| `{{PAPER}}` | `a4paper` or `letterpaper`: `page_format` in the profile, letter when it names neither, or `--format` of `ollama:pdf`. |
| `{{SUMMARY}}`, `{{EXPERIENCE}}`, `{{PROJECTS}}`, `{{EDUCATION}}`, `{{CERTIFICATIONS}}`, `{{SKILLS}}` | The six sections, written in the macros below. |

Every value is escaped for LaTeX, so an `_`, a `&` or a `%` in the profile
prints as itself. An address goes through a cleaner for addresses instead.
Any other placeholder refuses the template, as in
`{{AWARDS}} is not a placeholder of a LaTeX CV.` So does a name in double
braces written any other way, with spaces inside the braces, in lower case
or with a digit, such as `{{ NAME }}`, `{{Name}}` or `{{PHONE2}}`: the runner
would fill none of them, and each would print as its name. For one of the
placeholders above in another spelling, the reason gives the one to write.
A name begins with a letter or an underscore, so TeX code such as
`\setlength{\parindent}{{0pt}}` is no placeholder. TeX code that holds such
a name alone in double braces, such as `\def\x{{a}}`, reads as one and is
refused: break its line after the two braces with a `%`, which TeX reads as
the same code.

**Blocks.** The lines from `%%IF KEY` to `%%END KEY` are left out when KEY
has nothing to show, and the two marker lines are always removed. KEY is one
of the six sections, `TITLE`, `LOCATION`, `EMAIL`, `PHONE`, `LINKEDIN` or
`GITHUB`. Put each section's heading in its block, so that a section with
nothing to show goes with its heading, and each contact detail in a block of
its own, so that one the profile leaves out leaves no separator behind.
Blocks may nest, and never cross. A key that is misspelt or never closed
refuses the template.

**Macros.** The sections are written as calls of eleven macros, one call to
a line. The template defines all eleven in its preamble, and decides how
each one looks:

| Macro | What it is |
|---|---|
| `\cvSummary{text}` | A paragraph of the summary. |
| `\cvEntryHead{role}{company}{location}{dates}` | A job. Its blocks follow it. |
| `\cvEntryBlock{lead}{intro}{items}` | A group of bullets under a job: a title for the group, such as a product or a client; a sentence before the bullets; and the bullets, one `\cvItem` call to a line. |
| `\cvItem{text}` | A bullet. |
| `\cvEntrySep` | Between two jobs, and between two entries of the education. Never after the last. |
| `\cvProjects{intro}{items}` | The projects: a sentence before them, and one `\cvProject` call to a line. |
| `\cvProject{name}{url}{text}` | A project. The address is empty unless one of your files (`cv.md`, `config/profile.yml`, `article-digest.md`) holds it. |
| `\cvEducation{degree}{institution}{dates}` | An entry of the education. |
| `\cvCert{text}` | A certification. The default template lists them in an `itemize` that its body opens around `{{CERTIFICATIONS}}`. |
| `\cvSkill{category}{items}` | A line of skills. The category comes without its colon. |
| `\cvStop{.}` | Prints nothing: see below. |

An empty argument is written `{}`, so a template can test for one, as the
default template does with `\cvIfEmpty`. An argument that begins with `[`
begins with `{[}` instead, so that a macro ending in `\item #1` does not
read it as the label of the item. Bold, `**...**` in the JSON, is written
`{\bfseries ...}`, in the summary, a lead, an intro, a bullet, a project's
text and a certification, and nowhere else.

**`\cvStop`.** Define it as `\newcommand{\cvStop}[1]{}`. The runner writes
`\cvStop{.}` straight after each call of `\cvSummary`, `\cvEntryHead`,
`\cvItem`, `\cvProject`, `\cvEducation`, `\cvCert` and `\cvSkill`. The final
text check reads the `.tex` as text, and without a full stop there, a list of
tools after "using" ran on into the next bullet, and a CV true to `cv.md`
was blocked. Write `\cvStop{.}` after each heading of your template too, as
the default template does.

**Characters.** The runner writes `<`, `>`, `|`, the arrow (U+2192) and the
plus-minus sign (U+00B1) as themselves, and a tilde as `\~{}`: written as
commands, the final text check read them as names of tools. So the fonts of
a template have to print them:

- Under `pdflatex`, load `\usepackage[T1]{fontenc}` and
  `\usepackage{lmodern}`: in pdflatex's own encoding, `<`, `>` and `|` print
  as other characters. Declare the arrow with
  `\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}`. Declare the
  tilde as well, or T1 prints `\~{}` as an accent, which the text layer of
  the PDF, the part an applicant tracking system reads, gives as U+02DC:

  ```latex
  \ifcsname T1\string\~\endcsname
    \makeatletter
    \DeclareTextCompositeCommand{\~}{T1}{\@empty}{\textasciitilde}
    \makeatother
  \fi
  ```

  A character outside T1 that no line declares stops the compile, with the
  hint quoted above.
- Under `xelatex` or `tectonic`, which both run XeTeX, load `fontspec` and
  an OpenType font, and declare nothing: `\~{}` prints a tilde there. Do not
  load T1 under XeTeX, which finds no glyph past U+00FF in a T1 font and
  leaves out the en dash and the arrow.

The default template does both, by `\ifdefined\XeTeXrevision`.

TeX also prints a few characters, and a few runs of them, as other
characters, and the checks, which read the text before TeX does, never see
that: `--` prints as an en dash, `---` as an em dash, `<<` and `>>` as
guillemets, `,,` as a low double quote, and an apostrophe and a backquote
as curly single quotes.
Under XeTeX, whose fontspec turns on `Ligatures=TeX` for the main and the
sans-serif font, a straight double quote prints as a closing curly quote as
well, so every opening quote comes out as a closing one. These are TeX's
ligatures. A bullet that names `--dry-run` passes every check and prints an
en dash before `dry-run`, for the recruiter and for a parser alike, and so
does a line of `cv.md` whose parts are joined with `--`. A pack that wants
these characters printed as written turns the ligatures off: under XeTeX
with `Ligatures=TeXOff` in the options of its fonts, and under pdflatex,
where they are part of the font, with microtype's `\DisableLigatures`.

**`dates=year`.** The dates of each job and of each entry of the education
show their years only. A range shows its first year, an en dash and its
last year, or the word that ends it: `Mar 2019 - Present` shows 2019 and
Present. What follows that word is dropped, so
`Mar 2020 - Present (promoted 2022)` shows 2020 and Present as well, not
2022. A month before the last year, in any language and with a day or not,
is no such word: `Jan 2018 - March 1, 2020` shows 2018 and 2020. So one word
before a year is read as its month, unless it is a word for today, such as
Present, now, heute or actualidad, in a language career-ops has modes for:
`Mar 2020 - Present, 2022` shows 2020 and Present. A range within one year
shows that year: `Feb 2017 - Nov 2017` shows as `2017`. A date is split
once, at the first dash or `to` with a space on each side, or at a dash
written straight after a year and before a word or a year. Two things
cannot be told from the text. The words around a lone year are dropped:
`Expected 2027` shows as `2027`, which reads as a degree finished. And a
dash before two digits never splits, since `2011-12` is a month in ISO
form, so the academic year `2019-20` shows as `2019`. Write such a date
another way, or keep `dates=full`.

The checks read the dates as the template prints them, so a year that
`cv.md` contradicts is found, as in an HTML CV. That holds the locked
sections to the printed dates too: the locked-section check compares the
words of a section with `cv.md`'s, and a date that `cv.md` writes with a
month or a remark prints without it. So a pack with `dates=year` cannot be
used with a locked experience or education section, unless `cv.md` writes
every date in that section the way it prints, as years and at most a word
such as Present: otherwise the run, and `ollama:pdf`, stop with code
`LOCKED_SECTIONS`, and nothing is saved or compiled.

**babel.** If a pack loads babel, load it with `shorthands=off`, as in
`\usepackage[shorthands=off,english]{babel}`. A language's shorthands turn
characters of the text into commands: in German, `"a` prints as an a with
an umlaut. The CV would then print something other than the text that was
checked.

**A pack is code.** The engine runs its class and preamble with your
rights: it can read any file you can read and put it into the PDF, and it
runs until the time limit. Use only packs you wrote or read. Shell escape is
off for every engine, and pdflatex and xelatex install no package: the
runner turns MiKTeX's installer off. tectonic can still download the files
a document asks for from its online bundle. File reads are not restricted.
What the runner keeps inert is the text it writes into a template: every
value is escaped, and a value that would still hold a command stops the
render with
`The CV text would run a LaTeX command the renderer does not write:`. The
template itself is not checked.

### `ollama:apply`

`npm run ollama:apply -- <report>` drafts answers to application questions
for that report. It prints them and saves them to
`output/apply-<report-name>.md`. It never opens or submits a form. For a
report that scored below 4.0 it stops and tells you to repeat the command with
`--force`. It does the same for a report that advises against applying
whatever it scored: one that names a hard stop, gives the work-authorisation
tier No sponsorship, or rates the posting Suspicious. For a report whose
header states no score it can read, it says that the score check was not made
and goes on.

Because it never opens the form, it sees the form's questions only if you
give them to it:

```powershell
npm run ollama:apply -- 003 --questions questions.txt
```

`questions.txt` holds the questions as you pasted them from the form, as
UTF-8 or UTF-16 text. The model answers each one in order, and is told to
keep to a character limit a question states. Without `--questions` it answers
eight usual application questions. The saved file says which of the two it
answered.

Where the model closes an answer with a line such as
`Length: 998/1000 characters`, the runner replaces the first number with its
own count of the answer as saved, here `Length: 773/1000 characters`, because
a model cannot count characters. An answer longer than its limit is marked
`(over the limit: shorten it)`. Nothing else is counted: a length the model
states in other words, under an answer or in its closing notes, stays as the
model wrote it. In one live run the file said that a draft of 829 characters
was about 1,050 and had to be trimmed to fit 1,000.

With streaming on, the terminal shows the model's own numbers as they arrive,
and a line after the reply says how many were recounted. With `--no-save`
the answers are then printed once more as counted. Check the count against
the form all the same: forms differ in how they count line breaks.

Legal and personal fields, such as salary, work authorisation, sponsorship
and start date, come back as `Ask candidate: ...` with a suggestion from your
profile, for you to fill in.

A reply that is not a set of answers, such as one that asks for the form, is
asked for once more. If the second reply is not one either, the command fails
and nothing is saved.

The answers are drafts to read, change and paste yourself.

The model that writes them is not sent the whole report. Five parts are left
out, because what the evaluation made up about you there otherwise comes back
as fact in an answer: the proposed CV changes (Block E), the interview stories
(Block F), the draft answers (Block H), the Machine Summary and a cover letter
draft, when the report has one. The request also ends with a rule: state for a
job or a project only what `cv.md` states for it. What these two do and do not
achieve is measured under
[what not to trust](#what-was-tested-and-what-not-to-trust).

So the draft answers in the report itself (Block H) are not the ones to use.
They are written in the same pass as the interview stories and repeat their
inventions. Use the file this command writes.

### `ollama:run`

`npm run ollama:run -- <url>` or `npm run ollama:run -- --file <path>`
evaluates the posting. If the score is at least `--min-score` (default 4.0)
and the report does not advise against applying, it goes on to tailor the CV,
render the PDF and draft the answers. Then it merges the tracker.

`ollama:run` does not read standard input. Like `ollama:eval`, it refuses a
posting typed as one short line of text.

| Option | Meaning |
|---|---|
| `--file <path>` | Read the job description from a file. |
| `--posting-url <url>` | Where a posting given as a file lives. Recorded in the report and the tracker row. |
| `--questions <file>` | The form's questions, for the answers step. See [`ollama:apply`](#ollamaapply). A file that does not exist stops the run before the evaluation. |
| `--min-score X` | Score the posting needs for the CV, the PDF and the answers. Default 4.0. |
| `--no-merge` | Leave the tracker row in `batch/tracker-additions/`. |
| `--latex`, `--html`, `--template <name>` | The format of the CV, and the template of a CV in LaTeX: see [A CV in LaTeX](#a-cv-in-latex). |

A posting that scores below `--min-score` is evaluated, merged and left at
that, with exit code 0. `--min-score` is the only score threshold here:
answers are drafted for any posting that reaches it, where `ollama:apply`
alone asks for `--force` below 4.0.

The same holds for a posting whose report advises against applying, whatever
it scored: a hard stop in the report's Machine Summary, the work-authorisation
tier No sponsorship, or a posting rated Suspicious. The run says
`Stopping after the evaluation: the report advises against applying (...)`
and prints the two commands that make the CV and the answers anyway,
`node ollama-runner.mjs tailor <report> --pdf` and
`node ollama-runner.mjs apply <report> --force`. For a CV in LaTeX the first
of the two ends with `--latex`, followed by `--template <name>` when the run
took a template the profile does not choose. For a CV in HTML under a
profile that chooses LaTeX, it ends with `--html`. No value of
`--min-score` changes that.

One step that fails does not stop the others where they can still run. If the
PDF cannot be rendered, the answers are still drafted. If the CV cannot be
tailored, or the fact check still blocks it, there is no PDF step and the
answers are still drafted. Either way the exit code is 1 and the summary at
the end says which step failed.

### `ollama:scan`

Everything after `--` goes to `scan.mjs` unchanged, for example
`npm run ollama:scan -- --dry-run`. It is the same scanner as `npm run scan`
and it never calls a model. Its options are listed in
[SCRIPTS.md](SCRIPTS.md).

## Configuration

Everything has a default. Set a variable only to change something.

A variable can be set in three places. The shell wins, then the `.env` file in
your data root, then the `.env` file in the folder you run from.
`CAREER_OPS_ROOT` itself has to be set in the shell, because the runner needs
it to find that `.env`. [`.env.example`](../.env.example) has a commented block
with every variable below.

| Variable | Default | What it does |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where the runner connects. A trailing `/` or `/v1` is removed. |
| `OLLAMA_HOST` | not a client address | This is the Ollama server's own setting. The runner reads it only when `OLLAMA_BASE_URL` is unset. A real host is used. An address that means every interface (`0.0.0.0`, `::`) is used as `localhost` on its port when it names one, and ignored when it names none. |
| `OLLAMA_ALLOW_REMOTE` | unset | Must be `1` before the runner talks to any host other than `localhost`, `127.0.0.1` or `::1`. |
| `OLLAMA_MODEL` | automatic | The model to use. Unset, the runner takes the first installed name from the preference list. |
| `OLLAMA_MODEL_PREFERENCE` | `qwen3.8:27b,gemma4:26b-a4b-it-q4_K_M,glm-4.7-flash:latest,qwen3-coder:30b` | The order to try when no model is named. |
| `OLLAMA_NUM_CTX` | `65536` | Context window in tokens. Unset, it is raised automatically. Set, it is used as given. Details below. |
| `OLLAMA_NUM_PREDICT` | `16384` | Output cap in tokens. `28000` when thinking is on and this variable is unset. |
| `OLLAMA_THINK` | `auto` | `auto`, `on` or `off`. `auto` is off. |
| `OLLAMA_KEEP_ALIVE` | unset | How long Ollama keeps the model loaded after a request: a duration with a unit (`10m`, `1h`) or a number of seconds. Unset, the runner sends nothing and the server's own setting decides. This is also the server's own variable, so one value serves both. |
| `OLLAMA_TEMPERATURE` | `0.2` | From 0 to 2. |
| `OLLAMA_TIMEOUT_MS` | `900000` | Time limit for one whole request, the streamed reply included. Digits only. |
| `OLLAMA_RETRIES` | `1` | From 0 to 3. Asked for again: an empty reply, a tool call, a reply that fails its format check, a connection that drops in the middle of a reply. |
| `OLLAMA_SEED` | unset | Digits only. Unset, every request is a fresh sample. Set, a retry uses the next seed. |

The runner also honours the variables every career-ops script reads:
`CAREER_OPS_ROOT`, `CAREER_OPS_DATA_DIR`, `CAREER_OPS_PROFILE`,
`CAREER_OPS_ADDITIONS`, `CAREER_OPS_PIPELINE`, `CAREER_OPS_NO_PLAYWRIGHT` and
the `PRIVACY_*` settings. The data root is described under "Custom Data
Directory" in [DATA_CONTRACT.md](../DATA_CONTRACT.md).

### How the automatic choices are made

**Model.** `--model` wins, then `OLLAMA_MODEL`, then the first installed name
in `OLLAMA_MODEL_PREFERENCE`. `name` and `name:latest` are the same model.
Embedding models are never picked. There is no silent fallback: if the model
you named is not installed, or nothing on the preference list is, the run
stops and lists the chat models that are installed.

**Context window.** Before sending anything the runner estimates the prompt
and checks that the prompt plus 5%, plus room for the reply (up to 12,000
tokens), fits in `num_ctx`. If it does not fit and `OLLAMA_NUM_CTX` is unset,
the runner moves to 98304, then 131072, never above the model's own maximum.
If you set `OLLAMA_NUM_CTX` or `--num-ctx` yourself, the value is never
changed, and a prompt that does not fit stops the run before anything is
sent.

With thinking forced on, the room is the whole output cap, 28,000 tokens by
default, because the thinking and the reply come out of the same cap. An
evaluation prompt then no longer fits 65536, so the runner moves to 98304,
which takes more GPU memory and reloads the model.

**Thinking.** With `auto`, thinking is off for every model that can think. A
model without the thinking capability is sent no thinking setting at all.
`--think=on` or `OLLAMA_THINK=on` forces it on. Nothing measured supports that
for any model: the reasons are under [Models](#models).

**Fixed settings.** Every request sets the temperature (0.2 by default). Left
alone, Ollama uses each model's own temperature, which is 1.0 for the three
models measured here. `keep_alive` is sent only when you set
`OLLAMA_KEEP_ALIVE`: see [Freeing the GPU](#freeing-the-gpu).

**Asking again.** A reply is asked for again, up to `OLLAMA_RETRIES` times,
when it is empty, when it is a tool call, when it fails its format check or
when the connection dropped in the middle of it. A request without a seed is
simply sent again. A seed you set moves on by one. A reply cut at the output
cap, a truncated prompt, a missing model and a timeout are not asked for
again: they would fail the same way. Apart from that, a request whose
connection fails before the reply starts is sent once more by itself.

The runner uses Ollama's native `/api/chat` endpoint. The OpenAI-compatible
`/v1` endpoint ignored both a per-request context size and `think: false` when
it was probed on 0.40.16, so on that route the context is whatever the server
default happens to be.

## Models

### The probe

Measured on 2026-11-17 with Ollama 0.40.16 on an RTX 5090 (32,606 MiB) under
Windows 11, before the runner was written. The task was one evaluation of an
invented 696-word job description against the example CV and profile. The
prompt came to between 38,798 and 40,703 tokens, depending on the model's
tokenizer. Settings: `num_ctx` 65536, temperature 0.2, `num_predict` 12000
unless the row says otherwise.

**Each configuration was run once.** Other programs were using the same Ollama
server at the time, which inflates the model load times. Scores also move
between runs: the probe recorded 2.5, 3.2, 3.3 and 3.5 from `qwen3.8:27b` for
the same job description. Read the table as a first look, not a benchmark.

| Model | Thinking | Time for one evaluation | Output | GPU memory | What happened |
|---|---|---|---|---|---|
| `qwen3.8:27b` | off | 123 s, of which 48 s was loading the model | 8,196 tokens at 134 tokens/s | 20,389 MiB | Finished. All seven report blocks and a valid score summary. Score 2.5, Skip. The only run that treated the blocker as a hard stop. |
| `qwen3.8:27b` | on, `num_predict` 16000 | 138 s | 16,000 tokens at 136 tokens/s | 20,389 MiB | Cut off at the output cap before the score summary. 33,679 characters of thinking came first. |
| `gemma4:26b-a4b-it-q4_K_M` | on | 37 s, model already loaded | 5,892 tokens at 182 tokens/s | 18,099 MiB | Finished. Valid summary, score 3.5, Consider. It flagged the blocker and did not treat it as a hard stop. |
| `gemma4:26b-a4b-it-q4_K_M` | off | 37 s, of which 17 s was loading the model | 2,689 tokens at 174 tokens/s | 18,099 MiB | Finished with a valid summary, but contradicted itself: it flagged the blocker, then reported no hard stops and scored the job 4.8, Apply. |
| `glm-4.7-flash:latest` | off | 92 s, of which 19 s was loading the model | 8,598 tokens at 144 tokens/s | 21,482 MiB | Finished. Valid summary, score 4.0, Apply. It marked work authorisation as not needed, so it missed the blocker. Longest report: 38,189 characters. |
| `deepseek-r1:32b` | default | stopped by hand after 20 minutes | about 3 tokens/s | does not fit: 83% on the GPU, the rest in system memory | Unusable at this context size. |
| `mistral-small:latest`, `phi4-reasoning:plus` | | not run | | | Their maximum context is 32,768 tokens, less than the prompt. The runner refuses them. |
| `qwen3-coder:30b` | | not run | | | Fourth on the default preference list because it is installed and its context is large enough (262,144). Its evaluations are untested. |

The blocker: the posting is remote within UTC-1 to UTC+3 only and says the
employer does not sponsor visas, while the example candidate is authorised to
work in the United States only.

The other chat models installed on this machine were not measured.
`npm run ollama:models` shows whether each can hold the prompt. The two 70B
models (`llama3.1:70b-instruct-q4_K_M`, `deepseek-r1:70b`) are 39.6 GiB of
weights each, more than the card has, so they would run partly on the CPU.
`ollama:models` still lists them as `fits`: that column is about the context
window, not about GPU memory.

### Thinking, measured again with the runner

The probe ran `gemma4` once with thinking on and once with it off, and the
run with thinking on looked better. So the runner first turned thinking on for
`gemma4`. The live test measured that setting again, with the runner's own
prompt at `num_ctx` 65536:

| `gemma4:26b-a4b-it-q4_K_M` | Result |
|---|---|
| Thinking on, in the runner | The first reply had no summary block. The second stopped at the output cap with no reply text. Exit 1 after 189 seconds. |
| Thinking on, 3 more runs | 2 of 3 spent the whole output budget thinking: 25,520 output tokens, no reply, 144 to 150 seconds. 1 of 3 wrote a report in 30 seconds. |
| Thinking off, 3 runs | 3 of 3 wrote a complete report in 19 to 26 seconds, with a score of 4.8 each time. None carried the score summary block. |
| Thinking off, tailoring, 3 runs | 3 of 3 wrote a complete CV in 22 to 27 seconds. |

`qwen3.8:27b` and `glm-4.7-flash` were already off. Nothing measured supports
thinking for any model, so `auto` now means off.

### What the defaults owe to these runs

- **`qwen3.8:27b` is the default, with thinking off.** With thinking on it
  spent the whole 16,000-token output cap and never reached the summary.
  Without thinking, a full report took between 8,200 and 10,800 tokens across
  the probe's runs.
- **`gemma4` is second, with thinking off.** It is the fast option: 19 to 26
  seconds for a report in the live test. Its reports are short. In the probe
  the one it wrote with thinking on was about a third as long as the default
  model's (11,133 characters against 32,053), and the one without thinking
  was shorter still. It leaves out the score summary block in most of its
  reports, so the runner reads its verdict from the rest of the report. Its
  one probe run without thinking contradicted itself.
- **`glm-4.7-flash` is third.** Its report was well formed and it missed the
  blocker in its one run.
- **`num_ctx` is 65536.** The prompt is about 40,000 tokens, and a full report
  was 5,500 to 9,000 more in the live rounds from the third on. It was 7,700 to
  10,800 before the model was told not to copy the posting, in the second
  live round and in the probe. A window of 49152
  leaves about 9,000 tokens for the reply, which is too tight. At 32768 Ollama
  cut the prompt: see [Troubleshooting](#troubleshooting).
- **`num_predict` is 16384.** A cap of 6000 cut the `qwen3.8:27b` report off
  in its last block.

Memory for `qwen3.8:27b` by context size, all of it on the GPU:

| `num_ctx` | GPU memory |
|---|---|
| 49152 | 19,349 MiB |
| 65536 | 20,389 MiB |
| 131072 | 24,549 MiB |

At 98304, the window an evaluation moves to when thinking is forced on, the
model also loaded fully on the GPU (checked on 2026-11-18). That day's memory
reading is left out of the table: other programs were using the GPU, so it
does not compare with the three above.

Once the model is loaded, expect about a minute per evaluation with
`qwen3.8:27b`: 52 to 74 seconds for the five evaluations of the third live
round that found the model loaded, at 5,500 to 8,800 tokens of report, and 58
to 79 seconds for the three each of the fourth, the fifth and the sixth round
that came after the round's first, at 5,800 to 9,000 tokens. The
probe estimated 75 to 100 seconds (about 14 seconds to read the prompt, then
129 to 136 tokens per second). Expect about half a minute with `gemma4`, and
about 40 seconds when it has to be loaded first, as in the last three rounds.

## Troubleshooting

### The server is not reachable

```text
Ollama not reachable at http://localhost:11434. Start it (ollama serve) or set OLLAMA_BASE_URL.
```

Start Ollama (the desktop app, or `ollama serve`) and run
`npm run ollama:doctor`. If your server is somewhere else, set
`OLLAMA_BASE_URL`.

A request whose connection fails before the reply starts is sent once more by
itself. You see this message when the second try failed too.

### `OLLAMA_HOST=0.0.0.0`

`OLLAMA_HOST` is the Ollama server's setting. `0.0.0.0` tells the server to
listen on every network interface. It is not an address a client can connect
to, so the runner ignores it and uses `http://localhost:11434`. This machine
has `OLLAMA_HOST=0.0.0.0` set. Nothing needs changing for the runner.

When the value names a port, the runner uses that port on this machine:
`OLLAMA_HOST=0.0.0.0:8080` gives `http://localhost:8080`, which is where such a
server listens.

Two things follow. With that setting, other machines on your network can
reach your Ollama server; that is Ollama's behaviour and the runner does not
alter it. And if you want the runner to use an Ollama server on another
machine, `OLLAMA_HOST` is the wrong variable: set
`OLLAMA_BASE_URL=http://<host>:11434` and `OLLAMA_ALLOW_REMOTE=1`.

### The address answers with a redirect

```text
Ollama API error: HTTP <status>: <address> answered with a redirect to <other address>. Redirects are not followed: only the configured address was checked, and requests to it carry your CV. Set OLLAMA_BASE_URL (or --url) to the Ollama server itself.
```

An Ollama server never redirects. Something else answers on the address: a
proxy, a tunnel, another program on the port. The runner sends nothing to the
address the redirect names. Point `OLLAMA_BASE_URL` at the server itself.

### The model is not installed

```text
Model "<name>" is not installed in Ollama. Installed chat models: <list>. Set OLLAMA_MODEL=<installed model> or run: ollama pull <name>
```

The run stops before sending anything. Either pull the model or pick an
installed one with `--model` or `OLLAMA_MODEL`. `npm run ollama:models` shows
what you have.

When you named no model and nothing on the preference list is installed, the
message starts with `None of the preferred models are installed` instead, and
lists the installed chat models in the same way.

The legacy script behaves differently. `node ollama-eval.mjs` defaults to
`llama3.3`, which is not installed here. It sends the request and fails with
`Ollama API error: HTTP 404` and `model 'llama3.3' not found`.

### The prompt was truncated

Ollama does not refuse a prompt that is longer than the context window. It
cuts the prompt and answers anyway. Measured on this machine with the legacy
script at its default `num_ctx` of 32768: Ollama processed about 16,400
tokens of a prompt of about 38,000, the script saved a normal-looking report,
and that report scored the job 3.3. The same job scored 3.8 once the whole
prompt fitted.

The runner guards against this twice. Before sending, it checks that the
prompt fits and raises `num_ctx` or stops (see
[Configuration](#configuration)). After the reply, it compares the number of
prompt tokens Ollama reports with its own estimate, and fails the run if
Ollama processed less than 85% of it:

```text
Ollama processed <n> prompt tokens but ~<m> were sent: the prompt was truncated. Raise OLLAMA_NUM_CTX.
```

If either check stops a run:

- Unset `OLLAMA_NUM_CTX` and drop `--num-ctx`, so the runner can size the
  window itself.
- Or raise the value, up to the model's maximum.
- Or use a model with a larger maximum. `mistral-small` and
  `phi4-reasoning:plus` stop at 32,768 tokens and cannot hold this prompt.
- If you forced thinking on, leave it off. The window then has to hold 12,000
  tokens of reply, not the whole output cap.

### The reply was cut off

```text
Reply stopped with done_reason "length" after <n> chars. Raise OLLAMA_NUM_PREDICT (now <x>) or OLLAMA_NUM_CTX (now <y>).
```

The model reached the output cap, or the context window filled up. The run
fails and nothing is saved. It is not asked for again: the same settings
would fail the same way.

- Raise `OLLAMA_NUM_PREDICT`.
- If the context window is the limit, raise `OLLAMA_NUM_CTX`.

When the model used the whole cap for thinking and never began its answer,
the message is a different one:

```text
Reply stopped with done_reason "length" before the answer began: the model spent its output budget thinking (<n> chars of it). Leave thinking off: drop --think on or OLLAMA_THINK=on. Off is the default.
```

Do what it says. More room does not help: `gemma4` used 25,520 tokens this
way. If you did not switch thinking on, the model thinks whatever it is
asked, and the message tells you to use another model instead.

### The model called a tool

```text
Ollama returned a tool call instead of a text reply.
```

The prompts hold instructions written for an agent with tools, and a local
model sometimes answers by calling one. The runner tells the model that it
has no tools and asks again when it calls one anyway, so you see this message
only when every attempt ended that way. Run the command again. If it keeps
happening, try another model with `--model`.

### The model reloads between requests

Ollama loads a model for one context size. A request that asks for a different
size makes it unload the model and load it again. In the probe a clean load
took 8 to 19 seconds, and up to 106 seconds while other programs were
competing for the server.

On this machine that matters because the runner is not the only client. Since
the server last started on 2026-11-14, its log shows models loaded at seven
different context sizes, from 4096 to 131072, and only some of those requests
came from the probe. The Ollama desktop app is set to a default context of
131072, so a request that names no size gets that. The server handles one
request at a time (`OLLAMA_NUM_PARALLEL` is 1), so requests also queue.

The runner uses one context size for a whole run. `ollama:doctor` warns when
the model is currently loaded at another size. You have two choices:

- Accept one reload when you switch between the runner and another program.
- Set `OLLAMA_NUM_CTX=131072`, so the runner asks for the same size as the
  desktop app default. For `qwen3.8:27b` that costs 24,549 MiB instead of
  20,389 MiB.

The legacy script asks for 32768 by default, so it always forces a reload
after the runner, and the other way round.

### Freeing the GPU

The runner does not hold the model in GPU memory. After a run, Ollama keeps
the model loaded for as long as its own setting says, five minutes unless you
changed it, and then unloads it. To look, and to unload it now:

```powershell
ollama ps                   # what is loaded, how much of it is on the GPU, and until when
ollama stop qwen3.8:27b     # unload that model now
```

To keep a model loaded longer between commands, set `OLLAMA_KEEP_ALIVE`, for
example to `30m`. That saves a reload, 8 to 19 seconds in the probe. The price
is the memory: 20,389 MiB for `qwen3.8:27b` at the default window, held for
that long after every command.

### A request times out

```text
Request timed out after 900s. Raise OLLAMA_TIMEOUT_MS.
```

The limit is `OLLAMA_TIMEOUT_MS`, 15 minutes by default, for one whole
request. The probe's requests took between 37 and 166 seconds, so a timeout
usually means the model does not fit on the GPU. `deepseek-r1:32b` ran at
about 3 tokens per second at this context size. `ollama:doctor` warns when a
loaded model is only partly on the GPU, and says how far the window can go
down or that a smaller model is needed. Use a smaller model, or raise the
limit.

There is a second limit that `OLLAMA_TIMEOUT_MS` cannot move:

```text
Ollama sent nothing for 5 minutes, and Node's fetch ends a silent connection at that point whatever OLLAMA_TIMEOUT_MS is.
```

Node gives up on a connection that stays silent for five minutes, and the
server can be silent that long before the first token: while it loads the
model, while it reads the prompt, or while it is busy with another program's
request, since it handles one request at a time here. The runner does not ask
again. In a pipeline run the row stays pending, and two timeouts in a row end
the run. Run the command again when the server is idle.

### The job page cannot be read

Evaluating a URL reads the posting first. Boards with an API are read
directly. Other pages need the Playwright browser:
`npx playwright install chromium`. If a page still cannot be read, save its
text to a file and run
`npm run ollama:eval -- --file jd.txt --posting-url <url>`.

A URL that names this machine or a private network is refused before any
request: `postings are read from the public web only`.

### `cv.md` or the profile is missing

The runner stops before contacting the server. Either your personal setup is
not finished (run `npm run doctor`) or `CAREER_OPS_ROOT` points at a folder
that does not hold your files.

### A pause of a minute, then a lock timeout

This does not happen on this machine. It happens where the path of the temp
folder or of the data root holds a character outside ASCII, for example a
Windows user name such as José.

What you see: after the reply an evaluation waits 60 seconds and prints
`Could not release report reservation` with
`Timed out waiting for tracker lock`. It leaves a file `NNN-RESERVED.md` in
`reports/`, and the next evaluation fails with `Could not save evaluation`.
Or `ollama:pipeline` ticks its first row and reports `pipeline lock timeout`
for every row after it. Those rows stay pending and are evaluated again on the
next run.

The cause is in Node, not in this runner: on Node 24.11 for Windows,
`fs.rmSync` removes nothing under such a path and reports no error
(nodejs/node issue 61067). career-ops keeps its tracker lock in the temp
folder and its pipeline lock beside `data/pipeline.md`, and releases both that
way, so `node merge-tracker.mjs` fails the same way. A later Node 24 is
reported to fix it, which was not checked here. Until then, point `TEMP` and
`TMP` at a folder whose path is plain ASCII, and keep the data root on such a
path.

## Privacy

**What is contacted.** The CV, the profile and the job description go to the
Ollama server and nowhere else. A redirect from that server's address is
refused, not followed. Job boards and posting pages are read when you scan or
evaluate a URL, and a URL that names this machine or a private network is
refused.

**Another machine.** The runner refuses any host other than `localhost`,
`127.0.0.1` or `::1` unless you set `OLLAMA_ALLOW_REMOTE=1`. If you point it
at another machine, your CV travels over the network to that machine. Every
command that asks the model something then says so in one line that names the
host, so a setting you forgot in a `.env` does not stay invisible.

**The PDF step.** A tailored CV is refused before it is saved when it holds a
script, a frame, or an image or a style taken from a file or from another
address. Rendered as it came, such markup could pull a local file into the
PDF, or make Windows open a connection to the host it names. The saved file
also carries a content policy: a browser that opens it loads only what is
inside the file, plus your photo when the profile names one as a local file.

**A `.env` in the folder you run from** is read after the one in your data
root. `NODE_OPTIONS` and the two browser settings
(`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`, `CAREER_OPS_BROWSER_AUTODETECT`) are
passed on to the scripts the runner starts only when the shell set them, not
from a `.env`.

The privacy guard described in [PRIVACY.md](PRIVACY.md) inspects every request
the runner sends, local ones included. It has no exemption for `localhost`.
That has one consequence worth knowing before you turn the guard on:

- With no `config/privacy.yml`, which is the state of a fresh checkout, the
  guard has nothing configured and blocks nothing.
- **A filled-in `config/privacy.yml` blocks local calls too.** If the file
  lists your name, email or phone, every request that contains `cv.md` is
  refused. That is every evaluation, every tailored CV and every set of
  application answers. The sample file also turns on email and phone
  detection, which trips on the CV header and on recruiter addresses in job
  descriptions.

For local-only use, leave `config/privacy.yml` absent. If you need the file
for the hosted scripts, either set `enabled: false` in it while you run the
local pipeline, or list only values that are not in your CV. `enabled: false`
switches the guard off for the hosted scripts as well, so put it back before
you use one.

Do not use `PRIVACY_REDACT=1` with the runner. That mode swaps protected
values for placeholder tokens on the way out and restores them only in replies
read as a whole. The runner reads the reply as a stream, so the tokens would
end up in your reports and your application answers. The runner prints a
warning when the mode is on. `ollama:tailor` goes further: a CV that comes
back with such a token in it is refused, and nothing is saved.

The profile option `privacy.redact_contact_info` is a different mechanism,
and it works here. With it set, `ollama:tailor` and `ollama:apply` send
placeholders in place of your name and contact details, in the CV, the
profile and the report, and put the real ones back before they save. An
evaluation sends the CV as it is, as the legacy script does.

Changing the guard so that it trusts `localhost` would be a change to a
privacy control. This fork does not make it.

## What still needs an agent CLI

The runner covers scan, evaluate, pipeline, tailoring the CV in HTML or in
LaTeX, PDF and application answers. These parts of career-ops have no local
path and still need an agent CLI such as Claude Code:

| Feature | Why |
|---|---|
| Cover letters | `npm run cover-letter` only renders a finished JSON payload to PDF. The payload is written by an agent following `modes/cover.md`. |
| A CV in career-ops' own LaTeX templates | `modes/latex.md` fills `templates/cv-template.tex`, and `modes/latex-tex.md` tailors a `.tex` you keep yourself, both through an agent. The runner's `--latex` writes into templates of its own: see [A CV in LaTeX](#a-cv-in-latex). |
| Ranking the pipeline | `rank-pipeline.mjs` hands the work to an installed agent CLI. It has no HTTP path. |
| Bulk tailoring | `batch-tailor.mjs` launches Claude Code and nothing else. Locally, `npm run ollama:pipeline -- --tailor --pdf` tailors as it evaluates, and `ollama:tailor` handles one existing report at a time. |
| The web UI | `npm run ui` starts an agent CLI for every AI action. It has no Ollama engine. |
| Parallel batch runs | `batch/batch-runner.sh` drives agent CLI workers. `ollama:pipeline` is the local equivalent and runs one posting at a time. |
| The other modes | Interview preparation, deep research, outreach, follow-ups and the rest are prompts for an interactive agent. |

## Agent CLI on local models (not tested)

**No career-ops session was run this way for this fork.** This section
records what the installed tools offer, so you know where to start if you want
the features in the table above without a hosted model.

Ollama 0.40.16 can start an agent CLI against a local model:

```powershell
ollama launch claude --model qwen3.8:27b
```

`ollama launch --help` lists `claude` and `opencode` among its integrations,
and the server answers Anthropic-style requests on `/v1/messages` (one
one-word request returned HTTP 200). Whether a full career-ops session works
this way is unknown. What is known:

- The agent needs a model with the `tools` capability. `qwen3.8:27b`, `gemma4`,
  `glm-4.7-flash` and `qwen3-coder:30b` have it.
- This route cannot set the context size per request. It gets the server
  default, which is 131072 on this machine because the Ollama desktop app is
  set to that. On another machine, check that setting first.
- It will be much slower than the runner. An interactive evaluation loads
  about 53,000 tokens of instructions and takes 14 to 25 model turns (figures
  from [RUNNING_ON_A_BUDGET.md](RUNNING_ON_A_BUDGET.md), measured with Claude
  models).

The batch runner accepts a local model through OpenCode:

```bash
batch/batch-runner.sh --cli opencode --model qwen3.8:27b
```

OpenCode is not installed on this machine. Without it the batch runner falls
back to `ollama launch opencode ... -y`, and `-y` answers every confirmation
prompt of the launcher with yes, whatever it asks. The batch runner also
passes its whole prompt as one command-line argument, which may exceed the
Windows limit of about 32,000 characters. Neither was tried.

The CLIs career-ops supports are listed in
[SUPPORTED_CLIS.md](SUPPORTED_CLIS.md).

## The legacy script

`ollama-eval.mjs` is the upstream single-posting evaluator. It is untouched and
still runs with `node ollama-eval.mjs`. `npm run ollama:eval` no longer points
at it.

Its defaults do not work on this machine, so set two variables by hand:

```powershell
$env:OLLAMA_MODEL = 'qwen3.8:27b'
$env:OLLAMA_NUM_CTX = '65536'
node ollama-eval.mjs --file jd.txt
Remove-Item Env:OLLAMA_MODEL, Env:OLLAMA_NUM_CTX
```

- `OLLAMA_MODEL` defaults to `llama3.3`, which is not installed here.
- `OLLAMA_NUM_CTX` defaults to 32768, which is smaller than the prompt. The
  script warns, sends the prompt anyway, and Ollama truncates it.

The last line matters. The runner reads the same two variables, and an
`OLLAMA_NUM_CTX` left in the shell switches off its automatic sizing.

Compared with the runner, the legacy script takes pasted text or `--file`
only. It does not fetch a URL, does not check that the model exists, does not
control thinking and does not retry. Its time limit is 300 seconds unless you
set `OLLAMA_TIMEOUT_MS`.

## Keeping the fork in sync with upstream

Merge upstream with git, never with career-ops' updater:

```powershell
git fetch upstream
git merge upstream/main
```

`upstream` is `career-ops-hq/career-ops`; `origin` is your fork. On this
Windows checkout the merge needs one more line before it and one after it:
see [The six skill files](#the-six-skill-files).

**Commit the fork's work first**, and only the fork's work. The fork itself is
committed on the branch `ollama-local`, so this is about what you have changed
in its files since then. Git will not merge over uncommitted changes to the
same files, and until the work is committed no updater command can be undone
safely: see [Do not run the updater](#do-not-run-the-updater).

Add the fork's files by name. `git add -A` and `git commit -a` would also
commit six files that are not the fork's, which the next section explains.

```powershell
git add -- package.json .gitignore .env.example doctor.mjs docs/RUNNING_ON_A_BUDGET.md docs/SCRIPTS.md docs/PRIVACY.md docs/SUPPORTED_CLIS.md docs/OLLAMA.md config/local-paths.txt .claude/rules/ollama-fork.md ollama-runner.mjs lib/ollama-config.mjs lib/ollama-client.mjs lib/ollama-doctor.mjs lib/ollama-cv-latex.mjs lib/ollama-cv-latex-default.tex tests/ollama-config.test.mjs tests/ollama-client.test.mjs tests/ollama-doctor.test.mjs tests/ollama-fork-guard.test.mjs tests/ollama-runner.test.mjs tests/ollama-runner-pipeline.test.mjs tests/ollama-runner-tailor.test.mjs tests/ollama-runner-steps.test.mjs tests/ollama-cv-latex.test.mjs tests/helpers/ollama-runner-eval-harness.mjs tests/helpers/ollama-runner-harness.mjs
```

### The six skill files

On this checkout `git status` lists six modified files that this fork did not
edit: `SKILL.md` in `.claude/skills/career-ops/`,
`.opencode/skills/career-ops/`, `.qwen/skills/career-ops/`,
`.antigravitycli/skills/career-ops/`, `.grok/skills/career-ops/` and
`.kimi/skills/career-ops/`. Upstream tracks each one as a symlink to
`.agents/skills/career-ops/SKILL.md`. Windows did not create the symlinks, so
career-ops wrote a copy of that file into each of the six. Each tool finds its
`/career-ops` command there.

**Do not commit them.** `git add -A` and `git commit -a` record each one as a
symlink whose target is the whole file. After that `git status` no longer
lists them, and `git checkout --` no longer brings the symlink back.

**A merge does not update them.** It updates
`.agents/skills/career-ops/SKILL.md` and leaves the six copies as they were,
so those six tools keep the `/career-ops` command from before the merge.
`npm run doctor` does not notice a copy that is out of date.
`tests/ollama-fork-guard.test.mjs` does, and the full gate then reports six
more failures. So put the symlink entries back before every merge, and write
the copies again after it:

```powershell
git checkout -- .claude/skills/career-ops/SKILL.md .opencode/skills/career-ops/SKILL.md .qwen/skills/career-ops/SKILL.md .antigravitycli/skills/career-ops/SKILL.md .grok/skills/career-ops/SKILL.md .kimi/skills/career-ops/SKILL.md
git fetch upstream
git merge upstream/main
node -e "import('./scaffolder/bin/skill-entrypoints.mjs').then(m=>console.log(m.materializeSkillEntrypoints('.')))"
```

The last line prints the six paths it wrote. It has to be run also after a
merge that stopped on a conflict, once the conflict is resolved. Until it has
run, each of the six files holds only its link target, 43 bytes, and its tool
loads an empty `/career-ops`, while `git status` lists none of them.

The first line has a second reason: with the copies in place, a merge in
which upstream changes one of the six entries stops with
`Your local changes to the following files would be overwritten by merge`.

`npm run doctor` and `docs/FAQ.md` also name
`node update-system.mjs apply --confirm` as a repair for these files. Do not
use it here.

### Where a merge can conflict

Conflicts can only come from the eight upstream files this fork edits:

| File | What this fork changed |
|---|---|
| `package.json` | the ten `ollama` scripts |
| `.gitignore` | the `!config/local-paths.txt` line and its comment |
| `.env.example` | one commented `OLLAMA_*` block |
| `doctor.mjs` | the `--ollama` flag, the step that runs the Ollama checks, and the way the run ends after them |
| `docs/RUNNING_ON_A_BUDGET.md`, `docs/SCRIPTS.md` | the `ollama` scripts and Path B, described for this fork, with pointers to this page |
| `docs/PRIVACY.md`, `docs/SUPPORTED_CLIS.md` | added lines that point to this page |

Four of the eight only add lines: `.gitignore`, `.env.example`,
`docs/PRIVACY.md` and `docs/SUPPORTED_CLIS.md`. The other four also replace
upstream lines. A change upstream makes to one of those lines, or a line it
adds next to one, conflicts there:

- `package.json`: the `ollama:eval` script.
- `doctor.mjs`: the `KNOWN_FLAGS` line, and the two `process.exit` calls at
  the end of `main()`, which are `endRun` here.
- `docs/SCRIPTS.md`: the `npm run ollama:eval` row of the table and the
  `ollama:eval` example.
- `docs/RUNNING_ON_A_BUDGET.md`: three lines of Path B.

In `package.json`, upstream's half of such a conflict holds its own
`"ollama:eval": "node ollama-eval.mjs"` line beside the script it added. Keep
the fork's ten lines and the script upstream added, and delete that one
upstream line. Left in, it is a second `ollama:eval` entry, and npm runs
whichever comes last.

After a conflict is resolved, run the last line of the block under
[The six skill files](#the-six-skill-files) and then
`node --test tests/ollama-fork-guard.test.mjs`. The guard fails on a second
`ollama:eval` entry. It also fails on a skill entry that holds only its link
target, and names that last line as the repair: a merge that stopped on a
conflict is where the line gets left out.

Everything else is new, and upstream has no file of the same name today: the
runner (`ollama-runner.mjs`), its four modules (`lib/ollama-config.mjs`,
`lib/ollama-client.mjs`, `lib/ollama-doctor.mjs`, `lib/ollama-cv-latex.mjs`),
the default template of a CV in LaTeX (`lib/ollama-cv-latex-default.tex`),
`config/local-paths.txt`, the rule for agents
(`.claude/rules/ollama-fork.md`), this page, and nine test suites under
`tests/` (`ollama-config`, `ollama-client`, `ollama-runner`,
`ollama-runner-pipeline`, `ollama-runner-tailor`, `ollama-runner-steps`,
`ollama-cv-latex`, `ollama-doctor` and `ollama-fork-guard`) with the two
helpers they share, `tests/helpers/ollama-runner-eval-harness.mjs` and
`tests/helpers/ollama-runner-harness.mjs`.

After a merge, check the result:

```powershell
npm install
npm run lint
node validate-system-paths-coverage.mjs
node test-all.mjs --only ollama
node test-all.mjs --quick
```

`--only ollama` runs the nine new suites and the four upstream suites for the
legacy script. The last command is the full gate and takes about three minutes
here.

If the fork guard says that this section does not name a suite, check who
added the file. A `tests/ollama-*.test.mjs` file that came with the merge is
upstream's: add its name to `UPSTREAM_SUITES` in
`tests/ollama-fork-guard.test.mjs`, leave the list above as it is, and correct
the number of upstream suites in the sentence before this one. A helper under
`tests/helpers/` and an `ollama:` script that came with a merge go into
`UPSTREAM_HELPERS` and `UPSTREAM_SCRIPTS` in the same file. The guard's
message names the set and prints the entry as it has to be spelled.

`doctor.mjs` loads the Ollama checks from `lib/ollama-doctor.mjs` only when it
reaches them, so a merge or a commit that leaves out one of the three
modules the checks use (`lib/ollama-doctor.mjs`, `lib/ollama-client.mjs`,
`lib/ollama-config.mjs`) does not break `npm run doctor`. In that state
`npm run doctor -- --ollama` reports
`Ollama checks are not part of this checkout`, and
`tests/ollama-fork-guard.test.mjs` fails on the missing file. The doctor
never loads `lib/ollama-cv-latex.mjs`. Without it no runner command starts,
and without `lib/ollama-cv-latex-default.tex` a CV in LaTeX in the default
template is refused. The guard fails on either.

On this machine a full run of untouched upstream code (`main` at `287c3e59`,
283 commits after v1.40.0, run on 2026-11-17 with the six skill copies up to
date) already reports four failures: three need Windows symlink permission
(the web PDF write-scope check, `tests/employers-inbox-only` and
`tests/pipeline-lock-symlink`) and `tests/batch-runner-profile-override` fails
on a directory read. `tests/reply-watch` failed in one run of three. Compare
against that list instead of expecting zero.

This fork adds no failure of its own to the full gate, as long as the six
skill copies are up to date. Its rule for agents is
in `.claude/rules/ollama-fork.md` and not in `CLAUDE.md` for that reason:
upstream's gate checks twice that `CLAUDE.md` holds nothing but the
`@AGENTS.md` line and one comment, and with the rule written there two checks
failed (`CLAUDE.md must contain only @AGENTS.md plus an optional Claude-only
placeholder comment` and `AGENTS.md misses onboarding state docs or CLAUDE.md
is not a thin wrapper`). If either message appears, something was added to
`CLAUDE.md`: move it to the rule file.

### Do not run the updater

Do not run `npm run update`, `node update-system.mjs apply` or
`/career-ops update` in this checkout, with or without `--force`.
`/career-ops update` is the same updater behind a preview: its mode file runs
`node update-system.mjs apply --confirm`. When an agent says a career-ops
update is available, answer no and merge with git.

`.claude/rules/ollama-fork.md` tells Claude Code the same. Claude Code loads
every rule in that folder that has no `paths` front matter at the start of a
session, beside `CLAUDE.md`. Other agent tools (Codex, OpenCode and the rest)
do not read it and will still offer the update, so the answer there is yours
to give.

To check that the rule reaches Claude Code, start a new session and make this
the first message: "Without opening a file or running a command, which file
tells you how this checkout takes changes from upstream, and what is its first
line?" The answer should be `.claude/rules/ollama-fork.md` and
`# This checkout is the ollama-local fork`. An answer that needed a file read
or a command proves nothing, because this page holds the same two commands.
The question itself has not been asked that way. What was seen on 2026-11-18:
the Claude Code sessions that worked in this checkout had the rule's text
among their instructions from their first message, without opening the file.

The updater is career-ops' own. It does not use your git remotes: it fetches
the official repository and replaces the system files with the released
versions. What that does to this fork was run in scratch clones against a
simulated release.

**A plain update half-updates the fork and then blocks the git merge.** It
keeps your version of each file this fork edited or added among the system
files, writes a `.bak` copy beside it, and says that the file will no longer
receive upstream changes. In the run that was 14 files: the seven system
files the fork had edited by then, this page and the six test suites. Then:

- It stages the release's files on `ollama-local`, does not commit them, and
  prints `Update complete`.
- `VERSION` is at the new release while `package.json`, `doctor.mjs`,
  `.env.example` and the four docs stay at the old one. A dependency or a
  script the release added never arrives.
- The update check goes quiet, because it compares `VERSION` only.
- `git merge upstream/main` stops with
  `Your local changes to the following files would be overwritten by merge`.
  Committing the staged files, as the updater suggests, made the next merge
  conflict, in `VERSION` and in the file the release had added among others.

`tests/ollama-fork-guard.test.mjs` does not notice this state. It still
passed.

**`rollback` and `apply --force` restore committed state only.** With the
fork's work uncommitted, `node update-system.mjs rollback` after an update put
upstream's text back over the fork's edits. `package.json` was left with one
`ollama` script, upstream's, which runs the legacy evaluator. `doctor.mjs` had
no `--ollama`. The new files were still there, so the checkout looked intact,
and the output reported only `Rollback complete`. `apply --force` removes the
same edits in one step. With the work committed first, a rollback brought all
of it back.

**If the updater ran anyway**, undo it with `node update-system.mjs rollback`
and then merge with git. With the fork's work committed, that is all. With
the work uncommitted, the rollback also removes the fork's edits, as described
above. They are not lost for good:

- Each kept file has a `.bak` copy beside it that holds the fork's version.
  Copy each `.bak` over its file. This is the safer way.
- Or restore the copy the update saved. After an update from v1.40.0 that is
  the two lines below. Without the first one the second stops on `.gitignore`.
  A later update from the same version replaces that saved copy.

```powershell
git checkout -- .gitignore
git stash apply refs/backup-pre-update-wip/1.40.0
```

What does protect the fork's files:

- The runner, its four modules, the default template of a CV in LaTeX and
  the rule for agents are declared in
  [`config/local-paths.txt`](../config/local-paths.txt). No path the updater
  ships covers them today, so it does not write to them, declared or not. The
  declaration does two other things. It keeps
  `node validate-system-paths-coverage.mjs` passing once the files are
  committed. And if a release ever ships a file of the same name, the updater
  stops with an error that names the entry. The `!config/local-paths.txt` line
  in `.gitignore` is what lets this fork commit that file.
- In `.gitignore` the updater only appends rules it is missing, so the
  negation line stays.
- `tests/ollama-fork-guard.test.mjs` fails after `apply --force` and after a
  merge resolved the wrong way, when the `ollama` scripts or the `--ollama`
  flag are gone.

Before it restarts itself, every update replaces the updater and the files it
loads with the release's copies, with or without `--force`. Today that is ten
files: `update-system.mjs`, `lib/update-manifest.mjs`, `path-resolver.mjs`,
`cv-templates.mjs`, `scaffolder/bin/skill-entrypoints.mjs`,
`lib/is-main-module.mjs`, `lib/template-manifest.mjs`,
`lib/yaml-document.mjs`, `providers/_html-entities.mjs` and
`providers/_html-entity-table.mjs`. This fork edits none of them.

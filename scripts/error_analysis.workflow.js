// Root-cause analysis of the eval failures.
//
// Launched via the Workflow tool. `args` is a light list [{qid, ticker}, ...] for
// the failing questions. The full per-failure record lives in
// results/_error_input.json (question, gold answer + evidence, agent answer,
// judge reasoning, AND retrieval diagnostics: was the gold evidence retrievable,
// did the agent's own searches surface it, where the best-overlap chunk ranked).
//
// Per failure: an ANALYST classifies the root cause and proposes a fix; then an
// independent VERIFIER checks that diagnosis against the same record (adversarial
// second opinion). Pipeline, so each failure flows analyst -> verifier on its own.

export const meta = {
  name: 'eval-error-analysis',
  description: 'Root-cause each eval failure (retrieval miss vs reasoning vs nondeterminism vs gold-convention) and verify the diagnosis with a second agent',
  phases: [
    { title: 'Diagnose', detail: 'analyst classifies each failure' },
    { title: 'Verify', detail: 'independent check of each diagnosis' },
  ],
}

let parsed = args
if (typeof parsed === 'string') {
  try { parsed = JSON.parse(parsed) } catch (e) { log('args parse failed: ' + String(e)) }
}
const items = Array.isArray(parsed) ? parsed : []
if (!items.length) {
  log('No failures passed via args (typeof args=' + typeof args + ').')
  return { error: 'no items', argsType: typeof args }
}

const CATEGORIES =
  'retrieval_miss | reasoning_error | agent_nondeterminism | answer_formulation | gold_convention_or_ambiguity | tool_limitation | other'

const ANALYSIS = {
  type: 'object',
  properties: {
    qid: { type: 'string' },
    primary_category: { type: 'string', description: 'One of: ' + CATEGORIES },
    evidence_retrievable: {
      type: 'string',
      enum: ['yes', 'partial', 'no'],
      description: 'Did our retrieval actually surface the gold evidence (use agent_retrieval_covered_gold, best_gold_overlap, and top_chunks_for_question)?',
    },
    agent_defensible: {
      type: 'boolean',
      description: 'Is the agent answer arguably acceptable despite mismatching the gold (e.g. a sign convention), vs. clearly wrong?',
    },
    diagnosis: { type: 'string', description: '2-3 sentences: WHY it failed, grounded in the record.' },
    specific_fix: { type: 'string', description: 'The concrete change that would most likely fix this class of failure.' },
    severity: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
  required: ['qid', 'primary_category', 'evidence_retrievable', 'agent_defensible', 'diagnosis', 'specific_fix', 'severity'],
}

const VERIFY = {
  type: 'object',
  properties: {
    qid: { type: 'string' },
    agrees: { type: 'boolean', description: 'Do you agree with the analyst\'s primary_category and whether the agent is defensible?' },
    corrected_category: { type: 'string', description: 'If you disagree, the better category (one of: ' + CATEGORIES + '); else repeat the analyst\'s.' },
    note: { type: 'string', description: 'One line: what you confirmed or corrected.' },
  },
  required: ['qid', 'agrees', 'corrected_category', 'note'],
}

function analystPrompt(it) {
  return [
    'You are doing ROOT-CAUSE analysis on a failed eval question for a SEC-10-K agent.',
    'Read the JSON array file results/_error_input.json and take the element whose "qid" == "' + it.qid + '".',
    'It contains: question, gold_answer, gold_evidence, agent_answer, judge_reason, judge_issues,',
    'agent_retrieve_queries, agent_retrieval_covered_gold (did the agent\'s OWN searches surface the gold',
    'numbers), best_gold_overlap + best_gold_overlap_rank (how well the single best chunk for the raw',
    'question covers the gold numbers, and at what rank), and top_chunks_for_question.',
    '',
    'Decide the SINGLE primary reason it failed:',
    '- retrieval_miss: the gold evidence was NOT findable by our retrieval (low overlap / not in top chunks).',
    '- reasoning_error: the evidence WAS available but the agent computed or concluded wrong.',
    '- agent_nondeterminism: the agent clearly had the right data and the capability, but produced a wrong/variant answer this run (would likely pass on another sample).',
    '- answer_formulation: the agent had the right info but hedged, refused, or mis-stated it (e.g. answered "$0" or "cannot determine").',
    '- gold_convention_or_ambiguity: the agent answer is arguably correct; the gold uses a specific convention/wording the agent reasonably differs from.',
    '- tool_limitation: a missing/stubbed tool forced a worse path.',
    '',
    'Use evidence_retrievable to separate a SEARCH problem from a REASONING problem. Then give a concrete',
    'specific_fix and a severity. Be precise and grounded in the record for qid "' + it.qid + '".',
  ].join('\n')
}

function verifyPrompt(it, a) {
  return [
    'You are independently CHECKING a root-cause diagnosis for a failed eval question.',
    'Read results/_error_input.json, take the element whose "qid" == "' + it.qid + '", and review it yourself.',
    '',
    'The analyst concluded:',
    '  primary_category: ' + (a ? a.primary_category : '(none)'),
    '  evidence_retrievable: ' + (a ? a.evidence_retrievable : '(none)'),
    '  agent_defensible: ' + (a ? a.agent_defensible : '(none)'),
    '  diagnosis: ' + (a ? a.diagnosis : '(none)'),
    '',
    'Do you agree with the category and the defensible call? If not, give the better category. Be skeptical',
    'and literal: check the retrieval-coverage fields before accepting "reasoning_error" over "retrieval_miss",',
    'and check the gold vs agent figures before accepting "gold_convention_or_ambiguity".',
  ].join('\n')
}

const results = await pipeline(
  items,
  (it) => agent(analystPrompt(it), { schema: ANALYSIS, phase: 'Diagnose', label: `diagnose:${String(it.qid).slice(-5)}` }),
  (analysis, it) =>
    agent(verifyPrompt(it, analysis), { schema: VERIFY, phase: 'Verify', label: `verify:${String(it.qid).slice(-5)}` })
      .then((v) => ({ ...analysis, qid: it.qid, ticker: it.ticker, verify: v })),
)

const ok = results.filter(Boolean)
log(`Analyzed ${ok.length} failures; verifier agreed on ${ok.filter((r) => r.verify && r.verify.agrees).length}.`)

return {
  total: ok.length,
  failures: ok.map((r) => ({
    qid: r.qid,
    ticker: r.ticker,
    category: r.primary_category,
    verified_category: r.verify ? r.verify.corrected_category : r.primary_category,
    verifier_agreed: r.verify ? r.verify.agrees : null,
    evidence_retrievable: r.evidence_retrievable,
    agent_defensible: r.agent_defensible,
    severity: r.severity,
    diagnosis: r.diagnosis,
    specific_fix: r.specific_fix,
    verifier_note: r.verify ? r.verify.note : null,
  })),
}

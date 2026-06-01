// Independent multi-grader audit of the hardened LLM judge.
//
// Launched via the Workflow tool. `args` is a light index list:
//   [{ idx, qid, ticker }, ...]  (one per eval question)
// The actual question data lives in results/_audit_blind.json (a JSON array,
// same order as `idx`) which deliberately OMITS the judge's verdict, so graders
// cannot anchor on it.
//
// For every question, THREE graders independently read their item from the blind
// file and grade the agent answer against the gold answer + evidence. Majority of
// 3 = the panel verdict. The caller joins these panel verdicts back to the
// hardened judge's verdicts (in results/eval_run.json) to compute inter-rater
// agreement and surface disagreements. This is the "don't trust your own judge"
// check: real independent second/third/fourth opinions.

export const meta = {
  name: 'judge-independent-audit',
  description: 'Independent 3-grader panel re-grades every eval answer blind to the judge verdict; reports per-question majority verdicts to compare against the judge',
  phases: [{ title: 'Audit', detail: '3 independent blind graders per question' }],
}

let parsed = args
if (typeof parsed === 'string') {
  try {
    parsed = JSON.parse(parsed)
  } catch (e) {
    log('args was a string that failed to JSON.parse: ' + String(e))
  }
}
const items = Array.isArray(parsed) ? parsed : []
if (!items.length) {
  log('No audit items passed via args (typeof args=' + typeof args + ') — nothing to audit.')
  return { error: 'no items', argsType: typeof args }
}

const VERDICT = {
  type: 'object',
  properties: {
    qid: { type: 'string' },
    correct: { type: 'boolean', description: 'Is the AGENT ANSWER correct vs the GOLD ANSWER?' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    failure_mode: {
      type: 'string',
      description: 'If incorrect: one of wrong_figure | missing_item | wrong_sign | hedge_no_answer | wrong_conclusion | other. If correct: none.',
    },
    reason: { type: 'string', description: 'One line justifying the grade.' },
  },
  required: ['qid', 'correct', 'confidence', 'failure_mode', 'reason'],
}

function graderPrompt(it, k) {
  return [
    'You are an INDEPENDENT financial-statement grader for answers about SEC 10-K filings.',
    'Read the JSON array file results/_audit_blind.json and take the element at index ' + it.idx + ' (its',
    '"qid" must equal "' + it.qid + '"). That element has: question, gold_answer, gold_evidence, agent_answer.',
    '',
    'Decide whether the AGENT ANSWER is correct against the GOLD ANSWER (authoritative).',
    'Rules: CORRECT only if the agent reaches the SAME conclusion AND the SAME key figures',
    '(within ~1%). A missing, zero, or "cannot find" figure when the gold gives a real value is',
    'INCORRECT. A wrong sign or wrong direction of change is INCORRECT. Extra correct detail is fine.',
    'Judge only this one item; do not read or rely on any other grade.',
    '',
    'Return the verdict for qid "' + it.qid + '". You are independent grader #' + (k + 1) + ' — be strict and literal about figures and direction.',
  ].join('\n')
}

phase('Audit')
const panels = await parallel(
  items.map((it) => () =>
    parallel(
      [0, 1, 2].map((k) => () =>
        agent(graderPrompt(it, k), {
          schema: VERDICT,
          phase: 'Audit',
          label: `audit:${String(it.qid).slice(-5)}#${k + 1}`,
        })
      )
    ).then((votes) => {
      const valid = votes.filter(Boolean)
      const correctVotes = valid.filter((v) => v.correct).length
      const panel_correct = correctVotes >= Math.ceil(valid.length / 2)
      return {
        qid: it.qid,
        ticker: it.ticker,
        panel_correct,
        correctVotes,
        nVotes: valid.length,
        votes: valid.map((v) => ({
          correct: v.correct,
          confidence: v.confidence,
          failure_mode: v.failure_mode,
          reason: v.reason,
        })),
      }
    })
  )
)

const results = panels.filter(Boolean)
log(`Graded ${results.length} questions with a 3-grader blind panel.`)

return {
  total: results.length,
  panel_pass: results.filter((r) => r.panel_correct).length,
  per_question: results.map((r) => ({
    qid: r.qid,
    ticker: r.ticker,
    panel_correct: r.panel_correct,
    votes: `${r.correctVotes}/${r.nVotes}`,
    grader_notes: r.votes.map((v) => `${v.correct ? 'OK' : 'no'}/${v.confidence}/${v.failure_mode}: ${v.reason}`),
  })),
}

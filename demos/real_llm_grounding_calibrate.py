"""Real-LLM mode - calibrate the entailment groundedness threshold (offline, no LLM).

A real model paraphrases, so its answers rarely quote the evidence verbatim. In
real mode, check_groundedness(semantic=True) also accepts a sentence that a local
NLI model (config.NLI_MODEL) judges entailed by the evidence with probability
>= config.GROUNDING_MIN_ENTAILMENT, provided all its numbers occur in the
evidence. This demo scores labelled paraphrases, invented claims and
contradictions (eval/grounding_calibration.py) against their KB document and
checks that the threshold lies in the observed gap. It then scores sentences a
real model actually wrote (all verified correct) and shows why, when the LLM
reviewer runs, NLI is recorded but not enforced: no threshold separates them.

Needs the NLI model (~360 MB, downloaded on first run); no API key, no LLM.
Run:  python -m demos.real_llm_grounding_calibrate
"""
import config

from demos._transcript import transcript
from eval.grounding_calibration import CONTRADICTS, INVENTED, REAL_CORRECT, SUPPORTED
from guardrails import NLI_PREMISE_PREFIX, entailment_support, numbers_in, premise_units
from rag.generate import answer
from rag.kb import load_documents

DOCS = {d["doc_id"]: d["text"] for d in load_documents()}


def score(cases: list[tuple[str, str]]) -> list[tuple[str, float, set[float]]]:
    out = []
    for doc_id, sentence in cases:
        out += entailment_support([sentence], premise_units([DOCS[doc_id]]), numbers_in(DOCS[doc_id]))
    return out


def accepted(prob: float, missing: set[float]) -> bool:
    return prob >= config.GROUNDING_MIN_ENTAILMENT and not missing


def show(label: str, rows) -> None:
    print(f"\n{label}")
    for sentence, prob, missing in sorted(rows, key=lambda r: -r[1]):
        flag = f"  numbers not in evidence: {sorted(missing)}" if missing else ""
        print(f"  P(entail)={prob:.3f} {'ACCEPT' if accepted(prob, missing) else 'reject'} | {sentence}{flag}")


def main() -> None:
    print(f"NLI model: {config.NLI_MODEL}")
    print(f"Premises: each sentence of the document, each adjacent pair, and the whole document, "
          f"each prefixed {NLI_PREMISE_PREFIX!r}.")
    supported, invented, contradicts = score(SUPPORTED), score(INVENTED), score(CONTRADICTS)
    show(f"SUPPORTED paraphrases ({len(supported)})", supported)
    show(f"INVENTED claims ({len(invented)})", invented)
    show(f"CONTRADICTIONS ({len(contradicts)})", contradicts)

    lo_sup = min(p for _, p, _ in supported)
    hi_bad = max(p for _, p, _ in invented + contradicts)
    print(f"\nsupported min {lo_sup:.3f} | invented/contradiction max {hi_bad:.3f} | "
          f"gap ({hi_bad:.3f}, {lo_sup:.3f}) midpoint {(lo_sup + hi_bad) / 2:.4f}")
    assert hi_bad < config.GROUNDING_MIN_ENTAILMENT <= lo_sup, "threshold not inside the observed gap"
    n_ok = sum(accepted(p, m) for _, p, m in supported)
    n_bad = sum(accepted(p, m) for _, p, m in invented + contradicts)
    print(f"Chosen GROUNDING_MIN_ENTAILMENT = {config.GROUNDING_MIN_ENTAILMENT}: {n_ok}/{len(supported)} "
          f"paraphrases accepted, {n_bad}/{len(invented) + len(contradicts)} invented or contradicting claims accepted.")

    # Real model output, scored against the evidence the generator actually saw.
    real = []
    for query, sentence in REAL_CORRECT:
        evidence = answer(query).evidence  # retrieval only; deterministic, offline
        real += entailment_support([sentence], premise_units(evidence), numbers_in(" ".join(evidence) + " " + query))
    show(f"REAL model sentences, all verified CORRECT ({len(real)})", real)
    lo_real = min(p for _, p, _ in real)
    refused = sum(not accepted(p, m) for _, p, m in real)
    print(f"\nReal correct sentences: min {lo_real:.3f}; {refused}/{len(real)} would be refused by the strict gate.")
    assert lo_real < hi_bad, "real correct sentences no longer overlap the invented ones; revisit the decision"
    print(f"They overlap the invented claims (max {hi_bad:.3f}): correct answers that need world knowledge the")
    print("policy text doesn't state ('a dress is apparel', 'Bangalore is a metro') score near 0, so NO")
    print("threshold separates them. Decision: when the Autogen reviewer (a real LLM) has run, it judges")
    print("support and the final check enforces only the numbers rule, recording NLI scores as advisory.")
    print(f"The strict gate (threshold {config.GROUNDING_MIN_ENTAILMENT}) applies only when review is disabled.")


if __name__ == "__main__":
    with transcript("real_llm_grounding_calibrate"):
        main()

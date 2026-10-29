"""Maintenance assistant: the LLM EXPLAINS the model's structured result, it never produces the diagnosis."""
import json
import os

SYSTEM_PROMPT = """You are the maintenance assistant of a bearing-fault Digital Twin.
Rules:
1. The JSON 'model_result' is the ground truth from the ML/DL system. Never change or contradict the predicted fault.
2. Never invent sensor values, measurements, maintenance history or dates. If something is not in the
   model_result or the retrieved context, say you do not have it.
3. Technical claims must come from the retrieved context. Cite sources as [source].
4. Clearly separate 'Model prediction' from 'Recommendation'. Recommendations are decision support for a
   human engineer, not commands.
5. Mention uncertainty when confidence is below 90% or when physics evidence does not support the prediction.
Be concise."""


def build_user_prompt(question, result, contexts):
    ctx = "\n\n".join(f"[{c['source']}] {c['text']}" for c in contexts) or "(no documents retrieved)"
    return (f"model_result:\n{json.dumps(result, indent=2, default=str)}\n\n"
            f"retrieved_context:\n{ctx}\n\nquestion: {question}")


def template_answer(question, result, contexts):
    """Deterministic fallback (no LLM). Only restates structured facts and retrieved text."""
    parts = []
    if result:
        ph = result.get("physics", {})
        sup = ph.get("supports_prediction")
        parts.append(f"Model prediction: {result['fault'].replace('_', ' ')} with {result['confidence']:.0%} confidence "
                     f"(uncertainty: {result.get('uncertainty', 'n/a')}). Health score {result['health']:.0f}/100, "
                     f"alert level: {result['alert']['level']}.")
        if sup is True:
            parts.append("Physics evidence: the envelope spectrum shows a peak at the characteristic frequency of the predicted fault.")
        elif sup is False:
            parts.append("Physics evidence: NO clear peak at the predicted fault's characteristic frequency, so verify manually.")
        parts.append(f"Recommendation (decision support): {result.get('recommendation', 'n/a')}")
    if contexts:
        parts.append("Relevant reference notes:\n" + "\n".join(f"- [{c['source']}] {' '.join(c['text'].split())[:260]}..." for c in contexts[:3]))
    elif not result:
        parts.append("I have no analysed window yet and no matching reference documents.")
    return "\n\n".join(parts)


class MaintenanceAssistant:
    def __init__(self, retriever, provider=None, model=None, top_k=4):
        self.r = retriever
        self.provider = provider or os.getenv("LLM_PROVIDER", "template")
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-5-5")
        self.k = top_k

    def answer(self, question, result=None):
        ctx = self.r.search(question + " " + (result["fault"] if result else ""), self.k)
        used, text = "template", None
        if self.provider == "anthropic":
            try:
                import anthropic
                msg = anthropic.Anthropic().messages.create(
                    model=self.model, max_tokens=700, system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": build_user_prompt(question, result, ctx)}])
                text, used = "".join(b.text for b in msg.content if b.type == "text"), "anthropic"
            except Exception as e:                     # missing key / no network -> safe fallback
                text = template_answer(question, result, ctx) + f"\n\n(LLM unavailable, showing template answer: {type(e).__name__})"
        if text is None:
            text = template_answer(question, result, ctx)
        return {"answer": text, "sources": sorted({c["source"] for c in ctx}), "provider": used}

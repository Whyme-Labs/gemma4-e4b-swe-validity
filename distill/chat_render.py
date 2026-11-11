"""vLLM 0.19.1's chat rendering of a logged OpenAI request, offline (shared by build_sft.py and copy_fidelity.py)."""
import json


def content_format(tok):  # vLLM picks "openai" (a list of text parts) or "string" from the template's jinja AST
    try:
        from vllm.renderers.hf import _detect_content_format
    except ImportError:
        return "string"
    return _detect_content_format(tok.get_chat_template(None, tools=[{}]), default="string")


def conversation(req, fmt):  # vLLM's chat preprocessing: text parts as the template expects, tool-call arguments parsed
    conv = []
    for m in req["messages"]:
        m = dict(m); c = m.get("content")
        texts = [] if c is None else [c] if isinstance(c, str) else [
            p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
        m["content"] = [{"type": "text", "text": t} for t in texts] if fmt == "openai" else "\n".join(texts)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": json.loads(tc["function"]["arguments"])
                                if isinstance(tc["function"].get("arguments"), str) else tc["function"].get("arguments")}}
                               for tc in m["tool_calls"]]
        conv.append(m)
    return conv


def render(tok, req):
    text = tok.apply_chat_template(conversation(req, content_format(tok)), tools=req.get("tools"), add_generation_prompt=True,
                                   tokenize=False, **(req.get("chat_template_kwargs") or {}))
    return tok(text, add_special_tokens=False)["input_ids"]

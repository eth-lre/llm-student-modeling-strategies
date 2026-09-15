import json
import os
import re
from openai import OpenAI
import requests
from pprint import pprint

def prompt_openai(client: OpenAI, system_prompt: str, user_prompt: str, model_config: dict[str, str]) -> str:
    """ Send prompt to OpenAI client, returns the response """
    if isinstance(client, OpenAI):
        response = client.chat.completions.create(
            model=model_config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **model_config.get("completion_kwargs", {})
        )
        return response.choices[0].message.content.strip()
    else:
        raise ValueError("Unsupported model type")
    
def prompt_vllm(client: OpenAI, system_prompt: str, user_prompt: str, model_config: dict[str, str]) -> tuple[str, str]:
    """ Send prompt to a vLLM OpenAI-compatible server. Returns (reasoning, content).
    When vLLM is started with --reasoning-parser, the parsed reasoning is exposed
    on message.reasoning_content (extra field beyond the standard OpenAI schema). """
    response = client.chat.completions.create(
        model=model_config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **model_config.get("completion_kwargs", {})
    )
    msg = response.choices[0].message
    content = (msg.content or "").strip()
    extra = getattr(msg, "model_extra", None) or {}
    reasoning = extra.get("reasoning_content") or getattr(msg, "reasoning_content", None) or ""

    # Fallback: if vLLM didn't split reasoning (e.g. Gemma's "thought" turn isn't matched by any
    # built-in parser), split at the *last* "Distractor1:" line — everything before is the
    # thinking block, everything from there is the final answer.
    if not reasoning and content:
        matches = list(re.finditer(r"(?:^|\n)\s*Distractor\s*1\s*:", content))
        if matches:
            split_idx = matches[-1].start()
            inline_reasoning = content[:split_idx].strip()
            answer_block = content[split_idx:].strip()
            if inline_reasoning and answer_block:
                reasoning = inline_reasoning
                content = answer_block

    if not content:
        raise ValueError(f"vLLM returned empty content; reasoning preview: {reasoning[:300]!r}")
    return reasoning, content


def prompt_deepseek(system_prompt: str, user_prompt: str, model_config: dict[str, str]) -> tuple[str|None,str]:
    """ Send prompt to Deepseek API, returns the reasoning content (if available) as well as the response """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"
    }

    payload = {
        "model": model_config["model"],
        "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        **model_config.get("completion_kwargs", {})
    }

    # response = requests.post(model_config["base_url"], headers=headers, json=payload, timeout=600)
    # pprint(response)
    # data = response.json()
    # pprint(data)


    chunks = []
    with requests.post(model_config["base_url"], headers=headers, json=payload, stream=True, timeout=600) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                chunks.append(chunk)


    reasoning = ""
    response = ""
    for chunk_bytes in chunks:
        chunk_text = chunk_bytes.decode("utf-8")
        for line in chunk_text.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":  # Some streaming APIs signal end like this
                continue
            try:
                data_json = json.loads(data_str)
            except json.JSONDecodeError:
                continue  # ignore malformed lines

            for choice in data_json.get("choices", []):
                delta = choice.get("delta", {})
                reasoning += delta.get("reasoning_content") or ""
                response += delta.get("content") or ""

    # reasoning = data["choices"][0]["message"].get("reasoning_content", None)
    # response = data["choices"][0]["message"]["content"]

    return reasoning,response


def prompt_openrouter(system_prompt: str, user_prompt: str, model_config: dict[str, str], stream: bool = False) -> tuple[str, str]:
    """
    Call OpenRouter API directly via requests so all payload fields (including
    repetition_penalty) are guaranteed to reach the provider.
    extra_body fields from model_config are flattened into the top-level payload.
    When stream=True, partial content is captured even if the model hits its token limit.
    Returns (reasoning, response).
    """
    api_key = os.getenv(model_config["api_key_var"])
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    completion_kwargs = model_config.get("completion_kwargs", {})
    extra_body = completion_kwargs.get("extra_body", {})
    payload = {
        "model": model_config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **{k: v for k, v in completion_kwargs.items() if k != "extra_body"},
        **extra_body,  # flatten so repetition_penalty etc. are top-level
        "stream": stream,
    }

    if stream:
        chunks = []
        with requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=payload, stream=True, timeout=600
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    chunks.append(chunk)

        reasoning = ""
        content = ""
        finish_reason = None
        for chunk_bytes in chunks:
            for line in chunk_bytes.decode("utf-8").splitlines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    data_json = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                for choice in data_json.get("choices", []):
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta", {})
                    content += delta.get("content") or ""
                    reasoning += delta.get("reasoning") or ""

        if not content:
            preview = repr(reasoning) if reasoning else "<no reasoning either>"
            raise ValueError(
                f"Model returned empty content (finish_reason={finish_reason!r}). "
                f"Partial reasoning ({len(reasoning)} chars): {preview}"
            )
        return reasoning, content

    else:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=payload, timeout=600
        )
        resp.raise_for_status()
        data = resp.json()

        message = data["choices"][0]["message"]
        content = message.get("content")
        reasoning = message.get("reasoning") or ""

        if content is None:
            raise ValueError("Model returned None content — try again with stream=True to capture partial output")

        return reasoning, content
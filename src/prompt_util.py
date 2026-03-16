import re
import json
import os
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


def prompt_openrouter(client: OpenAI, system_prompt: str, user_prompt: str, model_config: dict[str, str]) -> tuple[str, str]:
    """
    Call OpenRouter API (OpenAI-compatible) and extract reasoning if available.
    Returns (reasoning, response).
    """
    response = client.chat.completions.create(
        model=model_config.get("model"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        **model_config.get("completion_kwargs", {})
    )
    
    # Extract reasoning if available (in response metadata)
    reasoning = ""
    content = response.choices[0].message.content
    
    # Try to extract reasoning from response if model supports thinking
    if hasattr(response.choices[0].message, "reasoning"):
        reasoning = response.choices[0].message.reasoning
    else:
        # Fallback: check for reasoning blocks in content
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
    
    return reasoning, content
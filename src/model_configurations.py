# USED FOR EXPERIMENTS, DO NOT CHANGE
gpt_4_1_mini_config = {
    "base_url": "https://api.openai.com/v1/",
    "api_key_var": "OPENAI_API_KEY",
    "model": "gpt-4.1-mini-2025-04-14",
    "completion_kwargs": {
        "max_completion_tokens": 2024,
        "temperature": 0.7,
    }
}

gpt_4_1_mini_det_config = { # for correctness
    "base_url": "https://api.openai.com/v1/",
    "api_key_var": "OPENAI_API_KEY",
    "model": "gpt-4.1-mini-2025-04-14",
    "completion_kwargs": {
        "max_completion_tokens": 1024,
        "temperature": 0.0,
        "top_p": 0.0
    }
}

gpt_4o_mini_config = { # for correctness
    "base_url": "https://api.openai.com/v1/",
    "api_key_var": "OPENAI_API_KEY",
    "model": "gpt-4o-mini-2024-07-18",
    "completion_kwargs": {
        "max_completion_tokens": 1024,
        "temperature": 0.0,
    }
}

gpt_5_mini_config = { 
    "base_url": "https://api.openai.com/v1/",
    "api_key_var": "OPENAI_API_KEY",
    "model": "gpt-5-mini-2025-08-07",
    "completion_kwargs": {
        "max_completion_tokens": 16*1024,
        "temperature": 1.0,
        "reasoning_effort": "low", # minimal, low, medium, high
    }
}

gpt_3_5_config = {
    "base_url": "https://api.openai.com/v1/",
    "api_key_var": "OPENAI_API_KEY",
    "model": "gpt-3.5-turbo-1106",
    "completion_kwargs": {
        "max_completion_tokens": 2024,
        "temperature": 0.7,
    }
}

# we use http requests here because using OpenAI API has limited support :/
deepseek_reasoner = { # latest reasoning model
    "base_url": "https://api.deepseek.com/chat/completions",
    "api_key_var": "DEEPSEEK_API_KEY",
    "model": "deepseek-reasoner"
}

deepseek_chat = { # latest non-reasoning model
    "base_url": "https://api.deepseek.com/chat/completions",
    "api_key_var": "DEEPSEEK_API_KEY",
    "model": "deepseek-chat"
}

gpt_4_1 = { # smartest non-reasoning model
    "base_url": "https://api.openai.com/v1/",
    "api_key_var": "OPENAI_API_KEY",
    "model": "gpt-4.1-2025-04-14",
    "completion_kwargs": {
        "max_completion_tokens": 16000,
        "temperature": 0.0,
        "top_p": None
    }
}

gpt_4_1_det = { # smartest non-reasoning model
    "base_url": "https://api.openai.com/v1/",
    "api_key_var": "OPENAI_API_KEY",
    "model": "gpt-4.1-2025-04-14",
    "completion_kwargs": {
        "max_completion_tokens": 1000,
        "temperature": 0.0,
        "top_p": 0.0
    }
}

openrouter_glm_4_7_reason = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_var": "OPENROUTER_API_KEY",
    "model": "z-ai/glm-4.7",
    "completion_kwargs": {
        "max_tokens": 16000,
        "temperature": 0.0,
    }
}

openrouter_glm_4_7_chat = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_var": "OPENROUTER_API_KEY",
    "model": "z-ai/glm-4.7",
    "completion_kwargs": {
        "max_tokens": 8000,
        "temperature": 0.0,
        "extra_body": {"reasoning": {"enabled": False}}
    }
}

openrouter_gpt_oss_20b_reason = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_var": "OPENROUTER_API_KEY",
    "model": "openai/gpt-oss-20b",
    "completion_kwargs": {
        "max_tokens": 16000,
        "temperature": 0.0,
        "extra_body": {"reasoning": {"enabled": True}}
    }
}



# CAN CHANGE IF NEEDED

gpt_4_1_nano_config = {
    "base_url": "https://api.openai.com/v1/",
    "api_key_var": "OPENAI_API_KEY",
    "model": "gpt-4.1-nano-2025-04-14",
    "completion_kwargs": {
        "max_completion_tokens": 1024,
        "temperature": 1.0,
    }
}

gpt_5 = { # smartest reasoning model
    "base_url": "https://api.openai.com/v1/",
    "api_key_var": "OPENAI_API_KEY",
    "model": "gpt-5-2025-08-07"
}


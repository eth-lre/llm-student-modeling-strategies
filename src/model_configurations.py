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

deepseek_reasoner = { # V3.2 with thinking enabled
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_var": "OPENROUTER_API_KEY",
    "model": "deepseek/deepseek-v3.2",
    "completion_kwargs": {
        "extra_body": {
            "reasoning": {"enabled": True},
            "provider": {"only": ["alibaba"]},
        },
    },
}

deepseek_chat = { # V3.2 with thinking disabled
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_var": "OPENROUTER_API_KEY",
    "model": "deepseek/deepseek-v3.2",
    "completion_kwargs": {
        "extra_body": {
            "reasoning": {"enabled": False},
            "provider": {"only": ["alibaba"]},
        },
    },
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
        "extra_body": {
            "reasoning": {
                "enabled": True
            },
            "provider": {
                "zdr": True
            }
        }
    }
}

openrouter_glm_4_7_flash_reason = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_var": "OPENROUTER_API_KEY",
    "model": "z-ai/glm-4.7-flash",
    "completion_kwargs": {
        "max_tokens": 8000,
        "temperature": 0.5,
        "repetition_penalty": 1.3,
        "frequency_penalty": 2.0,
        "extra_body": {
            "reasoning": {
                "enabled": True
            },
            "provider": {
                "zdr": True
            }
        }
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

openrouter_gpt_oss_120b_reason = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_var": "OPENROUTER_API_KEY",
    "model": "openai/gpt-oss-120b",
    "completion_kwargs": {
        "max_tokens": 16000,
        "temperature": 0.0,
        "extra_body": {
            "repetition_penalty": 1.1,
            "reasoning": {
                "enabled": True
            },
            "provider": {
                "zdr": True
            }
        }
    }
}

openrouter_gpt_oss_20b_reason = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_var": "OPENROUTER_API_KEY",
    "model": "openai/gpt-oss-20b",
    "completion_kwargs": {
        "max_tokens": 16000,
        "temperature": 0.0,
        "extra_body": {
            "repetition_penalty": 1.1,
            "reasoning": {
                "enabled": True
            },
            "provider": {
                "zdr": True
            }
        }
    }
}

gpt_5 = { # smartest reasoning model
    "base_url": "https://api.openai.com/v1/",
    "api_key_var": "OPENAI_API_KEY",
    "model": "gpt-5-2025-08-07",
    "completion_kwargs": {
        "max_completion_tokens": 16*1024,
        "reasoning_effort": "medium", # minimal, low, medium, high
        "service_tier": "flex", # ~50% off, slower queue; eligible for gpt-5 / o-series
    }
}


openrouter_gemma_4_31b_reason = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_var": "OPENROUTER_API_KEY",
    "model": "google/gemma-4-31b-it",
    "completion_kwargs": {
        "max_tokens": 16000,
        "temperature": 0.0,
        "extra_body": {
            "reasoning": {
                "enabled": True
            },
            "provider": {
                "zdr": True
            }
        }
    }
}

gemini_2_5_pro = { # smartest reasoning model from Google
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "api_key_var": "GOOGLE_API_KEY",
    "model": "gemini-2.5-pro",
    "completion_kwargs": {
        "max_completion_tokens": 16*1024,
        "temperature": 0.0,
        "reasoning_effort": "medium", # low, medium, high (mapped to thinking budget)
        "service_tier": "flex", # ~50% off, async/best-effort
    }
}

gemini_2_5_flash = { # cheaper Google reasoning model
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "api_key_var": "GOOGLE_API_KEY",
    "model": "gemini-2.5-flash",
    "completion_kwargs": {
        "max_completion_tokens": 16*1024,
        "temperature": 0.0,
        "reasoning_effort": "medium", # low, medium, high (mapped to thinking budget)
        "service_tier": "flex", # ~50% off, async/best-effort
    }
}

vllm_gemma_4_31b = {
    "base_url": "http://localhost:8000/v1",
    "api_key_var": None,
    "model": "google/gemma-4-31b-it",
    "completion_kwargs": {
        "max_tokens": 16000,
        "temperature": 0.0,
        # Gemma thinking-mode is gated by the chat template, enable via vLLM extra_body
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
        },
    },
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




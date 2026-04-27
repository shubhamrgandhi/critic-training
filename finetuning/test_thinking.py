"""Quick test to inspect raw vLLM response for thinking/reasoning content.

Auto-detects the served model name. Tests with enable_thinking=True.
"""
from openai import OpenAI
import json

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8071/v1")

# Auto-detect model name
models = client.models.list()
model_name = models.data[0].id
print(f"Model: {model_name}\n")

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2? Think step by step."},
    ],
    temperature=0.6,
    max_completion_tokens=2048,
    n=1,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

choice = response.choices[0]
msg = choice.message
content = msg.content or ""

print(f"HAS <think>: {'<think>' in content}")
print(f"HAS </think>: {'</think>' in content}")
print(f"reasoning_content: {repr(getattr(msg, 'reasoning_content', None))[:300]}")
print(f"\n=== message.content (first 1000 chars) ===")
print(content[:1000])

import openai
import os

client = openai.OpenAI(
    api_key=os.environ['AI_GATEWAY_KEY'],
    base_url="https://ai-gateway.andrew.cmu.edu"  # Your LiteLLM Proxy URL
)

response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ]
)

print(response.choices[0].message.content)
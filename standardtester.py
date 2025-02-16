import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "deepseek/deepseek-r1:free",
    "messages": [
      {
        "role": "user",
        "content": "<appropriate_prompt_template>",
      }
    ],
    
  })
)

print(response.json()["choices"][0]["message"]["content"])
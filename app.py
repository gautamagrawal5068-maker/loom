from fastapi import FastAPI
from pydantic import BaseModel
from mlx_lm import load, generate

app = FastAPI(title="LOOM API")

MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
ADAPTER = "/Users/gautamagrawal/loom-finetune/adapters/loom-v2"

print("Loading LOOM model...")
model, tokenizer = load(MODEL, adapter_path=ADAPTER)
print("LOOM model loaded.")


class EmailRequest(BaseModel):
    context: str
    recipient_type: str = "general"
    purpose: str = "general"
    tone: str = "natural"
    length: str = "medium"
    style_profile: str = ""


@app.get("/")
def health_check():
    return {
        "name": "LOOM",
        "status": "online"
    }


@app.post("/generate")
async def generate_email(request: EmailRequest):

    prompt = f"""You are LOOM, an AI email-writing assistant.

Your job is to write an email for the current user.

IMPORTANT:
The writing style belongs to the CURRENT USER.
Never assume the user's style is Gautam's style.
Use the provided style profile to match the user's natural writing.

User's writing style:
{request.style_profile}

Email context:
{request.context}

Recipient type:
{request.recipient_type}

Purpose:
{request.purpose}

Requested tone:
{request.tone}

Requested length:
{request.length}

Rules:
- Write one complete email only.
    - Start the email once.
    - Never repeat a greeting, sentence, paragraph, or sign-off.
    - Stop after the email is complete.
    - Do not generate multiple versions.
    - Do not mention the style profile.
    - Do not explain your reasoning.
    - Write only the email.
"""

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=150,
    )

    return {
        "status": "success",
        "email": response
    }

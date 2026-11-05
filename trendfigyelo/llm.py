"""Gemini-hívás egy helyen, újrapróbálással kvótahiba (429) esetén."""
import time


def generate(client, model, prompt, json_output=False, temperature=None, retries=3, wait=20):
    config = {}
    if json_output:
        config["response_mime_type"] = "application/json"
    if temperature is not None:
        config["temperature"] = temperature
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config or None)
            return resp.text or ""
        except Exception as e:  # a google-genai ClientError-t dob, a kódja a szövegben is benne van
            if attempt < retries and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                print(f"Kvótahiba, {wait * (attempt + 1)} mp várakozás, majd újrapróbálom…")
                time.sleep(wait * (attempt + 1))
                continue
            raise

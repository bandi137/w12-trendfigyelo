"""Gemini-hívás egy helyen, újrapróbálással átmeneti hibák esetén.

- Kvótahiba (429, RESOURCE_EXHAUSTED): hosszabb várakozás (20, 40, 60, 80 mp), mert a percenkénti keret
  csak percek alatt áll helyre.
- Átmeneti szerverhiba (500, 502, 503, 504: pl. „UNAVAILABLE … high demand”): rövidebb, duplázódó várakozás
  (5, 10, 20, 40 mp). Ezek a Gemini túlterheltsége miatt jönnek, és általában rövid idő alatt elmúlnak.
- Minden más hiba (pl. hibás kérés, rossz kulcs) azonnal továbbmegy.
Ha minden próbálkozás elfogy, érthető magyar hibaüzenet jön (az eredeti hiba hozzáfűzve).
"""
import time

TRANSIENT = ("500", "502", "503", "504", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED", "high demand")
QUOTA = ("429", "RESOURCE_EXHAUSTED")


def _status(e):
    """A hiba HTTP-kódja, ha a google-genai megadja (ServerError/ClientError .code), különben None."""
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code if isinstance(code, int) else None


def _kind(e):
    code, text = _status(e), str(e)
    if code == 429 or any(k in text for k in QUOTA):
        return "quota"
    if (code is not None and 500 <= code < 600) or any(k in text for k in TRANSIENT):
        return "transient"
    return None


def generate(client, model, prompt, json_output=False, temperature=None, retries=4, wait=20, server_wait=5,
             sleep=None):
    config = {}
    if json_output:
        config["response_mime_type"] = "application/json"
    if temperature is not None:
        config["temperature"] = temperature
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config or None)
            return resp.text or ""
        except Exception as e:  # a google-genai ClientError/ServerError-t dob, a kód a szövegben is benne van
            kind = _kind(e)
            if kind is None:
                raise
            if attempt == retries:
                what = ("a percenkénti vagy napi kvóta elfogyott (429)" if kind == "quota" else
                        "a Gemini-szolgáltatás most túlterhelt vagy átmenetileg nem érhető el (5xx)")
                raise RuntimeError(f"A nyelvi modell hívása {retries + 1} próbálkozás után sem sikerült: {what}. "
                                   "Várj néhány percet, és futtasd újra a cellát (vagy kérdezz újra). "
                                   f"Eredeti hiba: {str(e)[:200]}") from e
            delay = wait * (attempt + 1) if kind == "quota" else server_wait * 2 ** attempt
            label = "Kvótahiba" if kind == "quota" else "A Gemini most túlterhelt"
            print(f"{label} ({_status(e) or '?'}), {delay} mp várakozás, majd újrapróbálom "
                  f"({attempt + 1}/{retries})…")
            (sleep or time.sleep)(delay)

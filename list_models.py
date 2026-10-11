"""List Gemini models available to the API key in .env."""
import os
import sys

import httpx
from dotenv import load_dotenv

URL = "https://generativelanguage.googleapis.com/v1beta/models"


def fetch(api_key: str = "") -> list:
    load_dotenv()
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add your Google AI Studio key to .env.")

    models = []
    page_token = None
    while True:
        params = {"pageToken": page_token} if page_token else {}
        response = httpx.get(URL, headers={"x-goog-api-key": api_key}, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        models.extend(data.get("models", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            return models


def select(models: list, term: str = "") -> list:
    rows = []
    for model in models:
        if "generateContent" not in model.get("supportedGenerationMethods", []):
            continue
        name = model.get("name", "").removeprefix("models/")
        if name and term.lower() in name.lower():
            rows.append(name)
    return sorted(rows)


def main(argv: list) -> int:
    term = " ".join(arg for arg in argv if not arg.startswith("--"))
    try:
        models = fetch()
    except (httpx.HTTPError, RuntimeError) as e:
        print(f"Could not download Gemini models: {e}")
        return 1
    rows = select(models, term)
    if not rows:
        print("No Gemini models matched.")
        return 0
    for model in rows:
        print(model)
    print(f"\n{len(rows)} model(s). Set GEMINI_MODEL in .env.")
    print("For photo search, choose a model that supports image input.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

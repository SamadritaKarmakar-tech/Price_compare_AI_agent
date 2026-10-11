"""Gemini client via Google's OpenAI-compatible API, with tolerant JSON parsing."""
import json
import re

import openai
from openai import OpenAI

from . import config

_client = None


class LLMError(RuntimeError):
    """A problem with the Gemini call, worded so the user knows what to fix."""


def _get() -> OpenAI:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add your Google AI Studio key to the .env file.")
        _client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=config.GEMINI_API_KEY,
        )
    return _client


def _create(model, messages, max_tokens, temperature):
    return _get().chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)


def _friendly(e: Exception, model: str) -> Exception:
    """Turn raw Gemini API errors into messages that say what to fix."""
    if isinstance(e, openai.NotFoundError):
        return LLMError(f"Gemini does not provide the model '{model}'. Check the model name and update GEMINI_MODEL "
                        "in your .env file.")
    if isinstance(e, openai.AuthenticationError):
        return LLMError("Google Gemini rejected your API key. Check GEMINI_API_KEY in your .env file.")
    if isinstance(e, openai.PermissionDeniedError):
        return LLMError(f"Google Gemini refused access to '{model}'. Check key access and model availability.")
    if isinstance(e, openai.RateLimitError):
        return LLMError("Google Gemini rate limit reached. Wait a minute or check your API quota.")
    if isinstance(e, openai.APIConnectionError):
        return LLMError("Could not reach Google Gemini. Check your internet connection.")
    if isinstance(e, openai.APIStatusError):
        if e.status_code == 400:
            return LLMError(f"Google Gemini rejected the request for '{model}'. Check the model name and request.")
        if e.status_code == 429:
            return LLMError("Google Gemini quota or rate limit reached. Check your API quota, then try again.")
        return LLMError(f"Google Gemini returned an error ({e.status_code}). Check your key, quota, and model.")
    return e


def chat(messages, model=None, max_tokens=1500, temperature=0.2) -> str:
    model = model or config.GEMINI_MODEL
    if not model:
        raise LLMError("GEMINI_MODEL is not set. Add a Gemini model name to your .env file.")
    try:
        resp = _create(model, messages, max_tokens, temperature)
    except Exception as e:
        raise _friendly(e, model) from None
    if not resp.choices:
        raise LLMError(f"Google Gemini returned an empty answer for '{model}'. Try again.")
    return (resp.choices[0].message.content or "").strip()


def _salvage_list(t: str):
    """If a model hit its token limit mid-array, keep the objects that finished."""
    if not t.lstrip().startswith("["):
        return None
    cut = t.rfind("}")
    for _ in range(60):
        if cut == -1:
            return None
        try:
            return json.loads(t[:cut + 1] + "]")
        except json.JSONDecodeError:
            cut = t.rfind("}", 0, cut)
    return None


def parse_json(text: str):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    pairs = [("{", "}"), ("[", "]")]
    pairs.sort(key=lambda p: t.find(p[0]) if p[0] in t else len(t))  # outermost bracket type first
    for opener, closer in pairs:
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    salvaged = _salvage_list(t)
    if salvaged is not None:
        return salvaged
    raise ValueError("Model did not return valid JSON")


def chat_json(messages, model=None, max_tokens=1800):
    """Ask for JSON; retry once with a stricter reminder if parsing fails."""
    text = chat(messages, model=model, max_tokens=max_tokens, temperature=0.1)
    try:
        return parse_json(text)
    except ValueError:
        retry = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON, no prose, no code fences."},
        ]
        return parse_json(chat(retry, model=model, max_tokens=max_tokens, temperature=0))

import os
import importlib
import requests
from dotenv import load_dotenv

# Load .env located in src
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Retrieve the SerpAPI key from the environment using the correct key name
SERPAPI_API_KEY = os.getenv("serpapi_key")
print("Loaded SERPAPI_API_KEY:", SERPAPI_API_KEY)

def _find_google_search_class():
    candidates = [
        "serpapi.google_search_results",
        "google_search_results",
        "google_search_results.google_search_results",
        "serpapi",
    ]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            cls = getattr(mod, "GoogleSearch", None)
            if cls is not None:
                return cls, name
        except Exception:
            continue
    return None, None

GoogleSearch, _module_used = _find_google_search_class()

def _serpapi_http_search(query: str) -> dict:
    if not SERPAPI_API_KEY:
        raise RuntimeError("SERPAPI_API_KEY missing; set it in src/.env or environment.")
    url = "https://serpapi.com/search"
    params = {"q": query, "engine": "google", "api_key": SERPAPI_API_KEY, "num": 3}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def _get_snippets(data: dict):
    return [
        (r.get("snippet") or r.get("title") or r.get("link"))
        for r in data.get("organic_results", [])[:3]
        if (r.get("snippet") or r.get("title") or r.get("link"))
    ]

def get_google_answer(query: str) -> str:
    if not SERPAPI_API_KEY:
        return "SERPAPI_API_KEY missing. Add it to src/.env or environment."

    # Prefer library if available
    if GoogleSearch is not None:
        try:
            client = GoogleSearch({"q": query, "engine": "google", "api_key": SERPAPI_API_KEY, "num": 3})
            data = client.get_dict()
            snippets = _get_snippets(data)
            if snippets:
                return "\n\n".join(snippets)
            return data.get("answer") or data.get("search_information", {}).get("query_displayed") or "No answer found."
        except Exception:
            # fallback to HTTP on any client error
            pass

    # HTTP fallback
    try:
        data = _serpapi_http_search(query)
        snippets = _get_snippets(data)
        return "\n\n".join(snippets) if snippets else "No answer found."
    except Exception as e:
        return (
            "SERPAPI client not installed and HTTP fallback failed. "
            "Install google-search-results into the active venv and ensure the environment is correct. "
            f"Error: {e}"
        )
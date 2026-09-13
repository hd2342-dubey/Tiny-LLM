"""
Convenience entry point: `tiny-llm-serve` starts the API + web UI on
http://localhost:8000 locally.

Reads $PORT when set (Render, Railway, Heroku-style platforms inject
this at runtime and expect the app to bind to it) and falls back to
8000 for local development.
"""

import os

import uvicorn


def main():
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()

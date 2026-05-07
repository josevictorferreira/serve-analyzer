"""Allow running backend as python -m web.backend."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("web.backend.app:app", host="0.0.0.0", port=8000, reload=False)

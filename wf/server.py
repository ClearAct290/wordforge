"""WordForge web server: FastAPI app + uvicorn, optional HTTPS, optional HTTP->HTTPS redirect."""
import collections
import ssl  # noqa: F401  (fail early if Python was built without TLS support)
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ROOT

WEB = ROOT / "web"


class GenerateRequest(BaseModel):
    definition: str
    pos: str


def create_app(cfg: dict, engine):
    srv, https = cfg["server"], cfg["https"]
    app = FastAPI(title="WordForge", docs_url=None, redoc_url=None, openapi_url=None)

    hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
    hits_lock = threading.Lock()
    limit = int(srv.get("rate_limit_per_minute", 0))

    def rate_ok(ip: str) -> bool:
        if limit <= 0:
            return True
        now = time.monotonic()
        with hits_lock:
            q = hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            if len(hits) > 10_000:                      # keep the table bounded
                for k in [k for k, v in hits.items() if not v][:5000]:
                    hits.pop(k, None)
        return True

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if https["enabled"] and https.get("hsts", True):
            resp.headers["Strict-Transport-Security"] = "max-age=31536000"
        return resp

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/license", include_in_schema=False)
    def license_page():
        return FileResponse(WEB / "license.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "device": engine.hw.describe(), "params": engine.params}

    @app.post("/api/generate")
    def generate(req: GenerateRequest, request: Request):
        ip = request.client.host if request.client else "?"
        if not rate_ok(ip):
            raise HTTPException(429, "Too many requests. Please wait a moment.")
        definition = req.definition.strip()
        if not definition:
            raise HTTPException(400, "Enter a definition first.")
        if len(definition) > 300:
            raise HTTPException(400, "Keep the definition under 300 characters.")
        try:
            result = engine.generate(definition, req.pos)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # noqa: BLE001
            print(f"[server] generation error: {type(e).__name__}: {e}", file=sys.stderr)
            raise HTTPException(500, "Could not generate a word. Try rephrasing the definition.")
        return JSONResponse({**result, "definition": definition}, headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
    return app


def _start_redirect(cfg: dict):
    """Tiny HTTP listener that 301-redirects everything to the HTTPS site."""
    srv, https = cfg["server"], cfg["https"]
    domain = srv.get("domain", "").strip()

    class Redirect(BaseHTTPRequestHandler):
        def _go(self):
            host = domain or (self.headers.get("Host", "localhost").rsplit(":", 1)[0] if self.headers.get("Host") else "localhost")
            port = "" if int(https["port"]) == 443 else f":{https['port']}"
            self.send_response(301)
            self.send_header("Location", f"https://{host}{port}{self.path}")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = _go

        def log_message(self, *a):  # silence
            pass

    httpd = ThreadingHTTPServer((srv["host"], int(srv["port"])), Redirect)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http-redirect").start()
    print(f"[server] http://{srv['host']}:{srv['port']} redirects to HTTPS")


def serve(cfg: dict):
    import uvicorn
    from .engine import Engine

    srv, https = cfg["server"], cfg["https"]
    kwargs = {"host": srv["host"], "port": int(srv["port"]), "log_level": srv.get("log_level", "info")}

    if https["enabled"]:
        cert, key = Path(str(https["certfile"])).expanduser(), Path(str(https["keyfile"])).expanduser()
        if not (https["certfile"] and https["keyfile"]) or not cert.is_file() or not key.is_file():
            raise SystemExit("[server] https.enabled = true but certfile/keyfile are missing or not found. "
                             "Set [https] certfile and keyfile to your PEM files (see config.toml).")
        kwargs.update(port=int(https["port"]), ssl_certfile=str(cert), ssl_keyfile=str(key))
        if https.get("keyfile_password"):
            kwargs["ssl_keyfile_password"] = https["keyfile_password"]

    print("[server] loading model ...")
    engine = Engine(cfg)
    print(f"[server] model ready on {engine.hw.describe()} ({engine.params / 1e6:.1f}M parameters)")
    app = create_app(cfg, engine)

    if https["enabled"] and https.get("redirect_http", True):
        try:
            _start_redirect(cfg)
        except OSError as e:
            print(f"[server] could not start the HTTP redirect on port {srv['port']}: {e}")

    scheme = "https" if https["enabled"] else "http"
    host = srv.get("domain") or ("localhost" if srv["host"] in ("0.0.0.0", "::") else srv["host"])
    default = 443 if https["enabled"] else 80
    shown = "" if kwargs["port"] == default else f":{kwargs['port']}"
    print(f"[server] WordForge is running: {scheme}://{host}{shown}/")
    try:
        uvicorn.run(app, **kwargs)
    except PermissionError:
        raise SystemExit(f"[server] no permission to bind port {kwargs['port']}. On Linux/macOS run as root, "
                         f"grant the capability (setcap cap_net_bind_service) or pick a port above 1023 in config.toml.")
    except OSError as e:
        raise SystemExit(f"[server] could not bind {kwargs['host']}:{kwargs['port']}: {e}")

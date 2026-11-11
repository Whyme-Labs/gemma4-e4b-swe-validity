"""Logging proxy between the harness (LiteLLM) and vLLM, for training-data provenance.

Forwards every request on LISTEN (one port, or a range FIRST-LAST, one port per sample) to vLLM on UPSTREAM unchanged, except that chat completions ask vLLM for token ids
(`return_token_ids`). Each chat completion is appended to LOG (gzip JSON lines): the request body, vLLM's response
(with `prompt_token_ids` and per-choice `token_ids`), the time taken and the port it came in on. The response is returned to the caller as
vLLM sent it; LiteLLM ignores the extra fields.

  python proxy.py LISTEN_PORT UPSTREAM_PORT LOG
"""
import asyncio, gzip, json, signal, sys, time

from aiohttp import ClientSession, ClientTimeout, web

first, _, last = sys.argv[1].partition("-")
LISTEN, UPSTREAM, LOG = range(int(first), int(last or first) + 1), int(sys.argv[2]), sys.argv[3]
log = gzip.open(LOG, "at", encoding="utf-8")
lock = asyncio.Lock()


async def forward(request: web.Request) -> web.Response:
    body = await request.read()
    chat = request.method == "POST" and request.path.endswith("/chat/completions")
    if chat:
        req = json.loads(body)
        req["return_token_ids"] = True
        body = json.dumps(req).encode()
    t = time.time()
    async with request.app["session"].request(
            request.method, f"http://127.0.0.1:{UPSTREAM}{request.path_qs}", data=body,
            headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}) as r:
        out = await r.read()
        status, ctype = r.status, r.headers.get("Content-Type", "application/json")
    if chat:
        try: resp = json.loads(out)
        except ValueError: resp = {"unparsed": out.decode(errors="replace")[:2000]}
        async with lock:
            log.write(json.dumps({"port": request.transport.get_extra_info("sockname")[1], "t": t, "seconds": round(time.time() - t, 3), "status": status, "request": req,
                                  "response": resp}) + "\n")
            log.flush()
    return web.Response(body=out, status=status, content_type=ctype.split(";")[0])


async def main():
    app = web.Application(client_max_size=256 * 2**20)
    app["session"] = ClientSession(timeout=ClientTimeout(total=None))
    app.router.add_route("*", "/{tail:.*}", forward)
    runner = web.AppRunner(app); await runner.setup()
    for port in LISTEN: await web.TCPSite(runner, "127.0.0.1", port).start()
    print(f"proxy {sys.argv[1]} -> {UPSTREAM}, log {LOG}", flush=True)
    stop = asyncio.Event()  # SIGTERM (run.py's terminate) closes the gzip stream, so the log ends with a valid trailer
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    await stop.wait()
    async with lock: log.close()


asyncio.run(main())

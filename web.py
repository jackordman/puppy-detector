#!/usr/bin/env python3
"""Phone-friendly page to report an accident and confirm the finder's candidates.
Serves http://homepi.local:8080 (stdlib only). Everything it writes goes under OUT_DIR.

    web.py            serve on PORT (default 8080)
    web.py selftest   exercise the JSON API against a temp OUT_DIR
"""
import json
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import events
import find_events as fe

OUT = fe.OUT
PORT = int(os.environ.get("PORT", "8080"))
LOCK = threading.Lock()  # ponytail: one clip at a time, the Pi has one YOLO to spare

PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Puppy cam</title><link rel=manifest href=/manifest.json><link rel=apple-touch-icon href=/icon.png>
<meta name=apple-mobile-web-app-capable content=yes><meta name=apple-mobile-web-app-status-bar-style content=black><meta name=theme-color content=#111111>
<style>
body{font:18px/1.4 system-ui,sans-serif;margin:0;padding:12px;background:#111;color:#eee;max-width:720px;margin:auto}
h2{margin:18px 0 8px;font-size:20px;color:#aaa}
button{font:inherit;padding:14px 18px;margin:4px 4px 4px 0;border:0;border-radius:10px;background:#333;color:#eee}
button.primary{background:#2a7;color:#000;font-weight:600}button.danger{background:#a33}button.small{padding:8px 12px;font-size:15px}
input{font:inherit;padding:10px;border-radius:8px;border:1px solid #444;background:#222;color:#eee;width:100%;box-sizing:border-box;margin:4px 0}
.card{background:#1c1c1c;border-radius:12px;padding:10px;margin:10px 0}.card img{width:100%;border-radius:8px;display:block}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.row input{flex:1;min-width:150px;width:auto}.row img{width:96px;height:72px;object-fit:cover;border-radius:8px}
.muted{color:#888;font-size:15px}#status{min-height:1.4em;color:#fc6}
</style></head><body>
<h2>Found one</h2>
<div><button class=primary onclick="find('1 hours ago')">Last 1 h</button><button class=primary onclick="find('2 hours ago')">2 h</button>
<button class=primary onclick="find('4 hours ago')">4 h</button><button class=primary onclick="find('8 hours ago')">8 h</button></div>
<div class=row><input type=datetime-local id=from><input type=datetime-local id=to></div>
<button onclick="find(document.getElementById('from').value,document.getElementById('to').value)">Search this window</button>
<div id=status></div><div id=cands></div>
<h2>Events</h2><div id=events></div>
<script>
const $=id=>document.getElementById(id);
const api=(p,b)=>fetch(p,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{}).then(r=>r.json());
let lastSince='1 hours ago';
async function find(since,until){ if(!since)return; lastSince=since; $('status').textContent='Searching...'; $('cands').innerHTML='';
  const c=await api('/api/find?since='+encodeURIComponent(since)+(until?'&until='+encodeURIComponent(until):''));
  $('status').textContent=c.error||(c.length?'Tap the one that is it':'Nothing in that window, try wider');
  $('cands').innerHTML=c.map(x=>`<div class=card>${x.sheet?`<img src="/files/${x.sheet}">`:'<div class=muted>no recording for this time</div>'}
    <div class=row><div style="flex:1"><b>${x.start.slice(11,19)}</b> <span class=muted>${x.secs}s still</span></div>
    <button class=primary onclick="pick('${x.id}')">This one</button></div></div>`).join('')
    +(c.length?`<button onclick="none()">None of these</button>`:'');}
function none(){const h=parseInt(lastSince)||1; find((h*2)+' hours ago');}
async function pick(id){ $('status').textContent='Cutting the clip, 20 to 60 s...';
  const r=await api('/api/confirm',{id}); if(r.error){$('status').textContent=r.error;return;}
  $('status').innerHTML=`Saved. Was it <button class=primary onclick="type('${id}','poop')">poop</button> or <button class=primary onclick="type('${id}','pee')">pee</button>?`;
  $('cands').innerHTML=''; loadEvents();}
async function type(id,t){ await api('/api/type',{id,type:t}); $('status').textContent=''; loadEvents();}
async function del(id){ if(!confirm('Delete this event?'))return; await api('/api/delete',{id}); loadEvents();}
async function loadEvents(){ const e=await api('/api/events');
  $('events').innerHTML=e.length?e.map(x=>`<div class="card row">${x.thumb?`<img src="/files/${x.thumb}">`:''}
    <div style="flex:1"><b>${x.time.replace('T',' ').slice(0,16)}</b><br><span class=muted>${x.type||'untyped'}${x.confirmed?'':' · unconfirmed ('+x.source+')'}</span></div>
    ${x.confirmed?`<button class="danger small" onclick="del('${x.id}')">Delete</button>`
    :`<button class="primary small" onclick="type('${x.id}','poop')">Poop</button><button class="primary small" onclick="type('${x.id}','pee')">Pee</button><button class=small onclick="dismiss('${x.id}')">Not it</button>`}</div>`).join('')
    :'<div class=muted>No events yet</div>';}
async function dismiss(id){ await api('/api/dismiss',{id}); loadEvents();}
loadEvents();
</script></body></html>"""


def api_find(q):
    return fe.find(q["since"][0], (q.get("until") or [None])[0])


def api_confirm(b):
    with LOCK:
        dst, n = fe.clip(b["id"], float(b.get("pad", 10)))
    sheet = next(iter((OUT / "candidates").glob(f"{b['id']}_*.jpg")), None)
    thumb = None
    if sheet:
        thumb = OUT / "events" / f"{b['id']}.jpg"
        thumb.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sheet, thumb)
    time = fe.datetime.strptime(b["id"], "%Y-%m-%d_%H%M%S").isoformat(timespec="seconds")
    events.remove(b["id"])
    return events.append(id=b["id"], time=time, confirmed=True, clip=str(dst.relative_to(OUT)),
                         thumb=str(thumb.relative_to(OUT)) if thumb else None, crops=n)


def api_type(b):
    if b["type"] not in ("poop", "pee"):
        raise ValueError("type must be poop or pee")
    events.update(b["id"], type=b["type"], confirmed=True)
    return {"ok": True}


def api_dismiss(b):
    events.update(b["id"], dismissed=True)  # kept as a labelled false positive for training
    return {"ok": True}


def api_delete(b):
    ev = next((e for e in events.load() if e["id"] == b["id"]), None)
    if ev and ev.get("clip"):
        (OUT / ev["clip"]).unlink(missing_ok=True)
        shutil.rmtree(OUT / ev["clip"].removesuffix(".mp4"), ignore_errors=True)
    events.remove(b["id"])
    return {"ok": True}


def api_events():
    return sorted((e for e in events.load() if not e.get("dismissed")), key=lambda e: e["time"], reverse=True)


class H(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif u.path in ("/manifest.json", "/icon.png"):
                p = Path(__file__).with_name(u.path[1:])
                self.send(200, p.read_bytes(), "application/manifest+json" if p.suffix == ".json" else "image/png")
            elif u.path == "/api/find":
                self.send(200, api_find(parse_qs(u.query)))
            elif u.path == "/api/events":
                self.send(200, api_events())
            elif u.path.startswith("/files/"):
                p = (OUT / u.path[7:]).resolve()
                if not p.is_relative_to(OUT.resolve()) or not p.is_file():
                    return self.send(404, {"error": "not found"})
                self.send(200, p.read_bytes(), "image/jpeg" if p.suffix == ".jpg" else "application/octet-stream")
            else:
                self.send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self.send(500, {"error": str(e)})

    def do_POST(self):
        if self.headers.get("Content-Type") != "application/json":  # blocks cross-site form posts
            return self.send(403, {"error": "json only"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or b"{}"))
        fn = {"/api/confirm": api_confirm, "/api/type": api_type, "/api/dismiss": api_dismiss, "/api/delete": api_delete}.get(self.path)
        if not fn:
            return self.send(404, {"error": "not found"})
        try:
            self.send(200, fn(body))
        except Exception as e:  # noqa: BLE001
            self.send(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            print(fmt % args, flush=True)


def selftest():
    import tempfile
    import urllib.request
    with tempfile.TemporaryDirectory() as td:
        global OUT
        OUT = events.OUT = fe.OUT = Path(td)
        events.FILE = OUT / "events.jsonl"
        (OUT / "detections").mkdir()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_port}"
        def call(p, b=None):
            req = urllib.request.Request(base + p, json.dumps(b).encode() if b else None, method="POST" if b else "GET",
                                         headers={"Content-Type": "application/json"} if b else {})
            try:
                return json.loads(urllib.request.urlopen(req).read())
            except urllib.error.HTTPError as e:
                return json.loads(e.read())
        get, post = call, call
        assert b"Found one" in urllib.request.urlopen(base + "/").read()
        assert "icons" in get("/manifest.json")
        assert get("/api/find?since=1%20hours%20ago") == []
        assert get("/api/events") == []
        events.append(id="x", time="2026-11-03T12:00:00", source="classifier", thumb="events/x.jpg")
        assert get("/api/events")[0]["confirmed"] is False
        post("/api/type", {"id": "x", "type": "pee"})
        assert get("/api/events")[0] == {**get("/api/events")[0], "type": "pee", "confirmed": True}
        events.append(id="y", time="2026-11-03T13:00:00", source="classifier")
        post("/api/dismiss", {"id": "y"})
        assert [e["id"] for e in get("/api/events")] == ["x"] and events.load()[1]["dismissed"]
        assert "error" in post("/api/type", {"id": "x", "type": "<img onerror=1>"})
        post("/api/delete", {"id": "x"})
        assert get("/api/events") == []
        plain = urllib.request.Request(base + "/api/delete", b"{}", method="POST")
        try:
            urllib.request.urlopen(plain)
            raise AssertionError("plain POST accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 403
        assert "error" in post("/api/confirm", {"id": "2026-01-01_000000"})
        assert get("/files/../etc/passwd") == {"error": "not found"}
        srv.shutdown()
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        print(f"serving on :{PORT}", flush=True)
        ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()

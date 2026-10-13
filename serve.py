#!/usr/bin/env python3
"""
Brick Sorter local helper.

Browsers block web pages from calling some sites directly (Rebrickable, and
possibly Brickognize), especially when the page is opened as a file. This
helper runs the app from your own computer and passes its requests on, so
everything works in the browser.

Run it from the folder that has brick-sorter.html:

    python serve.py

Your browser opens at http://localhost:8642. Keep this window open while you
use the app; press Ctrl + C to stop. Only standard Python is needed.

To use it from your phone on the same Wi-Fi, run `python serve.py --lan` and
open the address it prints. The phone's browser won't run the trained detector
on the graphics card, and some features may need the page to be served over
https on phones.
"""

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import bricklink            # optional: live BrickLink prices (see bricklink.py)
except ImportError:
    bricklink = None
UPSTREAM = {
    "rebrickable": os.environ.get("BRICK_SORTER_RB_URL", "https://rebrickable.com/api/v3/lego"),
    "brickognize": os.environ.get("BRICK_SORTER_BG_URL", "https://api.brickognize.com"),
}
USER_AGENT = "BrickSorter-local-helper/1.0"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def log_message(self, fmt, *args):          # keep the window quiet except for problems
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, status, body, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bricklink(self, tail):
        """BrickLink calls are signed here with your keys; the page only sees the answers."""
        from urllib.parse import urlparse, parse_qs
        u = urlparse("/" + tail)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        what = u.path.strip("/")
        if bricklink is None:
            return self._send(200, json.dumps({"configured": False, "reason": "bricklink.py is missing"}).encode())
        if what == "status":
            try:
                ok = bricklink.configured()
                return self._send(200, json.dumps({"configured": ok, "currency": bricklink.currency()}).encode())
            except bricklink.BrickLinkError as e:
                return self._send(200, json.dumps({"configured": False, "reason": str(e)}).encode())
        try:
            if what == "price":
                data = bricklink.price(q.get("kind", "part"), q["no"], q.get("color_id") or None,
                                       q.get("condition", "U"), q.get("guide", "sold"), q.get("currency") or None)
            elif what == "supersets":
                data = bricklink.supersets(q.get("kind", "part"), q["no"], q.get("color_id") or None)
            elif what == "subsets":
                data = bricklink.subsets(q.get("kind", "fig"), q["no"])
            else:
                return self._send(404, b'{"error": "unknown BrickLink request"}')
        except KeyError:
            return self._send(400, b'{"error": "missing item number"}')
        except bricklink.BrickLinkError as e:
            return self._send(502, json.dumps({"error": str(e)}).encode())
        return self._send(200, json.dumps(data).encode())

    def _proxy(self, method):
        rest = self.path[len("/proxy/"):]
        service, _, tail = rest.partition("/")
        if service == "ping":
            return self._send(200, b'{"ok": true}')
        if service == "bricklink":
            return self._bricklink(tail)
        if service not in UPSTREAM:
            return self._send(404, b'{"error": "unknown service"}')
        url = UPSTREAM[service] + "/" + tail
        body = None
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if method == "POST":
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            if self.headers.get("Content-Type"):
                headers["Content-Type"] = self.headers["Content-Type"]
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                self._send(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read() or b"{}", e.headers.get("Content-Type", "application/json"))
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            msg = json.dumps({"error": f"The helper couldn't reach {service}: {getattr(e, 'reason', e)}"})
            self._send(502, msg.encode())

    def do_GET(self):
        if self.path.startswith("/proxy/"):
            return self._proxy("GET")
        if self.path in ("/", "/index.html"):
            self.path = "/brick-sorter.html"
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/proxy/"):
            return self._proxy("POST")
        self._send(405, b'{"error": "not allowed"}')


def lan_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 80))       # no packets are sent; this just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--lan", action="store_true", help="also allow other devices on your network (e.g. your phone)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    args = ap.parse_args()

    if not (HERE / "brick-sorter.html").exists():
        sys.exit(f"brick-sorter.html isn't in {HERE}. Put serve.py in the same folder as brick-sorter.html.")
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    try:
        server = ThreadingHTTPServer((host, args.port), Handler)
    except OSError:
        sys.exit(f"Port {args.port} is already in use. Is the helper already running? "
                 f"Otherwise try: python serve.py --port {args.port + 1}")
    url = f"http://localhost:{args.port}/"
    print(f"Brick Sorter is running at {url}")
    if args.lan:
        ip = lan_address()
        if ip:
            print(f"On your phone (same Wi-Fi): http://{ip}:{args.port}/")
    print("Keep this window open while you use it. Press Ctrl + C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

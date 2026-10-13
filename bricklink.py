"""
BrickLink API access for Brick Sorter (used by serve.py and brick_sorter.py).

BrickLink signs every request with four keys from your BrickLink account. Put
them in bricklink_keys.json next to this file:

    {
      "consumer_key": "...",
      "consumer_secret": "...",
      "token_value": "...",
      "token_secret": "...",
      "currency": "AUD"
    }

or set BRICKLINK_CONSUMER_KEY, BRICKLINK_CONSUMER_SECRET, BRICKLINK_TOKEN_VALUE
and BRICKLINK_TOKEN_SECRET. The keys stay on your computer; the web app only
ever sees prices.

Only standard Python is needed. Answers are cached in bricklink_cache.json for
a day, so checking the same parts again doesn't use up BrickLink's daily
request allowance.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS_FILE = HERE / "bricklink_keys.json"
CACHE_FILE = HERE / "bricklink_cache.json"
API = os.environ.get("BRICK_SORTER_BL_URL", "https://api.bricklink.com/api/store/v1")
CACHE_SECONDS = 24 * 3600
TYPES = {"part": "PART", "fig": "MINIFIG", "minifig": "MINIFIG", "set": "SET", "sticker": "PART"}


class BrickLinkError(Exception):
    pass


def _enc(s):
    return urllib.parse.quote(str(s), safe="~-._")


def oauth_header(method, url, params, consumer_key, consumer_secret, token, token_secret,
                 nonce=None, timestamp=None):
    """OAuth 1.0 Authorization header (HMAC-SHA1), as BrickLink requires."""
    oauth = {
        "oauth_consumer_key": consumer_key,
        "oauth_token": token,
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(timestamp or int(time.time())),
        "oauth_nonce": nonce or secrets.token_hex(12),
        "oauth_version": "1.0",
    }
    pairs = sorted((_enc(k), _enc(v)) for k, v in list(params.items()) + list(oauth.items()))
    base = "&".join([method.upper(), _enc(url), _enc("&".join(f"{k}={v}" for k, v in pairs))])
    key = f"{_enc(consumer_secret)}&{_enc(token_secret)}"
    oauth["oauth_signature"] = base64.b64encode(hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()).decode()
    return 'OAuth realm="", ' + ", ".join(f'{_enc(k)}="{_enc(v)}"' for k, v in oauth.items())


def load_keys():
    keys = {}
    if KEYS_FILE.exists():
        try:
            keys = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        except ValueError as e:
            raise BrickLinkError(f"bricklink_keys.json isn't valid JSON: {e}")
    for name in ("consumer_key", "consumer_secret", "token_value", "token_secret"):
        env = os.environ.get("BRICKLINK_" + name.upper())
        if env:
            keys[name] = env
    return keys


def configured():
    k = load_keys()
    return all(k.get(n) for n in ("consumer_key", "consumer_secret", "token_value", "token_secret"))


_cache_lock = threading.Lock()
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _save_cache():
    try:
        CACHE_FILE.write_text(json.dumps(_cache), encoding="utf-8")
    except OSError:
        pass


def get(path, params=None, use_cache=True):
    """GET an API path like /items/PART/3001/price. Returns the 'data' part."""
    params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
    ck = path + "?" + urllib.parse.urlencode(sorted(params.items()))
    if use_cache:
        with _cache_lock:
            hit = _load_cache().get(ck)
            if hit and time.time() - hit["t"] < CACHE_SECONDS:
                return hit["data"]
    keys = load_keys()
    if not all(keys.get(n) for n in ("consumer_key", "consumer_secret", "token_value", "token_secret")):
        raise BrickLinkError("BrickLink keys aren't set up. Add bricklink_keys.json (see the README).")
    url = API + path
    auth = oauth_header("GET", url, params, keys["consumer_key"], keys["consumer_secret"],
                        keys["token_value"], keys["token_secret"])
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(full, headers={"Authorization": auth, "Accept": "application/json",
                                                "User-Agent": "BrickSorter/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8") or "{}")
        except ValueError:
            raise BrickLinkError(f"BrickLink returned {e.code}")
    except (urllib.error.URLError, TimeoutError) as e:
        raise BrickLinkError(f"Couldn't reach BrickLink: {getattr(e, 'reason', e)}")
    meta = body.get("meta") or {}
    code = int(meta.get("code") or 0)
    if code and not 200 <= code < 300:
        msg = meta.get("message") or ""
        hint = ""
        if "IP" in msg.upper():
            hint = " Your internet address may have changed: register the new one for your BrickLink token."
        elif code == 401:
            hint = " Check the keys in bricklink_keys.json."
        raise BrickLinkError(f"BrickLink said {code} {msg}: {meta.get('description', '')}.{hint}".strip())
    data = body.get("data")
    if use_cache:
        with _cache_lock:
            _load_cache()[ck] = {"t": time.time(), "data": data}
            _save_cache()
    return data


def currency():
    return load_keys().get("currency") or os.environ.get("BRICKLINK_CURRENCY") or ""


def price(kind, no, color_id=None, condition="U", guide="sold", currency_code=None):
    """Price guide summary for one item. kind: part, fig or set."""
    t = TYPES.get(kind, kind.upper())
    d = get(f"/items/{t}/{urllib.parse.quote(str(no), safe='')}/price", {
        "color_id": color_id if t == "PART" else None,
        "guide_type": guide, "new_or_used": condition,
        "currency_code": currency_code or currency() or None,
    })
    d = d or {}
    return {
        "no": no, "type": t, "condition": condition, "guide": guide,
        "currency": d.get("currency_code", currency_code or currency()),
        "avg": _num(d.get("avg_price")), "qty_avg": _num(d.get("qty_avg_price")),
        "min": _num(d.get("min_price")), "max": _num(d.get("max_price")),
        "sold_times": d.get("unit_quantity"), "quantity": d.get("total_quantity"),
    }


def supersets(kind, no, color_id=None):
    """Items (sets and minifigures) that contain this item: [{type, no, name, quantity}]."""
    t = TYPES.get(kind, kind.upper())
    d = get(f"/items/{t}/{urllib.parse.quote(str(no), safe='')}/supersets", {"color_id": color_id}) or []
    out = []
    for group in d:
        for e in group.get("entries") or []:
            it = e.get("item") or {}
            out.append({"type": it.get("type"), "no": it.get("no"), "name": it.get("name"),
                        "quantity": e.get("quantity"), "appear_as": e.get("appear_as") or e.get("appears_as"),
                        "color_id": group.get("color_id")})
    return out


def subsets(kind, no):
    """Parts of an item (e.g. a minifigure): [{type, no, name, color_id, quantity}]."""
    t = TYPES.get(kind, kind.upper())
    d = get(f"/items/{t}/{urllib.parse.quote(str(no), safe='')}/subsets") or []
    out = []
    for group in d:
        for e in group.get("entries") or []:
            it = e.get("item") or {}
            out.append({"type": it.get("type"), "no": it.get("no"), "name": it.get("name"),
                        "color_id": e.get("color_id"), "quantity": e.get("quantity")})
    return out


def _num(v):
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None

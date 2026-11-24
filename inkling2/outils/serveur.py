# serveur.py : façade OpenAI devant strata-ink2 (portage HIP), pour pi, Harn ou tout client OpenAI.
#
#   .venv/bin/python -I outils/serveur.py --model poids --engine build/strata-ink2 [--port 4799] [--ctx 131072]
#                                         [--spec] [--kv f16|i8r] [--api-key-file F] [--alias NOM]
#                                         [--vision build/inkling2-vision --mmproj poids/mmproj-BF16.gguf]
#
# Le moteur tourne en --serve, protocole par lignes de Strata : on lui envoie « GEN <max> k=v… <ids> », il répond
# « PP … » pendant le préremplissage, « T <id> » par token, puis « DONE <gen> <prompt> <prompt_ms> <decode_ms> <fin> » ;
# « STOP » l'interrompt. Il garde son cache d'un appel à l'autre tant que le nouveau prompt prolonge l'ancien : un
# agent qui ajoute des tours à la conversation ne repaie pas le préremplissage.
#
# Réponse d'Inkling : des messages <|message_model|>[NOM]<|content_KIND|>CORPS<|end_message|>, KIND valant thinking
# (-> reasoning_content), text (-> content) ou invoke_tool_json (CORPS = {"name":…, "args":{…}} -> tool_calls), puis
# <|content_model_end_sampling|>. Une seule requête à la fois (le moteur n'a qu'un fil) : les autres attendent.
#
# Images (avec --vision et --mmproj) : le gabarit officiel met <|content_image|><|unused_200054|><|end_message|> à
# la place de chaque image. inkling2-vision (l'encodeur de llama.cpp, seul) en tire N vecteurs, un par carré de
# 40 x 40 pixels ; « IMG <fichier> » les donne au moteur, qui répond « IMG <premier id> <N> », et ces N ids
# remplacent le token. Une image déjà vue (même SHA-256) garde ses ids : le cache du prompt la reconnaît.
# Seules les URL data: sont acceptées (pas de téléchargement par la façade).
import argparse, base64, hashlib, json, os, re, subprocess, sys, tempfile, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

HEAD, THINK, TEXT, CALL = "<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>"
END, STOP = "<|end_message|>", "<|content_model_end_sampling|>"
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
IMAGE_TOKEN = "<|unused_200054|>"
IMAGE_TYPES = ("image", "input_image", "image_url")
IMG_ROWS_MAX = 100_000   # vecteurs d'image gardés par le moteur (~0,9 Go de mémoire vive) avant un IMGRESET


class GenerationTag(jinja2.ext.Extension):
    """{% generation %}…{% endgeneration %} de transformers (masque d'entraînement) : le corps tel quel."""
    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


class BadRequest(ValueError):
    pass


class Model:
    def __init__(self, a):
        self.tok = Tokenizer.from_file(f"{a.model}/tokenizer.json")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols", GenerationTag])
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def fail(m):
            raise BadRequest(m)
        env.globals["raise_exception"] = fail
        self.template = env.from_string(open(f"{a.model}/chat_template.jinja", encoding="utf-8").read())
        self.special = {self.tok.token_to_id(t): t for t in (HEAD, THINK, TEXT, CALL, END, STOP)}
        self.eos = self.tok.token_to_id(STOP)
        cmd = [a.engine, "--model", a.model, "--serve", "--chunk", "512", "--ctx", str(a.ctx), "--kv", a.kv]
        if a.spec:
            cmd += ["--spec", "1"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        line = self.proc.stdout.readline().split()
        if not line or line[0] != "READY":
            sys.exit(f"le moteur n'a pas démarré : {line}")
        self.ctx = int(line[1])
        self.lock = threading.Lock()
        self.image_id = self.tok.token_to_id(IMAGE_TOKEN)
        self.images, self.img_rows, self.vision = {}, 0, None   # SHA-256 -> (premier id, N)
        if a.vision and a.mmproj:
            cmd = [a.vision, "--mmproj", a.mmproj, "--serve"]
            if a.image_max_tokens:
                cmd += ["--max-tokens", str(a.image_max_tokens)]
            self.vision = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           text=True, bufsize=1)
            if self.vision.stdout.readline().strip() != "READY":
                sys.exit("l'encodeur d'image n'a pas démarré")
            self.tmp = tempfile.mkdtemp(prefix="inkling2-images-")

    # --- images ---
    def image_bytes(self, p):
        src = p.get("image_url", p.get("image"))
        if isinstance(src, dict):
            src = src.get("url")
        if not isinstance(src, str) or not src.startswith("data:"):
            raise BadRequest("image : seule une URL data: (base64) est acceptée")
        head, _, body = src.partition(",")
        if ";base64" not in head:
            raise BadRequest("image : URL data: attendue en base64")
        try:
            return base64.b64decode(body, validate=False)
        except ValueError:
            raise BadRequest("image : base64 illisible")

    def image_ids(self, data):
        """Les ids que le moteur a donnés à cette image (encodée et chargée la première fois). Sous self.lock."""
        h = hashlib.sha256(data).hexdigest()
        if h in self.images:
            return self.images[h]
        src, out = os.path.join(self.tmp, h), os.path.join(self.tmp, h + ".inkv")
        with open(src, "wb") as f:
            f.write(data)
        try:
            self.vision.stdin.write(f"{src}\t{out}\n")
            self.vision.stdin.flush()
            r = self.vision.stdout.readline().split(maxsplit=1)
            if not r:
                raise RuntimeError("l'encodeur d'image s'est arrêté")
            if r[0] != "OK":
                raise BadRequest(f"image refusée : {r[1].strip() if len(r) > 1 else '?'}")
            n = int(r[1].split()[0])
            if self.img_rows + n > IMG_ROWS_MAX:   # le moteur oublie toutes les images, et son cache avec
                self.proc.stdin.write("IMGRESET\n")
                self.proc.stdin.flush()
                if self.proc.stdout.readline().strip() != "IMGRESET":
                    raise RuntimeError("le moteur n'a pas oublié les images")
                self.images, self.img_rows = {}, 0
            self.proc.stdin.write(f"IMG {out}\n")
            self.proc.stdin.flush()
            f = self.proc.stdout.readline().split()
            if not f or f[0] != "IMG":
                raise RuntimeError(f"le moteur a refusé l'image : {' '.join(f)}")
            first, n = int(f[1]), int(f[2])
        finally:
            for x in (src, out):
                if os.path.exists(x):
                    os.remove(x)
        self.images[h] = (first, n)
        self.img_rows += n
        return first, n

    def expand(self, ids, images):
        """Chaque <|unused_200054|> du prompt devient les ids des vecteurs de son image. Sous self.lock."""
        if not images:
            return ids
        out, k = [], 0
        for t in ids:
            if t == self.image_id:
                if k >= len(images):
                    raise BadRequest("plus de tokens image que d'images")
                first, n = self.image_ids(images[k])
                out.extend(range(first, first + n))
                k += 1
            else:
                out.append(t)
        if k != len(images):
            raise BadRequest(f"{len(images)} images, {k} places dans le prompt")
        return out

    # --- prompt ---
    def prompt(self, req):
        """(ids, images) : les ids du gabarit, un <|unused_200054|> par image, et les octets des images dans l'ordre."""
        messages, images = [], []
        for m in req.get("messages") or []:
            role = m.get("role")
            if role == "developer":
                role = "system"
            out = {"role": role}
            c = m.get("content")
            if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_TYPES for p in c):
                if not self.vision:
                    raise BadRequest("images : lancer la façade avec --vision et --mmproj")
                parts = []   # le gabarit rend chaque morceau dans son propre bloc, l'image à sa place
                for p in c:
                    if isinstance(p, str):
                        parts.append({"type": "text", "text": p})
                    elif p.get("type") in ("text", "input_text"):
                        parts.append({"type": "text", "text": p.get("text") or ""})
                    elif p.get("type") in IMAGE_TYPES:
                        images.append(self.image_bytes(p))
                        parts.append({"type": "image"})
                    else:
                        raise BadRequest(f"morceau de message non pris en charge : {p.get('type')}")
                c = parts
            elif isinstance(c, list):
                parts = []
                for p in c:
                    if isinstance(p, str):
                        parts.append(p)
                    elif p.get("type") in ("text", "input_text"):
                        parts.append(p.get("text") or "")
                    else:
                        raise BadRequest(f"morceau de message non pris en charge : {p.get('type')} (ce moteur lit le texte et les images)")
                c = "".join(parts)
            out["content"] = c if c is not None else ""
            if m.get("reasoning_content"):
                out["reasoning_content"] = m["reasoning_content"]
            if m.get("name"):
                out["name"] = m["name"]
            if m.get("tool_call_id"):
                out["tool_call_id"] = m["tool_call_id"]
            if m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    fn = dict(tc.get("function") or {})
                    args = fn.get("arguments")
                    if isinstance(args, str):   # OpenAI envoie une chaîne JSON ; le gabarit veut l'objet
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError:
                            raise BadRequest(f"arguments d'outil illisibles : {args[:80]}")
                    fn["arguments"] = args if isinstance(args, dict) else {}
                    calls.append({"id": tc.get("id"), "type": "function", "function": fn})
                out["tool_calls"] = calls
            messages.append(out)
        if not messages:
            raise BadRequest("messages vide")
        kw = dict(req.get("chat_template_kwargs") or {})
        effort = req.get("reasoning_effort", kw.get("reasoning_effort"))
        if isinstance(req.get("reasoning"), dict) and req["reasoning"].get("effort"):
            effort = req["reasoning"]["effort"]
        if effort is not None:
            effort = str(effort).lower()
            effort = {"off": "none", "disabled": "none", "false": "none", "maximum": "max"}.get(effort, effort)
            if effort not in EFFORTS:
                raise BadRequest(f"reasoning_effort inconnu : {effort} (none, minimal, low, medium, high, xhigh, max)")
            kw["reasoning_effort"] = effort
        text = self.template.render(messages=messages, tools=req.get("tools") or None, add_generation_prompt=True, **kw)
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if not images and self.image_id in ids:   # le texte d'un client ne doit pas ouvrir une place d'image
            raise BadRequest(f"{IMAGE_TOKEN} est réservé aux images")
        return ids, images

    # --- génération ---
    def generate(self, req, prompt, on_token, cancelled):
        kv = {"temperature": req.get("temperature", 1.0), "top_p": req.get("top_p"), "top_k": req.get("top_k"),
              "min_p": req.get("min_p"), "seed": req.get("seed")}
        args = " ".join(f"{k}={v}" for k, v in kv.items() if v is not None)
        with self.lock:
            ids = self.expand(*prompt)
            if len(ids) >= self.ctx:
                raise BadRequest(f"prompt de {len(ids)} tokens, contexte {self.ctx}")
            max_new = req.get("max_completion_tokens") or req.get("max_tokens") or self.ctx
            max_new = max(1, min(int(max_new), self.ctx - len(ids)))
            self.proc.stdin.write(f"GEN {max_new} {args} {','.join(map(str, ids))}\n")
            self.proc.stdin.flush()
            stop_sent = False
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("le moteur s'est arrêté")
                f = line.split()
                if not f:
                    continue
                want_stop = False
                if f[0] == "T":
                    want_stop = not on_token(int(f[1]))
                elif f[0] == "DONE":
                    return {"gen": int(f[1]), "prompt": int(f[2]), "prompt_ms": float(f[3]), "decode_ms": float(f[4]),
                            "finish": f[5]}
                elif f[0] == "ERR":
                    raise BadRequest(line[4:].strip())
                if (want_stop or cancelled()) and not stop_sent:   # client parti : le moteur s'arrête au token suivant
                    self.proc.stdin.write("STOP\n")
                    self.proc.stdin.flush()
                    stop_sent = True


class Reply:
    """Lit les tokens d'Inkling au fil de l'eau et en tire des deltas OpenAI."""

    def __init__(self, model):
        self.m, self.sec, self.body, self.sent, self.calls = model, None, [], 0, []
        self.reasoning, self.content = [], []

    def _text(self, final=False):
        s = self.m.tok.decode(self.body, skip_special_tokens=False)
        if not final and s.endswith("�"):   # un caractère coupé entre deux tokens : attendre la suite
            return ""
        new, self.sent = s[self.sent:], len(s)
        return new

    def _flush(self):
        out = []
        if self.sec in ("think", "text"):
            t = self._text(final=True)
            if t:
                out.append(("reasoning" if self.sec == "think" else "content", t))
        elif self.sec == "call":
            raw = self.m.tok.decode(self.body, skip_special_tokens=False).strip()
            try:
                j = json.loads(raw)
                call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                        "function": {"name": j.get("name", ""), "arguments": json.dumps(j.get("args", {}), ensure_ascii=False)}}
            except json.JSONDecodeError:   # appel mal formé : rendu comme texte, pour que le client le voie
                out.append(("content", raw))
            else:
                self.calls.append(call)
                out.append(("call", call))
        self.sec, self.body, self.sent = None, [], 0
        return out

    def feed(self, t):
        mark = self.m.special.get(t)
        if mark is None:
            if self.sec is None:
                return []           # le NOM entre l'en-tête et le contenu (déjà dans le JSON de l'appel)
            self.body.append(t)
            if self.sec == "call":
                return []
            d = self._text()
            return [("reasoning" if self.sec == "think" else "content", d)] if d else []
        if mark in (THINK, TEXT, CALL):
            out = self._flush()
            self.sec = {THINK: "think", TEXT: "text", CALL: "call"}[mark]
            return out
        return self._flush()        # HEAD, END, STOP

    def finish(self):
        return self._flush()


def run(a):
    model = Model(a)
    key = open(a.api_key_file).read().strip() if a.api_key_file else None
    name = a.alias

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *x):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _auth(self):
            if key and self.headers.get("Authorization", "") != f"Bearer {key}":
                self._json(401, {"error": {"message": "clé d'API invalide", "type": "authentication_error"}})
                return False
            return True

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if not self._auth():
                return
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "local",
                                                                    "meta": {"n_ctx_train": 1048576, "n_ctx": model.ctx}}]})
            if self.path == "/props":
                return self._json(200, {"default_generation_settings": {"n_ctx": model.ctx}, "model_alias": name})
            self._json(404, {"error": {"message": "inconnu"}})

        def do_POST(self):
            if not self._auth():
                return
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                return self._json(404, {"error": {"message": "inconnu"}})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                prompt = model.prompt(req)
            except (BadRequest, json.JSONDecodeError, jinja2.TemplateError) as e:
                return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            stream, rid, created = bool(req.get("stream")), "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
            reply, gone = Reply(model), [False]

            def chunk(delta, finish=None, extra=None):
                o = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": name,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if extra:
                    o.update(extra)
                try:
                    self.wfile.write(f"data: {json.dumps(o, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    gone[0] = True

            ncall = [0]

            def emit(events):
                if not stream:
                    return
                for kind, v in events:
                    if kind == "reasoning":
                        chunk({"reasoning_content": v})
                    elif kind == "content":
                        chunk({"content": v})
                    else:
                        chunk({"tool_calls": [{"index": ncall[0], "id": v["id"], "type": "function", "function": v["function"]}]})
                        ncall[0] += 1

            def on_token(t):
                events = reply.feed(t)
                for kind, v in events:
                    (reply.reasoning if kind == "reasoning" else reply.content if kind == "content" else []).append(v)
                emit(events)
                return not gone[0]

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                chunk({"role": "assistant", "content": ""})
            try:
                st = model.generate(req, prompt, on_token, lambda: gone[0])
            except BadRequest as e:
                if stream:
                    chunk({}, "error")
                    return
                return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            last = reply.finish()
            for kind, v in last:
                (reply.reasoning if kind == "reasoning" else reply.content if kind == "content" else []).append(v)
            emit(last)
            finish = "tool_calls" if reply.calls else {"stop": "stop", "length": "length"}.get(st["finish"], "stop")
            timings = {"prompt_n": st["prompt"], "prompt_ms": st["prompt_ms"],
                       "prompt_per_second": st["prompt"] * 1000 / max(st["prompt_ms"], 1e-3),
                       "predicted_n": st["gen"], "predicted_ms": st["decode_ms"],
                       "predicted_per_second": st["gen"] * 1000 / max(st["decode_ms"], 1e-3)}
            usage = {"prompt_tokens": st["prompt"], "completion_tokens": st["gen"], "total_tokens": st["prompt"] + st["gen"]}
            if stream:
                chunk({}, finish, {"usage": usage, "timings": timings})
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
                return
            msg = {"role": "assistant", "content": "".join(reply.content) or (None if reply.calls else "")}
            if reply.reasoning:
                msg["reasoning_content"] = "".join(reply.reasoning)
            if reply.calls:
                msg["tool_calls"] = reply.calls
            self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": name,
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage, "timings": timings})

    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"prêt : http://{a.host}:{a.port}/v1 ({name}, contexte {model.ctx})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4799)
    p.add_argument("--ctx", type=int, default=131072)
    p.add_argument("--kv", default="f16")
    p.add_argument("--spec", action="store_true")
    p.add_argument("--api-key-file")
    p.add_argument("--alias", default="inkling-2-mini")
    p.add_argument("--vision", help="build/inkling2-vision : l'encodeur d'image")
    p.add_argument("--mmproj", help="poids/mmproj-BF16.gguf (unsloth/Inkling-2-Mini-GGUF)")
    p.add_argument("--image-max-tokens", type=int, help="plafond de vecteurs par image (par défaut : celui du mmproj)")
    run(p.parse_args())

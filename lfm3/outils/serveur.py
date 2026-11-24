# serveur.py : façade OpenAI devant lfm3-engine, pour pi, Harn ou tout client OpenAI.
#
#   .venv/bin/python -I outils/serveur.py --model poids --engine build/lfm3-engine --gguf poids/LFM3-1.7B-QAD-Q4_0.gguf
#                                         [--port 4799] [--ctx 131072] [--api-key-file F] [--alias NOM]
#
# Le moteur tourne en --serve (protocole de strata-ink2) : « GEN <max> k=v… <ids> », réponses « PP … », « T <id> »,
# « DONE <gen> <prompt> <prompt_ms> <decode_ms> <fin> » ; « STOP » l'interrompt. Il garde son cache tant que le
# nouveau prompt prolonge l'ancien (renvoyer reasoning_content pour que le gabarit réécrive la même chose).
#
# Réponse de LFM3 (gabarit officiel) : le prompt finit par « <think> » (raisonnement) ou « <think></think> » ; le
# modèle écrit son raisonnement, « </think> », puis le texte et d'éventuels appels :
#   <tool_call>\n<function=NOM>\n<parameter=CLÉ>\nVALEUR\n</parameter>\n…</function>\n</tool_call>
# puis <|im_end|>. Une requête à la fois (le moteur n'a qu'un fil) : les autres attendent.
import argparse, json, re, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

THINK_END, CALL_START, CALL_END, IM_END, EOT = "</think>", "<tool_call>", "</tool_call>", "<|im_end|>", "<|endoftext|>"
# generation_config.json de LiquidAI : température 0,1, top-k 50, pénalité de répétition 1,1
DEFAULTS = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1}


class GenerationTag(jinja2.ext.Extension):
    """{% generation %}…{% endgeneration %} de transformers (masque d'entraînement) : le corps tel quel."""
    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


class BadRequest(ValueError):
    pass


def parse_call(raw):
    """<function=NOM>\\n<parameter=CLÉ>\\nVALEUR\\n</parameter>… -> (nom, arguments) ; None si illisible."""
    m = re.search(r"<function=([^>\n]+)>", raw)
    if not m:
        return None
    args = {}
    for k, v in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", raw, re.S):
        try:
            args[k] = json.loads(v)
        except json.JSONDecodeError:
            args[k] = v
    return m.group(1).strip(), args


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
        self.bos = "<|startoftext|>"
        self.special = {self.tok.token_to_id(t): t for t in (THINK_END, CALL_START, CALL_END, IM_END, EOT)}
        cmd = [a.engine, "--model", a.gguf, "--serve", "--ctx", str(a.ctx)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        line = self.proc.stdout.readline().split()
        if not line or line[0] != "READY":
            sys.exit(f"le moteur n'a pas démarré : {line}")
        self.ctx = int(line[1])
        self.lock = threading.Lock()

    def prompt(self, req):
        """(ids, réflexion) : les ids du gabarit officiel, et si le prompt ouvre un raisonnement."""
        messages = []
        for m in req.get("messages") or []:
            role = m.get("role")
            if role == "developer":
                role = "system"
            out = {"role": role}
            c = m.get("content")
            if isinstance(c, list):
                parts = []
                for p in c:
                    if isinstance(p, str):
                        parts.append(p)
                    elif p.get("type") in ("text", "input_text"):
                        parts.append(p.get("text") or "")
                    else:
                        raise BadRequest(f"morceau de message non pris en charge : {p.get('type')} (ce moteur lit le texte)")
                c = "".join(parts)
            out["content"] = c if c is not None else ""
            for k in ("reasoning_content", "name", "tool_call_id"):
                if m.get(k):
                    out[k] = m[k]
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
        effort = req.get("reasoning_effort", kw.pop("reasoning_effort", None))
        if isinstance(req.get("reasoning"), dict) and req["reasoning"].get("effort"):
            effort = req["reasoning"]["effort"]
        if effort is not None and "enable_thinking" not in kw:
            kw["enable_thinking"] = str(effort).lower() not in ("none", "minimal", "off", "disabled", "false")
        thinking = kw.get("enable_thinking", True)
        text = self.template.render(messages=messages, tools=req.get("tools") or None, add_generation_prompt=True,
                                    bos_token=self.bos, **kw)
        return self.tok.encode(text, add_special_tokens=False).ids, bool(thinking)

    def generate(self, req, ids, on_token, cancelled):
        with self.lock:
            if len(ids) >= self.ctx:
                raise BadRequest(f"prompt de {len(ids)} tokens, contexte {self.ctx}")
            max_new = req.get("max_completion_tokens") or req.get("max_tokens") or self.ctx
            max_new = max(1, min(int(max_new), self.ctx - len(ids)))
            kv = dict(DEFAULTS)
            for k in ("temperature", "top_p", "top_k", "min_p", "seed"):
                if req.get(k) is not None:
                    kv[k] = req[k]
            rp = req.get("repeat_penalty", req.get("repetition_penalty"))
            if rp is not None:
                kv["repeat_penalty"] = rp
            args = " ".join(f"{k}={v}" for k, v in kv.items())
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
                    return {"gen": int(f[1]), "prompt": int(f[2]), "prompt_ms": float(f[3]), "decode_ms": float(f[4]), "finish": f[5]}
                elif f[0] == "ERR":
                    raise BadRequest(line[4:].strip())
                if (want_stop or cancelled()) and not stop_sent:
                    self.proc.stdin.write("STOP\n")
                    self.proc.stdin.flush()
                    stop_sent = True


class Reply:
    """Lit les tokens de LFM3 au fil de l'eau et en tire des deltas OpenAI."""

    def __init__(self, model, thinking):
        self.m, self.sec, self.body, self.sent, self.calls = model, ("think" if thinking else "text"), [], 0, []
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
            raw = self.m.tok.decode(self.body, skip_special_tokens=False)
            got = parse_call(raw)
            if got is None:   # appel mal formé : rendu comme texte, pour que le client le voie
                out.append(("content", raw))
            else:
                call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                        "function": {"name": got[0], "arguments": json.dumps(got[1], ensure_ascii=False)}}
                self.calls.append(call)
                out.append(("call", call))
        self.body, self.sent = [], 0
        return out

    def feed(self, t):
        mark = self.m.special.get(t)
        if mark is None:
            self.body.append(t)
            if self.sec == "call":
                return []
            d = self._text()
            return [("reasoning" if self.sec == "think" else "content", d)] if d else []
        if mark == THINK_END:
            out = self._flush()
            self.sec = "text"
            return out
        if mark == CALL_START:
            out = self._flush()
            self.sec = "call"
            return out
        if mark == CALL_END:
            out = self._flush()
            self.sec = "text"
            return out
        return self._flush()        # <|im_end|>, <|endoftext|>

    def finish(self):
        return self._flush()


def run(a):
    model = Model(a)
    key = open(a.api_key_file).read().strip() if a.api_key_file else None
    name = a.alias

    class Hd(BaseHTTPRequestHandler):
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
                                                                    "meta": {"n_ctx_train": 262144, "n_ctx": model.ctx}}]})
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
                ids, thinking = model.prompt(req)
            except (BadRequest, json.JSONDecodeError, jinja2.TemplateError) as e:
                return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            stream, rid, created = bool(req.get("stream")), "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
            reply, gone = Reply(model, thinking), [False]

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

            def keep(events):
                for kind, v in events:
                    (reply.reasoning if kind == "reasoning" else reply.content if kind == "content" else []).append(v)
                emit(events)

            def on_token(t):
                keep(reply.feed(t))
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
                st = model.generate(req, ids, on_token, lambda: gone[0])
            except BadRequest as e:
                if stream:
                    chunk({}, "error")
                    return
                return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            keep(reply.finish())
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
            content = "".join(reply.content).strip("\n")
            msg = {"role": "assistant", "content": content or (None if reply.calls else "")}
            if reply.reasoning:
                msg["reasoning_content"] = "".join(reply.reasoning).strip("\n")
            if reply.calls:
                msg["tool_calls"] = reply.calls
            self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": name,
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage, "timings": timings})

    srv = ThreadingHTTPServer((a.host, a.port), Hd)
    print(f"prêt : http://{a.host}:{a.port}/v1 ({name}, contexte {model.ctx})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="dossier : tokenizer.json, chat_template.jinja")
    p.add_argument("--engine", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4799)
    p.add_argument("--ctx", type=int, default=131072)
    p.add_argument("--api-key-file")
    p.add_argument("--alias", default="lfm3-1.7b-qad")
    run(p.parse_args())

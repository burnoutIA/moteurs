# prompt.py encode <dossier poids> <texte> [effort] : ids du prompt de chat (gabarit officiel), séparés par des virgules
# prompt.py decode <dossier poids> <ids…>          : le texte de ces ids
import json, sys
from tokenizers import Tokenizer
import jinja2
cmd, d = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file(f"{d}/tokenizer.json")
if cmd == "encode":
    import re
    src = open(f"{d}/chat_template.jinja").read()
    src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)   # balise propre à transformers
    env = jinja2.Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False, **k)
    text = env.from_string(src).render(messages=[{"role": "user", "content": sys.argv[3]}],
        add_generation_prompt=True, reasoning_effort=sys.argv[4] if len(sys.argv) > 4 else "none")
    print(",".join(map(str, tok.encode(text, add_special_tokens=False).ids)))
else:
    ids = [int(x) for x in " ".join(sys.argv[3:]).replace(",", " ").split()]
    print(tok.decode(ids, skip_special_tokens=False))

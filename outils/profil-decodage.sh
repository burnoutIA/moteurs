#!/bin/bash
# ROCM=<racine ROCm> profil-decodage.sh <dossier bin llama.cpp> <modèle .gguf> [tokens] [options llama-server…]
#
# Profile le décodage d'un moteur llama.cpp HIP avec rocprofv3 et dit où part le temps d'un token :
# nombre de noyaux par token, part du temps où la carte calcule vraiment, écart médian entre deux
# noyaux, et les noyaux qui coûtent le plus. Sert à décider s'il faut un moteur sur mesure
# (README.md de ce dépôt) : si la carte calcule moins de ~60 % du temps, le goulot est
# l'enchaînement des noyaux, et seule la fusion d'opérations le lève.
#
# Le profileur ralentit le moteur (~30 %) : les chiffres absolus sont pessimistes, les proportions
# sont justes. Le port 4799 doit être libre, et la carte assez vide pour charger le modèle.
set -e
BIN=$(realpath "$1"); MODEL=$(realpath "$2"); N=${3:-200}; shift 3 2>/dev/null || shift $#
ROCM=${ROCM:-/opt/rocm}
OUT=$(mktemp -d "${TMPDIR:-/tmp}/profil-decodage.XXXXXX")

LD_LIBRARY_PATH=$ROCM/lib:$BIN "$ROCM/bin/rocprofv3" --kernel-trace -f csv -d "$OUT/trace" -- \
  "$BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port 4799 --jinja --no-webui --parallel 1 \
  -ngl 999 --fit off -fa on -c 8192 --device ROCm0 "$@" > "$OUT/serveur.log" 2>&1 &
prof=$!
for _ in $(seq 1 300); do
  python3 -I -c 'import urllib.request;urllib.request.urlopen("http://127.0.0.1:4799/health")' 2>/dev/null && break
  kill -0 $prof 2>/dev/null || { echo "le moteur s'est arrêté :"; tail -5 "$OUT/serveur.log"; exit 1; }
  sleep 1
done
NS=$(python3 -I - "$N" <<'EOF'
import json, sys, urllib.request
body = {"messages": [{"role": "user", "content": "Écris en Python une classe LRUCache complète avec tests."}],
        "max_tokens": int(sys.argv[1]), "reasoning_effort": "none", "temperature": 0, "ignore_eos": True}
t = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:4799/v1/chat/completions",
    json.dumps(body).encode(), {"Content-Type": "application/json"})))["timings"]
print(f'{t["predicted_n"]} {t["predicted_ms"] * 1e6:.0f}')
EOF
)
# rocprofv3 se remplace par le serveur (même pid) : l’arrêter proprement fait écrire la trace.
kill -TERM $prof
wait $prof 2>/dev/null || true
python3 -I - $NS "$(find "$OUT/trace" -name '*_kernel_trace.csv' | head -1)" <<'EOF'
import collections, csv, re, sys
n, span, path = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
rows = sorted(((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"])
               for r in csv.DictReader(open(path))), key=lambda x: x[0])
end = rows[-1][1]
dec = [x for x in rows if x[0] > end - span * 0.98]       # la fenêtre du décodage, à la fin de la trace
wall = (dec[-1][1] - dec[0][0]) / 1e9
busy = sum(e - s for s, e, _ in dec) / 1e9
gaps = sorted(g for g in (dec[i + 1][0] - dec[i][1] for i in range(len(dec) - 1)) if g > 0)
print(f"{n} tokens · {len(dec) / n:.0f} noyaux par token · la carte calcule {busy / wall * 100:.0f} % du temps"
      f" · écart médian entre noyaux {gaps[len(gaps) // 2] / 1e3:.1f} µs")
agg = collections.defaultdict(lambda: [0, 0])
for s, e, k in dec:
    k = re.sub(r"\(.*", "", k)[:100]
    agg[k][0] += 1
    agg[k][1] += e - s
print(f"{'ms/token':>9} {'par token':>9} {'durée':>8}  noyau")
for k, (c, d) in sorted(agg.items(), key=lambda x: -x[1][1])[:20]:
    print(f"{d / 1e6 / n:9.2f} {c / n:9.0f} {d / c / 1e3:6.1f}µs  {k}")
EOF
rm -rf "$OUT"

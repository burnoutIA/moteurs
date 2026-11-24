#!/bin/bash
# lancer.sh [options de outils/serveur.py] : l'API OpenAI d'Inkling 2 Mini sur http://127.0.0.1:4799/v1
# (exemple : ./lancer.sh --ctx 131072 --port 4799 ; ajouter --spec pour le drafter)
# Images : prises en charge dès que build-vision.sh a compilé build/inkling2-vision et que poids/mmproj-BF16.gguf est là.
cd "$(dirname "$0")"
VISION=()
if [ -x build/inkling2-vision ] && [ -f poids/mmproj-BF16.gguf ]; then
  VISION=(--vision build/inkling2-vision --mmproj poids/mmproj-BF16.gguf)
fi
exec .venv/bin/python -I outils/serveur.py --model poids --engine build/strata-ink2 "${VISION[@]}" "$@"

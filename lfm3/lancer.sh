#!/bin/bash
# lancer.sh [options de outils/serveur.py] : l'API OpenAI de LFM3 1.7B QAD sur http://127.0.0.1:4799/v1
# (exemple : ./lancer.sh --ctx 131072 --port 4799)
cd "$(dirname "$0")"
exec .venv/bin/python -I outils/serveur.py --model poids --engine build/lfm3-engine --gguf poids/LFM3-1.7B-QAD-Q4_0.gguf "$@"

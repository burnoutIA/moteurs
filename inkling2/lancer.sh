#!/bin/bash
# lancer.sh [options de outils/serveur.py] : l'API OpenAI d'Inkling 2 Mini sur http://127.0.0.1:4799/v1
# (exemple : ./lancer.sh --ctx 131072 --port 4799 ; ajouter --spec pour le drafter)
cd "$(dirname "$0")"
exec .venv/bin/python -I outils/serveur.py --model poids --engine build/strata-ink2 "$@"

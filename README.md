# moteurs

Des moteurs d'inférence écrits pour **un seul modèle**, quand les moteurs généralistes laissent
trop de vitesse de côté. Pour l'instant, un seul : **Inkling 2 Mini** (Thinking Machines,
Apache 2.0), servi par `strata-ink2` sur une carte graphique de 24 Go ou plus.

## Inkling 2 Mini : près de deux fois plus vite que llama.cpp

Mesures sur le checkpoint officiel (NVFP4 + MXFP8), 128k de contexte, sans spéculation :

| Carte | llama.cpp | strata-ink2 | après 8k de prompt | préremplissage |
|---|---|---|---|---|
| NVIDIA RTX 3090 (CUDA d'origine) | 54 tok/s | **94 tok/s** | 89 tok/s | 723 tok/s |
| AMD Radeon AI PRO R9700 (portage HIP) | 33 tok/s | **68 tok/s** | 65 tok/s | 510 tok/s |

Avec 128k de contexte, le moteur prend 21,4 Go sur la carte.

**Pourquoi c'est plus rapide.** Inkling repasse trois fois dans les mêmes couches (6 à 17) pour
chaque token. llama.cpp découpe ce travail en plus de 7 000 petits noyaux par token, et la carte
passe plus de la moitié du temps à attendre entre deux. `strata-ink2` fusionne ces opérations en
~1 200 noyaux, joués dans un seul graphe par token.

**Validation.** En décodage glouton, les sorties ne divergent de llama.cpp que sur des
quasi-égalités : le token choisi par llama.cpp est toujours notre deuxième choix, à quelques
points de probabilité près. Le moteur d'origine avait été validé contre transformers en fp32.
Détails dans [inkling-2-mini.md](inkling-2-mini.md).

## Installer

Il faut Linux, une carte de 24 Go et ~20 Go de disque pour les poids. Tout se passe dans
[`inkling2/`](inkling2/).

```bash
cd inkling2
node outils/telecharger-poids.mjs poids          # 20 Go depuis Hugging Face, avec reprise
python3 -m venv .venv && .venv/bin/pip install tokenizers jinja2
```

Puis compiler le moteur, selon la carte :

```bash
# NVIDIA (testé sur RTX 3090, CUDA 13.4) : CUDA_ARCH=89 pour une RTX 40, 120 pour une RTX 50
CUDA_ARCH=86 ./build-cuda.sh

# AMD RDNA4 (RX 9070, R9700 ; testé avec ROCm 10.1) : GPU_TARGET=gfx1200 pour une RX 9060
ROCM=/opt/rocm ./build.sh
```

Et lancer l'API compatible OpenAI :

```bash
./lancer.sh --ctx 131072 --port 4799
curl http://127.0.0.1:4799/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Bonjour"}],"reasoning_effort":"low"}'
```

L'API gère le flux SSE, les appels d'outils, le raisonnement séparé (`reasoning_content`), et
l'effort de raisonnement (`none` à `max`). Elle garde le cache du prompt d'un tour à l'autre.
Option `--spec` : le drafter MTP du modèle, utile en contexte court et contre-productif sur les
longs prompts.

## Limites

- Pas de vision : le moteur ne lit que le texte.
- Une requête à la fois.
- Pas de version allégée : le moteur lit le checkpoint officiel, pas les GGUF quantifiés.
- Préremplissage plus lent que llama.cpp (calcul en fp32 strict).
- Linux seulement. Côté NVIDIA, seule la 3090 a été essayée.

## Contenu

| Chemin | Rôle |
|---|---|
| `inkling2/amont/` | sources CUDA d'origine de `strata-ink2`, inchangées (Strata, commit `5050dca`) |
| `inkling2/src/ink2_engine.hip` | le portage HIP pour RDNA4 |
| `inkling2/outils/serveur.py` | l'API compatible OpenAI devant le moteur |
| `inkling-2-mini.md` | architecture, mesures, choix du portage, validation, pièges |
| [METHODE.md](METHODE.md) | quand un moteur sur mesure vaut la peine, et comment s'y prendre |
| `outils/profil-decodage.sh` | profil du décodage sur AMD (noyaux par token, temps de calcul) |

## Crédits et licence

`strata-ink2` est l'œuvre de Niko1221 et des contributeurs de
[Strata](https://github.com/baptisterajaut/Strata) (branche `feat/inkling2`, licence MIT). Ce
dépôt ajoute le portage HIP, l'API et les scripts. Le tout est sous licence MIT (voir
[LICENSE](LICENSE)). Les poids d'Inkling 2 Mini sont sous Apache 2.0, chez
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini).

# LFM3 1.7B (QAD Q4_0) : fiche du moteur sur mesure

Dernière mise à jour : 24 novembre 2026. État : **moteur écrit et validé, API OpenAI**, HIP (R9700) et
CUDA (RTX 3090). Décodage +24 à +44 % sur llama.cpp, préremplissage devant sur la R9700. Pas encore
branché dans Harn. Aucun moteur sur mesure n'existait pour LFM3 (ni Strata, ni ailleurs) : écrit de zéro.

## Sources

- Modèle : [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B), licence LFM 1.0.
- GGUF : [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF).
  `LFM3-1.7B-QAD-Q4_0.gguf` (1 049 615 680 octets, SHA-256 `43a7d09a…72a7452`) est le point de
  contrôle « Quantization-Aware Distillation » : entraîné pour retrouver la justesse perdue en
  Q4_0. Échantillonnage conseillé par LiquidAI : température 0,1, top-k 50, pénalité de
  répétition 1,1.
- Piège : la fiche Harn « LFM3 1.7B QAD » (`lfm3-1-7b-gguf-bf16`) installe le **BF16**
  (3,47 Go), pas le fichier QAD, avec l'échantillonnage 0,7 / top-p 0,95 / top-k 20.

## Architecture

Dense, 24 couches : 13 à convolution courte, 6 en attention globale, 5 en attention glissante.
Vocabulaire de 128k lié (entrée et sortie), ~15 % des paramètres.

## Plancher, plafond, profil

Mesure : `llama-server`, une requête de code (LRUCache), 512 tokens en glouton, `ignore_eos`,
sans spéculation, 3 passages après un échauffement (médiane retenue).

| Carte | Fichier | Moteur | Décodage | Plafond (octets / bande passante) | Part du plafond |
|---|---|---|---|---|---|
| R9700 (640 Go/s) | BF16 | llama.cpp b12064 Vulkan | 140 tok/s | ~184 | 76 % |
| R9700 | BF16 | llama.cpp b12065 HIP | 119 | | |
| R9700 | BF16 | RDNA Boosts 64f2e88 HIP | 137 | | |
| R9700 | QAD Q4_0 | llama.cpp b12064 Vulkan | **279** | ~610 | 46 % |
| R9700 | QAD Q4_0 | llama.cpp b12065 HIP | 241 | | |
| R9700 | QAD Q4_0 | RDNA Boosts 64f2e88 HIP | 236 | | |
| RTX 3090 (936 Go/s) | BF16 | llama.cpp b12651 CUDA 12 | 186 | ~270 | 69 % |
| RTX 3090 | QAD Q4_0 | llama.cpp b12651 CUDA 12 | **366** | ~890 | 41 % |

llamAmpere v0.5 ne connaît pas l'architecture `lfm3`.

En BF16, un moteur sur mesure ne ferait pas beaucoup mieux (au-dessus de 60 % du plafond). En QAD
Q4_0, il reste plus de la moitié de la vitesse sur la table, sur les deux cartes.

Profil (R9700, RDNA Boosts HIP, QAD Q4_0, `outils/profil-decodage.sh`, 300 tokens) :
- **381 noyaux par token**, la carte calcule **61 %** du temps, 3,7 µs médians entre deux noyaux ;
- `mul_mat_vec_q_ksplit` : 2,50 ms/token en 106 appels, soit ~420 Go/s (65 % du débit) ;
- ensuite des normes (`rms_norm_f32`, `rms_norm_q8_1_f32`, ~47 chacune), `quantize_q8_1`,
  `k_bin_bcast`, l'attention flash (11), la convolution (13), des copies et concaténations.

Deux leviers : la fusion (381 noyaux, ~1,4 ms de trous) et un produit matrice-vecteur Q4_0 plus
proche du débit de la carte.

## Le moteur (`lfm3/`)

| Fichier | Rôle |
|---|---|
| `src/lfm3_engine.cu` | le moteur, une seule source CUDA et HIP (rocWMMA / `nvcuda::wmma`, hipBLAS / cuBLAS) |
| `build.sh` | HIP : `ROCM=<racine> [GPU_TARGET=gfx1201] ./build.sh` |
| `build-cuda.sh` | CUDA : `[CUDA_ARCH=86] [CXX=g++-14] ./build-cuda.sh` |
| `outils/serveur.py` | façade OpenAI (gabarit officiel, `<think>`, appels d'outils au format `<tool_call><function=…>`) |
| `lancer.sh` | `./lancer.sh --ctx 131072 [--port 4799]` |

Refaire de zéro : le GGUF QAD et les fichiers du tokenizer de `LiquidAI/LFM3-1.7B` (`tokenizer.json`,
`chat_template.jinja`, `config.json`…) dans `poids/`, `python3 -m venv .venv && .venv/bin/pip install
tokenizers==0.23.3 jinja2==3.1.6`, puis `build.sh` ou `build-cuda.sh`.

### Choix

- **Décodage** : un graphe de **112 noyaux par token** (llama.cpp : 381). La carte calcule 80-84 % du
  temps (llama.cpp : 61 %).
  - Poids Q4_0 réorganisés au chargement : échelles à part, quartets en 16 octets alignés, lus en
    128 bits par fil ; l'activation est quantifiée en q8 par blocs de 32 (comme llama.cpp) et tenue
    dans les registres de chaque warp ; produit `dp4a` / `sudot4`.
  - Chaque produit matrice-vecteur commence par la norme dont il a besoin (prologue : norme du bloc
    précédent, résidu, norme d'entrée), refaite par chaque bloc de la grille. Le fil t tient les
    éléments t + 256k : chaque warp tient un bloc de 32 valeurs, la quantification se fait par
    échanges entre fils (K = 7 680 : en mémoire partagée, moins cher que 450 échanges par fil).
  - Épilogues : SwiGLU (lignes gate/up entrelacées) ; la convolution courte entière (lignes b, c, x
    entrelacées par canal : b·x, fenêtre de 7, c·conv, état mis à jour).
  - Sortie liée Q6_K réorganisée (groupes de 32 : quartets bas, 2 bits hauts, échelles int8), arg-max
    par atomique 64 bits : ~606 Go/s, 95 % du débit de la carte.
  - Attention : bloc (morceau de positions, tête KV), les 4 têtes de requête ensemble, norme q/k et
    RoPE sur place ; courte (≤ 1 024 positions) : lecture directe, morceaux de 32 ; longue : tuiles de
    32 clés en mémoire partagée pour les 4 têtes, morceaux de 128, jusqu'à 128 morceaux. Le dernier
    bloc d'une tête KV combine les morceaux et applique la porte (ticket atomique, pas de noyau en plus).
  - Essayé et écarté : précharger les poids pendant le prologue (registres, −16 %), 4 lignes par warp
    hors gate/up, plus de 4 blocs par CU (le prologue refait coûte plus qu'il ne rapporte).
- **Préremplissage** : paquets de 2 048 tokens ; une copie fp16 des poids (+3 Go) et hipBLAS / cuBLAS
  pour les produits ; normes, convolution (en séquence dans le temps, en parallèle sur les canaux),
  SwiGLU par noyaux groupés ; attention sur les unités matricielles (WMMA 16×16×16) : bloc (16 tokens,
  tête KV), S = Q·Kᵀ, softmax en ligne (une paire de fils par ligne), O dans les registres, corrigé
  seulement quand le maximum d'une ligne monte ; la tête de sortie pour la dernière position seulement.
  Un échauffement au démarrage charge les noyaux de hipBLAS (sans lui, le premier prompt paie ~100 ms).
- **Échantillonnage** : glouton par l'arg-max de la carte ; sinon sur l'hôte (logits copiés, top-k,
  top-p, min-p, pénalité de répétition sur les 64 derniers tokens) : ~10 % du temps de décodage.

### Validation (24/11)

Contre llama.cpp (Vulkan b12064 sur la R9700, CUDA b12651 sur la 3090), même suite de tokens : la
référence génère 200 tokens en glouton avec ses 8 meilleurs log-probas par position, notre moteur note
la même suite (préremplissage groupé du prompt, puis décodage).

| Prompt | R9700 : top-1 identique | 3090 : top-1 identique | \|Δ log-proba\| moyen (4 meilleurs) |
|---|---|---|---|
| code (29 tokens) | 198/200 | 196/200 | 0,065 |
| prose (22) | 195/200 | 195/200 | 0,055 |
| fichier (4 370) | 198/200 | 199/200 | 0,060 |
| fichiers (17 891) | 199/200 | 200/200 | 0,06 |

Chaque désaccord tombe sur une quasi-égalité (écart < 0,3 entre les deux premiers chez llama.cpp).
Le chemin groupé et le chemin token par token donnent les mêmes 200 tokens ; le moteur est
déterministe d'une exécution à l'autre. Façade : texte, raisonnement séparé, appel d'outil puis réponse
au résultat, essayés.

### Mesures (24/11, glouton, `--ctx 32768`)

| | R9700 : llama.cpp Vulkan | R9700 : nous | 3090 : llama.cpp CUDA | 3090 : nous |
|---|---|---|---|---|
| Décodage, contexte court | 279 tok/s | **~395** | 366 | **514** |
| Décodage après 4 370 tokens | ~260 | **370** | 367 | **485** |
| Décodage après 17 891 tokens | 249 | **308** | 340 | **433** |
| Préremplissage, 4 370 tokens | 17 000 (RDNA Boosts : 20 000) | **24 400** | 20 300 | 20 800 |
| Préremplissage, 17 891 tokens | 15 500 | **16 900** | 19 200 | 14 200 |

Mémoire : 5,8 Gio à 131k (poids Q4_0 1 Go, copie fp16 3 Go, cache KV f16 des 6 couches globales).
Piège de mesure : sur la R9700, la mémoire est au repos à 96 MHz ; un préremplissage de 300 ms peut
finir avant que les fréquences montent (mesures bimodales, 13k ou 19k). `LFM3_REPEAT=N` refait le
préremplissage N fois et garde le meilleur.

### Reste

1. Brancher dans Harn (fiche à part, comme `inkling-2-mini-ink2`), banc et test d'intelligence.
2. Échantillonnage sur la carte (top-k), pour ne plus copier 512 Ko de logits par token.
3. 3090, préremplissage long : produits en int8 sur les tensor cores (llama.cpp : MMQ, deux fois le débit
   du fp16) et une attention mieux réglée pour Ampere.
4. Cache KV en q8 pour les longs contextes (la moitié des octets lus au décodage).

# Inkling 2 Mini sur R9700 : fiche du moteur sur mesure

Dernière mise à jour : 21 novembre 2026. État : **portage HIP validé, servi en API OpenAI** (~67 tok/s contre 33 ; 131k de contexte dans 21,4 Gio). Reste : branchement dans Harn, préremplissage.

## Sources

- Modèle : [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini),
  Apache 2.0. Checkpoint officiel NVFP4 + MXFP8 en un seul fichier safetensors (19,5 Go), plus
  `mtp.safetensors` (636 Mo, le drafter) et le tokenizer tiktoken.
- Moteur de départ : `strata-ink2`, branche `feat/inkling2` de
  [baptisterajaut/Strata](https://github.com/baptisterajaut/Strata/tree/feat/inkling2), sous
  licence MIT (Niko1221 et les contributeurs de Strata). Fichiers : `engine/src/inkling2/ink2_engine.cu`
  (3 800 lignes, commit 5050dca) et `ink2ref.cu` (chemin de référence). Validé contre
  transformers#51050 en fp32 : KL 1,4e-12. CUDA uniquement : `nvcuda::wmma`, cuBLAS, et une
  compilation conditionnée à `STRATA_ENABLE_CUDA`. Suivi dans le ticket Strata #6442.
- Moteur général de référence : le mélange Harn `mix-inkling2-loop`, soit llama.cpp 44c20b1 +
  PR #32218 (DSpark) + unslothai#385 (attention relative découpée), avec #32704 déjà dans master.

## Architecture (config.json officiel)

| | |
|---|---|
| Couches | 24 ; les couches 6 à 17 tournent **3 fois** avec les mêmes poids, soit 48 passages par token (`loop_config` : core 6-18, 3 passes, `gate_rank` 64) |
| Dimension | 2 304 ; 24 têtes de requête, 8 têtes KV, `head_dim` 128 |
| Attention | 20 couches locales (fenêtre 1 024) et 4 globales (5, 11, 17, 23) ; biais de position relatif (`d_rel` 16, `rel_extent` 1 024) ; `qk_l2_clip` 22,63, `rel_l2_clip` 8 ; mise à l'échelle logarithmique (`log_scaling_n_floor` 8 192, alpha 0,1) ; norme des valeurs sans gamma |
| Divers | convolution courte (`sconv_kernel_size` 4) ; résidus d'attention par blocs de 6 couches, valeurs normalisées |
| MoE | 128 experts, 8 actifs, aucun partagé ; `intermediate_size` 1 152 ; softcaps : gate 4, up 25 ; `route_scale` 8, biais de gate |
| Sortie | softcap final 30 ; vocabulaire 201 024 (200 058 utiles) |
| Drafter (mtp) | 3 couches (0 et 2 locales), blocs de **8 tokens**, tête Markov RNN de rang 256 ; branché sur les couches 5, 23, 41 et 48 du passage déroulé |
| Contexte | jusqu'à 1M de tokens |
| Échantillonnage | température 1 (generation_config) ; l'effort de raisonnement minimal ne coupe pas tout le raisonnement |

## Plancher, plafond, profil (R9700, ROCm 10.1)

**Plancher**, avec le banc Harn et le moteur général : 39,5 tok/s de moyenne (code 44,3, texte
35,3), 120k de contexte en KV f16, DSpark n3, IQ 115. Sans spéculation : 33 tok/s.

**Plafond**, estimé : les experts actifs pèsent 3 × 2 304 × 1 152 poids × 8 experts × 48
passages, soit ~1,7 Go en NVFP4. L'attention pèse ~19M poids × 48 passages, soit ~1 Go en Q8_0.
Ça fait ~2,7 Go lus par token à 640 Go/s, soit **~4,2 ms/token, environ 235 tok/s au maximum**.
Le cache de la carte ne garde pas les 12 couches en boucle d'un passage au suivant (~240 Mo).
Nous sommes à 15 % du plafond.

**Profil** (`outils/profil-decodage.sh`, le 21/11, sans spéculation) :
- **7 265 noyaux par token** ; **la carte ne calcule que 40 % du temps** ; écart médian de
  3,4 µs entre deux noyaux ;
- les graphes HIP sont bien utilisés (« CUDA Graph id N reused » des centaines de fois) : le
  coût de lancement côté processeur n'est pas en cause ;
- `mul_mat_vec_q` (les experts) prend 6,7 ms/token en 377 appels. Viennent ensuite 1 494
  `k_bin_bcast` de 1,3 µs et des centaines de petits noyaux de mise à l'échelle, copie,
  concaténation, norme et convolution. Ce sont eux, et les écarts qui les séparent, qui
  coûtent le plus cher.

**Conclusion** : la fusion d'opérations est le seul vrai levier. `strata-ink2` en est la preuve
sur une 3090, avec ~1 200 noyaux par token : 90 tok/s en contexte court, 80 après 16k, 114 avec
son drafter en glouton après 16k.

## Le portage (dossier `inkling2/`)

| Fichier | Rôle |
|---|---|
| `amont/` | sources d'origine figées au commit `5050dcaf0a2de0f0ecc68030f9b44db0dd265be9` (`amont/COMMIT`), avec la licence MIT |
| `src/ink2_engine.hip` | le moteur porté |
| `src/safetensors_lite.hpp` | lecteur safetensors de Strata, inchangé |
| `build.sh` | compile `build/strata-ink2` (`ROCM=…`, `GPU_TARGET=gfx1201` par défaut) |
| `outils/telecharger-poids.mjs` | checkpoint officiel dans `poids/` (avec reprise) |
| `outils/prompt.py` | prompt de chat en ids avec le gabarit officiel, et ids vers texte (venv `.venv` : tokenizers, jinja2) |
| `outils/serveur.py` | façade OpenAI devant `strata-ink2 --serve` |
| `lancer.sh` | lance la façade : `./lancer.sh --ctx 131072 [--port 4799] [--spec]` |

Refaire de zéro :
```
cd inkling2
node outils/telecharger-poids.mjs poids                 # 19,5 Go
python3 -m venv .venv && .venv/bin/pip install tokenizers jinja2
ROCM=<racine ROCm> ./build.sh                          # ~1 min (ROCm de Harn : runtime/rocm/10.1.0)
.venv/bin/python -I outils/prompt.py encode poids "Bonjour" > build/ids.txt
build/strata-ink2 --model poids --tokens-file build/ids.txt --n 200 --ctx 8192 --chunk 512 [--spec 1]
.venv/bin/python -I outils/prompt.py decode poids <les ids de la ligne gen:>
```

Ce que le portage a changé, et pourquoi :
1. `hipify-perl` sur `ink2_engine.cu` : le gros du travail, sans retouche.
2. `nvcuda::wmma` devient rocWMMA, avec `rocwmma::hfloat16_t`, qui vaut `__half` : les fragments
   et les pointeurs `half` restent tels quels.
3. `<hipblas.h>` devient `<hipblas/hipblas.h>`.
4. Les masques de `__shfl_*_sync` et `__syncwarp` passent de `0xffffffffu` à `0xffffffffull` :
   HIP exige un masque de 64 bits. RDNA4 compile en wave32, donc les 32 bits du bas suffisent.
5. `hipGraphInstantiate(&x, g, 0)` devient `hipGraphInstantiateWithFlags(&x, g, 0)` : la
   signature de HIP est celle de l'ancien CUDA.
6. `HIP_SYMBOL(x)` devient `(const void*) &x` : avec le flux par thread (`_spt`), HIP veut
   l'adresse.
7. Attention du préremplissage (`kb_attn`) : la tuile de clés `FK` passe de 64 à 32. Avec 64, le
   noyau demande 83 Ko de mémoire partagée, alors qu'une RDNA4 en donne au plus 64 Ko par bloc.
   Avec 32, il en faut 59 Ko. Les boucles écrites en dur pour 4 colonnes de 16 suivent
   maintenant `FK / 16`. Le calcul est le même, seul l'ordre des tuiles change.

**Piège de compilation** : hipcc met le ROCm de Harn derrière `/usr/include`, où le ROCm 7.1 du
système a ses propres en-têtes `hip/`. rocWMMA prend alors le `hip_fp8.h` du système, et les
erreurs `__ocml_*` apparaissent. Un `-I` sur le dossier de Harn, même par un lien symbolique, est
écarté comme doublon. `build.sh` copie donc `include/hip` dans `build/rocm-include`.

### Mesures du portage (R9700, 21/11, checkpoint officiel, KV f16)

| | llama.cpp (mélange) | strata-ink2 HIP |
|---|---|---|
| Décodage, contexte court, sans spéculation | 33 tok/s | **68-70 tok/s** |
| Décodage avec drafter | 39,5 (DSpark n3, moyenne du banc) | 72 tok/s (`--spec 1` : 3,3 tokens par vérification, mais 283 tokens sur 400 en repli simple) |
| Décodage après un prompt de 7 243 tokens | ~43 à 35k (DSpark) | **64,8 tok/s** |
| Préremplissage, 7 243 tokens | ~830 tok/s | 510 tok/s (SGEMM en fp32 strict, attention en fp32) |
| VRAM, 32k de contexte | | 18,4 Gio |

Sorties vérifiées à l'œil : du code Python juste, et un résumé fidèle de `src/bench.mjs`, de Harn.

### Validation contre llama.cpp (21/11)

Mêmes ids de prompt, décodage glouton, 256 tokens. Référence : le mélange llama.cpp avec
UD-Q8_K_XL (experts NVFP4, attention BF16). La sortie de llama.cpp est prise sur `/completion`,
avec `temperature 0`, `top_k 1`, `samplers ["top_k"]` et `ignore_eos`.

| Prompt | Tokens identiques avant la divergence | Notre p(top-1) contre p(choix de llama.cpp), à la divergence |
|---|---|---|
| prose (24 tokens) | 44 | 0,523 contre 0,475 |
| TypeScript (37) | 101 | 0,238 contre 0,221 |
| Python (32) | 81 | 0,227 contre 0,225 |
| code long (7 243) | 22 | 0,422 contre 0,361 |

À chaque fois, le choix de llama.cpp est notre deuxième token, à quelques points près : ce sont
des quasi-égalités qu'un écart d'arrondi fait basculer. Les deux moteurs calculent la même
distribution. **Piège** : `--dump-logits` écrit des lignes de **200 058** floats (vocabulaire
utile), pas 201 024. Une lecture avec la mauvaise largeur décale tous les indices de 966. Les
logits complets ne sortent que par le chemin token par token (`--chunk 0`).

### La façade OpenAI (`outils/serveur.py`)

- `POST /v1/chat/completions`, avec ou sans flux (SSE), et `GET /v1/models`, `/health`,
  `/props`. Clé d'API facultative (`--api-key-file`).
- Prompt : le gabarit officiel avec Jinja en bac à sable. La balise `{% generation %}` de
  transformers est rendue telle quelle. `developer` devient `system`. Les arguments d'outils,
  chaînes JSON côté OpenAI, sont décodés : le gabarit veut des objets. `reasoning_effort` (ou
  `reasoning.effort`, ou `chat_template_kwargs`) prend none, minimal, low, medium, high, xhigh
  ou max. Sans effort donné, le gabarit met 0,9.
- Sortie, lue token par token sur les marqueurs spéciaux : `content_thinking` donne
  `reasoning_content`, `content_text` donne `content`, `content_invoke_tool_json` donne
  `tool_calls` (le JSON `{"name","args"}`, avec `arguments` resérialisé en chaîne). Un appel
  illisible est rendu comme texte. Un caractère UTF-8 coupé entre deux tokens est retenu.
- `timings` au format llama.cpp (`predicted_per_second`, etc.), comme le banc de Harn les lit,
  plus `usage`.
- Client parti : `STOP` est envoyé au moteur, qui s'arrête au token suivant.
- Une requête à la fois, puisque le moteur n'a qu'un fil. Le cache du prompt est gardé quand la
  conversation prolonge la précédente : dans une boucle d'agent, seul le nouveau tour est
  préchargé (200 tokens en 310 ms, mesuré).
- Pas d'images : le moteur n'a pas la vision, et la façade répond 400.

Essais du 21/11, tous réussis : une question simple (raisonnement séparé, réponse juste) ; un
appel d'outil `read_file` aux arguments propres ; le résultat de l'outil réinjecté, puis une
réponse juste ; un flux SSE (premier delta en 0,35 s, 64,2 tok/s) ; une coupure du client suivie
d'une nouvelle requête servie normalement. À noter : avec `max_tokens` petit, la réponse peut
rester vide, parce que même `none` réfléchit un peu avant de répondre.

## Plan

1. ~~Récupérer la branche à un commit figé et noter ce commit ici.~~ Fait : 5050dca.
2. ~~Porter `ink2_engine.cu` en HIP pour gfx1201~~ (fait, voir plus haut) :
   - hipify ;
   - `nvcuda::wmma` vers rocWMMA ;
   - cuBLAS vers hipBLAS ;
   - vérifier la largeur de warp et les `__shfl_*_sync` ;
   - garder le décodage capturé en un seul graphe ;
   - décompression NVFP4 et MXFP8 dans les noyaux.
3. ~~Valider~~ (fait, voir plus haut) : logits contre llama.cpp UD-Q8_K_XL (`--dump-logits`), qui
   garde les mêmes experts NVFP4 et l'attention en BF16. Puis le tokenizer, puis le décodage
   glouton identique avec et sans `--spec`.
3b. Accélérer : le préremplissage (GEMM en f16 ou bf16 avec rocWMMA, plutôt que SGEMM strict),
   la vérification du drafter (33 ms pour 8 tokens sur 3090 ; à mesurer ici) et le KV int8
   (`--kv i8r`) pour les longs contextes.
3c. ~~Serveur~~ (fait : `outils/serveur.py`). `--serve` parle le protocole de Strata. Il faut une façade OpenAI (gabarit de chat,
   tokenizer, flux SSE, outils, marqueurs de raisonnement) : la reprendre du serveur Strata, ou
   l'écrire.
4. Mesurer avec `profil-decodage` adapté, puis le banc Harn.
5. Brancher dans Harn. Le code est écrit (21/11), il reste l'installation et le banc.
   - fiche `inkling-2-mini-ink2` (`engine: 'ink2'`), avec 4 fichiers du checkpoint officiel aux
     empreintes figées ;
   - `installInk2` (src/runtimes.mjs) : archive de ce dépôt à un commit figé (`INK2.commit`),
     `build.sh` dans la bulle (ROCm de Harn en lecture, le seul dossier `inkling2/` en écriture),
     puis un venv uv avec tokenizers et jinja2 ;
   - lancement (src/loading.mjs) : `serveur.py` sur le port 4755, avec la clé du moteur de Harn
     et l'id Harn comme alias ;
   - banc : le moteur sans spéculation et avec `--spec`, le meilleur des deux à 128k.
   Pour monter de version : pousser ici, puis changer `INK2.commit` dans Harn.

## Journal

- **21/11** : banc du mélange llama.cpp à 39,5 tok/s. Graphes HIP vérifiés et actifs. Profil :
  40 % de calcul, 7 265 noyaux par token. Basalt écarté (Blackwell uniquement). Vulkan pas
  essayé (il faudrait compiler le mélange avec le SDK Vulkan). Décision : porter `strata-ink2`.
- **21/11, plus tard** : portage HIP compilé et lancé (7 corrections, détaillées plus haut).
  68-70 tok/s au départ, 64,8 tok/s après 7k de prompt. Le drafter n'apporte presque rien
  (72 tok/s).
- **21/11, soir** : validation contre llama.cpp (divergences toutes sur des quasi-égalités) ;
  façade OpenAI écrite et testée (outils, flux, coupure) ; 131k de contexte dans 21,4 Gio, à
  ~67 tok/s.
- **21/11, 17 h** : installé par Harn (fiche `inkling-2-mini-ink2`, compilé dans la bulle en
  ~10 s). Banc Harn : **64,2 tok/s** (code 63,8, texte 65,4), 57,4 à ~35k de contexte,
  préremplissage à 574 tok/s. Avec le drafter (`--spec`), 59,2 : il coûte plus qu'il ne
  rapporte, et Harn garde le moteur sans. **QI 120** (llama.cpp UD-Q4_K_XL : 115). Outils,
  raisonnement et contexte à 125, code 121, honnêteté 83 : le modèle invente encore une
  rencontre entre Jean Valjean et d'Artagnan.

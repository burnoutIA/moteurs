# Moteurs sur mesure : la recette

Portage et attribution : `inkling2/src/ink2_engine.hip` est le moteur `strata-ink2` de Strata (MIT, Niko1221 et les contributeurs de Strata), porté en HIP ; l'original est dans `inkling2/amont/`. Voir `LICENSE`.

Un moteur sur mesure est un programme écrit pour **un seul modèle sur une seule famille de
cartes**. On le fait seulement quand les moteurs généraux (llama.cpp, vLLM, Strata) laissent
une grosse part de la vitesse sur la table et qu'aucune correction simple ne la rend. Chaque
moteur est une dette, puisqu'il faut le maintenir à chaque version du modèle. On n'en écrit donc
que pour un modèle utilisé tous les jours.

Ce dépôt est à part de Harn : Harn installe des moteurs aboutis, la recherche se fait ici. Une fiche par modèle, dans ce dossier, garde tout ce qu'il faut pour refaire le moteur : les faits
d'architecture, les mesures, les choix, les pièges. Elle se met à jour à chaque étape. **Un moteur
sans fiche à jour est un moteur perdu.**

| Modèle | Fiche | État |
|---|---|---|
| Inkling 2 Mini | [inkling-2-mini.md](inkling-2-mini.md) | validé, API OpenAI : ~67 tok/s contre 33 ; branchement dans Harn à faire |

## 1. Décider : le moteur général est-il vraiment le goulot ?

1. Faire le banc de Harn avec le meilleur moteur général disponible : release, PR pas encore
   fusionnées, mélange (`requiredPr` en liste). On obtient ainsi un plancher honnête.
2. Calculer le plafond : octets lus par token (poids actifs, plus le cache KV), divisés par la
   bande passante de la carte. Pour un MoE, ce sont les experts actifs, pas tout le modèle. Si le
   banc atteint déjà 60 % du plafond, aucun moteur ne fera beaucoup mieux : on s'arrête là.
3. Profiler le décodage :
   ```
   outils/profil-decodage.sh <dossier bin du moteur> <modèle.gguf> 200
   ```
   Le script donne le nombre de noyaux par token, la part du temps où la carte calcule et les
   noyaux qui coûtent le plus.
   - **La carte calcule moins de ~60 % du temps** : le goulot est l'enchaînement des noyaux.
     Sur AMD, il faut compter environ 3,4 µs entre deux noyaux, même dans un graphe HIP.
     Seule la fusion d'opérations lève ce goulot, et c'est le cas d'un moteur sur mesure.
   - **Un noyau domine** : le corriger dans le moteur général, par une PR ou un patch dans un
     mélange, coûte moins cher qu'un moteur.
4. Vérifier que les graphes HIP sont bien utilisés : lancer avec `GGML_CUDA_GRAPH_DEBUG=1 -v` et
   compter les « CUDA Graph id N reused ». S'ils sont désactivés, la correction est là, pas dans
   un moteur.

## 2. Partir de l'existant

Chercher d'abord un moteur sur mesure déjà écrit, même pour une autre carte : une branche Strata,
un noyau de vLLM, le code de référence de transformers. Le porter coûte bien moins cher que
l'écrire, et il arrive avec sa validation. Vérifier la licence (MIT, Apache 2.0) et garder
l'attribution dans les sources.

Porter de CUDA vers HIP (RDNA3/RDNA4, wave32) :
- `hipify-perl` traduit l'essentiel de l'API (flux, mémoire, graphes, événements) ;
- `nvcuda::wmma` devient rocWMMA (fragments 16×16×16, RDNA4 géré), ou les intrinsèques
  `__builtin_amdgcn_wmma_*` ;
- cuBLAS devient hipBLAS (ou hipBLASLt) ;
- les formats propres à NVIDIA (NVFP4, MXFP8) n'ont pas de calcul matériel sur RDNA : il faut
  les décompresser dans le noyau, vers f16 ou int8, au moment du calcul ;
- `__shfl_*_sync` avec un masque devient `__shfl_*` : la largeur de warp est 32 sur RDNA en
  wave32, il faut la vérifier partout où le code suppose 32.

## 3. Valider avant de mesurer

Un moteur rapide et faux ne sert à rien. Dans l'ordre :
1. logits contre la référence (transformers en fp32, ou le moteur général) sur un texte de plus de
   1 000 tokens : KL moyenne et top-1 identique à chaque position ;
2. tokenizer contre celui de Hugging Face, sur un millier de chaînes ;
3. décodage glouton identique au décodage sans spéculation ;
4. le test d'intelligence de Harn, à comparer avec le profil du moteur général.

## 4. Brancher dans Harn

Le moteur parle l'API OpenAI (`/v1/chat/completions`, flux SSE, outils). Harn le compile dans la
bulle après accord de l'utilisateur, comme les forks, et le banc le compare au moteur général.
Le plus rapide **à qualité égale** gagne. Le moteur général reste installé en secours.

## 5. Ce que la fiche d'un modèle doit contenir

- les sources : dépôt, commit, licence, ce qui a été repris ;
- l'architecture exacte (le `config.json`, recopié) et ce qui est inhabituel ;
- le plancher (le banc du moteur général), le plafond calculé, le profil ;
- les choix faits et pourquoi (fusion, formats, mémoire) ;
- comment compiler, lancer et valider, avec les commandes ;
- les mesures après chaque étape, et les pièges rencontrés.

# FaceCam — face swap temps réel pour macOS

Application macOS (Apple Silicon) qui remplace en direct le visage de la webcam par celui d'une photo,
et renvoie le résultat dans une **caméra virtuelle** utilisable dans Zoom, Meet, Teams, OBS…

## Modèles utilisés (tous ONNX, exécutés via CoreML)

| Rôle | Modèle | Source |
|---|---|---|
| Détection + points clés | `buffalo_l` (SCRFD 10G + ArcFace r50) | insightface |
| Swap | `inswapper_128_fp16.onnx` (fp32 en secours) | facefusion-assets |
| Amélioration (optionnelle) | `gfpgan_1.4.onnx` | facefusion-assets |

Ils sont dans `models/` (≈1,4 Go, hors git). Pour les retélécharger : voir `scripts/download_models.sh`.

## Lancer en développement

```bash
./run.sh
```

1. **Choisir une photo…** : visage source (le plus grand visage de l'image est utilisé).
2. **Démarrer** : ouvre la webcam. macOS demande l'autorisation caméra la première fois.
3. **Démarrer la caméra virtuelle** : publie le flux dans « OBS Virtual Camera ».
   Puis sélectionne cette caméra dans Zoom/Meet/Teams.

Prérequis caméra virtuelle (une seule fois) : installer OBS (`brew install --cask obs`), le lancer,
cliquer **Démarrer la caméra virtuelle** et autoriser l'extension dans
*Réglages Système › Général › Ouverture de session et extensions › Extensions de caméra*.

## Performances mesurées (M4, 1280x720, un visage)

| Étape | Latence |
|---|---|
| Détection (SCRFD) | ~5 ms |
| inswapper fp16 (CoreML MLProgram) | ~49 ms |
| Collage ROI | ~15 ms |
| **Total swap** | **~74 ms → ~13 fps** |
| + GFPGAN | +~270 ms → ~3 fps |

L'amélioration GFPGAN est donc à réserver aux cas où la netteté prime sur la fluidité.
Le menu **Résolution** (1080p à 640x480) agit sur la capture ; le coût du swap dépend peu de la résolution.

Pipeline : la détection de l'image N+1 tourne pendant le swap de l'image N (+19 % mesuré, 17 → 20 fps à vide),
ce qui garde aussi le Neural Engine occupé : sous-cadencé dès qu'il attend, il passe de 49 ms à 65–76 ms par swap.
Le menu **Calcul du swap** est sur « Neural Engine seul » par défaut : dans l'app, caméra ouverte, c'est 89 ms stable (11 fps),
contre 63 à 238 ms en « Auto » (Neural Engine + GPU) parce que la part GPU subit la concurrence de la capture et de l'affichage.
À vide, « Auto » est plus rapide (48 ms), d'où des benchmarks hors app trompeurs.

Variabilité résiduelle : caméra ouverte, les parties GPU du modèle (mode Auto) ralentissent et oscillent ; le secteur n'y change rien.

Charge CPU : ~0,2 cœur en swap (le calcul est sur le Neural Engine). Deux pièges évités dans le code :
le produit matriciel du vecteur d'identité est précalculé (sinon OpenBLAS occupe 8 cœurs en attente active),
et les threads ONNX Runtime ne tournent pas en attente active. Journal : `~/Library/Logs/FaceCam.log`.

## Empaqueter en .app

```bash
./scripts/build_app.sh
```

Produit `dist/FaceCam.app` (modèles inclus). L'app contient `NSCameraUsageDescription`,
nécessaire pour que macOS affiche la demande d'autorisation caméra.

## Structure

- `app/engine.py` — chargement des modèles, détection, swap, collage ROI, GFPGAN
- `app/camera.py` — capture AVFoundation dans un thread
- `app/vcam.py` — sortie caméra virtuelle (pyvirtualcam / extension OBS)
- `app/main.py` — interface PySide6
- `scripts/test_swap.py` — test hors caméra : `python scripts/test_swap.py source.jpg cible.jpg sortie.jpg`

## Éthique

N'utilise que des visages pour lesquels tu as le consentement, et préviens tes interlocuteurs.
L'usurpation d'identité est illégale.

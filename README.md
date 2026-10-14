# FaceCam — face swap temps réel pour macOS

Application macOS (Apple Silicon) qui remplace en direct le visage de la webcam par celui d'une photo,
et renvoie le résultat dans une **caméra virtuelle** utilisable dans Zoom, Meet, Teams, OBS…

## Modèles utilisés (tous ONNX, exécutés via CoreML)

| Rôle | Modèle | Source |
|---|---|---|
| Détection + points clés | `buffalo_l` (SCRFD 10G + ArcFace r50) | insightface |
| Swap | `inswapper_128_int8.mlpackage` (CoreML natif, généré localement) ; repli `inswapper_128_fp16.onnx` | facefusion-assets + `scripts/convert_inswapper_coreml.py` |
| Amélioration (optionnelle) | `gfpgan_1.4.onnx` | facefusion-assets |

Ils sont dans `models/` (≈1,4 Go, hors git). Pour les retélécharger : voir `scripts/download_models.sh`.

### Modèle de swap CoreML natif (recommandé)

Le swap est 1,6 à 2,5 fois plus rapide avec une version CoreML native d'inswapper aux poids int8 :

```bash
.venv/bin/pip install torch onnx2torch coremltools
.venv/bin/python scripts/convert_inswapper_coreml.py      # ONNX -> PyTorch -> CoreML fp16 (+ fusion instance_norm)
.venv/bin/python scripts/quantize_inswapper_coreml.py     # -> inswapper_128_int8.mlpackage (Qualité) et inswapper_128_p6.mlpackage (Rapide)
```

Le menu **Modèle de swap** choisit entre *Qualité* (poids int8, 48 ms, écart 0,003 vs fp16) et *Rapide*
(palette 6 bits, 39 ms, écart 0,009, invisible à l'œil). Rechargement à chaud, préférence mémorisée.

Pourquoi ça marche : le réseau pèse 349 GFLOP mais surtout ~250 Mo de poids par inférence ; sur le Neural Engine il est
limité par la bande passante mémoire, et les poids int8 la divisent par deux (120 ms fp16 natif → 48 ms, écart moyen 0,003).
Le moteur utilise ce modèle s'il est présent, sinon ONNX Runtime (`FACECAM_NO_COREML=1` pour forcer le repli).
`scripts/build_app.sh` précompile les paquets en `.mlmodelc` (`models/compiled/`) : l'app n'embarque que ces versions
compilées, chargées en 2 à 3 s, car charger un `.mlpackage` depuis le bundle prenait plus d'une minute.

## Modes

- **Remplacer le visage (swap)** : le visage de la photo source remplace le tien.
- **Flouter / Pixelliser / Masquer** : anonymisation sans photo source, tous les visages détectés, ellipse à bords doux,
  curseur d'intensité. Seule la détection tourne : ~30 fps (plus de 60 fps à vide).

## Lancer en développement

```bash
./run.sh
```

1. **Choisir une photo…** : visage source (le plus grand visage de l'image est utilisé).
2. **Démarrer** : ouvre la webcam. macOS demande l'autorisation caméra la première fois.
3. **Démarrer la caméra virtuelle** : publie le flux dans « OBS Virtual Camera ».
   Puis sélectionne cette caméra dans Zoom/Meet/Teams.

### Caméra virtuelle « FaceCam Camera » (sans OBS)

Extension caméra CoreMediaIO dédiée, dans `FaceCamCamera/` : une app hôte native minimale (« FaceCam Camera »)
embarque l'extension et l'installe ; FaceCam pousse ensuite ses images dans le flux d'entrée de la caméra via
`libfacecam_cmio.dylib` (client CoreMediaIO en Swift, appelé par ctypes).

```bash
./FaceCamCamera/build_and_install.sh     # nécessite un compte Apple connecté dans Xcode (équipe 57LHMQV3Y5)
open "/Applications/FaceCam Camera.app"   # « Installer l'extension », puis autoriser dans Réglages Système
```

Réglages Système › Général › Ouverture de session et extensions › Extensions de caméra. La caméra « FaceCam Camera »
(1280x720, 30 fps) apparaît alors dans Zoom, Meet, FaceTime… ; sans flux de FaceCam, elle diffuse une mire sombre.

## Performances mesurées (M4, 1280x720, un visage)

| Étape | Latence |
|---|---|
| Détection (SCRFD) | ~5 ms |
| inswapper CoreML natif, Neural Engine seul | 48 ms (int8) / 39 ms (palette 6 bits) ; ONNX Runtime : 79 ms |
| Collage ROI | ~15 ms |
| **Total swap** | **~63 ms → 15,5 fps (Qualité), 19 fps (Rapide), à vide** |
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

Produit `dist/FaceCam.app` (modèles inclus, macOS 15 minimum à cause des modèles CoreML palettisés).
L'app contient `NSCameraUsageDescription`, nécessaire pour que macOS affiche la demande d'autorisation caméra.
La signature est ad hoc : son empreinte change à chaque build et macOS refuse alors la caméra sans rien demander.
Le script lance `tccutil reset Camera fr.bruno.facecam` après la signature pour que l'autorisation soit redemandée.
Pour une autorisation durable, signer avec un certificat stable (Developer ID ou certificat auto-signé).

## Structure

- `app/engine.py` — chargement des modèles, détection, swap, collage ROI, GFPGAN
- `app/camera.py` — capture AVFoundation dans un thread ; caméras identifiées par leur uniqueID AVFoundation,
  l'index OpenCV étant recalculé à l'ouverture (l'ordre de la liste change avec la caméra par défaut du système)
- `app/vcam.py` + `app/facecam_cmio.py` — sortie vers la caméra virtuelle FaceCam Camera (client CoreMediaIO)
- `FaceCamCamera/` — extension caméra CoreMediaIO (Swift) + app hôte, projet XcodeGen
- `app/main.py` — interface PySide6
- `scripts/test_swap.py` — test hors caméra : `python scripts/test_swap.py source.jpg cible.jpg sortie.jpg`

## Éthique

N'utilise que des visages pour lesquels tu as le consentement, et préviens tes interlocuteurs.
L'usurpation d'identité est illégale.

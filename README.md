Détection des engins— YOLOv8 
---

Projet de détection d’objets pour Ambulance, Bus, Car, Motorcycle, Truck à partir du dataset Kaggle abdallahwagih/cars-detection.
Entraînement avec Ultralytics YOLOv8, prédiction sur images et vidéos avec pourcentage de confiance, résumé par classe et export CSV.



Fonctionnalités
---

Entraînement finetune YOLOv8 sur 5 classes.

Prédictions sur images & vidéos.

Boîtes + confiance (%) sur les sorties.

Panneau récap par classe (compte + confiance moyenne).

Export detections.csv pour la vidéo.

Sorties rangées dans outputs/images/ et outputs/videos/.



Arborescence
---

Projet_DS_vs_code/
├─ detection_engins/
│  ├─ train.py                      # entraînement
│  ├─ charger_dataset_kaggle.py     # download Kaggle (+ affiche chemin)
│  ├─ chemin_data_yaml.py           # génère data_cars_local.yaml (chemins absolus)
│  ├─ predict_image_stat_v2.py      # prédiction image + % + résumé
│  ├─ detect_video_stats.py         # prédiction vidéo + % + CSV (codec fallback)
│  ├─ data_cars_local.yaml          # YAML dataset (local, sans 'path')
│  ├─ runs/                         # dossiers Ultralytics (train/val/predict)
│  └─ outputs/
│     ├─ images/                    # images annotées
│     └─ videos/                    # vidéos annotées + CSV
└─ ...

Prérequis
---

Windows + VS Code (extension Python)

Python 3.10+ (testé en 3.13)

(Optionnel) CUDA si GPU NVIDIA

(Optionnel) ffmpeg / VLC pour la lecture/convertibilité des vidéos


Installation rapide
---

Ouvrir un terminal PowerShell à la racine du projet :

1) Créer & activer le venv
python -m venv detection_engins\.venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\detection_engins\.venv\Scripts\Activate.ps1

2) Dépendances
python -m pip install --upgrade pip setuptools wheel
pip install ultralytics opencv-python pyyaml tqdm matplotlib kagglehub

3) PyTorch (choose one)
CPU :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
GPU (CUDA 12.1) :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


Dans VS Code : Ctrl+Shift+P → Python: Select Interpreter → choisissez ...\detection_engins\.venv\Scripts\python.exe.



Télécharger & préparer le dataset
---

cd detection_engins
python .\charger_dataset_kaggle.py
python .\chemin_data_yaml.py   # génère data_cars_local.yaml avec chemins ABSOLUS


Exemple de YAML généré :

train: C:/Users/.../.cache/kagglehub/datasets/abdallahwagih/cars-detection/versions/1/Cars Detection/train/images
val:   C:/Users/.../.cache/kagglehub/datasets/abdallahwagih/cars-detection/versions/1/Cars Detection/valid/images
test:  C:/Users/.../.cache/kagglehub/datasets/abdallahwagih/cars-detection/versions/1/Cars Detection/test/images
nc: 5
names: ['Ambulance','Bus','Car','Motorcycle','Truck']


NB : Sous Windows, éviter path: relatif avec espaces (“Cars Detection”) → utiliser des chemins absolus comme ci-dessus.


Entraînement
---

Script (recommandé)
python .\train.py

Ou en CLI
yolo train model=yolov8n.pt data="data_cars_local.yaml" epochs=100 imgsz=640 batch=16 workers=0 project=runs name=cars_all2


Poids : runs/cars_all2/weights/best.pt

Éval & courbes :
yolo val model="runs/cars_all2/weights/best.pt" data="data_cars_local.yaml" plots=True

Résultats observés (réf. CPU, 100 epochs, yolov8n.pt) : mAP50≈0.62 global (Ambulance0.90, Bus0.67, Car0.45, Motorcycle0.64, Truck~0.42).


Prédictions — Images
---

Dossier de sortie
New-Item -ItemType Directory -Force .\outputs\images | Out-Null

CLI
yolo predict `
  model="runs/cars_all2/weights/best.pt" `
  source="C:/CHEMIN/vers/une_image_ou_un_dossier" `
  conf=0.25 save=True save_conf=True `
  project="./outputs/images" name="test_$(Get-Date -Format yyyyMMdd_HHmmss)" exist_ok=True

Script avec % + résumé
python .\predict_image_stat_v2.py
écrit outputs/images/<nom>_annot_pct.jpg

Prédictions — Vidéos / Webcam
Dossier de sortie
New-Item -ItemType Directory -Force .\outputs\videos | Out-Null

Script avec % + panneau + CSV (codec fallback)
python .\detect_video_stats.py
écrit:
- outputs/videos/video_annot_percent.(mp4|avi)
- outputs/videos/detections.csv


VS Code n’aperçoit pas toujours les MP4. Ouvrir avec Windows :
start "" ".\outputs\videos\video_annot_percent.mp4"


Paramètres utiles (inférence)
---

conf : seuil de confiance (baisser à 0.10–0.15 pour petits objets).

iou : NMS (monter à 0.60–0.70 pour scènes denses).

imgsz : 960–1280 pour mieux capter les véhicules lointains (plus lent).

classes : filtrer ex. classes=1,4 (Bus & Truck).

vid_stride : sous-échantillonnage vidéo.


Améliorer la précision
---

Réentraîner avec yolov8s.pt + imgsz=768/960.

Rééquilibrer le dataset (plus de Bus/Truck en trafic dense & lointain).

Vérifier les labels (boîtes serrées, pas d’oubli).

Inspecter confusion matrix (yolo val ... plots=True).


Dépannage
---

Activate.ps1 bloqué →
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass puis & .\.venv\Scripts\Activate.ps1

yolo non reconnu →
pip install -U ultralytics ou utilisez .\.venv\Scripts\yolo.exe ...

best.pt introuvable →
Get-ChildItem -Recurse -Filter best.pt puis mettez le chemin exact.

Erreur dataset “images not found … Cars Detection” →
utiliser data_cars_local.yaml avec chemins absolus (pas de path: relatif).

Vidéo illisible →
ouvrir avec le lecteur système, ou convertir :
ffmpeg -y -i video_annot_percent.avi -c:v libx264 -pix_fmt yuv420p video_annot_percent_h264.mp4


Licence & crédits
---

Dataset : Kaggle abdallahwagih/cars-detection.

Code : basé sur Ultralytics YOLOv8.


Checklist “ça marche”
---

- venv activé ((.venv) visible dans le terminal)

-  data_cars_local.yaml pointe vers les bons chemins absolus

-  Entraînement OK → runs/.../weights/best.pt présent

- Images annotées dans outputs/images/…

- Vidéo annotée + detections.csv dans outputs/videos/…
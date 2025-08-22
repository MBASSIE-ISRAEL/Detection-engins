from ultralytics import YOLO
from pathlib import Path

# 'emplacement le plus probable
cand1 = Path(r"C:\Users\israe\Documents\Projet_DS_vs_code\detection_engins\runs\cars_all2\weights\best.pt")
cand2 = Path(r"C:\Users\israe\Documents\Projet_DS_vs_code\runs\cars_all2\weights\best.pt")

model_path = cand1 if cand1.exists() else cand2
if not model_path.exists():
    # fallback: chercher automatiquement
    hits = list(Path(r"C:\Users\israe\Documents\Projet_DS_vs_code").rglob("runs/*/weights/best.pt"))
    if not hits:
        raise FileNotFoundError("best.pt introuvable. Vérifie le dossier 'runs/'.")
    model_path = hits[0]

m = YOLO(str(model_path))
print("Modèle chargé :", model_path)
print("Classes :", m.names)  # ['Ambulance','Bus','Car','Motorcycle','Truck']

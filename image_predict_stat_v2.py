# predict_image_with_stats_v2.py
from ultralytics import YOLO
import cv2, os, numpy as np
from collections import Counter
from pathlib import Path

MODEL = r"C:\Users\israe\Documents\Projet_DS_vs_code\detection_engins\runs\cars_all2\weights\best.pt"  
IMAGE = r"C:\Users\israe\Documents\Projet_DS_vs_code\detection_engins\image_test_4.jpg"   

OUT_DIR = Path(r"C:/Users/israe/Documents/Projet_DS_vs_code/detection_engins/outputs/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / (Path(IMAGE).stem + "_annot_pct.jpg")

CONF = 0.20   # 0.15 pour capter les petites voitures lointaines
IOU  = 0.45 # coexistance des boites proches
IMGSZ = 1356 # plus de détails sur les voitures lointaines

CLASSES = [0, 1, 2, 3, 4] # classes des engins
model = YOLO(MODEL)
res = model.predict(source=IMAGE, conf=CONF, iou=IOU, imgsz=IMGSZ, classes=CLASSES , verbose=False, augment=True)[0]
img = cv2.imread(IMAGE); h, w = img.shape[:2]
names = model.names

boxes = res.boxes
if boxes is None or boxes.cls is None or len(boxes) == 0:
    print("Aucune détection."); exit()

cls_ids = boxes.cls.cpu().int().tolist()
confs   = boxes.conf.cpu().tolist()
xyxy    = boxes.xyxy.cpu().numpy().astype(int)

#  Dessin des boîtes avec pourcentage
for (x1,y1,x2,y2), cid, conf in zip(xyxy, cls_ids, confs):
    label = f"{names[cid]} {conf*100:.1f}%"
    color = (0,255,0)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, y0), (x1 + tw + 6, y0 + th + 6), color, -1)
    cv2.putText(img, label, (x1 + 3, y0 + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

#  Stats par classe 
counts = Counter(cls_ids)
total  = sum(counts.values())
lines  = []
for cid in sorted(counts):
    part = 100 * counts[cid] / total
    avgc = 100 * np.mean([c for i,c in enumerate(confs) if cls_ids[i]==cid])
    lines.append(f"{names[cid]}: {counts[cid]} ({part:.1f}%)  conf moy {avgc:.1f}%")

# Dessiner un panneau récap en haut-gauche
pad, lh = 10, 28
panel_w = int(max([cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for t in lines]) + 2*pad)
panel_h = int(lh*len(lines) + 2*pad)
overlay = img.copy()
cv2.rectangle(overlay, (10,10), (10+panel_w, 10+panel_h), (0,0,0), -1)
img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
for i, t in enumerate(lines):
    cv2.putText(img, t, (10+pad, 10+pad + (i+1)*lh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

cv2.imwrite(OUT, img)
print(f"Image annotée: {OUT}")
print("Résumé :")
for t in lines: print("-", t)

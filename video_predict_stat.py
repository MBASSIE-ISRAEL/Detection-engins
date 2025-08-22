from ultralytics import YOLO
import cv2, os, csv, numpy as np
from collections import Counter
from pathlib import Path


MODEL = r"C:/Users/israe/Documents/Projet_DS_vs_code/detection_engins/runs/cars_all2/weights/best.pt" # chemin vers notre model
SOURCE = r"C:\Users\israe\Documents\Projet_DS_vs_code\detection_engins\video_test_4.mp4"   # chemin vers la vidéo

OUT_DIR = Path(r"C:/Users/israe/Documents/Projet_DS_vs_code/detection_engins/outputs/videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MP4 = OUT_DIR / "video_annot_percent.mp4"
OUT_CSV = OUT_DIR / "detections.csv"

CONF = 0.20
IOU = 0.45
IMGZ = 960

model = YOLO(MODEL)
names = model.names

cap = cv2.VideoCapture(0 if str(SOURCE)=="0" else SOURCE)
assert cap.isOpened(), f"Impossible d'ouvrir {SOURCE}"
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w, h = int(cap.get(3)), int(cap.get(4))
writer = cv2.VideoWriter(OUT_MP4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# CSV: frame, class, conf, x1,y1,x2,y2
csv_f = open(OUT_CSV, "w", newline="", encoding="utf-8")
wrt = csv.writer(csv_f); wrt.writerow(["frame","class","conf","x1","y1","x2","y2"])

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = model.predict(source=frame, conf=CONF, iou=IOU, imgsz=IMGZ, verbose=False)[0]

    annotated = frame.copy()
    counts = Counter()
    confs_by_class = {}

    if res.boxes is not None and len(res.boxes) > 0:
        cls_ids = res.boxes.cls.cpu().int().tolist()
        confs = res.boxes.conf.cpu().tolist()
        xyxy  = res.boxes.xyxy.cpu().numpy().astype(int)

        for (x1,y1,x2,y2), cid, conf in zip(xyxy, cls_ids, confs):
            label = f"{names[cid]} {conf*100:.1f}%"
            color = (0,255,0)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
            (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y0 = max(0, y1 - th - 6)
            cv2.rectangle(annotated,(x1,y0),(x1+tw+6,y0+th+6),color,-1)
            cv2.putText(annotated,label,(x1+3,y0+th+1),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)

            counts[cid] += 1
            confs_by_class.setdefault(cid, []).append(conf*100)
            wrt.writerow([frame_idx, names[cid], f"{conf*100:.2f}", x1,y1,x2,y2])

    # panneau récap par classe
    if counts:
        total = sum(counts.values())
        lines = []
        for cid in sorted(counts):
            part = 100*counts[cid]/total
            avgc = np.mean(confs_by_class[cid])
            lines.append(f"{names[cid]}: {counts[cid]} ({part:.1f}%), conf moy {avgc:.1f}%")

        pad, lh = 10, 28
        panel_w = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for t in lines) + 2*pad)
        panel_h = int(lh*len(lines) + 2*pad)
        overlay = annotated.copy()
        cv2.rectangle(overlay,(10,10),(10+panel_w,10+panel_h),(0,0,0),-1)
        annotated = cv2.addWeighted(overlay,0.35,annotated,0.65,0)
        for i,t in enumerate(lines):
            cv2.putText(annotated,t,(10+pad,10+pad+(i+1)*lh-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv2.LINE_AA)

    writer.write(annotated)
    frame_idx += 1

cap.release(); writer.release(); csv_f.close()
print(f" Vidéo annotée: {os.path.abspath(OUT_MP4)}")
print(f" CSV des détections: {os.path.abspath(OUT_CSV)}")

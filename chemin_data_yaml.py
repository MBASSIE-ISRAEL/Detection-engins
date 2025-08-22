import os, yaml
root = r"C:/Users/israe/.cache/kagglehub/datasets/abdallahwagih/cars-detection/versions/1/Cars Detection"
cfg = {
  # on met des chemins ABSOLUS
  "train": f"{root}/train/images",
  "val":   f"{root}/valid/images",
  "test":  f"{root}/test/images",
  "nc": 5,
  "names": ['Ambulance','Bus','Car','Motorcycle','Truck']
}
with open("data_cars_local.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print("Écrit :", os.path.abspath("data_cars_local.yaml"))

# Ici on charge notre dataset depuis Kaggle

import kagglehub, glob, os
path = kagglehub.dataset_download("abdallahwagih/cars-detection")
print("Path:", path)

cands = glob.glob(os.path.join(path, "**", "data.yaml"), recursive=True)
print("Trouvé data.yaml:", cands)

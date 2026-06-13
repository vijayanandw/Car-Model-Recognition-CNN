from scipy.io import loadmat
import os
import shutil

meta = loadmat("car_devkit/devkit/cars_meta.mat")
annos = loadmat("car_devkit/devkit/cars_train_annos.mat")

class_names = meta["class_names"][0]
annotations = annos["annotations"][0]

output_dir = "dataset"

os.makedirs(output_dir, exist_ok=True)

for item in annotations:
    image_name = item[5][0]
    class_id = int(item[4][0][0])

    class_name = class_names[class_id - 1][0]

    class_folder = os.path.join(output_dir, class_name)
    os.makedirs(class_folder, exist_ok=True)

    src = os.path.join("cars_train", "cars_train", image_name)

    if os.path.exists(src):
        shutil.copy(src, class_folder)

print("Dataset Prepared Successfully!")
from scipy.io import loadmat

meta = loadmat("car_devkit/devkit/cars_meta.mat")

class_names = meta['class_names']

print("Total Classes:", len(class_names[0]))

for i in range(10):
    print(i+1, ":", class_names[0][i][0])
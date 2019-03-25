import autokeras as ak
from autokeras.image.image_supervised import load_image_dataset

x_train, y_train = load_image_dataset(csv_file_path="data/train.1.csv",
                                      images_path="data/train")

# print(x_train.shape)
# print(y_train.shape)
x_train = x_train.reshape(x_train.shape + (1,))
# y_train = y_train.reshape(y_train.shape + (1,))

clf = ak.ImageClassifier(verbose=True, augment=False)

clf.fit(x_train, y_train, time_limit=12 * 60 * 60)

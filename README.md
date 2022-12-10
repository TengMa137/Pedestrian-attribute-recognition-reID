# Pedestrian-attribute-recognition-reID

This is a course project, the aim of which is to solve task 1: Pedestrian attribute recognition and task 2: re-identification using deep learning methods. Market-1501 dataset is used in this project.

# Run the model
Go into directory and run 'main.py': ~/DLproject$ python3 main.py.

For task 1, function validate() provides metrics in validation, test() generates 'classification_test.csv'.

For task 2, function reid(gallerypath="./dataset/train_reid", querypath="./dataset/val_reid") for evaluating reID in training set,
function reid(gallerypath="./dataset/test", querypath="./dataset/queries", test = 1) generates 'reid_test.txt'

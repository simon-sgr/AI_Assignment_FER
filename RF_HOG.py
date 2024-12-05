import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

rf_classifier_ck = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced',
                                          criterion='gini', min_impurity_decrease=0.01, bootstrap=False,
                                          max_features='log2', min_samples_leaf=4, min_samples_split=2)

rf_classifier_jaffe = RandomForestClassifier(n_estimators=500, random_state=42, max_features='log2', max_depth=10,
                                             min_samples_split=10, min_samples_leaf=4, n_jobs=-1, bootstrap=False,
                                             class_weight=None, criterion='entropy', min_impurity_decrease=0.01)


def detect_face(file_path):
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=10, minSize=(40, 40))

    cropped_face = None
    for (x, y, w, h) in faces:
        if cropped_face is None:
            cropped_face = img[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 256, 0), 2)

    return img, faces, cropped_face


def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys"):
    hog_features = []
    for img in images:
        img_resized = cv2.resize(img, (64, 64))
        features = hog(
            img_resized,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            visualize=False,
        )
        hog_features.append(features)
    return np.array(hog_features)


def load_and_detect_faces(folder_path):
    images = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
                    for (x, y, w, h) in faces:
                        face_region = img[y:y + h, x:x + w]
                        images.append(face_region)
                        labels.append(label)
    return images, labels


def train_and_evaluate_model(train_folder, test_folder, model, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), block_norm="L2-Hys"):
    x_train, y_train = load_and_detect_faces(train_folder)
    x_test, y_test = load_and_detect_faces(test_folder)

    x_train_features = extract_hog_features(x_train, orientations, pixels_per_cell, cells_per_block, block_norm)
    x_test_features = extract_hog_features(x_test, orientations, pixels_per_cell, cells_per_block, block_norm)

    # tree depth
    # n_trees = [10, 50, 100, 200, 500]
    # training_loss = []
    # testing_loss = []
    # 
    # for n in n_trees:
    #     model.set_params(n_estimators=n)
    #     model.fit(x_train_features, y_train)
    #
    #     y_train_pred = model.predict(x_train_features)
    #     y_test_pred = model.predict(x_test_features)
    #
    #     train_loss = 1 - accuracy_score(y_train, y_train_pred)
    #     test_loss = 1 - accuracy_score(y_test, y_test_pred)
    #
    #     training_loss.append(train_loss)
    #     testing_loss.append(test_loss)
    #
    # # Plot loss graph
    # plt.figure(figsize=(10, 6))
    # plt.plot(n_trees, training_loss, label='Training Loss', marker='o', color='blue')
    # plt.plot(n_trees, testing_loss, label='Testing Loss', marker='x', color='orange')
    # plt.xlabel('Number of Trees')
    # plt.ylabel('Loss')
    # plt.title('Training and Testing Loss vs Number of Trees')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    model.fit(x_train_features, y_train)

    y_pred = model.predict(x_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_labels = sorted(list(set(y_train)))
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)

    # RandomSearchCV
    # from sklearn.model_selection import GridSearchCV
    #
    # param_grid = {
    #     'n_estimators': [100, 200, 500, 1000],
    #     'max_depth': [10, 20, 30, 50, None],
    #     'min_samples_split': [2, 5, 10, 20],
    #     'min_samples_leaf': [1, 2, 4, 10],
    #     'max_features': ['sqrt', 'log2', None],
    #     'bootstrap': [True, False],
    #     'class_weight': [None, 'balanced'],
    #     'criterion': ['gini', 'entropy', 'log_loss'],
    #     'min_impurity_decrease': [0.0, 0.01, 0.1],
    # }
    #
    # # Perform Grid Search
    # grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(x_train_features, y_train)
    # print("Best parameters:", grid_search.best_params_)

    return accuracy, conf_matrix_df


def preprocess_and_extract_features(image_path, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), block_norm="L2-Hys"):
    # @Simon Schöggler: use cropped face for better results
    _, _, cropped_face = detect_face(image_path)
    if cropped_face is None:
        raise ValueError("No face detected in the image.")

    gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY) if len(cropped_face.shape) == 3 else cropped_face
    face_resized = cv2.resize(gray_face, (64, 64))

    features = hog(
        face_resized,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=False,
    )

    return features.reshape(1, -1)


def predict_emotion(image_path, model, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm="L2-Hys"):
    features = preprocess_and_extract_features(image_path, orientations, pixels_per_cell, cells_per_block, block_norm)
    return model.predict(features)[0]


def select_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")],
    )

    if file_path:
        try:
            detected_img, faces, cropped_face = detect_face(file_path)
            gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY) if len(cropped_face.shape) == 3 else cropped_face
            face_resized = cv2.resize(gray_face, (64, 64))
            _, hog_image = hog(
                face_resized,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L1",
                visualize=True,
            )

            input_img = cv2.imread(file_path)
            input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
            cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            hog_image_rgb = cv2.cvtColor(np.uint8(hog_image * 255 / np.max(hog_image)),
                                         cv2.COLOR_GRAY2RGB)

            input_img_pil = Image.fromarray(input_img_rgb).resize((300, 300))
            detected_img_pil = Image.fromarray(detected_img_rgb).resize((300, 300))
            cropped_face_pil = Image.fromarray(cropped_face_rgb).resize((300, 300))
            hog_img_pil = Image.fromarray(hog_image_rgb).resize((300, 300))

            input_img_tk = ImageTk.PhotoImage(input_img_pil)
            detected_img_tk = ImageTk.PhotoImage(detected_img_pil)
            cropped_face_tk = ImageTk.PhotoImage(cropped_face_pil)
            hog_img_tk = ImageTk.PhotoImage(hog_img_pil)

            input_image_label.config(image=input_img_tk)
            input_image_label.image = input_img_tk

            detected_image_label.config(image=detected_img_tk)
            detected_image_label.image = detected_img_tk

            cropped_face_label.config(image=cropped_face_tk)
            cropped_face_label.image = cropped_face_tk

            hog_image_label.config(image=hog_img_tk)
            hog_image_label.image = hog_img_tk

            if len(faces) > 0:
                result_label.config(text=f"Detected {len(faces)} face(s).", fg="green")
                predict_ck.config(text=f"CK: {predict_emotion(file_path, rf_classifier_ck, 9, (8, 8),
                                                              (2, 2), "L1")}", fg="green")
                predict_jaffe.config(
                    text=f"JAFFE: {predict_emotion(file_path, rf_classifier_jaffe, 10, (8, 8), (2, 2), "L1")}",
                    fg="green")
            else:
                result_label.config(text="No faces detected.", fg="red")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    else:
        messagebox.showinfo("Info", "No image selected.")


root = tk.Tk()
root.title("Simon Schöggler - FER")

instructions_label = tk.Label(root, text="Select an Image for FER.", font=("Arial", 14))
instructions_label.pack(pady=10)

select_button = tk.Button(root, text="Select Image", command=select_image, font=("Arial", 12), bg="blue", fg="white")
select_button.pack(pady=10)

image_frame = tk.Frame(root)
image_frame.pack(pady=10)

input_image_label = tk.Label(image_frame)
input_image_label.pack(side=tk.LEFT, padx=10)

detected_image_label = tk.Label(image_frame)
detected_image_label.pack(side=tk.LEFT, padx=10)

cropped_face_label = tk.Label(image_frame)
cropped_face_label.pack(side=tk.LEFT, padx=10)

hog_image_label = tk.Label(image_frame)
hog_image_label.pack(side=tk.BOTTOM, padx=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

predict_ck = tk.Label(root, text="", font=("Arial", 12))
predict_ck.pack(pady=5)

predict_jaffe = tk.Label(root, text="", font=("Arial", 12))
predict_jaffe.pack(pady=5)

train_folder_ck = "CK_dataset/train"
test_folder_ck = "CK_dataset/test"
train_folder_jaffe = "JAFFE/train"
test_folder_jaffe = "JAFFE/test"

accuracy_ck, conf_matrix_ck = train_and_evaluate_model(train_folder_ck, test_folder_ck, rf_classifier_ck, 9, (8, 8),
                                                       (2, 2), "L1")
print(f"CK Accuracy: {accuracy_ck * 100:.2f}%")
print("Confusion Matrix (CK):")
print(conf_matrix_ck)

accuracy_jaffe, conf_matrix_jaffe = train_and_evaluate_model(train_folder_jaffe, test_folder_jaffe, rf_classifier_jaffe,
                                                             10, (8, 8), (2, 2), "L1")
print(f"JAFFE Accuracy: {accuracy_jaffe * 100:.2f}%")
print("Confusion Matrix (JAFFE):")
print(conf_matrix_jaffe)

root.mainloop()

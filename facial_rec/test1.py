import cv2
import matplotlib.pyplot as plt

imagePath = input("Enter the image path: ")
img = cv2.imread(imagePath)

if img is None:
    print("Error: Could not read the image. Please check the file path.")
    exit()

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_classifier.empty():
    print("Error: Could not load the cascade classifier.")
    exit()

faces = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

if len(faces) == 0:
    print("No faces detected.")

else:
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

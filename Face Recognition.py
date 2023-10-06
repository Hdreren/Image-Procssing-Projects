import cv2
import matplotlib.pyplot as plt

miranda = cv2.imread("miranda.jpg", 0)
plt.figure(), plt.imshow(miranda, cmap = "gray"), plt.axis("off")

face_cascade = cv2.CascadeClassifier("raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_default.xml")
face_rect = face_cascade.detectMultiScale(miranda)

for (x,y,w,h) in face_rect:
    cv2.rectangle(miranda, (x,y), (x+w, y+h), (255,255,255), 15)

plt.figure(), plt.imshow(miranda, cmap = "gray"), plt.axis("off"), plt.show()






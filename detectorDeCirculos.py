import cv2 as cv
import numpy as np

def resize_image(img, max_width=800, max_height=600):
    h, w = img.shape[:2]
    
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
    elif h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
    
    return img

img = cv.imread("imagem.jpg")
img = resize_image(img, max_width=800)
output = img.copy()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.equalizeHist(gray)
gray_blur = cv.GaussianBlur(gray, (9, 9), 2)

# edges = cv.Canny(gray_blur, 50, 150)

circles = cv.HoughCircles(
    gray_blur,
    cv.HOUGH_GRADIENT,
    dp=1.2,
    minDist=450,
    param1=100,
    param2=70,
    minRadius=50,
    maxRadius=500
)

if circles is not None:
    circles = np.uint16(np.around(circles))

    for (x, y, r) in circles[0, :]:
        cv.circle(output, (x,y), r, (0, 255, 0), 2) # pra marcar o círculo em si
        cv.circle(output, (x, y), 2, (0, 0, 255), 3) # só pra marcar o centro

cv.imshow("Detecção de círculos", output)
cv.waitKey(0)
cv.destroyAllWindows()
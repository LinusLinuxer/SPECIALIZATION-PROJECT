import cv2

img = cv2.imread(
    "/home/linus/Dokumente/1_Universitaet/10_SoSe_25/SPECIALIZATION-PROEJECT/img_preprocessing/hide from code/bsb00050531.0011.jpeg"
)
if img is None:
    print("Error: Unable to read the image file. Please check the file path.")
    exit()

print(type(img))

# Shape of the image
print("Shape of the image", img.shape)

# [rows, columns]
crop = img[50:180, 100:300]

cv2.imshow("original", img)
cv2.imshow("cropped", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

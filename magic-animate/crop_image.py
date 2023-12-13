import cv2

input_image = 'data/image/input/902-2.jpg'
output_image = 'data/image/902-2_crop.jpg'

img = cv2.imread(input_image)
# Set your own crop region, according to the image
img = img[-832:-400, :]
img = cv2.resize(img, (512, 512))
cv2.imwrite(output_image, img)
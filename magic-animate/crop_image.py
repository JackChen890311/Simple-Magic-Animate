import cv2

input_image = 'data/image/input/bao5.jpg'
output_image = 'data/image/bao5_crop.jpg'

img = cv2.imread(input_image)
print('Original Shape: ', img.shape)

# Set your own crop region, according to the image
# img = img[200:-300, :]

img = cv2.resize(img, (512, 512))
cv2.imwrite(output_image, img)
import cv2

input_video = 'data/video/dense/d1cen_dense.mp4'
output_video = 'data/video/d1cen_pad.mp4'

captura = cv2.VideoCapture(input_video)
fps = captura.get(cv2.CAP_PROP_FPS)
width = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frame = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
final_size = 512
print('Original size: %d x %d' % (width, height))
print('Target size: %d x %d' % (final_size, final_size))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (final_size, final_size))
max_size = max(width, height)

for i in range(total_frame):
    ret, frame = captura.read()
    if ret == False:
        break

    delta_w = max_size - width
    delta_h = max_size - height
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # Set color to pad
    color = [81, 0, 65]
    assert frame[0][0].tolist() == color # comment this line if raise error
    frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=frame[0][0].tolist())

    frame = cv2.resize(frame, (final_size, final_size))
    video.write(frame)

video.release()
captura.release()

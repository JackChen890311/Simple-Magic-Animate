import cv2

input_video = 'data/video/dense/hiit2_dense.mp4'
output_video = 'data/video/dense/hiit2c_dense.mp4'

captura = cv2.VideoCapture(input_video)
fps = captura.get(cv2.CAP_PROP_FPS)
width = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frame = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

# Set new size here
new_width = width // 3
new_height = height
final_size = (new_width, new_height)
print('Original size: %d x %d' % (width, height))
print('Target size: %d x %d' % final_size)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, final_size)
max_size = max(width, height)

for i in range(total_frame):
    ret, frame = captura.read()
    if ret == False:
        break
    # Add also here
    frame = frame[:, new_width:2*new_width, :]
    assert frame.shape[1] == final_size[0]
    assert frame.shape[0] == final_size[1]
    video.write(frame)

video.release()
captura.release()

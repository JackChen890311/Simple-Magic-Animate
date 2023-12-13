import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.extractor import CompoundExtractor, create_extractor

from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)

cfg = get_cfg()
add_densepose_config(cfg)

# model_name = 'densepose_rcnn_R_50_FPN_s1x'
model_name = 'densepose_rcnn_R_101_FPN_DL_s1x'
input_video = 'data/022.mp4'
output_video = 'data/output/022_dense.mp4'

cfg.merge_from_file(f"configs/{model_name}.yaml")
cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

cfg.MODEL.WEIGHTS = f"{model_name}.pkl"
predictor = DefaultPredictor(cfg)

VISUALIZERS = {
    "dp_contour": DensePoseResultsContourVisualizer,
    "dp_segm": DensePoseResultsFineSegmentationVisualizer,
    "dp_u": DensePoseResultsUVisualizer,
    "dp_v": DensePoseResultsVVisualizer,
    "bbox": ScoredBoundingBoxVisualizer,
}

vis_specs = ['dp_segm']
visualizers = []
extractors = []
for vis_spec in vis_specs:
    vis = VISUALIZERS[vis_spec]()
    visualizers.append(vis)
    extractor = create_extractor(vis)
    extractors.append(extractor)
visualizer = CompoundVisualizer(visualizers)
extractor = CompoundExtractor(extractors)

context = {
    "extractor": extractor,
    "visualizer": visualizer
}

visualizer = context["visualizer"]
extractor = context["extractor"]

captura = cv2.VideoCapture(input_video)
total_frame = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(captura.get(cv2.CAP_PROP_FPS))


def predict(img):
    outputs = predictor(img)['instances']
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    data = extractor(outputs)
    image = np.ones_like(image)
    image[:,:,0] *= 84 #81
    image[:,:,1] *= 0  # 0
    image[:,:,2] *= 68 #65
    image_vis = visualizer.visualize(image, data)

    return image_vis
    
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

cnt = 0
while captura.isOpened():
    print('Progress: %d / %d' % (cnt, total_frame))
    cnt += 1
    ret, frame = captura.read()
    if not ret:
        break

    if cnt < 60:
        continue
    result = predict(frame)
    out.write(result)
    if cnt == 360:
        break
 
captura.release()
out.release()

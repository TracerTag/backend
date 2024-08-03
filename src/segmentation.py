import onnxruntime
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import svgwrite
from scipy.spatial import ConvexHull


class Segmentation:
    def __init__(self, image_path):
        sam = sam_model_registry["vit_b"](checkpoint="/mnt/models/sam_vit_b_01ec64.pth")

        self.predictor = SamPredictor(sam)

        onnx_model_path = "/mnt/models/sam_onnx.onnx"

        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)

        self.image = cv2.imread(image_path)
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image)
        self.image_embedding = self.predictor.get_image_embedding().numpy()

    def segment_image(self, pos_x, pos_y):
        input_point = np.array([[pos_x, pos_y]])
        input_label = np.array([1])

        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = self.predictor.transform.apply_coords(onnx_coord, self.image.shape[:2]).astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": self.image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(self.image.shape[:2], dtype=np.float32)
        }
        masks, _, low_res_logits = self.ort_session.run(None, ort_inputs)
        masks = masks > self.predictor.model.mask_threshold
        masks = np.concatenate(masks)

        contours = []
        for mask in masks:
            mask_points = np.asarray(np.where(mask)).T
            mask_convex_hull = ConvexHull(mask_points)
            mask_contour = mask_points[mask_convex_hull.vertices]
            contours.append(mask_contour)

        contours_to_svg(contours, 'output.svg', (self.image.shape[1], self.image.shape[0]))
        return masks


def contours_to_svg(contours, svg_filename, image_shape):
    """ This function takes a list of countour, the desired svg filename and size and writes the countours as SVG """
    dwg = svgwrite.Drawing(svg_filename, profile='tiny', size=image_shape)

    for contour in contours:
        # reverse points since coords are reversed by cv2
        points = [(point[1], point[0]) for point in contour]
        path = 'M ' + ' '.join([f'{p[0]},{p[1]}' for p in points]) + ' Z'
        dwg.add(dwg.path(d=path, fill='none', stroke='black', stroke_width=1))

    dwg.save()

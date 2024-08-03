class DetectedObject:
    """This class represents a single object detected in the image provided by the user"""
    def __init__(self, contour_points, object_class):
        self.contour_points = contour_points
        self.object_class = object_class

    def to_svg_path(self):
        """This method creates the object SVG path from the provided contour points"""
        # Extract the points from the mask provided
        points = [(point[0], point[1]) for point in self.contour_points[0]]
        # Create the actual path based on the points
        svg_path = 'M ' + ' '.join([f'{p[0]},{p[1]}' for p in points]) + ' Z'

        return svg_path


def detect_objects_in_image(model, image_path):
    """This function receives the path to an image and returns the list of """

    classes_names = model.names

    # Run the model to retrieve the detected objects
    yolo_results = model(image_path, conf=0.5)

    # Here we will accumulate the objects that were found by the model
    objects = []

    # Loop over the available results
    for result in yolo_results:
        # If no boxes are found in this result let's skip it
        if result.boxes is None:
            continue

        # Otherwise let's loop over the boxes length. Please note this equals
        # to the masks length with a 1:1 correspondence
        for i in range(0, len(result.boxes)):
            # Extract the class ID of the object
            object_class_id = result.boxes[i].cls.numpy()[0]

            # Extract the corresponding class name
            object_class = classes_names[object_class_id]

            # Then let's get the mask of the object
            object_mask = result.masks[i].xy

            # Create the detected object
            detected_object = DetectedObject(object_mask, object_class)

            # Add it to our list
            objects.append(detected_object)

    return objects

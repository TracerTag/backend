def detect_objects_in_image(model, image_path):
    """This function receives the path to an image and returns the list of """

    # Run the model to retrieve the detected objects
    yolo_results = model(image_path, conf=0.5)

    # The result could be an array of results, let's accumulate them
    boxes = []

    for yolo_result in yolo_results:
        # Get all the boxes from the current result
        yolo_result_boxes = yolo_result.boxes

        # If at least a box was retrieved
        if yolo_result_boxes is not None:
            boxes.extend(yolo_result_boxes)

    return boxes

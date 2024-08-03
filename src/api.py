from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
from ultralytics import YOLO
import detect_objects
from segmentation import Segmentation
import logging

app = FastAPI()

UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(exist_ok=True)

model = YOLO("/mnt/models/yolov10x.pt")
classes_names = model.names

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded image to the directory
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        seg = Segmentation(file_path)
        # Return an array of ultralytics boxes (containing all the detect objects)
        boxes = detect_objects.detect_objects_in_image(model, file_path)
        for box in boxes:
            corner_high = {"x": box.xyxy[0][0], "y": box.xyxy[0][1]}
            corner_low = {"x": box.xyxy[0][2], "y": box.xyxy[0][3]}
            middle_point = {"x": (corner_high["x"] + corner_low["x"])/2, "y": (corner_high["x"] + corner_low["x"])/2}

            import logging
            logger = logging.getLogger('uvicorn.error')
            logger.error(f"middlepoint - {middle_point['x']} - {middle_point['y']}")
            # seg.segment_image(middle_point["x"], middle_point["y"])
            seg.segment_image(middle_point['x'], middle_point['y'])

        return JSONResponse(content={"filename": file.filename, "message": f"Image uploaded successfully - {len(boxes)}"})

    except Exception as e:
        return JSONResponse(content={"message": "There was an error uploading the image", "error": str(e)}, status_code=500)
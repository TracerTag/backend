from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
from ultralytics import YOLO
from export_objects import export_to_svg
from detect_objects import detect_objects_in_image
import logging
import io

app = FastAPI()

UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(exist_ok=True)

model = YOLO("/mnt/models/yolov9e-seg.pt")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded image to the directory
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        detected_objects = detect_objects_in_image(model, image_path=file_path)

        # Export the objects to svg
        dwg = export_to_svg(detected_objects)

        svg_output = io.StringIO()
        dwg.write(svg_output)
        svg_output.seek(0)

        return Response(content=svg_output.getvalue(), media_type="image/svg+xml")

    except Exception as e:
        return JSONResponse(content={"message": "There was an error uploading the image", "error": str(e)}, status_code=500)
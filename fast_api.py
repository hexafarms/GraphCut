from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File
from cut import segment_api as segment_gc

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hexa": "Farm"}

@app.post("/segment_gc")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"fast_api/input/{file.filename}"

    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    ############################ Current Best Model ###############################
    histograms= "/ground_data/histograms.npy"
    ################################################################################

    input_dir = f"fast_api/input/{file.filename}"
    output_dir = "fast_api/output/"

    segment_gc(histograms, input_dir, output_dir)
    image = open(output_dir+f"/{file.filename}", 'rb')

    return StreamingResponse(image, media_type=("image/jpeg"or"image/png"))
    

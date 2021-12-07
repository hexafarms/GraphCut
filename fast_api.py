from fastapi.responses import StreamingResponse, Response
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from cut import segment_api as segment_gc

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hexa": "Farm"}

@app.post("/segment_gc")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"fast_api/input/{file.filename}"

    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    ############################ Current Best Model ###############################
    histograms= "/weights/v0/histograms.npy"
    ################################################################################

    ############################### Camera Ratio ##################################
    # 1/10 means 1pixel is 10cm^2 
    ratio = (1/13)**2
    ################################################################################

    input_dir = f"fast_api/input/{file.filename}"
    output_dir = "fast_api/output/"

    area = segment_gc(histograms, input_dir, output_dir, ratio)
    result = {f'Leaf area of {file.filename}': f'{int(area)} cm^2'}
    image = open(output_dir+f"/{file.filename}", 'rb')

    return Response(content=image, headers=result, media_type=("image/jpeg"or"image/png"))
    

import os
import multipart
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage 
from google.oauth2 import service_account
from pydantic import BaseModel 
from google.auth import default,compute_engine
from google.auth.transport import requests
from utils.utils import *
from config import ENV_CONFIG
os.environ["OPENAI_API_KEY"] = ENV_CONFIG["openai_api_key"]

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
storage_client = storage.Client()
credentials,_ = default()

class FileNameRequest(BaseModel):
    filename: str
    filetype: str

class ComplianceRequest(BaseModel):
    gcs_path: str
    selectedMarket: str
    selectedChannel: str
    selectedProduct: str


# @app.post("/get_result/")
# async def getResult(gcs_path:str= Form(...),selectedProduct:str= Form(...),selectedMarket:str= Form(...),selectedChannel:str= Form(...)):

#     try:
#         project_id= 'ig-aimt-sandbox'
#         bucketName = 'quantiphi-ig-mkt-compliance-upload-bucket'
#         expiration = datetime.timedelta(hours=1)
#         credentials = service_account.Credentials.from_service_account_file('service-account.json')

#         # Create a client object with the fetched credentials
#         client = storage.Client(credentials=credentials)


#         print('file details',gcs_path,selectedProduct,selectedMarket,selectedChannel)
#         data = [
#             {

#                 "FileName":"",

#                 "Type": "pdf/docx/jpeg/gif/mp4",

#                 "Timestamp":"",

#                 "gcs_original_path":"",

#                 "gcs_output_path":"",

#                 "images":

#                     [

#                             {

#                                 "font_prediction":[],
#                                 "gcs_path" : "gs://quantiphi-ig-mkt-compliance-upload-bucket/upload/20240520155346_non_comp_by_text.jpg",
#                                 "non_compliance_list":

#                                 [

#                                     {"title":"Logo Colour","text":"logo is small"},

#                                     {"title":"Incorrect Font","text":"risk warning center aligned"},

#                                     {"title":"Logo Colour","text":"logo is small"},

#                                     {"title":"Incorrect Font","text":"risk warning center aligned"},

#                                     {"title":"Logo Colour","text":"logo is small"},

#                                     {"title":"Incorrect Font","text":"risk warning center aligned"},

#                                     {"title":"Logo Colour","text":"logo is small"},

#                                     {"title":"Incorrect Font","text":"risk warning center aligned"},

#                                     {"title":"Logo Colour","text":"logo is small"},

#                                     {"title":"Incorrect Font","text":"risk warning center aligned"},

#                                     {"title":"Logo Colour","text":"logo is small"},

#                                     {"title":"Incorrect Font","text":"risk warning center aligned"}

#                                 ]

#                             },

#                             {

#                                 "font_prediction":[],
#                                 "gcs_path" : "gs://quantiphi-ig-mkt-compliance-upload-bucket/upload/20240520155346_non_comp_by_text.jpg",
#                                 "non_compliance_list":

#                                     [

#                                         {"title":"Logo Colour","text":"logo is small"},

#                                         {"title":"Incorrect Font","text":"risk warning center aligned"},

#                                         {"title":"Logo Colour","text":"logo is small"}

#                                     ]

#                             }

#                     ]

#             }
#         ]
#         for image in data[0]['images']:
#             gcs_path_img = image['gcs_path']
#              # Check if the gcs_path starts with 'gs://'
#             if not gcs_path_img.startswith('gs://'):
#                 raise ValueError("Invalid GCS path. The path must start with 'gs://'")

#             # Remove 'gs://' prefix and split the path
#             _, path = gcs_path_img.split('gs://', 1)
#             bucket_name, object_name = path.split('/', 1)
#              # Get the bucket
#             bucket1 = client.bucket(bucket_name)
#             # Get the blob object representing the object in the bucket
#             blob1 = bucket1.blob(object_name)
#             # Generate a signed URL for the blob
#             signed_url = blob1.generate_signed_url(
#                 version="v4",
#                 expiration=expiration,
#                 method="GET"
#             )
#             image['signed_url']=signed_url
#             image['summary']={
#                 'ComplianceRisk' : 0,
#                 'spellingMistake': 0,
#                 'brandRisk': len(image['non_compliance_list'])
#             }

#         return { 
#             "status":"success",
#             "results":data}

#     except Exception as e:

#         return {
#             "status":"error",
#             "error": str(e)}

@app.post("/generate-signed-url")
async def generate_signed_url(file_request: FileNameRequest):
    # Generate a signed URL for uploading the file to GCS
    try:
        project_id= ENV_CONFIG['project_id']
        bucketName = ENV_CONFIG['bucket_name']
        expiration = datetime.utcnow() + timedelta(hours=1)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        auth_request =  requests.Request()
        credentials.refresh(auth_request)
        signing_credentials = compute_engine.IDTokenCredentials(auth_request,"",service_account_email=credentials.service_account_email)
        # Append the unique hex code to the filename
        filename_with_hex = f"upload/{timestamp}_{file_request.filename}"
        bucket = storage_client.bucket(bucketName)
        blob = bucket.blob(filename_with_hex)
        gcs_path = f"gs://{bucketName}/{filename_with_hex}"
        signed_url = blob.generate_signed_url(
        version='v4',  # Specify version as 'v4'
        expiration=expiration,
        method='PUT',  # Specify write permission
        content_type=file_request.filetype,
        credentials=signing_credentials
        # Specify action as 'write'
        )  # URL expires in 1 hour (3600 seconds)
        return { 

            "status":"success",
            "gcs_path":gcs_path,
            "signed_url":signed_url}
    except Exception as e:

        return {

            "status":"error",

            "error": str(e)}

@app.get("/")
def ping():
    return {"Hello": "World"}

@app.post("/compliance_check")
def compliance_check(compliance_request: ComplianceRequest):
    project_id= ENV_CONFIG['project_id']
    bucketName = ENV_CONFIG['bucket_name']
    expiration = timedelta(hours=1)
    
    
    IMAGE_FORMATS = ('jpg', 'png', 'gif')
    TEXT_FORMATS = ('pdf', 'docx')
    VIDEO_FORMATS = (".mp4")

    try:
        # Load file from gcs
        auth_request =  requests.Request()
        credentials.refresh(auth_request)
        signing_credentials = compute_engine.IDTokenCredentials(auth_request,"",service_account_email=credentials.service_account_email)
        
        file_path = compliance_request.gcs_path
        file_name = file_path.split("/")[-1]

        country = compliance_request.selectedMarket
        medium = compliance_request.selectedChannel
        product = compliance_request.selectedProduct

        if file_path.endswith(IMAGE_FORMATS):
            print("Starting Image pipeline for file: {}".format(file_name))
            data = image_compliance_check(file_path, medium)

        else:
            print("Starting Text pipeline for file: {}".format(file_name))
            data = text_compliance_check(file_path, source="text",
                                         medium=medium)

        for image in data['images']:
            gcs_path_img = image['gcs_path']
            # Check if the gcs_path starts with 'gs://'
            if not gcs_path_img.startswith('gs://'):
                raise ValueError("Invalid GCS path. The path must start with 'gs://'")

            # Remove 'gs://' prefix and split the path
            _, path = gcs_path_img.split('gs://', 1)
            bucket_name, object_name = path.split('/', 1)
            # Get the bucket
            bucket1 = storage_client.bucket(bucket_name)
            # Get the blob object representing the object in the bucket
            blob1 = bucket1.blob(object_name)
            # Generate a signed URL for the blob
            signed_url = blob1.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET",
                credentials=signing_credentials
            )
            image['signed_url']=signed_url
        final_output = {
                "status":"success",
                "results":[data]
        }

        return final_output

    except Exception as error:
        print(error)
        traceback.print_exc()

        final_output = {
            "error": "Compliance Check failed",
            "results": [],
            "status": 'error'
        }

        return final_output

import json
import os
import cv2
import traceback
from utils.image_utils.logo import *
from utils.image_utils.typography import *
from utils.text_compliance import *
from vertexai.generative_models import GenerativeModel
from datetime import datetime, timedelta
from config import ENV_CONFIG
from google.cloud import storage
from PIL import Image, ImageDraw
import imageio
import fitz  # PyMuPDF
from google.cloud import storage
import pypandoc


model_gen = GenerativeModel(model_name=ENV_CONFIG["gemini_model"])
ig_list = ["IG", "IG\n", "IG Group"]
scale_factor = 1.3
output_path = ENV_CONFIG['buffer_path']
ocr_processor_display_name = "ig-ocr" # Must be unique per project, e.g.: "My Processor"
ocr_processor_id = "165d647a387cb584" # Create processor before running sample
ocr_processor_version = "pretrained-ocr-v2.0-2023-06-02" # Optional. Processor version to use

form_processor_display_name = "form-parser-ig"
form_processor_id = "ff445d9f10fdd25e" # Create processor before running sample
form_processor_version = "pretrained-form-parser-v2.0-2022-11-10"

def upload_json_to_gcs(data, destination_blob_name):
    """Uploads a JSON file to the bucket."""
        
    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(ENV_CONFIG["bucket_name"])

    # Create a blob object from the bucket
    blob = bucket.blob(destination_blob_name)

    # Convert data to JSON
    json_data = json.dumps(data, default=str)

    # Upload JSON data to the blob
    blob.upload_from_string(json_data, content_type='application/json')

    print(f"File {destination_blob_name} uploaded")


def download_from_gcs(bucket_name, source_blob_name, destination_file_name): 
    """Downloads a file from the bucket.""" 
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")
    return


def image_compliance_check(path, medium):
    
    saved_folder_path = ""
    try:
        file_name = os.path.basename(path)
        file = os.path.splitext(file_name) 
        file1 = file[0] + file[1]
        type_file = file[1].replace('.', '')
        
        now = datetime.now()

        data_ = {'Filename': file1,
                'Type': type_file,
                'Timestamp': now,
                'gcs_original_path': path,
                'gcs_output_path': "gs://{}/processed/{}".format(ENV_CONFIG["bucket_name"], file[0]),
                "images": []
                }
        
        print(data_)

        #put a condition to chek gif file and then run the compliance check for each frame. else once.
        if path.endswith(".gif"):
            file_paths = frames_from_gif(path)
        else:
            file_paths = [path]

        for file_path in file_paths:
            
            try:
                # filename_download = file_path.split("/")[-1]
                bucket_name = file_path.split("/")[2]
                source_blob_name = '/'.join(file_path.split("/")[3:])
                print(bucket_name, source_blob_name)
                destination_file_name = 'tmp/' + source_blob_name
                print(destination_file_name)
                if not os.path.exists('/'.join(destination_file_name.split("/")[:-1])):
                    os.makedirs('/'.join(destination_file_name.split("/")[:-1]))
                download_blob(ENV_CONFIG["bucket_name"], source_blob_name, destination_file_name)

                res = detect_text_and_logos(destination_file_name)

                doc = process_document_ocr_sample(ENV_CONFIG['project_id'],
                    ENV_CONFIG['location'],
                    ocr_processor_id,
                    ocr_processor_version,
                    destination_file_name,
                    ENV_CONFIG['mime_type_image']) 

                doc_json = documentai.Document.to_json(doc)
                data = json.loads(doc_json)
                sort_and_clean_data(data)

                form_doc = process_document_form_sample(
                        ENV_CONFIG['project_id'],
                        ENV_CONFIG['location'],
                        form_processor_id,
                        form_processor_version,
                        destination_file_name,
                        ENV_CONFIG['mime_type_image'],
                    )

                table_status = check_if_table(form_doc)
                logo_print_statements, logo_text_color, distance_logo_rw, ig_logo_count = final_logo_print_statement(res, data, destination_file_name)
                bbox_50_precent=None
                overline, headline, bodyline, head_avg, head_line_regex_check=typo_output(data, bbox_50_precent, res)
                if head_line_regex_check:
                    words_adjacent_headline=second_check_for_heading(data, head_line_regex_check)
                    bbox_50_precent=get_final_bbox_50_precent(words_adjacent_headline, destination_file_name)
                    overline, headline, bodyline, head_avg, head_line_regex_check=typo_output(data, bbox_50_precent, res)
                else:
                    pass
                overline_print_statements, font_prediction_overline_list =overline_print_statement(overline, data, destination_file_name, head_avg)
                headline_print_statements, font_prediction_headline_list=headline_print_statement(data, headline, destination_file_name)
                rw_print_statements, risk_warning_align, distance_final_riskwarning_line,risk_response,riskwarning_coordinates = risk_warning_print_statement(data, res, ig_logo_count, destination_file_name, head_avg)
                logo_print_statements = rw_logo_position_check(distance_logo_rw, risk_response, logo_print_statements, destination_file_name, data, riskwarning_coordinates, ig_logo_count)
                url_print_statements= url_check(form_doc, data, logo_text_color, riskwarning_coordinates, destination_file_name)
                date_print_statements = date_check(form_doc, destination_file_name)
                align_print_statements = alignment_check(data,overline, headline, bodyline, table_status, risk_warning_align, distance_final_riskwarning_line, destination_file_name)

                final = {
                    "Logo" : logo_print_statements,
                    "Typography": {"Overline": overline_print_statements, 
                                "Headline": headline_print_statements,
                                "Riskwarning": rw_print_statements},
                    "Entities": {"URL": url_print_statements,
                                "Date": date_print_statements },
                    "Alignment": align_print_statements,
                }

                final_statements_to_print(logo_print_statements,overline_print_statements, headline_print_statements, url_print_statements, date_print_statements, rw_print_statements, align_print_statements, final)
                
                # Identify paragraphs to be passed through the text pipeline
                para_bbox_text_list = text_bbox_para_list(doc)

                # Call the text compliance pipeline
                # text_non_compliance_list, boxes_to_draw, summary = text_compliance_check(input_=para_bbox_text_list, source="image", medium=medium)
                text_non_compliance_list, boxes_to_draw = [], []
                summary = {
                    "RiskWarnings": 0,
                    "CFD": 0,
                    "Compliance": 0
                }
                                           
                # TODO
                # draw boxes using boxes_to_draw
                image_upload_path = save_file_with_cordinates(final, destination_file_name, ENV_CONFIG['bucket_name'],  ENV_CONFIG['processed_path'], ENV_CONFIG['buffer_path'], boxes_to_draw)
                print(f"==========={image_upload_path}=============")

                non_com_sum_count = count_of_non_compliance(final)
                final_non_compliance_list = non_compliance_list_function(final)

                summary.update({"Branding": non_com_sum_count})

                images_dict = {
                    "font_prediction": font_prediction_overline_list + font_prediction_headline_list,
                    "gcs_path": image_upload_path,
                    "non_compliance_list": final_non_compliance_list + text_non_compliance_list,
                    "summary": summary
                }

                data_['images'].append(images_dict)
                
            except Exception:
                print("Error during image compliance check for file path: {}".format(file_path))
                traceback.print_exc()
        print("UPLOADING JSON FILE")

        destination_blob_name = "processed/{}/{}.json".format(file[0], file[0])       
        print(destination_blob_name, "destination_blob_name")
        upload_json_to_gcs(data_, destination_blob_name)

    except Exception:
        print("Error during image compliance check for file: {}".format(path))
        traceback.print_exc()
        data_ = {}

    return data_


def text_compliance_check(input_, source, medium):
    
    def call_text_pipeline(paragraphs_list):
                    
        text_result = []
        boxes_to_draw = []
        summary = {}
        count = 0
        print("Total paragraphs to be processed: {}".format(len(paragraphs_list)))
        try:
            for para_info in paragraphs_list:
                non_compliance_list, summary = get_text_output(para_info["text"], medium)

                if non_compliance_list:
                    text_result.extend(non_compliance_list)
                    boxes_to_draw.append(para_info["BoundingBox"])
                
                count += 1
                if count % 5 == 0:
                    print("***********Completed {} paragraphs*************".format(count))

        except Exception as e:
            print("Error during text compliance check")
            traceback.print_exc()
            text_result = []
            boxes_to_draw = []    
            summary = {}

        return text_result, boxes_to_draw, summary
        
    # Input is a file path of PDF or docx

    print("Source of text is: {}".format(source))
    if source == "text":
        
        try:
            file_name = os.path.basename(input_)
            file = os.path.splitext(file_name) 
            file1 = file[0] + file[1]
            type_file = file[1].replace('.', '')
            
            now = datetime.now()

            data_ = {'Filename': file1,
                'Type': type_file,
                'Timestamp': now,
                'gcs_original_path': input_,
                'gcs_output_path': "gs://{}/processed/{}".format(ENV_CONFIG["bucket_name"], file[0]),
                "images": []
                }
                
            # Step 1:
            if input_.endswith("pdf"):
                print("{} is a pdf. Downloading from gcs....".format(input_))
                # Download pdf to local: pdf gcs file path --> get a local pdf path
                base_name = os.path.basename(input_).rsplit('.', 1)[0]
                temp_dir = os.path.join("/tmp", base_name)
                os.makedirs(temp_dir, exist_ok=True)

                destination_filename = os.path.join(temp_dir, f"{base_name}.pdf")
                source_blob_name = "".join(a.split(ENV_CONFIG["bucket_name"])[-1].strip("/"))

                pdf_output = download_from_gcs(ENV_CONFIG["bucket_name"], source_blob_name, destination_filename)

            elif input_.endswith("docx"):
                print("{} is a docx. Converting docx to pdf....".format(input_))
                # Convert docx to PDF: docx gcs file path --> get a local pdf path
                pdf_output = convert_docx_to_pdf(input_)
            
            print("PDF saved in local directory: {}".format(pdf_ouput))

            # Step 2:
            # extract text from pdf --> get a list of gcs paths for each page in pdf
            print("Splitting pdf into pages and converting them to images......")
            gcs_bucket_path = "gs://{}/{}".format(ENV_CONFIG['bucket_name'], ENV_CONFIG['buffer_path'])
            image_page_path_list = split_pdf_and_upload_to_gcs(pdf_ouput, gcs_bucket_path)
            print("PDF splitted into {} pages and their images uploaded to buffer folder".format(len(image_page_path_list)))

            # Step 3: Pass each pdf page through text extraction API
            print("Extracting text from the converted images....")
            for path in image_page_path_list:
                try:
                    print("Processing uploaded buffer image: {}....".format(path))
                    document = process_document_ocr_sample(ENV_CONFIG['project_id'],
                                                            ENV_CONFIG['location'],
                                                            ocr_processor_id,
                                                            ocr_processor_version,
                                                            path,
                                                            ENV_CONFIG['mime_type_image']) 

                    # Identify paragraphs to be passed through the text pipeline
                    para_bbox_text_list = text_bbox_para_list(document)
                    print("Text extracted successfully!")

                    text_non_compliance_list, boxes_to_draw, summary = call_text_pipeline(para_bbox_text_list)
                    print("Text compliance check completed!")

                    print("Drawing non compliance boxes if any and uploading to processed folder....")
                    gcs_processed_image_path = process_image_from_gcs(path, boxes_to_draw, "processed")
                    print("Image uploaded!")

                    text_non_compliance_info = {
                                                "font_prediction": [],
                                                "gcs_path": gcs_processed_image_path,
                                                "non_compliance_list:": text_non_compliance_list,
                                                "summary": summary
                                            }

                    print("Completed processing buffer image: {}......".format(path))
                
                except Exception:
                    traceback.print_exc()
                    print("Error during text compliance check while processing file path: {}".format(path))
                    gcs_processed_image_path = process_image_from_gcs(path, [], "processed")

                    text_non_compliance_info = {
                            "font_prediction": [],
                            "gcs_path": gcs_processed_image_path,
                            "non_compliance_list:": [],
                            "summary": {}
                        }

                    print("Uploaded buffer image: {}......".format(path))

                data_['images'].append(text_non_compliance_info)
                
            print("Uploading final data json to bucket")
            destination_blob_name = "processed/{}/{}.json".format(file[0], file[0])       
            upload_json_to_gcs(data_, destination_blob_name)
            print("Uploaded final data")

            return data_
        
        except Exception:
            print("Error during text compliance check for file: {}".format(file_path))
            traceback.print_exc()
            return {}

    # Input is a list of extracted text with bounding boxes
    elif source == "image":
        try:
            print("Calling text pipeline on paragraphs identified from image....")
            text_result, boxes_to_draw, summary = call_text_pipeline(input_)
            print("Text non compliance identified.....")
            return text_result, boxes_to_draw, summary
        except Exception:
            print("Error during text compliance check")
            traceback.print_exc()
            return [], [], {}

#################################### Handling GIFs ###################################################################################

def frames_from_gif(gcs_gif_path):
    try:
        print("*********started downlaoding gif in local dir**************")
        gif_local_path = download_gif(gcs_gif_path, local_dir = 'tmp/')
        print(f"*******gif donloaded here: path->{gif_local_path}*********************")
        print("*****************getting the frames from gif*******")
        gif_as_list = imageio.mimread(gif_local_path)
        local_gif_frames_dir = os.path.basename(gcs_gif_path).split('.')[0]
        print(f"***************saving the gif frames as png in {local_gif_frames_dir}**********")
        gif_base_name = os.path.basename(gcs_gif_path).split('.')[0]
        save_images_with_opencv(gif_as_list, local_gif_frames_dir, gif_base_name)
        print(f"************ uploading frames in buffer blob in gcs backend bucket into subfolder:{local_gif_frames_dir}******")
      
        gcs_frame_paths_list = process_and_upload_gif_frames(gif_base_name, local_dir=local_gif_frames_dir, gcs_bucket_name='quantiphi-ig-mkt-compliance-upload-bucket') 
        print("***frames saved in gcs successfully!*******frame_paths list is being returned**********")
        # return frames_path
    except Exception:
        print("Error while changing the gif to images")
        traceback.print_exc()
    return gcs_frame_paths_list


def save_images_with_opencv(images, output_dir, gif_base_name):
    #processed_filepath = []
    """
    Saves a list of images as PNG files using OpenCV."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
   
    # Save each image as a PNG file
    for i, image in enumerate(images):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filename = os.path.join(output_dir, f"{gif_base_name}_frame_{i+1}.png")
        cv2.imwrite(filename, image_rgb)
        #processed_filepath.append(filename)
    print(f"frames saved into {output_dir} locally")
    
    
def download_gif_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(destination_file_name)
    #print("Downlaoding gif from cloud bucket and storing locally")
    return destination_file_name
def download_gif(gcs_gif_path, local_dir = 'tmp/'):
    bucket_name = gcs_gif_path.split('/')[2]
    #print(bucket_name)
    source_blob_name = '/'.join(gcs_gif_path.split('/')[3:])
    base_name = os.path.basename(source_blob_name)
    #defineing the local path to save the gif
    #print(base_name)
   
    os.makedirs(local_dir, exist_ok=True)
    local_gif_path = os.path.join(local_dir, base_name)
    #downloading gif from gcs to local
    #print(local_gif_path)
    downloaded_gif_path = download_gif_from_gcs(bucket_name, source_blob_name, local_gif_path)
    #print(downloaded_gif_path)
    return downloaded_gif_path   
def upload_gif_frames_to__gcs_buffer(bucket_name, source_dir, destination_subfolder):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    gcs_frame_paths = []
    
    for file_name in os.listdir(source_dir):
        local_file_path = os.path.join(source_dir, file_name)
         # Check if it's a file (not a directory)
        if os.path.isfile(local_file_path):
            # Construct the GCS destination path
            destination_blob_name = f"{destination_subfolder}/{file_name}"
           
            # Get the blob object for the destination file
            blob = bucket.blob(destination_blob_name)
           
            # Upload the file to GCS
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")
            gcs_path = f"gs://{bucket_name}/{destination_blob_name}"
            gcs_frame_paths.append(gcs_path)
    return gcs_frame_paths
def process_and_upload_gif_frames(gif_base_name, local_dir='tmp_1', gcs_bucket_name='quantiphi-ig-mkt-compliance-upload-bucket'):
    """Processes GIF frames and uploads them to GCS under a specified subfolder."""
    # Construct the subfolder name in GCS bucket
    destination_subfolder = f"buffer/{gif_base_name}"
   
    # Upload all files from the local directory to the specified GCS bucket and subfolder
    gcs_frame_paths = upload_gif_frames_to__gcs_buffer(gcs_bucket_name, local_dir, destination_subfolder)
    return gcs_frame_paths

############################################ Handling doc and pdf ##################################################


def convert_docx_to_pdf(input_file, output_file):
    extra_args = ['--pdf-engine=xelatex', '-V','geometry:margin=1in']
    output = pypandoc.convert_file(input_file, 'pdf', outputfile = output_file, extra_args = extra_args)
    print(f"**********Conversion from doc to pdf is Successful!**********")
    return output
# input_file = "sample3.docx"
# output_file = "doc_img/sample3.pdf"
# convert_docx_to_pdf(input_file, output_file)

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    return f"gs://{bucket_name}/{destination_blob_name}"

def split_pdf_and_upload_to_gcs(pdf_path, gcs_bucket_path):
    # Open the PDF file
    document = fitz.open(pdf_path)
   
    # Extract the base name of the PDF file without extension
    base_name = os.path.basename(pdf_path).rsplit('.', 1)[0]
   
    # Create a temporary directory to store individual pages
    temp_dir = os.path.join("/tmp", base_name)
    os.makedirs(temp_dir, exist_ok=True)
   
    # Initialize a list to store the GCS paths of the saved pages
    gcs_paths = []
   
    # Iterate over each page and save as a separate PDF
    for page_num in range(len(document)):
        page = document.load_page(page_num)  # Load the page
        single_page_pdf = fitz.open()  # Create a new PDF for the single page
        single_page_pdf.insert_pdf(document, from_page=page_num, to_page=page_num)
       
        # Save the single page PDF to a temporary file
        single_page_path = os.path.join(temp_dir, f"{base_name}_page_{page_num + 1}.pdf")
        single_page_pdf.save(single_page_path)

        image_created_from_pdf = pdf_to_image(single_page_path)
        image_save_path = os.path.join(temp_dir, f"{base_name}_page_{page_num + 1}.png")
        image_created_from_pdf.save(image_save_path)
       
        # Upload the single page PDF to GCS
        gcs_page_path = upload_to_gcs(
            bucket_name=gcs_bucket_path.split('/')[2],
            source_file_name=image_save_path,
            destination_blob_name=f"{gcs_bucket_path.split('/', 3)[3]}/{base_name}/{base_name}_page_{page_num + 1}.png"
        )
       
        # Add the GCS path to the list
        gcs_paths.append(gcs_page_path)
       
        # Close the single page PDF
        single_page_pdf.close()
   
    # Close the original PDF document
    document.close()
   
    return gcs_paths


def pdf_to_image(pdf_path):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page()
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

    pdf_document.close()

    return img


def draw_bounding_boxex_and_upload_to_gcs(file_path, boxes_to_draw):
        image = Image.open(file_path)
        draw = ImageDraw.Draw(image)     
        for b in boxes_to_draw:
            points = [(cord['x'], cord['y']) for cord in b]
            for i in range(len(points)):
                draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)
                
        image.save("file_name.png")


def download_image_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")


def upload_image_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"


def draw_bounding_boxes(image_path, bounding_boxes):
    """Draws bounding boxes on the image."""
    with Image.open(image_path) as im:
        draw = ImageDraw.Draw(im)
        for box in bounding_boxes:
            points = [(point['x'], point['y']) for point in box]
            draw.polygon(points, outline='red', width=3)
        im.save(image_path)
        print(f"Bounding boxes drawn on {image_path}")


def process_image_from_gcs(gcs_image_path, bounding_boxes, gcs_output_folder):
    # Parse the GCS path to get the bucket name and the source blob name
    bucket_name = gcs_image_path.split('/')[2]
    source_blob_name = '/'.join(gcs_image_path.split('/')[3:])
   
    # Extract the base name of the image file without extension and excluding the last 7 characters
    base_name_full = os.path.basename(source_blob_name).rsplit('.', 1)[0]
    base_name = base_name_full[:-7]  # Removing the last 7 characters
   
    # Define the local path to save the downloaded image
    local_image_path = f'/tmp/{base_name_full}.png'
   
    # Download the image from GCS to the local path
    download_image_from_gcs(bucket_name, source_blob_name, local_image_path)
   
    # Draw bounding boxes on the downloaded image
    if boxes_to_draw:
        draw_bounding_boxes(local_image_path, bounding_boxes)
   
    # Define the GCS path to save the processed image in the specified folder
    destination_blob_name = f"{gcs_output_folder}/{base_name}/{base_name_full}.png"
   
    # Upload the processed image from the local path to GCS
    gcs_processed_image_path = upload_image_to_gcs(bucket_name, local_image_path, destination_blob_name)
   
    # Return the GCS path of the uploaded processed image
    return gcs_processed_image_path
#########################################################################################################################
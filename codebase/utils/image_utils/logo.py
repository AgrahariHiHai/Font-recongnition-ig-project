import json
import math
import numpy as np
from google.cloud import vision, documentai, documentai_v1 as documentai
from typing import Optional, Sequence
from google.api_core.client_options import ClientOptions
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from google.cloud import storage
import warnings 



ocr_processor_display_name = "ig-ocr" # Must be unique per project, e.g.: "My Processor"
ocr_processor_id = "165d647a387cb584" # Create processor before running sample
ocr_processor_version = "pretrained-ocr-v2.0-2023-06-02" # Optional. Processor version to use

form_processor_display_name = "form-parser-ig"
form_processor_id = "ff445d9f10fdd25e" # Create processor before running sample
form_processor_version = "pretrained-form-parser-v2.0-2022-11-10"


from PIL import Image

# image = Image.open(f'{file_path}')
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

    
def detect_text_and_logos(path):
    # Initialize the client
    client = vision.ImageAnnotatorClient()

    # Read the image file
    with open(path, "rb") as image_file:
        content = image_file.read()

    # Create an image object
    image = vision.Image(content=content)

    # Text detection
    text_response = client.text_detection(image=image)
    texts = text_response.text_annotations
    # print(text_response)
    text_results = [{"text": text.description, "bounds": [{
                    'x': vertex.x, 
                    'y' : vertex.y} for vertex in text.bounding_poly.vertices]} for text in texts]
    
    for i in text_response.full_text_annotation.pages:
        language_results = [{
    "language_code" : i.property.detected_languages[h].language_code,
            "condifence": i.property.detected_languages[h].confidence
    } for h, j in enumerate(i.property.detected_languages)]
    # print(language_results)
    
    # Logo detection
    logo_response = client.logo_detection(image=image)
    logos = logo_response.logo_annotations
    logo_results = [{"logo": logo.description, "bounds": [{
                    'x': vertex.x, 
                    'y' : vertex.y} for vertex in logo.bounding_poly.vertices]} for logo in logos]
   
    
    # Object detection
    object_response = client.object_localization(image=image)
    objects = object_response.localized_object_annotations
    # print(object_response)
    object_result = [{"objects": obj.name, "bounds": [{
                    'x': vertex.x, 
                    'y' : vertex.y} for vertex in obj.bounding_poly.normalized_vertices]} for obj in objects]
    
    
    # Color detection
    color_response = client.image_properties(image=image)
    props = color_response.image_properties_annotation
    # object_result = [{"objects": obj.description, "bounds": [(vertex.x, vertex.y) for vertex in obj.bounding_poly.vertices]} for obj in objects]
    # for color in props.dominant_colors.colors:
    #     print(f"fraction: {color.pixel_fraction}")
    #     print(f"\tr: {color.color.red}")
    #     print(f"\tg: {color.color.green}")
    #     print(f"\tb: {color.color.blue}")
    #     print(f"\ta: {color.color.alpha}")
    color_results=[
    {"fraction": color.pixel_fraction,
    "r" : color.color.red,
     "g" : color.color.green,
     "b" : color.color.blue
    }
        for color in props.dominant_colors.colors
    ]

    # Store results in JSON format
    results = {"text": text_results, "logos": logo_results, "object_detection":object_result, "dominat_colors":color_results, "detected_languages" : language_results}

    # Print JSON string
    # json_str = json.dumps(results, indent=4)
    # print(json_str)

    # If there's an error message in the response, raise an exception
    if text_response.error.message:
        raise Exception(text_response.error.message)
    if logo_response.error.message:
        raise Exception(logo_response.error.message)
        
    return results
    # return text_response

def process_document_ocr_sample(
    project_id,
    location,
    ocr_processor_id,
    ocr_processor_version,
    file_path,
    mime_type,
) -> None:
    # Optional: Additional configurations for Document OCR Processor.
    # For more information: https://cloud.google.com/document-ai/docs/enterprise-document-ocr
    print(ocr_processor_id, ocr_processor_version)
    process_options = documentai.ProcessOptions(
        ocr_config=documentai.OcrConfig(
            enable_native_pdf_parsing=True,
            enable_image_quality_scores=True,
            enable_symbol=True,
            # OCR Add Ons https://cloud.google.com/document-ai/docs/ocr-add-ons
            premium_features=documentai.OcrConfig.PremiumFeatures(
                compute_style_info=True,
                enable_math_ocr=False,  # Enable to use Math OCR Model
                enable_selection_mark_detection=True,
            ),
        )
    )

    document = process_document(
        project_id,
        location,
        ocr_processor_id,
        ocr_processor_version,
        file_path,
        mime_type,
        process_options=process_options,
    )
    
    

    text = document.text

    
    return document

def process_document_form_sample(
    project_id: str,
    location: str,
    form_processor_id: str,
    form_processor_version: str,
    file_path: str,
    mime_type: str,
    # process_options=None
) -> documentai.Document:
    # Online processing request to Document AI
    document = process_document(
        project_id, location, form_processor_id, form_processor_version, file_path, mime_type, process_options=None
    )

    # Read the table and form fields output from the processor
    # The form processor also contains OCR data. For more information
    # on how to parse OCR data please see the OCR sample.

    text = document.text

    return document

def process_document(
    project_id,
    location,
    processor_id,
    processor_version,
    file_path,
    mime_type,
    process_options
) -> documentai.Document:
   
    # You must set the `api_endpoint` if you use a location other than "us".
    
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com"
        )
    )

    name = client.processor_version_path(
        project_id, location, processor_id, processor_version
    )
    
    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()
    # Configure the process request
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=image_content, mime_type=mime_type),
        # Only supported for Document OCR processor
        process_options=process_options,
    )

    result = client.process_document(request=request)

    # For a full list of `Document` object attributes, reference this page:
    # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
    
    return result.document


def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    Document AI identifies text in different parts of the document by their
    offsets in the entirety of the document"s text. This function converts
    offsets to a string.
    """
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    return "".join(
        text[int(segment.start_index) : int(segment.end_index)]
        for segment in layout.text_anchor.text_segments
    )

def _get_text(document,el):
    """Doc AI identifies form fields by their offsets
    in document text. This function converts offsets
    to text snippets.
    """
    response = ''
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in el['textAnchor']['textSegments']:
        # print(segment)
        if segment['startIndex']:
            start_index = segment['startIndex']
        else:
            start_index=0
    
        if segment['endIndex']:
            end_index = segment['endIndex']
        else:
            end_index=0
        
        response += document['text'][int(start_index):int(end_index)]
    return response, start_index, end_index


def open_gcs_image(gcs_path):
    gcs_prefix = 'gs://'
    bucket_name = gcs_path[len(gcs_prefix):].split('/')[0]
    file_p = gcs_path[len(f'{gcs_prefix}{bucket_name}/'):]
    # print(bucket_name)
    # Initialize a client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Get the blob (file) from the bucket
    blob = bucket.blob(file_p)

    # Download the blob to a bytes object
    image_bytes = blob.download_as_bytes()

    # Open the image using Pillow
    image = Image.open(io.BytesIO(image_bytes))

    return image


def is_point_inside_bbox(point, bbox):
    try:
        x, y = point["x"], point["y"]
        x_coords = [p["x"] for p in bbox]
        y_coords = [p["y"] for p in bbox]
        if min(x_coords) <= x <= max(x_coords) and min(y_coords) <= y <= max(y_coords):
            return True
        return False
    except Exception as e:
        return None

def is_bbox1_inside_bbox(main, bbox_list):
    try:
        for bbox in bbox_list:
            # print(bbox_list)
            all_inside = all(is_point_inside_bbox(point, bbox) for point in main)
            if all_inside:
                return True
        return False
    except Exception as e:
        return None

def logo_present_check(res, data):
    try:
        print("Checking if Logo present using function logo_present_check")
        ig_post=0
        ig_list = ["IG", "IG\n", "IG Group"]
        ig_logo_count = 0
        logo_print_statements=[]
        for i in res['logos']:  
            if i['logo'] == "IG" or i['logo'] == "IG Group":
                ig_logo_count += 1
                logo_coordinates = i['bounds']
            else:
                print("Logos detected apart IG", i['logo'])
                    # print("No IG logo")
        if ig_logo_count == 1:
            print("Only one IG Logo present: logo_present_check")
            
            logo_print_statements.append([f"Compliant only one IG logo present",None])
        

        if ig_logo_count >= 2:
            print("More than one IG Logo present: logo_present_check")
            print_state = "More than one IG logo present"
            logo_print_statements.append([f"More than one IG logo present",logo_coordinates])
            
            
        if ig_logo_count == 0:
            print("IG Logo not detected through vision AI, need to post-process: logo_present_check")
            for page in data['pages']:
                for para in page['paragraphs']:
                    el = para[0]['layout']
                    response_para, start_para, end_para = _get_text(data, el)
                    for token in page['tokens']:
                        el = para[0]['layout']
                        response_token, start_token, end_token = _get_text(data, el)
                        if int(start_para) == int(start_token) and int(end_para) == int(end_token) and response_para in ig_list:
                            logo_cordinates = para[0]['layout']['boundingPoly']['vertices']
                            res['logos'].append({"logo": "IG", "bounds": logo_cordinates})
                            logo_print_statements.append([f"Compliant only one IG logo present",None])  
                            ig_logo_count += 1
                            break
                        else:
                            ig_logo_count==0
                            # pass
        if ig_logo_count==0:
            print("IG Logo not detected through vision AI and post-processing: logo_present_check")
            logo_print_statements.append([f"Non-Compliant No IG logo present",None])


        

        print(f"return from logo_present_check; Count of IG logo : {ig_logo_count}, print statement : {logo_print_statements}")
        return ig_logo_count, logo_print_statements, res
    except Exception as e:
        print("Failing in detecting Logo using Vision AI, function logo_present_check", e)

def enlarge_bbox_logo(bbox, scale_factor):
    try:
        center_x = (bbox[0]["x"] + bbox[1]["x"]) / 2
        center_y = (bbox[0]["y"] + bbox[2]["y"]) / 2

        # Determine original width and height
        width = bbox[1]["x"] - bbox[0]["x"]
        height = bbox[2]["y"] - bbox[0]["y"]

        # Calculate new width and height
        new_width = width * scale_factor
        new_height = height * scale_factor

        # Calculate new coordinates for the corners
        new_bbox = [
            {"x": center_x - new_width / 2, "y": center_y - new_height / 2},
            {"x": center_x + new_width / 2, "y": center_y - new_height / 2},
            {"x": center_x + new_width / 2, "y": center_y + new_height / 2},
            {"x": center_x - new_width / 2, "y": center_y + new_height / 2}
        ]


        return new_bbox
    except Exception as e:
        print("Logo Enlargement of Coordinates by 30%, function enlarge_bbox_logo", e)

def check_overlap(res, new_logo):
    try:
        overlapping = []
        if res['text'] is not None or res['objects'] is not None:
            for i in res['text']:
                for point in i['bounds']:
                    if new_logo[0]['x'] <= point['x'] <= new_logo[1]['x'] and \
                        new_logo[0]['y'] <= point['y'] <= new_logo[3]['y']:
                        overlapping.append(i)
            for i in res['object_detection']:
                for point in i['bounds']:
                    if new_logo[0]['x'] <= point['x'] <= new_logo[1]['x'] and \
                        new_logo[0]['y'] <= point['y'] <= new_logo[3]['y']:
                        overlapping.append(i)
        else:
            return None
        return overlapping
    except Exception as e:
        print("Post Enlargement of Logo box, failing during check of any overlapping objects or text present at the space of 30%, function check_overlap", e)


def extract_dominant_colors(image, region_coordinates, num_colors=2):
    try:
        print("Extracting Dominat colors for Overline")
        # Convert PIL Image to numpy array
        image_array = np.array(image)

        # Define the bounding box of the region
        x_min = min(x for x, y in region_coordinates)
        y_min = min(y for x, y in region_coordinates)
        x_max = max(x for x, y in region_coordinates)
        y_max = max(y for x, y in region_coordinates)
        # print(x_min, y_min, x_max, y_max)
        # Crop the region from the image
        region = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Convert the region to numpy array
        region_array = np.array(region)

        # Check the shape of the region array
        # # print("Region array shape:", region_array.shape)
        # print(image.size)
        try:
            pixels = region_array[:, :]
            print("Pixels array shape before reshape 1:", pixels.shape)
            pixels = pixels.reshape((-1, 3))

        except Exception as e:
            try:
                pixels = region_array[:, :, 3]

                num_pixels = pixels.shape[0] // 3 * 3

                pixels = pixels[:num_pixels]
                print("Pixels array shape before reshape 2:", pixels.shape)

                # Reshape the region into a 2D array of pixels
                pixels = pixels.reshape((-1, 3))

            except Exception as e:
                pixels = region_array[:, :]

                num_pixels = pixels.shape[0] // 3 * 3

                pixels = pixels[:num_pixels]
                print("Pixels array shape before reshape 3:", pixels.shape)

                    # Reshape the region into a 2D array of pixels
                pixels = pixels.reshape((-1, 3))


            # Check the shape of the pixels array after reshaping
        # print("Pixels array shape after reshape:", pixels.shape)

        # Perform K-means clustering to find dominant colors
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        print("Dominat colors extracted :", dominant_colors)

        return dominant_colors
    except Exception as e:
        print("Failed to extract 2 dominat color for a given coordinates function extract_dominant_colors", e)
        return None


def classify_color(color):  
    try:
        red_range = [200, 255, 0, 80, 0, 80]  # Red: R: 200-255, G: 0-80, B: 0-80
        white_range = [240, 255, 240, 255, 240, 255]  # White: R, G, B: 240-255
        black_range = [0, 50, 0, 50, 0, 50]  # Black: R, G, B: 0-50

        if (red_range[0] <= color[0] <= red_range[1]) and (red_range[2] <= color[1] <= red_range[3]) and (red_range[4] <= color[2] <= red_range[5]):
            return "Red"
        elif (white_range[0] <= color[0] <= white_range[1]) and (white_range[2] <= color[1] <= white_range[3]) and (white_range[4] <= color[2] <= white_range[5]):
            return "White"
        elif (black_range[0] <= color[0] <= black_range[1]) and (black_range[2] <= color[1] <= black_range[3]) and (black_range[4] <= color[2] <= black_range[5]):
            return "Black"
        else:
            return "Other"
    except Exception as e:
        print("Function where range of pixel values present for Red, White, Black", e)
        return None

def logo_color_check(file_path, res):
    try:
        print("Checking if Logo colors detected using function logo_color_check")

        region_coordinates=[]
        # Open the image
        image = Image.open(f'{file_path}')
        if res['logos']:
            for i in res['logos']: 
                if i['logo'] == "IG" or i['logo'] == "IG Group":
                    # print("hi")
                    logo_cordinates= i['bounds']
                    region_coordinates=[(coord['x'], coord['y']) for coord in logo_cordinates]
                # else:
                #     return None
        else:
            return None


        # Extract dominant colors
        dominant_colors = extract_dominant_colors(image, region_coordinates, num_colors=2)

        print("Dominant Colors in Logo:")
        for color in dominant_colors:
            print(color)


        classified_colors = []
        for color in dominant_colors:
            classified_colors.append(classify_color(color))

        print("Classified Colors for Logo:")
        for i, color in enumerate(dominant_colors):
            print(f"Color {i + 1}: {color} - {classified_colors[i]}")
            
        print("returing classified colors using function logo_color_check: ", classified_colors)
            
        return classified_colors
    except Exception as e:
        print("Exception during logo color checks, logo_color_check, identifying all colors present in logo bbox", e)
        return None
    
    
def logo_font_color(data, res):
    try:
        print("Logo Font color check using function: logo_font_color")

        over=[]
        text_color=None
        red_range = [200, 255, 0, 80, 0, 80]  # Red: R: 200-255, G: 0-80, B: 0-80
        white_range = [240, 255, 240, 255, 240, 255]  # White: R, G, B: 240-255
        black_range = [0, 50, 0, 50, 0, 50]  # Black: R, G, B: 0-50
        coordinates_for_logo_color=[]

        for page in data['pages']:
            for token in page['tokens']:
                # for token in tokens:
                new_logo = token['layout']['boundingPoly']['vertices']
                el= token['layout']
                response_token, token_start_index, token_end_index=_get_text(data, el)
                # print(token_start_index, token_end_index)
                # print(type(response_token))
                # print(response_token)

                if 'IG\n' == response_token:
                    for i in res['logos']:
                        if i['logo'] == "IG" or i['logo'] == "IG Group":
                            for point in i['bounds']:
                                if new_logo[0]['x'] <= point['x'] <= new_logo[1]['x'] and \
                                        new_logo[0]['y'] <= point['y'] <= new_logo[3]['y']:
                                    print()
                                over.append(token)
                                coordinates_for_logo_color = i['bounds']

                                c=[token['styleInfo']['textColor']['red'], token['styleInfo']['textColor']['green'],token['styleInfo']['textColor']['blue']]

                                # print(color)
                                color=[]
                                r = int(c[0] * 255)
                                g = int(c[1] * 255)
                                b = int(c[2] * 255)
                                color.append([r , g, b])

                                text_color=classify_color(color[0])
                        print("Final text color detected for IG Logo", text_color)
        print("Returning ig font color using logo_font_color", text_color)
        return text_color

    except Exception as e:
        print("Exception during final check for identifying font and background check for Logo, function logo_font_color", e)
        return None




def logo_margin_alignment(file_path, res):
    try:
        print("Checking Logo alignment and position using function logo_margin_alignment")
        distance = None
        print_state=None
        status=None
        image = Image.open(file_path)

        # Determine the shorter edge
        shorter_edge = min(image.width, image.height)

        # Calculate the length of margins (5% of the shorter edge)
        margin_length = int(shorter_edge * 0.05)

        for i in res['logos']: 
            if i['logo'] == "IG" or i['logo'] == "IG Group":
                coordinates_json = i['bounds']
            # else:
            #     return None

        top_margin_y = margin_length
        bottom_margin_y = image.height - margin_length
        left_margin_x = margin_length
        right_margin_x = image.width - margin_length

        # Find the top-most y-coordinate and bottom-most y-coordinate of the logo
        top_most_y = min(point["y"] for point in coordinates_json)
        bottom_most_y = max(point["y"] for point in coordinates_json)

        # Find the left-most x-coordinate and right-most x-coordinate of the logo
        left_most_x = min(point["x"] for point in coordinates_json)
        right_most_x = max(point["x"] for point in coordinates_json)

        # Calculate the distances between logo edges and margin lines
        distance_top = abs(top_most_y - top_margin_y)
        distance_bottom = abs(bottom_margin_y - bottom_most_y)
        distance_left = abs(left_most_x - left_margin_x)
        distance_right = abs(right_margin_x - right_most_x)

        print("Distance between logo's top line and top margin line:", distance_top)
        print("Distance between logo's bottom line and bottom margin line:", distance_bottom)
        print("Distance between logo's left line and left margin line:", distance_left)
        print("Distance between logo's right line and right margin line:", distance_right)
        


        if -5 <= distance_right <= 5 and  -5 <= distance_bottom <= 5:
            print("Compliant Logo correctly placed at Left Bottom")
            status = "Compliant"
            print_state="Compliant Logo correctly placed at Left Bottom"
        elif -5 <= distance_left <= 5 and  -5 <= distance_bottom <= 5:
            print("Compliant Logo correctly placed at Right Bottom")
            status = "Compliant"
            print_state="Compliant Logo correctly placed at Right Bottom"
        elif distance_left == distance_right and  -5 <= distance_bottom <= 5:
            print("Compliant Logo correctly placed at Bottom Centre")
            status = "Compliant"
            print_state="Compliant Logo correctly placed at Bottom Centre"
        elif distance_top == distance_bottom and distance_left == distance_right:
            print("Compliant Logo correctly placed at centre of the image")
            status = "Compliant"
            print_state="Compliant Logo correctly placed at centre of the image"
        elif -5 <= distance_bottom <= 5 and distance_left == distance_right:
            print("Compliant Logo correct places at Bottom Centre")
            status = "Compliant"
            print_state="Compliant Logo correct places at Bottom Centre"
        elif distance_bottom > distance_right and -5 <= distance_right <=8:
            print("Non-Compliant for logo; because bottom of Logo is higher from margin")
            status = "Non-Compliant"
            print_state="Non-Compliant for logo; because bottom of Logo is higher from margin"
        elif distance_bottom > distance_left and -5 <= distance_left <=8:
            print("Non-Compliant for logo; because bottom of Logo is higher from margin")
            status = "Non-Compliant"
            print_state = "Non-Compliant for logo; because bottom of Logo is higher from margin"
        elif distance_right < 5 and distance_bottom < 5:
            print("Non-Compliant for logo; Adjust the logo position")
            status = "Non-Compliant"
            print_state="Non-Compliant for logo; Adjust the logo position"
        elif distance_left < 5 and distance_bottom < 5:
            print("Non-Compliant for logo; Adjust the logo position")
            status = "Non-Compliant"
            print_state="Non-Compliant for logo; Adjust the logo position"
        else:
            print("Non-Compliant for logo; Adjust the logo position")
            status = "Non-Compliant"
            print_state="Non-Compliant for logo; Adjust the logo position"
        if distance_left <= 8 or distance_right <= 8:
            if distance_left<=8:
                distance = distance_left
            else:
                distance = distance_right
        else:
            print("No distance detected for Logo")
            distance = None
        print("returning distance ")
        return print_state, status, distance
    except Exception as e:
        print("Exception in detecting Logo and margin line", e)
        return None, None, None

def logo_size(file_path, res):
    try:
        print("Checking Logo size using function logo_size")
        print_state=None
        status=None
        image = Image.open(file_path)

        # Determine the shorter edge
        shorter_edge = min(image.width, image.height)

        # Calculate the length of margins (5% of the shorter edge)
        margin_length = int(shorter_edge * 0.05)


        def calculate_distance(x1, y1, x2, y2):
            # return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) 
            # return math.dist([x2], [x1])
            return x2 - x1 
        if res['logos']:
            for i in res['logos']: 
                if i['logo'] == "IG" or i['logo'] == "IG Group":
                    coordinates_json=i['bounds'] 
                    logo_size = calculate_distance(coordinates_json[0]['x'], coordinates_json[0]['y'], coordinates_json[1]['x'], coordinates_json[1]['y'])

            # print("Logo size:", logo_size)
            small_logo_size = margin_length * 3.5
            # print("Margin", margin_length)

            # print("Actual Logo size:", logo_size)
            # print("Expected Logo szie:", small_logo_size)
            if small_logo_size-15 <= logo_size and logo_size <= small_logo_size+15:
                status = "Compliant"
                print_state = f"Complaint Size of the Logo is Correct"
            else:
                status = "Non-Compliant"
                if small_logo_size > logo_size:
                    print("Non-Compliant for logo size: Actual Logo size: {logo_size} Expected Logo szie: {small_logo_size} Increase the logo size")
                    print_state=f"Non-Compliant for logo size: Actual Logo size: {logo_size} Expected Logo szie: {small_logo_size} Increase the logo size"
                else:
                    print(f"Non-Compliant for logo size: Actual Logo size: {logo_size} Expected Logo szie: {small_logo_size} Decrese the logo size")
                    print_state=f"Non-Compliant for logo size: Actual Logo size: {logo_size} Expected Logo szie: {small_logo_size} Decrese the logo size"

                    # Logo size: 148
        else:
            print_state = None
            status=None
            print("No IG logo")
        return print_state, status
    except Exception as e:
        print("Exception during detecting logo size", e)
        return None, None


def final_logo_print_statement(res, data, file_path):
    logo_print_statements = []
    ig_logo_count, logo_print_statements, res= logo_present_check(res, data)
    if ig_logo_count == 1:
        for i in res['logos']: 
            if i['logo'] == "IG" or i['logo'] == "IG Group":
                if ig_logo_count == 1:
                    new_logo = enlarge_bbox_logo(i['bounds'], 1.3)
                    overlapping = check_overlap(res, new_logo)
                    overlapp = check_overlap(res, new_logo)

                    # deleting IG 
                    index = 0

                    while index < len(overlapping):
                        if 'IG' == overlapping[index]['text']:
                            del overlapping[index]
                        else:
                            index += 1

                    coordinates_for_30_clear_space=[]
                    if len(overlapping) == 1 or len(overlapping) == 0:
                        status = "Compliant"
                        logo_print_statements.append([f"Compliant, no objects are present and there is 30% clear space around logo",None])

                    else:
                        status = "Non-Compliant"
                        for i in overlapping[1:]:
                            coordinates_for_30_clear_space.append(i['bounds'])

                        logo_print_statements.append(["Non-Compliant, as objects are present and there is no 30% clear space around logo",coordinates_for_30_clear_space])

                    classified_colors = logo_color_check(file_path, res)
                    # print(type(classified_colors))
                    logo_text_color = logo_font_color(data, res)

                    must=['Red', 'White', 'Black']

                    if classified_colors != None and logo_text_color !=None:
                        background_color_logo = None
                        for color in classified_colors:
                            if color != logo_text_color:
                                background_color_logo = color
                                break

                        if background_color_logo in must and logo_text_color in must:

                        # if background_color:
                            # print("background_color:", background_color_logo)
                            # print("Font_color", logo_text_color)

                            if logo_text_color != background_color_logo:
                                status="Compliant"
                                logo_print_statements.append([f"Compliant! BG color: {background_color_logo} and IG logo color {logo_text_color}",None])
                            else:
                                status = "Non-Compliant"
                                # print("Invalid combination: Text color cannot be the same as background color.", coordinates_for_logo_color)
                                logo_print_statements.append([f"Non-Compliant! BG color: {background_color_logo} and IG logo color {logo_text_color}",i['bounds']])
                        else:
                            logo_print_statements.append([f"Non-Compliant! BG color: {background_color_logo} and IG logo color {logo_text_color}",i['bounds']])

                    else:
                        # return coordinates_for_logo_color
                        print("Error: No background color found in classified_colors.", coordinates_for_logo_color)



                    print_state_align_logo, status_align_logo, distance_logo_rw = logo_margin_alignment(file_path, res)
                    if status_align_logo == "Compliant":
                        # "Logo, Compliant")
                        logo_print_statements.append([print_state_align_logo,None])

                    else: 
                        status_align_logo == "Non-Compliant"
                        print("Non-Compliant Alignment because bottom of Logo is higher from margin")
                        logo_print_statements.append([print_state_align_logo,i['bounds']])



                    print_state_size_logo, status_for_logo_size = logo_size(file_path, res)
                    if status_for_logo_size == "Compliant":
                        print("Logo Size is Compliant")
                        logo_print_statements.append([print_state_size_logo,None])
                    elif status_for_logo_size == "Non-Compliant":
                        print("Logo Size is Non-Compliant")
                        
                        logo_print_statements.append([print_state_size_logo,i['bounds']])

                        # print(distance_logo_rw)
                        pass
                    else:
                        logo_text_color = None
                        distance_logo_rw = None
                        print("No-Logo")
    else:
        logo_text_color = None
        distance_logo_rw = None

    return logo_print_statements, logo_text_color, distance_logo_rw, ig_logo_count
import cv2
import numpy as np
import io
import os
from google.cloud import aiplatform
from config import ENV_CONFIG

project = ENV_CONFIG['endpoint_project_id']
endpoint_id = ENV_CONFIG['endpoint_id']
network_name = ENV_CONFIG["endpoint_network_name"]
endpoint_name = f"projects/{project}/locations/europe-west2/endpoints/{endpoint_id}"
endpoint = aiplatform.PrivateEndpoint(endpoint_name=endpoint_name)


def check_Font_type_compliance(image_path, result):
    """ Returns the final compliance chcek againts Font Type
            inputs:
                image: image in png of jpeg format
                bounding_boxes : list of tuples containg first and third cordinate for each word. eg- [(x1, y1, x2, y2), (x1, y1, x2, y2)]
            outputs:
                full image.
                text compliant against Font-Type/ Non-Compliant against Font Type.
                Text Patches with non-Complaint prediction score
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #response = get_response_from_vision(image_path)
    # all_vertices_around_text = get_all_vertices(response)
    # print(all_vertices_around_text)
    
    #bounding_boxes =  get_bounding_boxes(all_vertices_around_text)
    bounding_boxes = [i["BoundingBox"] for i in result]
    subimages_list = capture_subimages(image, bounding_boxes)
    #print(len(subimages_list))
    
    for i, sub_img in enumerate(subimages_list):
        print("%" *10, type(sub_img))
        if sub_img is not None:
            
            preprocessed_patch = preprocess_unseen_image(sub_img)

            pred_result, pred_score = model_prediction(preprocessed_patch)
        else:
            pred_result = "Compliant Font Type"
            pred_score = {}
        print()
        print("!" *20)
        print(pred_result, pred_score)

        result[i].update({"prediction": pred_result, "prediction_score": pred_score})

    return result        

def get_bounding_boxes(b_box_list):
    print("***********getting the BoundingBox******************")
    x1 , y1 = b_box_list[0]['x'] , b_box_list[0]['y']
    x2 , y2 = b_box_list[2]['x'] , b_box_list[2]['y']
    print("***********BoundingBox extracted!******************")
    return (x1, y1, x2, y2)

def capture_subimages(image , bounding_boxes):
    subimage_list = []
    # subimage =[]
    print("**********patch cropping strated based on BoundingBoxes*****************")
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        if (x2-x1 >=105) and (y2-y1>32) and (y2-y1 <= 128):
            subimage = image[y1-6:y2+6, x1-6:x2+6]
            subimage_list.append(subimage)
        else:
            subimage_list.append(None)
    print("********* *******text patch cropped successfully!***********************")
    return subimage_list

def preprocess_unseen_image(image):
    print("***********image preprocessing started***************")
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized_img = resize_with_padding(gray_image , target_size = (128, 512))
    normalized_img = normalize_image(resized_img)
    input_image = np.expand_dims(normalized_img, axis = -1)
    print("********************preprocessing done!***************")
    return input_image  

def normalize_image(image):
    normalized_img = image/255.0
    return normalized_img

def resize_with_padding(image, target_size=(128, 512)):
    """
    Resizes an image using padding to preserve its aspect ratio.
   
    Args:
    image: The input image in grayscale format.
    target_size: A tuple (height, width) specifying the target size.

    Returns:
    A resized image with padding to fit the target size.
    """
    # Calculate the original aspect ratio
    original_height, original_width = image.shape
    original_aspect_ratio = original_width / original_height

    # Calculate the target aspect ratio
    target_height, target_width = target_size
    target_aspect_ratio = target_width / target_height

    # Determine the new dimensions of the image
    if original_aspect_ratio > target_aspect_ratio:
        # Fit to width
        new_width = target_width
        new_height = int(target_width / original_aspect_ratio)
    else:
        # Fit to height
        new_height = target_height
        new_width = int(target_height * original_aspect_ratio)
   
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
   
    # Calculate padding
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top
    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left
   
    # Add padding to the resized image
    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
   
    return padded_image

def model_prediction(preprocessed_image): 
    try:
        instances = [preprocessed_image.tolist()]
        print("*****************loading endpoint for prediction********************")
        p = endpoint.predict(instances=instances)[0][0]
        print("*****************got the prediction successfully!********************")
    
        for i in range(len(p)):
            p[i] = round(p[i], 4)
        labels = ["Bad", "NonSentenceCase", "Good_1", "Good_2", "Bad_3", "Bad_2" ]
        result = {'Bad': p[0],"NonSentenceCase": p[1],'Good_1': p[2], 'Good_2': p[3], 'Bad_3': p[4], 'Bad_2': p[5]}
        result_m = np.argmax(np.array(p))
        

        if labels[result_m] in ("Good_1", "Good_2"):
            final = "Compliant Font Type"
        else:
            final = 'Recheck Font Type'
        
    except Exception:
        print("Error during font checker model prediction")
        traceback.print_exc()
        result = {}
        final = ""

    return final, result


# for each word in - headline and overline
# result = [{Typography :'headline/overline', text: 'Share', prediction: 'comp/non-comp', prediction_score:{'bad_':.50, 'good': 0.60, 'good_ii':.20}, bb: (xmin, ymin, xmax, ymax)},
#  {Typography :'headline/overline', text: 'market', prediction: 'comp/non-comp', prediction_score:{'bad_':.50, 'good': 0.60, 'good_ii':.20}, bb: (xmin, ymin, xmax, ymax)}]

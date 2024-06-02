import re
import cv2
from collections import Counter
from PIL import Image, ImageDraw
from utils.image_utils.logo import *
from utils.font_checker import *
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel

# https://cloud.google.com/document-ai/docs/samples/documentai-process-form-document
def check_if_table(form_doc):
    try:
        text = form_doc.text
        for page in form_doc.pages:
    #         print(f"\n\n**** Page {page.page_number} ****")

    #         print(f"\nFound {len(page.tables)} table(s):")
            if len(page.tables) > 1:
                # print("g")
                return True
            else:
                for table in page.tables:
                    num_columns = len(table.header_rows[0].cells)
                    num_rows = len(table.body_rows)
                    # print(f"Table with {num_columns} columns and {num_rows} rows:")
                    if num_columns and num_rows >= 3:
                        # print("h")
                        return True
                    else:
                        # print("i")
                        return False
    except Exception as e:
        print("Failed during Doc-Form parser table check for alignment", e)
        return None
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

def font_threshold(data):  
    try:
    
        max_font_size = 0
        for page in data['pages']:
            for token in page['tokens']:
                # for token in tokens:
                el = token['layout']
                response_line, line_start_index, line_end_index= _get_text(data, el)
                # print(response_line)
                size = token['styleInfo']['fontSize']
                # print(size)
                if size > max_font_size and response_line != "IG\n":
                    max_font_size=size
                    print(max_font_size)

        if max_font_size >= 40:
            threshold = 10
        if max_font_size >= 30:
            threshold = 8
        elif max_font_size >= 20:
            threshold = 5
        elif max_font_size >= 15:
            threshold = 3
        elif max_font_size >= 10:
            threshold = 2     
        elif max_font_size >= 5:
            threshold=1
        else:
            threshold=0

        return threshold
    except Exception as e:
        print("Failed while deciding the threshold value for grouping the text", e)
        return None


def grouping_tokens(data, threshold):    
    try:
        for page in data['pages']:
            for line in page['lines']:
                # print(line[0])

                el = line[0]['layout']
                # print(el)
                response_line, line_start_index, line_end_index=_get_text(data, el)

                first_token = None
                last_token = None
                grouped_tokens = []
                current_group = []
                for i, token in enumerate(page['tokens']):
                    # print(token['styleInfo']['fontSize'])
                    if i == 0:
                        current_group.append(token)
                    else:
                        prev_token = page['tokens'][i - 1]

                        if abs(token['styleInfo']['fontSize'] - prev_token['styleInfo']['fontSize']) <= threshold:
                            current_group.append(token)
                        else:
                            grouped_tokens.append(current_group)
                            current_group = [token]

                # Add the last group
                if current_group:
                    grouped_tokens.append(current_group)
        return grouped_tokens
    except Exception as e:
        print("Failed during Grouping the tokens using the threshold received", e)
        return None

def validate_headline_text(text):
    try:
        status = False
        pattern = r'^([-+*/%]|\+?\-?\d+(\.\d+)?[A-Z]{0,2}%?[-+*/%]?)$'
        pattern_1 = r'\b(1[0-9]{3}|2[0-9]{3})\b'
        if re.match(pattern, text):
            status = True
            if re.match(pattern_1, text):
                status = False
        else:
            status = False
            return False
        return status
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

def max_font_size_decision(data, grouped_token, res, bbox_50_precent, enlarged_bbox_list):
    try:
        print("Deciding Max font size / Headline or to Crop the image")
        max_font_size_group = None
        max_font_size = 0
        extra_logo_overlap=[]
        for group_id, group in enumerate(grouped_token):
            for i in res['logos']:
                el = group[0]['layout']
                response_line, line_start_index, line_end_index=_get_text(data, el)
                token_bounding = group[0]['layout']['boundingPoly']['vertices']
                new_font = i['bounds']
                for point in token_bounding:
                    if new_font[0]['x'] <= point['x'] <= new_font[1]['x'] and \
                            new_font[0]['y'] <= point['y'] <= new_font[3]['y']:
                        print("All response lines", response_line)
                        extra_logo_overlap.append(response_line)
        head_line_regex_check=[]
        for group_id, group in enumerate(grouped_token):
                group_font_size_sum = sum(token['styleInfo']['fontSize'] for token in group)
                avg_group_font_size = group_font_size_sum / len(group)
                el = group[0]['layout']
                response_line, line_start_index, line_end_index=_get_text(data, el)
                bbox1 = group[0]['layout']['boundingPoly']['vertices']
                overlapping_with_logo= is_bbox1_inside_bbox(bbox1, enlarged_bbox_list)
                after_cropping=False
                if bbox_50_precent: #{'x': 45, 'y': 368}, {'x': 930, 'y': 70}, {'x': 1040, 'y': 396},
                    print("50 % cropping applicable")
                    after_cropping=False
                    bbox_check_token_for_50=[(coord['x'], coord['y']) for coord in bbox1]
                    x_min = min(x for x, y in bbox_check_token_for_50)
                    x_max = max(x for x, y in bbox_check_token_for_50)
                    y_min = min(y for x, y in bbox_check_token_for_50)
                    y_max = max(y for x, y in bbox_check_token_for_50)
                    bbox_50 = [(coord['x'], coord['y']) for coord in bbox_50_precent]
                    x_min_50 = min(x for x, y in bbox_50)
                    x_max_50 = max(x for x, y in bbox_50)
                    y_min_50 = min(y for x, y in bbox_50)
                    y_max_50 = max(y for x, y in bbox_50)
                    # print(x_min_50,x_min, x_max, x_max_50, y_min_50, y_min, y_min, y_max_50)
                    if x_min_50 <= x_min and x_max <= x_max_50 and y_min_50 <= y_min and y_min <= y_max_50:
                        print("Falling under the 50% cropped image", response_line)
                        after_cropping=True
                    else:
                        after_cropping=False
                # print(after_cropping)
                                                                             
                regex_check = validate_headline_text(response_line.strip())
                if regex_check == True:
                    # print("Regex check true for: ", response_line.strip())
                    head_line_regex_check.append(group[0])
                if after_cropping == False and overlapping_with_logo == True:
                    print("Word overlapping with Logo and No 50% cropping for word", response_line, "Its size is", avg_group_font_size)
                    pass
                if after_cropping == True:
                    print("Word Present in that 50% cropped region", response_line, "Its size is", avg_group_font_size)
                    pass
                else:
                    print("Temp Headline :",response_line,  "size :" , avg_group_font_size)
                    if avg_group_font_size > max_font_size and regex_check != True  and response_line != "IG\n":
                        
                        max_font_size = avg_group_font_size
                        max_font_size_group = group_id
                        # print(max_font_size_group)
                        # print(response_line)
        print(f"Max font size decided : {max_font_size}")
        return max_font_size, max_font_size_group, head_line_regex_check
    except Exception as e:
        print("Failed during max_font_size_decision function", e)
        return None

def check_issupper(data, typography):
    try:
        status_list=[]
        issupper_check_for_typo=[]
        final_status = False
        if typography != None:
            for typography_word in typography:
                el = typography_word['layout']
                response_line, line_start_index, line_end_index=_get_text(data, el)
                strings = response_line.isupper()
                numeric = response_line.strip().isdigit()
                if numeric:
                    # print("Compliant Numeric")
                    status = True
                    status_list.append(status)

                elif strings:
                    # print("Compliant CAPS Compliant")
                    status = True
                    status_list.append(status)

                else:
                    # print("Non-Compliant All not in CAPS")
                    status = False
                    status_list.append(status)

                if all(element == True for element in status_list):
                    final_status = True
                else:
                    final_status = False 
        else:
            final_status = False
        return final_status, issupper_check_for_typo
    except Exception as e:
        print("Exception occured while checking if the passed typography is in all 'CAPS'", e)
        return None

def font_style_check(typography, file_path, data, type_):
    try:
        result  = []
        for i in typography:
            print(i)
            el = i['layout']
            text, s, e = _get_text(data, el)
            bbox = i['layout']['boundingPoly']['vertices']
            
            temp = {'text': text,
                    'Typography': type_,
                    'BoundingBox': get_bounding_boxes(bbox)
                   }

            result.append(temp)
        #get_bbox=get_bounding_boxes(bbox_all_vertices)
        
        retur = check_Font_type_compliance(file_path, result)
        print(retur)
        def check_majority(lst):
            counts = Counter(lst)
            maj_elem = max(counts, key=counts.get)
            print(counts, maj_elem, len(lst))
            if counts[maj_elem] > len(lst)/2:
                return maj_elem
            elif counts[maj_elem] == len(lst) / 2:
                return "Compliant Font Type"
            else:
                return None

        maj_elem = check_majority([i["prediction"] for i in retur])
        print("*" *100)
        print(maj_elem, retur)
        print("*" *100)
        return maj_elem, retur
    except Exception as e:
        print("Exception while looping font style", e)
        return None, []

def typography_bold(typography):
    try:
        bold=[]
        status = False
        for typo_word in typography:
            typography_value = typo_word['styleInfo']['bold']
            bold.append(typography_value)

        if all(element == True for element in bold):
            # print("Blodness true")
            status = True
        else:
            # print("Boldness False")
            status = False
        return status
    except Exception as e:
        print("Exception occured while checking if the passed typography is in all 'BOLD'", e)
        return None

def typography_text_bg_color(file_path, data, typography):
    try:
        image = Image.open(file_path)
        classified_colors = []
        text_color_for_typography = []

        for page in data['pages']:
            for typo_word in typography:   
                el= typo_word['layout']
                response_token, token_start_index, token_end_index=_get_text(data, el)
                # token_bounding_poly = [
                #                 {"x": point.x, "y": point.y} for point in typo_word['layout']['boundingPoly']['vertices']
                #             ]

                region_coordinates=[(coord['x'], coord['y']) for coord in typo_word['layout']['boundingPoly']['vertices']]
                # print(region_coordinates)

                dominant_colors = extract_dominant_colors(image, region_coordinates, num_colors=2)
                # print(dominant_colors)

                c=[typo_word['styleInfo']['textColor']['red'], typo_word['styleInfo']['textColor']['green'],typo_word['styleInfo']['textColor']['blue']]
                # print(color)
                r = int(c[0] * 255)
                g = int(c[1] * 255)
                b = int(c[2] * 255)



                text_color=classify_color([r,g,b])
                text_color_for_typography.append(text_color)


                for color in dominant_colors:
                    classified_colors.append(classify_color(color))



        return classified_colors, text_color_for_typography
    except Exception as e:
            print("Exception occured while typography_text_bg_color function", e)
            return None

def typography_alignement(data, typography, image):
    try:
        distance_l=None
        distance_r=None
        shorter_edge = min(image.width, image.height)
        margin_length = int(shorter_edge * 0.05)
        #     bottom_margin_y = image.height - margin_length

        #     bottom_most_y = max(point["y"] for point in coordinates_json)
        image_height = image.height

        distance_final_typography={}

        if headline != None:
            counter = 1
            # print("1")
            for page in data['pages']:
                for line in page['lines']:
                    el = line[0]['layout']
                    response_line, line_start_index, line_end_index=_get_text(data, el)
                    # print(response_line,line_start_index)


                    for typo_word in typography:
                        # counter = 1
                        el= typo_word['layout']
                        response_token, token_start_index, token_end_index=_get_text(data, el)
                        # print(response_token)
                        first_token = None
                        last_token = None


                        # print(token_start_index,line_start_index)
                        if int(token_start_index) == int(line_start_index) and response_line != "IG\n" and response_line != "IG" and response_line != "IG Group":
                            # print(token_start_index, line_start_index)
                            if first_token is None:
                                first_token = typo_word
                                el= first_token['layout']
                                response_first, start_index, end=_get_text(data, el)
                                # print(response_first)
                                first_token_bounding_poly= first_token['layout']['boundingPoly']['vertices']

                                distance_l = abs(margin_length - first_token_bounding_poly[0]["x"])
                        # print(response_token)
                        # print(token_end_index, line_end_index)
                        if int(token_end_index) == int(line_end_index) and response_line != "IG\n" and response_line != "IG" and response_line != "IG Group":
                            if last_token is None:


                                last_token = typo_word
                                el= last_token['layout']
                                response_last, start_index, end=_get_text(data, el)
                                # print(response_last)
                                last_token_bounding_poly= last_token['layout']['boundingPoly']['vertices']

                                distance_r = (image.width - margin_length) - last_token_bounding_poly[1]["x"]

                                final_distance = abs(distance_l - distance_r)
                                        # final.append(final_distance)
                                distance_final_typography[counter] = {"left": distance_l, 
                                                          "right": distance_r,
                                                          "final": final_distance,
                                                          "word" : response_line,
                                                                     "co-ordinates" : line[0]['layout']['boundingPoly']['vertices']}

                                counter += 1

        else:
            return None

        return distance_final_typography
    except Exception as e:
        print("Exception occurred while Typography alignment, mostly due to no start-index", e)
        return None


def alignment(distance_final):
    try:
        def check_majority(lst):
            counts = Counter(lst)
            maj_elem = max(counts, key=counts.get)
            if counts[maj_elem]>len(lst)/2:
                return maj_elem
            # elif counts[maj_elem] == len(lst)/2:
            #     return 
            else:
                return None

        if distance_final != None:
            threshold = 10
            align = []
            value_join=[]
            for key, value in distance_final.items():
                value_join.append(value['word'])
                if value['final'] >= threshold and value['left'] < value['right']:
                    # print("Left------>", value['word'])
                    align.append("Left")
                elif value['final'] <= threshold:
                    # print("Center---->", value['word'])
                    align.append("Center")
                elif value['final'] >= threshold and value['left'] > value['right']:
                    # print("Right----->", value['word'])
                    align.append("Right")

                else:
                    print("Invalid")
                    return None
            print(align)
            maj_elem=check_majority(align)
            # print("Final Sentence", " ".join(value_join))

            if len(align) >= 1:
                if all(element == align[0] for element in align): # or chec_majority(align) :
                # if  chec_majority(align) :
                    print(f"All are same Aligned for deciding Alignment for sentence", " ".join(value_join), " and it is" ,align[0])
                    return align[0]
                elif maj_elem:
                    print(f"Taking Majority for deciding Alignment for sentence", " ".join(value_join), "and it is", maj_elem)
                        # print("mismatch")
                    return maj_elem
                else:
                    return None
    except Exception as e:
        print("Exception occured while checking Typography alignment post finding distance", e)
        return None


def enlarge_bbox_list(bbox_list, scale_factor):
    try:
        enlarged_bbox_list = []
        for bbox in bbox_list:
            # Calculate center point of the bounding box
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

            enlarged_bbox_list.append(new_bbox)

        return enlarged_bbox_list
    except Exception as e:
        print("Exception occured while Enlarging the bounding boxes for text", e)
        return None

def typo_output(data, bbox_50_precent, res):
    threshold = font_threshold(data)
    grouped_token= grouping_tokens(data, threshold)
    # all_logos=[]
    bbox_list=[]
    for i in res['logos']:
        # all_logos.append(i['logo'])
        bbox_list.append(i['bounds'])
    
    scale_factor = 1.3

    # Enlarge bounding box list
    enlarged_bbox_list = enlarge_bbox_list(bbox_list, scale_factor)
    
    max_font_size, max_font_size_group,head_line_regex_check = max_font_size_decision(data, grouped_token, res, bbox_50_precent, enlarged_bbox_list)
    headline = grouped_token[max_font_size_group]

    headline_line=[]
    for token in headline:
        el=token['layout']
        rh, sh, eh = _get_text(data, el)
        for page in data['pages']:
            for line in page['lines']:
                el = line[0]['layout']
                rl, sl, el = _get_text(data, el)
                if int(sh) == int(sl):
                    headline_line.append(line[0])
                    fhrl = rl

    print("Final Headline")
    for i in headline_line:
        el = i['layout']
        rh, sh, eh = _get_text(data, el)
        print(rh)

    id = 0
    print(max_font_size_group)

    overline=[]
    ig_list = ["IG", "IG\n", "IG Group"]
    check_cat = max_font_size_group-1
    # overline
    if check_cat >=0 and check_cat < max_font_size_group: # overline < headline -----------> overline should always be above headline
        # print(check_cat)
        # overline = grouped_token[max_font_size_group -1]
        overline_line=[]
        for i in headline:
            el = i['layout']
            response_line, start_line, end_line = _get_text(data, el)
            head_line_bounds = i['layout']['boundingPoly']['vertices']
            head_top = min(point['y'] for point in head_line_bounds)
            # print(response_line)
            break
        for page in data['pages']:
            for para in page['paragraphs']:
                el = para[0]['layout']
                respinse, start_para, end_para = _get_text(data, el)
                para_bounds = para[0]['layout']['boundingPoly']['vertices']
                # print(start_line, e)
                if int(end_para) == int(start_line):
                    para_bottom = max(point['y'] for point in para_bounds)            
                    # print(para_bottom, head_top)
                    if head_top > para_bottom:
                        distance = abs(head_top-para_bottom)
                        # print(distance)
                        print("Temp Overline Context", respinse,"Comparing with Headline", response_line) 
                        # print(distance)
                        threshold_h = 100
                        if distance<threshold_h:
                            print(respinse, "Overline Context")
                            for token in page['tokens']:
                                el = token['layout']
                                response_token, s_token, e_token = _get_text(data, el)  
                                if int(start_para) <= int(s_token) and int(e_token) <= int(end_para):                
                                    overline.append(token)

                            for line in page['lines']:
                                el = line[0]['layout']
                                response_over_line, over_start_line, over_end_line = _get_text(data, el)  
                                if int(start_para) <= int(over_start_line) and int(over_end_line) <= int(end_para):                
                                    overline_line.append(line[0])
                        # if overline:
                        #     # if 
                        else:
                            overline=None

                    else:
                        overline=None

    check_cat = max_font_size_group+1
    if len(grouped_token) > max_font_size_group + 1:
        print("Bodyline present", ) 
        if check_cat > max_font_size_group: # body > headline -----------> body should always be below headline
            # print(check_cat)
            bodyline = grouped_token[max_font_size_group + 1]
            for body_token in bodyline:
                el = body_token['layout']
                rb, s, e = _get_text(data, el)
                if rb in ig_list and len(grouped_token) >= max_font_size_group+2:
                    bodyline = grouped_token[max_font_size_group + 2]
                else:
                    bodyline = grouped_token[max_font_size_group + 1]
        else:
            bodyline = None


    if headline != None:
        headline_font_sum = sum(token['styleInfo']['fontSize'] for token in headline)
        head_avg = headline_font_sum / len(headline)
    else:
        head_avg = None
    return overline, headline, bodyline, head_avg, head_line_regex_check

def get_top_40_bounding_box(file_path):
    # Load the image
    image = cv2.imread(file_path)

    # Get the dimensions of the image
    height, width, _ = image.shape


    # Calculate the height of the top 40%
    top_40_height = int(0.5 * height)

    # Define the coordinates of the bounding box
    top_left = {"x": 0, "y": 0}
    top_right = {"x": width, "y": 0}
    bottom_right = {"x": width, "y": top_40_height}
    bottom_left = {"x": 0, "y": top_40_height}

    bounding_box = [
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ]
    return bounding_box


def second_check_for_heading(data, head_line_regex_check):    
    words_adjacent=[]
    for i in head_line_regex_check:
        el = i['layout']
        rh, s, e = _get_text(data, el)
        # print(rh, len(rh))
        font_size = i['styleInfo']['fontSize']


            # words_adjacent.append(i)
        new_c=[i['styleInfo']['textColor']['red'], i['styleInfo']['textColor']['green'],i['styleInfo']['textColor']['blue']]
        # print(new_c)
        r = int(new_c[0] * 255)
        g = int(new_c[1] * 255)
        b = int(new_c[2] * 255)

        text_color=classify_color([r,g,b])
        # print(text_color, rh)
        head_string = validate_headline_text(rh)
        # first_head_bounding_poly= i['layout']['boundingPoly']['vertices']


        # # distance_l = abs(margin_length - first_token_bounding_poly[0]["x"])
        if text_color == "Other" and head_string == True and len(rh.strip())>=2:
            words_adjacent.append(i)

    if len(words_adjacent) >=3:
        words_adjacent_headline=[]
        max_font_headline=0
        for words in words_adjacent:
            el = words['layout']
            rh, s, e = _get_text(data, el)
            # print(rh, len(rh))
            font_size = words['styleInfo']['fontSize']
            # print(font_size, rh, words['layout']['confidence'])
            if max_font_headline < font_size:
                max_font_headline=font_size
        print("Max font-size",max_font_headline)

        for word in words_adjacent:
            el = word['layout']
            rh, s, e = _get_text(data, el)
            # print(rh, len(rh))
            font_size = word['styleInfo']['fontSize']
            # print(max_font_headline-2 , font_size, max_font_headline+2, word['layout']['confidence'], rh)
            if max_font_headline-2 <= font_size <= max_font_headline+2:
                if word['layout']['confidence']>0.9:
                    print("Final headline words", rh)
                    words_adjacent_headline.append(word) 
        if len(words_adjacent_headline) != 2:
            print("Post all checks couldnt find anything to ignore the image into 50%")
    elif len(words_adjacent) == 2:
        words_adjacent_headline=words_adjacent
    else:
        words_adjacent_headline = None
    return words_adjacent_headline


def get_final_bbox_50_precent(words_adjacent_headline, file_path):
    bbox_50_precent=None
    if words_adjacent_headline:
        print(len(words_adjacent_headline))
        if len(words_adjacent_headline) == 2:
            first_word=words_adjacent_headline[0] 
            second_word=words_adjacent_headline[1]
            first_head_bounding_poly= first_word['layout']['boundingPoly']['vertices']
            first_head_bbox=[(coord['x'], coord['y']) for coord in first_head_bounding_poly]
            x_min = min(x for x, y in first_head_bbox)


            second_head_bounding_poly= second_word['layout']['boundingPoly']['vertices']
            second_head_bbox=[(coord['x'], coord['y']) for coord in second_head_bounding_poly]
            x_min_2 = min(x for x, y in second_head_bbox)
            print(x_min, x_min_2)
            if x_min> x_min_2:
                distance_headline = abs(first_head_bounding_poly[0]["x"]-second_head_bounding_poly[1]['x'])
            else:
                distance_headline = abs(first_head_bounding_poly[1]["x"]-second_head_bounding_poly[0]['x'])


            print("50% ignoring part checking distance between 2 digits adjacent and Distance is",distance_headline)

            if 200 <= distance_headline <= 300:
                bbox_50_precent=get_top_40_bounding_box(file_path)
            else:
                bbox_50_precent=None
    return bbox_50_precent

def enlarge_bbox(bbox, scale_factor):
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
        print("Exception occured while Enlarging the bounding boxes for text", e)
        return None


def overline_print_statement(overline, data, file_path, head_avg):
    font_prediction_overline_list = []
    if overline:
        overline_print_state=[]
        # overline_cordinates_para=None
        # for i in overline:
            #     overline_cordinates.append(i['layout']['boundingPoly']['vertices'])
        for page in data['pages']:
            for para in page['paragraphs']:
                el = para[0]['layout']
                rp, sp, ep = _get_text(data, el)
                for i in overline:
                    el = i['layout']
                    response, s, e = _get_text(data, el)
                    if int(sp) <= int(s) and int(e) <= int(ep):
                        # print("hi")
                        overline_cordinates=para[0]['layout']['boundingPoly']['vertices']
                        # print(overline_cordinates_para)
            if overline_cordinates == None:
                # print("i")
                overline_cordinates=[]
                for i in overline:
                    overline_cordinates.append(i['layout']['boundingPoly']['vertices'])

            # print(overline_cordinates)


                # check if all are CAPs for overline
            status, _ = check_issupper(data, overline)
            if status:
                report = "Compliant for caps"
                overline_print_state.append(["Compliant; overline All caps ", None])
            else:
                over_ro=[]
                for i in overline:
                    el = i['layout']
                    r, s, e = _get_text(data, el)
                    over_ro.append(r)        
                print("Non-Compliant all words are not caps; overline",  "".join(over_ro))
                overline_print_state.append(["Non-Compliant all words are not caps; overline", overline_cordinates])


            status = typography_bold(overline)  
            over_bold =[]
            if status:
                for i in overline:
                    report = "Non-Compliant"
                    over_bold.append(i['layout']['boundingPoly']['vertices'])
                # print("Non-Compliant; overline words are bold")
                overline_print_state.append(["Non-Compliant; overline words are bold", overline_cordinates])


            else:
                over_ro=[]
                for i in overline:
                    el = i['layout']
                    r, s, e = _get_text(data, el)
                    over_ro.append(r)
                report = "For bold"
                print("Non-Compliant. overline",  "".join(over_ro))
                overline_print_state.append(["Compliant; overline words are not bold", None])



                # check of only one color is present for overline
            print(overline_cordinates)
            color_expland_cordinates=enlarge_bbox(overline_cordinates, 1.3)
            print(color_expland_cordinates)
            classified_colors, text_color_for_typography = typography_text_bg_color(file_path, data, overline)
            if all(element == "Red" for element in text_color_for_typography):
                text_color = text_color_for_typography[0]
                # print("Compliant for red color; overline")
                overline_print_state.append([f"Compliant; overline text in {text_color}", None])

            else:
                over_color=[]
                over_ro=[]
                for i in overline:
                    el = i['layout']
                    r, s, e = _get_text(data, el)
                    over_ro.append(r)        
                    report = "Non-Compliant"
                    over_color.append(i['layout']['boundingPoly']['vertices'])
                text_color = text_color_for_typography[0]

                print("Non-Compliant no red color; overline",  "".join(over_ro))
                overline_print_state.append([f"Non-Compliant not in red color, but in {text_color}", overline_cordinates])



                # check of size is correct for overline
            overline_font_sum = sum(token['styleInfo']['fontSize'] for token in overline)
            over_avg = overline_font_sum / len(overline)
            if head_avg/5 <= over_avg <= head_avg/3:
                # print("Compliant, overline font size is correct")
                overline_print_state.append(["Compliant; overline font size is correct", None])
            else:
                over_size=[]
                over_ro=[]
                for i in overline:
                    el = i['layout']
                    r, s, e = _get_text(data, el)
                    over_ro.append(r)
                    report = "Non-Compliant"
                    over_size.append(i['layout']['boundingPoly']['vertices'])
                print("Non-Compliant, wrong overline size",  "".join(over_ro))
                overline_print_state.append(["Non-Compliant; wrong overline size", overline_cordinates])


            status_font, font_prediction_overline_list = font_style_check(overline, file_path, data, type_="Overline")
            if status_font == "Compliant Font Type":
                overline_print_state.append([f"Compliant Font-Style for overline", None])
            elif status_font == "Recheck Font Type":
                over_ro=[]
                for i in overline:
                    el = i['layout']
                    r, s, e = _get_text(data, el)
                    over_ro.append(r)
                print("Non-Compliant for font-style", "".join(over_ro))
                overline_print_state.append([f"Non-Compliant font-style for overline", overline_cordinates])
            else:
                print("Invalid type recevided for overline font-style")
    else:
        # overline_print_state.append([f"Non-Compliant overline not found", overline_cordinates])
        overline_print_state = None
    return overline_print_state, font_prediction_overline_list

def headline_print_statement(data, headline, file_path):
    font_prediction_headline_list = []
    headline_print_state=[]
    headline_cordinates=None
    if headline:
        for page in data['pages']:
            for para in page['paragraphs']:
                el = para[0]['layout']
                rp, sp, ep = _get_text(data, el)   
                for i in headline:
                    el = i['layout']
                    response, s, e = _get_text(data, el)
                    if int(sp) <= int(s) and int(e) <= int(ep):
                        headline_cordinates=para[0]['layout']['boundingPoly']['vertices']
                        # print(para_cord)
        # if headline:
            headline_print_state=[]
            if headline_cordinates == None:
                for i in headline:
                    headline_cordinates=[]
                    headline_cordinates.append(i['layout']['boundingPoly']['vertices'])
            bold=[]      

            status = typography_bold(headline)
            if status:
                report = "Compliant"
                print("Compliant for bold; headline") 
                headline_print_state.append(["Compliant; headline for BOLD", None])
            else:
                report = "Non-Compliant"
                print("Non-Compliant for bold; headline" )#head_bold)
                headline_print_state.append(["Non-Compliant; headline not in bold", headline_cordinates])


            status_font, font_prediction_headline_list = font_style_check(headline, file_path, data, type_="Headline")
            if status_font == "Compliant Font Type":
                headline_print_state.append([f"Compliant for Headline font-style", None])
            elif status_font == "Recheck Font Type":
                print("Non-Compliant for font-style for headline")
                headline_print_state.append([f"Non-Compliant font-style for Headline", headline_cordinates])
            else:
                print("Invalid type recevided for headline Font-Style")

    else:
        headline=None
        headline_print_state.append([f"Non-Compliant Headline not found", headline_cordinates])
        
    return headline_print_state, font_prediction_headline_list

def extract_date(date_string):
    parts = date_string.split(',')
    
    # Remove any leading or trailing whitespace
    parts = [part.strip() for part in parts]
    
    for part in parts:
        # Check if the part contains a year (indicating it's the date part)
        if any(char.isdigit() for char in part):
            return part

def date_check(form_doc, file_path):
    try:
        image = Image.open(f'{file_path}')
        non_compliant_coord=[]
        non_compliant=[]
        compliant=[]
        print_statement=[]
        for i in form_doc.entities:
            text_color_for_typography=[]
            for j in i.properties:
                cordinates = [
                    {"x": point.x, "y": point.y} for point in j.page_anchor.page_refs[0].bounding_poly.normalized_vertices
                ]

                new_cord=[]
                for co in cordinates:
                    x = int(co['x'] * image.width)
                    y = int(co['y'] * image.height)   
                    new_cord.append({'x': x, 'y': y})
                response = ''
                response_token=''
                # print(j.type_)
                if j.type_ == "date_time":
                    mentioned_text=extract_date(j.mention_text)
                    print(("Mentioned text for date_time", j.mention_text))
                    pattern =  r"^\d{1,2}\s(january|february|march|april|may|june|july|august|september|october|november|december)\s\d{4}$"
                    pattern_1 = "^([0-9]|[1-9][0-9]):([0-5][0-9]|[6-9][0-9])(:([0-5][0-9]|[6-9][0-9]))?$"
                    if re.match(pattern, (mentioned_text).lower()):
                        status = "Compliance"
                        compliant.append(f"{(mentioned_text)}")
                    elif re.match(pattern_1, mentioned_text):
                        status = "Non-Compliance"
                        non_compliant.append(f"{(mentioned_text)}")
                    else:
                        status = "Non-Compliance"
                        non_compliant.append((mentioned_text))
                        non_compliant_coord.append(new_cord)
                        
                        # print_statement.append((f"{status}, {(j.mention_text)}", new_cord))
                else:
                    pass
        if compliant:
            print_statement.append([f"Complaint for {compliant}", None])
        if non_compliant:
            print_statement.append([f"Non-Complaint for {non_compliant}", non_compliant_coord])
            
        return print_statement
    except Exception as e:
        print("Exception occured while Date checking function", e)
        return None

def url_check(form_doc, data, logo_text_color, riskwarning_coordinates, file_path):# logo_text_color
    try:
        image = Image.open(f'{file_path}')
        must=['Red', 'White', 'Black']
        print_s=[]
        background_color = None
        text_color=None
        for i in form_doc.entities:
            text_color_for_typography=[]
            for j in i.properties:
                cordinates = [
                    {"x": point.x, "y": point.y} for point in j.page_anchor.page_refs[0].bounding_poly.normalized_vertices
                ]

                new_cord=[]
                for co in cordinates:
                    x = int(co['x'] * image.width)
                    y = int(co['y'] * image.height)   
                    new_cord.append({'x': x, 'y': y})
                response = ''
                response_token=''
                if j.type_ == "url" and "ig.com" in (j.mention_text).lower():
                    url_mentioned=j.mention_text
                    # if check_url_issupper:
                    #     print_s.append(["Compliant", None])
                    # else: 
                    #     print_s.append(["Non-Compliant, URL is not CAPs", new_cord])
                    # print(j.mention_text)
                    if riskwarning_coordinates != None:
                        overlapping_with_url= is_bbox1_inside_bbox(new_cord, riskwarning_coordinates)
                        print("overlapping_with_url", overlapping_with_url)
                        if overlapping_with_url == False:
                            l=j.text_anchor.text_segments
                            start= l[0].start_index
                            end = l[0].end_index
                            # response += form_doc.text[int(start):int(end)]

                            for page in data['pages']:
                                for token in page['tokens']:
                                    el=token['layout']

                                    response_token, start_token, end_token = _get_text(data, el)

                                    # response += data['text'][int(start):int(end)]

                                    # print(response)

                                    if start <= int(start_token)+3 and int(end_token) <= end+3 and len(response_token) != 1 and response_token.strip() != ".":
                                        # print(len(response_token), response_token)


                                        region_coordinates=[(coord['x'], coord['y']) for coord in new_cord]

                                        # print(image.size)
                                        # image = Image.open(f'{file_path}')

                                        dominant_colors = extract_dominant_colors(image, region_coordinates, num_colors=2)
                                        # print(dominant_colors)

                                        c=[token['styleInfo']['textColor']['red'], token['styleInfo']['textColor']['green'],token['styleInfo']['textColor']['blue']]
                                        # print(color)
                                        r = int(c[0] * 255)
                                        g = int(c[1] * 255)
                                        b = int(c[2] * 255)



                                        text_color=classify_color([r,g,b])
                                        text_color_for_typography.append(text_color)
                                        # print(text_color)

                                        classified_colors=[]
                                        for color in dominant_colors:
                                            classified_colors.append(classify_color(color))
                                        # if "Other" in classified_colors:
                                        #     classified_colors.remove("Other")
                                        # print(classified_colors)

                                        if classified_colors != None and text_color !=None:
                                            for color in classified_colors:
                                                if color != text_color:
                                                    background_color = color
                                                    # break
                        else:
                            check_url_issupper = url_mentioned.isupper()
                            if check_url_issupper:
                                print_s.append(["Compliant", None])
                                return print_s
                            else: 
                                print_s.append(("Non-Compliant, URL is not CAPs", new_cord))
                                print_s.append([f"Compliant URL in Riskwarning", None])
                                return print_s
                else:
                    print("Non-Compliant URL not IG url")
                    return print_s

                # else:
                #     # print_s.append([f"Non-Compliant URL not IG url", new_cord])
                #     pass
                #     # return print_s

            if background_color == "White" and text_color != logo_text_color and logo_text_color !=None:
                print_s.append([f"Non-Compliant URL color is {text_color} and LOGO color is {logo_text_color}", new_cord])

            elif background_color == "Red" and text_color != logo_text_color and logo_text_color !=None:
                print_s.append([f"Non-Compliant URL color is {text_color} and LOGO color is {logo_text_color}", new_cord])

            elif background_color == "Black" and text_color != logo_text_color and logo_text_color !=None:
                print_s.append([f"Non-Compliant URL color is {text_color} and LOGO color is {logo_text_color}", new_cord])

            elif background_color in must and text_color in must and logo_text_color==None:
                # print("h")
                if text_color != background_color:
                    status="Compliant"
                    print_s.append([f"Compliant BG color: {background_color} and URL color {text_color}",None])
                else:
                    status = "Non-Compliant"
                    # print("Invalid combination: Text color cannot be the same as background color.", coordinates_for_logo_color)
                    print_s.append([f"Non-Compliant! BG color: {background_color} and URL color {text_color}",new_cord])
            else:
                status = "Non-Compliant"
                    # print("Invalid combination: Text color cannot be the same as background color.", coordinates_for_logo_color)
                print_s.append([f"Non-Compliant! BG color: {background_color} and URL color {text_color}",new_cord])



        return print_s
    except Exception as e:
        print("Exception occured while checking URL", e)
        return None


def is_text_below_logo(logo_coords, text_coords):
    logo_bottom = max(coord['y'] for coord in logo_coords)
    text_below_logo = [coord for coord in text_coords if coord['y'] > logo_bottom]
    return len(text_below_logo) > 0


def rw_check_if_logo_not_present(file_path, data, ig_logo_count):
    try:
        text_present = None
        final_end_index = None

        image = Image.open(file_path)

        shorter_edge = min(image.width, image.height)
        margin_length = int(shorter_edge * 0.05)
        #     bottom_margin_y = image.height - margin_length

        #     bottom_most_y = max(point["y"] for point in coordinates_json)
        image_height = image.height


        def distance_to_bottom_text(margin_length, text_cordinates, image_height):
            bounding_boxes=[(coord['x'], coord['y']) for coord in text_cordinates]
            y = max(y for x, y in bounding_boxes)
            distance =  image_height -y - margin_length
            return distance


        if ig_logo_count == 0:
            for page in data['pages']:
                for token in page['tokens']:
                    text_cordinates = token['layout']['boundingPoly']['vertices']
                    distance = distance_to_bottom_text(margin_length, text_cordinates, image_height)
                    # print(distance)
                    el= token['layout']
                    response_token, start_token, end_token=_get_text(data, el)
                    # print(distance)

                    if distance < 10:
                        final_response = response_token
                        final_start_index = start_token
                        final_end_index = end_token
                        # print(final_end_index)
        return final_end_index

    except Exception as e:
        print("Exception occured while checking if Logo is not present and checking for Riskwarning", e)
        return None



def risk_warning_line_function(data, res, file_path, ig_logo_count):
    try:
        line_below_logo = []
        riskwarning_line=[]
        riskwarning_coordinates=None
        start_end=[]
        rw_bbox=[]
        # for page in data['pages']:
        for page in data['pages']:
            for line in page['lines']:
                el= line[0]['layout']
                response_line, start_index_line, end_line=_get_text(data, el)
                line_bounding_boxes = line[0]['layout']['boundingPoly']['vertices']

                for i in res['logos']:
                    if i['logo'] == "IG Group" or i['logo'] == "IG":
                        if is_text_below_logo(i['bounds'], line_bounding_boxes):
                            riskwarning = True
                            line_below_logo.append(response_line)
                            riskwarning_line.append(line[0])
                            start_end.append((start_index_line, end_line))
                            # print(riskwarning_line)
                final_end_index = rw_check_if_logo_not_present(file_path, data, ig_logo_count)
                # print(final_end_index)
                if ig_logo_count == 0 and final_end_index != None:
                    for paragraph in page['paragraphs']:
                        el= paragraph[0]['layout']
                        response_para, start_index_para, end_para=_get_text(data, el)
                        riskwarning_coordinates=[]
                        if final_end_index == end_para:
                            # print(response_para)
                            line_below_logo.append(response_para)
                            riskwarning_coordinates.append(paragraph[0]['layout']['boundingPoly']['vertices'])

                    for line in page['lines']:
                        el= line[0]['layout']
                        response_line, start_index_line, end_line=_get_text(data, el)         
                        if int(start_index_para) <= int(start_index_line) and int(end_line) <= int(end_para):
                            riskwarning_line.append(line[0])

        # print(final_end_index, end_para)

        # for paragraph in page['paragraphs']:
        #     el= paragraph[0]['layout']
        #     response_para, start_index_para, end_para=_get_text(data, el)
        #     # print(end_para)
        #     if start_end:
        #         start_end[0][0] == start_index_para
        #         # print(response_para)
        #         line_below_logo.append(response_para)
        #         riskwarning_coordinates.append(paragraph[0]['layout']['boundingPoly']['vertices'])
        #     elif final_end_index == end_para:
        #         # print(riskwarning_line)
        #         # print(response_para)
        #         line_below_logo.append(response_para)
        #         riskwarning_coordinates.append(paragraph[0]['layout']['boundingPoly']['vertices'])
        #         # print(riskwarning_coordinates)
        if riskwarning_line:
            for rw in riskwarning_line:
                el = rw['layout']
                res_rw, res_s, res_e = _get_text(data, el)
                rw_bbox.append(rw['layout']['boundingPoly']['vertices'])
                riskwarning_coordinates = find_bounding_box(rw_bbox)
                    
                    # if int(start_index_para) == int(res_s): 
                    #     riskwarning_coordinates.append(paragraph[0]['layout']['boundingPoly']['vertices'])

        # print(riskwarning_coordinates)
        return line_below_logo, riskwarning_line, riskwarning_coordinates
    except Exception as e:
        print("Exception occured while risk_warning_line_function, where tokens are collected", e)
        return None


def risk_typography_token(data, typography):
    # try:
        distance_final_typography={}
        rw_token=[]
        # bounding_rw=[]

        if typography != None:
            for page in data['pages']:
                # for line in page['lines']:
                for line in typography:
                    el = line['layout']
                    response_line, line_start_index, line_end_index=_get_text(data, el)
                    # print(response_line)
                    # print(line_start_index, line_end_index)
                    # bounding = line[0]['layout']['boundingPoly']['vertices']
                    # bounding_rw.append(bounding_rw)

                    for token in page['tokens']:
                        el= token['layout']
                        response_token, token_start_index, token_end_index=_get_text(data, el)
                        # print(response_token,token_start_index, token_end_index)

                        # print(line_start_index, token_start_index, token_end_index,line_end_index)
                        if int(line_start_index) <= int(token_start_index) and int(token_end_index) <= int(line_end_index):
                            # print("---------TRUE----------")


                            el= token['layout']
                            response_first, start_index, end=_get_text(data, el)
                            rw_token.append(token)
                        else:
                            pass
        return rw_token #bounding_rw
    # except Exception as e:
    #     print("Exception occured while risk_typography_token", e)
    #     return None

def risk_align(data, risk_response, riskwarning_line, file_path):
    try:
        # bounding_rw=[]
        for i in risk_response:
            yes_list=['Yes', 'yes', ' Yes', 'Yes ', ' Yes ']
            if i in yes_list:
                print("Risk Warning Present")
                rw_token = risk_typography_token(data, riskwarning_line)
                # print(rw_token)
                distance_final_riskwarning_line= typography_allignement(data, rw_token, file_path)
                print(distance_final_riskwarning_line)
                risk_warning_align = alignment(distance_final_riskwarning_line)
                print("================================================================")
                print("================================================================")
                print("================================================================")
                print(risk_warning_align)
                print("================================================================")
                print("================================================================")
                
                
              
                return risk_warning_align, rw_token, distance_final_riskwarning_line
            else:
                pass #return rw_token
    except Exception as e:
        print("Exception occured while checking RiskWarnig Alignment", e)
        return None


def rw_size(rw_token, head_avg, typography, riskwarning_coordinates):
    try:
        # bounding_rw=[]
        print_state=[]
        status=None
        for line in typography:
            bounding = line['layout']['boundingPoly']['vertices']
            rw_font_sum = sum(token['styleInfo']['fontSize'] for token in rw_token)
            rw_avg = rw_font_sum / len(rw_token)
            # print(head_avg/3.5, rw_avg, head_avg/2.2)
        if head_avg/3.5 <= rw_avg <= head_avg/2.2:
            status="Compliant"
            print_state=["Compliant, RiskWarning font size is correct", None]

        else:
            status="Non-Compliant"
            # bounding_rw.append(bounding)
            if rw_avg <= head_avg/3.5:
                print_state=[f"Non-Compliant, wrong riskwarning size Increase it", riskwarning_coordinates]
            elif rw_avg >= head_avg/2.5:
                print_state=[f"Non-Compliant, wrong riskwarning size Decrease it", riskwarning_coordinates]
            else:
                print_state=[f"Non-Compliant, wrong riskwarning size Adjust it", riskwarning_coordinates]

        return status, print_state
    except Exception as e:
        print("Exception occured while checking Riskwarning size", e)
        return None


def rw_logo_position_check(distance, risk_response, logo_print_statements,file_path, data, riskwarning_coordinates, ig_logo_count):
    try:
        print("Checking logo position post risk-warning check distance and Logo count", distance, ig_logo_count)
        # for logo
        # logo_print_statements=[]
        if distance != None:
            if distance <= 8 and risk_response:
                final_logo_postition = "Compliant"
                # logo_print_statements[2][0].remove(logo_print_statements)
                logo_print_statements[3][0] = 'Compliant for logo; Logos correctly placed post riskwarning'
                logo_print_statements[3][1] = None

            else:
                final_logo_postition = "Non-Compliant"
                logo_print_statements[3][0] = 'Non-Compliant for logo; Logos incorrectly placed, post risk warning'
                # logo_print_statements[2][1] = None
        else:
            final_end_index = rw_check_if_logo_not_present(file_path, data, ig_logo_count)

            # print(final_end_index)
            if ig_logo_count == 0 and final_end_index != None:
                for page in data['pages']:
                    for paragraph in page['paragraphs']:
                        el= paragraph[0]['layout']
                        response_para, start_index_para, end_para=_get_text(data, el)

                if final_end_index == end_para:
                    # print(response_para)
                    line_below_logo.append(response_para)
                    riskwarning_coordinates.append(paragraph['layout']['boundingPoly']['vertices'])

        return logo_print_statements
    except Exception as e:
        print("Exception occured while checking if Riskwarning is placed correctly with Logo", e)
        return None

def risk_warning_print_statement(data, res, ig_logo_count, file_path, head_avg):
    try:
        print("Initial check for finding risk-warning")
        model_gen = GenerativeModel(model_name="gemini-1.0-pro-vision")
        line_below_logo, riskwarning_line, riskwarning_coordinates=risk_warning_line_function(data, res, file_path, ig_logo_count)

        risk_response=[]
        risk_warning_list_check = ["CFDs are high risk products. Consider taking steps to manage your risk and ensure to monitor your positions. Refer to our PDS & TMD available on our website",
                           "Your capital is at risk. XX% of retail investor accounts lose money when trading CFDs with this provider. You should consider whether you can afford to take the high risk of losing your money."]
        for line in line_below_logo:
            for risk in risk_warning_list_check:
                prompt = f"check if {risk} matches {line}, give answer in 'Yes', 'No'"
                response = model_gen.generate_content([prompt, file_path])
                # print(response)
                risk_response.append(response.text)

        if riskwarning_line:
            print("Line below Logo", line_below_logo)
            risk_warning_align, rw_token, distance_final_riskwarning_line = risk_align(data, risk_response, riskwarning_line, file_path)
            status_rw_size, print_state_rw=  rw_size(rw_token,head_avg, riskwarning_line, riskwarning_coordinates)
        else:
            print_state_rw=[]
            risk_response=None
            risk_warning_align=None
            distance_final_riskwarning_line=None
            print_state_rw.append([f"Non-Compliant, No RiskWarning found", None])
        return print_state_rw, risk_warning_align, distance_final_riskwarning_line,risk_response,riskwarning_coordinates
    except Exception as e:
        print("Error during finding riskwarning lines", e)
        return None, None, None, None



def typography_allignement(data, typography, file_path):
    try:
        distance_l=None
        distance_r=None
        
        image = Image.open(f'{file_path}')
        
        
        shorter_edge = min(image.width, image.height)
        margin_length = int(shorter_edge * 0.05)
        #     bottom_margin_y = image.height - margin_length

        #     bottom_most_y = max(point["y"] for point in coordinates_json)
        image_height = image.height

        distance_final_typography={}

        if typography != None:
            counter = 1
            # print("1")
            for page in data['pages']:
                for line in page['lines']:
                    el = line[0]['layout']
                    response_line, line_start_index, line_end_index=_get_text(data, el)
                    # print(response_line,line_start_index)


                    for typo_word in typography:
                        # counter = 1
                        el= typo_word['layout']
                        response_token, token_start_index, token_end_index=_get_text(data, el)
                        # print(response_token)
                        first_token = None
                        last_token = None


                        # print(token_start_index,line_start_index)
                        if int(token_start_index) == int(line_start_index) and response_line != "IG\n" and response_line != "IG" and response_line != "IG Group":
                            # print(token_start_index, line_start_index)
                            if first_token is None:
                                first_token = typo_word
                                el= first_token['layout']
                                response_first, start_index, end=_get_text(data, el)
                                # print(response_first)
                                first_token_bounding_poly= first_token['layout']['boundingPoly']['vertices']

                                distance_l = abs(margin_length - first_token_bounding_poly[0]["x"])
                        # print(response_token)
                        # print(token_end_index, line_end_index)
                        if int(token_end_index) == int(line_end_index) and response_line != "IG\n" and response_line != "IG" and response_line != "IG Group":
                            if last_token is None:


                                last_token = typo_word
                                el= last_token['layout']
                                response_last, start_index, end=_get_text(data, el)
                                # print(response_last)
                                last_token_bounding_poly= last_token['layout']['boundingPoly']['vertices']

                                distance_r = (image.width - margin_length) - last_token_bounding_poly[1]["x"]




                                final_distance = abs(distance_l - distance_r)
                                        # final.append(final_distance)
                                distance_final_typography[counter] = {"left": distance_l, 
                                                          "right": distance_r,
                                                          "final": final_distance,
                                                          "word" : response_line,
                                                                     "co-ordinates" : line[0]['layout']['boundingPoly']['vertices']}

                                counter += 1


        else:
            return None

        return distance_final_typography
    except Exception as e:
        print("Exception occured while Typography alignment, mostly due to no start-index", e)
        return None

def sort_and_clean_data(data):
    try:
        def sort_entities(entities, key_name):
            sorted_entities = []
            for page in data['pages']:
                for entity in page[key_name]:
                    bounding_boxes = [(coord['x'], coord['y']) for coord in entity['layout']['boundingPoly']['vertices']]
                    y_min = min(y for x, y in bounding_boxes)
                    sorted_entities.append([entity, y_min])
                page[key_name].clear()
                sorted_entities.sort(key=lambda x: x[1])
                page[key_name] = sorted_entities[:]
            for page in data['pages']:
                for entity in page[key_name]:
                    del entity[1]

        sort_entities(data, 'lines')
        sort_entities(data, 'paragraphs')
    except Exception as e:
        print(e)
        return None

def get_cordinates_for_alignment(distance_final_typograph):
    try:
        print("trying to find co-ordinates for")
        cordinates=[]
        for i in distance_final_typograph:
            cordinates.append(distance_final_typograph[i]['co-ordinates'])

        return cordinates
    except Exception as e:
        print("Error occured while finding co-ordinates inside get_cordinates_for_alignment function", e, distance_final_typograph)
        return None

def alignment_check(data, overline, headline, bodyline, table_status, risk_warning_align, distance_final_riskwarning_line, file_path):
    try:
        print("Checking Alignment for file", file_path)
        print("overline, headline, bodyline, table_status, risk_warning_align, distance_final_riskwarning_line, file_path", risk_warning_align, distance_final_riskwarning_line, file_path)
        
        if overline:
            distance_final_overline = typography_allignement(data, overline, file_path)
            print(distance_final_overline)
            overline_align = alignment(distance_final_overline)
        else:
            print("No Overline")
            distance_final_overline = None
            overline_align = None
            overline_align_new = None

        if headline:
            print("Headline")
            distance_final_headline = typography_allignement(data, headline, file_path)
            print(distance_final_headline)
            headline_align = alignment(distance_final_headline)
        else:
            print("No Headline")
            distance_final_headline = None
            headline_align = None



        if bodyline:  
            print("Bodyline")
            distance_final_bodyline = typography_allignement(data, bodyline, file_path)
            print(distance_final_bodyline)
            bodyline_align = alignment({list(distance_final_bodyline.keys())[0] : list(distance_final_bodyline.values())[0]})
        else:
            print("No Bodyline")
            distance_final_bodyline = None
            bodyline_align = None       


            

        if risk_warning_align:
            print("risk_warning_coordinatesrisk_warning_coordinatesrisk_warning_coordinates")
            risk_warning_coordinates = get_cordinates_for_alignment(distance_final_riskwarning_line)
        else:
            risk_warning_coordinates=None

        if overline_align:
            overline_coordinates=get_cordinates_for_alignment(distance_final_overline)
        else:
            overline_coordinates=None

        if bodyline_align:
            bodyline_coordinates=get_cordinates_for_alignment(distance_final_bodyline)
        else:
            bodyline_coordinates=None


        print_state_align_typo = []

        # If all alignments are the same
        if overline_align == headline_align == bodyline_align == risk_warning_align:
            print_state_align_typo.append(["All same aligned Compliant", None])
        elif table_status:
            final_align = headline_align
            print_state_align_typo.append(["In-Correct Alignment, As tables are found", None])

        # If one alignment is different from the others
        elif (overline_align == headline_align == bodyline_align) and (risk_warning_align != headline_align):
            final_align = headline_align
            print_state_align_typo.append([f"Non-Compliant as Overline: {overline_align}, Headline: {headline_align}, Bodyline: {bodyline_align}, Riskwarning: {risk_warning_align}", risk_warning_coordinates])

        elif (overline_align == headline_align == risk_warning_align) and (bodyline_align != headline_align):
            final_align = headline_align
            print_state_align_typo.append([f"Non-Compliant as Overline: {overline_align}, Headline: {headline_align}, Bodyline: {bodyline_align}, Riskwarning: {risk_warning_align}", bodyline_coordinates])

        elif (headline_align == bodyline_align == risk_warning_align) and (overline_align != headline_align):
            final_align = headline_align
            print_state_align_typo.append([f"Non-Compliant as Overline: {overline_align}, Headline: {headline_align}, Bodyline: {bodyline_align}, Riskwarning: {risk_warning_align}", overline_coordinates])

        # If any one alignment is None and the rest are the same
        elif (overline_align is None) and (headline_align == bodyline_align == risk_warning_align):
            final_align = headline_align
            print_state_align_typo.append([f"Compliant as Overline: {overline_align}, Headline: {headline_align}, Bodyline: {bodyline_align}, Riskwarning: {risk_warning_align}", None])

        elif (headline_align is None) and (overline_align == bodyline_align == risk_warning_align):
            final_align = bodyline_align
            print_state_align_typo.append([f"Compliant as Overline: {overline_align}, Headline: {headline_align}, Bodyline: {bodyline_align}, Riskwarning: {risk_warning_align}", None])

        elif (bodyline_align is None) and (overline_align == headline_align == risk_warning_align):
            final_align = bodyline_align
            print_state_align_typo.append([f"Compliant as Overline: {overline_align}, Headline: {headline_align}, Bodyline: {bodyline_align}, Riskwarning: {risk_warning_align}", None])

        # If two alignments are different from the others
        elif (overline_align == headline_align) and (bodyline_align == risk_warning_align == None):
            final_align = headline_align
            print_state_align_typo.append([f"Compliant as Overline and Headline aligned, Bodyline and Riskwarning aligned; Overline: {overline_align}, Headline: {headline_align}, Bodyline: {bodyline_align}, Riskwarning: {risk_warning_align}", overline_coordinates])

        elif (headline_align == risk_warning_align) and (bodyline_align == overline_align == None):
            final_align = headline_align
            print_state_align_typo.append([f"Compliant as Overline and Riskwarning aligned, Headline and Bodyline aligned; Overline: {overline_align}, Headline: {headline_align}, Bodyline: {bodyline_align}, Riskwarning: {risk_warning_align}", risk_warning_cordinates])

        # If any alignment is "Right"
        elif "Right" in [overline_align, headline_align, bodyline_align, risk_warning_align]:
            final_align = headline_align
            print_state_align_typo.append([f"Non-Compliant as Right Aligned; Overline: {overline_align}, Heading: {headline_align}, Body line: {bodyline_align}, Riskwarning: {risk_warning_align}", None])


        # Default case
        else:
            final_align = headline_align
            print_state_align_typo.append([f"Non-Compliant as Overline: {overline_align}, Heading: {headline_align}, Body line: {bodyline_align}, Riskwarning: {risk_warning_align}", None])

        return print_state_align_typo
    except Exception as e:
        print("Error during checking for Alignment alignment_check function", e)
        return None

def final_statements_to_print(logo_print_statements,overline_print_statements, headline_print_statements, url_print_statements, date_print_statements, rw_print_statements, align_print_statements, final):
    try:
        print("Final Print Statements")
        non_com_sum = count_of_non_compliance(final)
        if logo_print_statements:
            for i in logo_print_statements:
                print(i[0])
        if headline_print_statements:
            for i in headline_print_statements:
                print(i[0])
        if overline_print_statements:
            for i in overline_print_statements:
                print(i[0])
        if url_print_statements:
            for i in url_print_statements:
                print(i[0])
        if date_print_statements:
            for i in date_print_statements:
                print(i[0])
        if rw_print_statements:
            # for i in print_state_rw:
                print(rw_print_statements[0])

        print(align_print_statements[0][0])
        print("Non Compliance Count",non_com_sum)
    except Exception as e:
        print("Error while print final statements", e)


def is_valid_points_structure(points):
    if not isinstance(points, list):
        return False
    for sublist in points:
        if not isinstance(sublist, list):
            return False
        for point in sublist:
            if not isinstance(point, dict):
                return False
            if "x" not in point or "y" not in point:
                return False
    return True

def find_bounding_box(points):


    # Extract all points from the nested list
    all_points = [point for sublist in points for point in sublist]

    # Calculate min and max values for x and y
    min_x = min(point["x"] for point in all_points)
    max_x = max(point["x"] for point in all_points)
    min_y = min(point["y"] for point in all_points)
    max_y = max(point["y"] for point in all_points)

    # Construct the corners of the bounding box
    bounding_box = [
        {"x": min_x, "y": min_y},  # Bottom-left corner
        {"x": max_x, "y": min_y},  # Bottom-right corner
        {"x": max_x, "y": max_y},  # Top-right corner
        {"x": min_x, "y": max_y}   # Top-left corner
    ]

    return bounding_box

def save_file_with_cordinates(final, file_path, bucket_name, processed_path, buffer_path, boxes_to_draw):
    try:
        print("-" *50)
        print(final)
        print("-" *50)
        file_name = os.path.basename(file_path)
        file = os.path.splitext(file_name) 
        file1=file[0]+file[1]
        print(file1)
        file_save = file[0]+file[1]
        print("Filename: ", file)
            # Initialize Google Cloud Storage client
        client = storage.Client()
        def create_folder(bucket_name, destination_folder_name):
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(bucket_name)
            blob = bucket.blob(destination_folder_name)

            blob.upload_from_string('')

            print('Created {} .'.format(destination_folder_name))

        folder = create_folder(bucket_name, f'{file[0]}')
        # print(folder)

        # Define your GCS bucket name
        # bucket_name = "test-ig"
        gcs_path = f'processed/{file[0]}/{file1}'
        # save_json = f'processed/{file[0]}/{file[0]}

        # Path to the image you want to save

        # Path to save the image in GCS
        # gcs_path = f"non_compliance_images/{file}"
        # dest_gcs_path = f"{processed_path}/{file[0]}"
        prefix = f'gs://{bucket_name}'
        
        dest_gcs_path = file_path[len(prefix):].replace(f'{buffer_path}', f'{processed_path}')



        image = Image.open(file_path)
        draw = ImageDraw.Draw(image)
        
        image_buffer = io.BytesIO()

        for k, v in final.items():
            # print(k)
            if k == "Logo":
                print("-------LOGO------")
                for logo_values in v:
                    if logo_values[1] != None:
                        points = [(co['x'], co['y']) for co in logo_values[1]]
                        for i in range(len(points)):
                            draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)
                    else:
                        pass
            # else:
                # pass
                            
            if k == "Typography":
                for typo_key, typo_values in v.items():
                    if typo_key == "Overline":
                        print('---------OVERLINE-----------')
                        if typo_values:
                            for over in typo_values:
                                if over[1] != None:
                                    points_ = over[1]
                                    if is_valid_points_structure(points_):
                                        bbox_overline = find_bounding_box(points_)
                                        points = [(co['x'], co['y']) for co in bbox_overline]
                                        for i in range(len(points)):
                                            draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)
                                    else:
                                        points = [(co['x'], co['y']) for co in points_]
                                        for i in range(len(points)):
                                            draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)
                                else:
                                    pass
                    #     else:
                    #         pass
                    # else:
                    #     pass
   
                    if typo_key == "Headline":
                        print("-----------HEADLINE-----------")
                        if typo_values:
                            for head in typo_values:
                                if head[1] != None:
                                    points_ = head[1]
                                    if is_valid_points_structure(points_):
                                        bounding_box = find_bounding_box(points_)
                                        points = [(co['x'], co['y']) for co in bounding_box]
                                        for i in range(len(points)):
                                            draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)
                                    else:
                                        points = [(co['x'], co['y']) for co in points_]
                                        for i in range(len(points)):
                                            draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)
                                else:
                                    pass
                    #     else:
                    #         pass
                    # else:
                    #     pass
                    if typo_key == "Riskwarning":
                        print("-------------RISKWARNING----------")
                        print(typo_values[1])
                        if typo_values[1]:
                            points_ = typo_values[1]
                            if is_valid_points_structure(points_):
                                print("Riskwarning is_valid_points_structure")
                                points_1 = find_bounding_box(points_)
                                points = [(co['x'], co['y']) for co in points_1]
                                for i in range(len(points)):
                                    draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)
                            else:
                                print("Riskwarning not is_valid_points_structure")                            
                                points = [(co['x'], co['y']) for co in points_]
                                for i in range(len(points)):
                                    draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)

#                             points = [(co['x'], co['y']) for co in typo_values[0][1]]
#                             for i in range(len(points)):
#                                 draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)
                        else:
                            pass
        
        
            if k == "Entities":
                for typo_key, typo_values in v.items():
                    
                    if typo_key == "URL":
                        if typo_values: 
                            print(typo_values[0][1])
                            for b in typo_values[0][1]:
                                print("-------URL List Non-compliance----------")
                                points = [(co['x'], co['y']) for co in b]
                                for i in range(len(points)):
                                    draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)
                        else:
                            pass
                                        
  
                    if typo_key == "DATE":
                        if typo_values: 
                            print(typo_values[0][1])
                            for b in typo_values[0][1]:
                                print("-------Date List Non-compliance----------")
                                points = [(co['x'], co['y']) for co in b]
                                for i in range(len(points)):
                                    draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)
                        else:
                            pass
       
            if k == "Alignment":
                print("-----------ALIGNMENT----------")
                if v[0][1] != None:
                    print("coord", v[0][1])
                    points = [(co['x'], co['y']) for co in v[0][1]]

                    for i in range(len(points)):
                        draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)
                else:
                    pass
            # else:
            #     pass
        for b in boxes_to_draw:
            print("-------TEXT----------")
            points = [(co['x'], co['y']) for co in b]
            for i in range(len(points)):
                draw.line((points[i], points[(i+1) % len(points)]), fill="red", width=3)


        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        # Upload the image to GCS
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_file(image_buffer, content_type="image/png")
        

        print(f"Image saved to GCS: gs://{bucket_name}/{gcs_path}")
        saved_folder = f'gs://{bucket_name}/{gcs_path}'
        return f'gs://{bucket_name}/{gcs_path}'
    except Exception as e:
        print("Exception during saving the file with cordinates", e)
        return ""


def count_non_compliant(print_statements):
    count_non_comp=0
    count_comp=0
    if print_statements:
        for id_, i in enumerate(print_statements):
            if print_statements[id_][0] == None:
                pass
            elif print_statements[id_][0].startswith("Compliant"):
                count_comp +=1
            elif print_statements[id_][0].startswith("Non-Compliant"):
                count_non_comp +=1
            # elif print_statements[id_][0] == None:
                # pass
    return count_non_comp,count_comp

def count_of_non_compliance(final):    
    try:
        non_com_to_count=[]
        non_com_print_statement={}
        for k,v in final.items():
            if v:
                if k == "Typography" or k == "Entities":
                    for s_k, s_v in v.items(): 
                        if s_v and s_k != "Riskwarning":
                            count_non_comp,_= count_non_compliant(s_v) 
                            # print(s_k,count_non_comp)
                            # print(sum(non_com_to_count))
                            non_com_to_count.append(count_non_comp)

                        elif s_k == "Riskwarning":
                            if s_v[0].startswith("Non-Compliant"):
                                non_com_to_count.append(1)

                                # print(s_k,count_non_comp)
                                # print(sum(non_com_to_count))
                            else:
                                pass
                        else:
                            pass
                elif k == "Alignment":
                    v[0][0].startswith("Non-Compliant")
                    # print(k,"1")
                    non_com_to_count.append(1)

                else:
                    count_non_comp,_=count_non_compliant(v)
                    # print(k, count_non_comp)
                    non_com_to_count.append(count_non_comp)

        non_com_sum=sum(non_com_to_count)
        return non_com_sum
    except Exception as e:
        return 0

def non_compliance_list_function(final):
    try:
        output = {}
        counter = {}

        # Function to add to the output with a counter for duplicates
        def add_to_output(category, issue):
            if category in counter:
                counter[category] += 1
            else:
                counter[category] = 1

            if counter[category] > 1:
                output[f"{category}_{counter[category]}"] = issue
            else:
                output[category] = issue

        # Process 'Logo' key
        for category, entries in final.items():
            print("-" *20, category, "-" *20)
            if category == 'Logo':
                for issue in entries:
                    print(issue[0])
                    if 'Non-Compliant' in issue[0]:
                        add_to_output(category, issue[0])
                    else:
                        pass
                        
            elif category == 'Entities' or category == 'Typography':
                for ent_key, ent_val in entries.items():
                    # print(".............................", ent_key, ent_val)
                    if ent_key == "Riskwarning":
                        if ent_val != None:
                            if 'Non-Compliant' in ent_val[0]:
                                add_to_output(ent_key, ent_val[0])
                            else:
                                pass
                        else:
                            pass
                        
                    if ent_val != None:
                        for item in ent_val:
                            if item !=None:
                                # print("-" *20, item[0])
                                if 'Non-Compliant' in item[0]:
                                    add_to_output(ent_key, item[0])
            else:  # Process 'Alignment' key
                for issue in entries:
                    if issue[0] != None:
                        print("Alignment")
                        print("-" *20, issue[0])
                        if 'Non-Compliant' in issue[0]:
                            add_to_output('Alignment', issue[0])
       
        output_data = [{'title': key, 'text': value} for key, value in output.items()]
        return output_data
    except Exception as e:
        print("Failed during collecting final non-compliance list", e)
        return []



# def non_compliance_list_function(final):
#     try:
#         output = {}
#         counter = {}

#         # Function to add to the output with a counter for duplicates
#         def add_to_output(category, issue):
#             if category in counter:
#                 counter[category] += 1
#             else:
#                 counter[category] = 1

#             if counter[category] > 1:
#                 output[f"{category}_{counter[category]}"] = issue
#             else:
#                 output[category] = issue

#         # Process 'Logo' key
#         for category, entries in final.items():
#             if category == 'Logo':
#                 for issue in entries:
#                     if 'Non-Compliant' in issue[0]:
#                         add_to_output(category, issue[0])
#             elif category == 'Entities' or category == 'Typography':
#                 for ent_key, ent_val in entries.items():
#                     for item in ent_val:
#                         if 'Non-Compliant' in item[0]:
#                             add_to_output(ent_key, item[0])
#             else:  # Process 'Alignment' key
#                 for issue in entries:
#                     if 'Non-Compliant' in issue[0]:
#                         add_to_output('Alignment', issue[0])

#         output_data = [{'title': key, 'print': value} for key, value in output.items()]
#         return output_data
#     except Exception as e:
#         print("Failed during collecting final non-compliance list")
#         return None

def text_bbox_para_list(doc):    
    try:
        doc_json=documentai.Document.to_json(doc)
        data = json.loads(doc_json)
        para_bbox_text_list= []
        for page in data['pages']:
            for para in page['paragraphs']:
                el = para['layout']
                response_para, s, e = _get_text(data, el)
                # print(len(response_para), len(response_para.split()))
                para_bbox_text_dict={}
                if len(response_para.split()) >= 4:
                    print("Paragraphs with more than 4 words present")
                    para_bbox_text_dict['text'] = response_para
                    para_bbox_text_dict['BoundingBox'] = para['layout']['boundingPoly']['vertices']
                    para_bbox_text_list.append(para_bbox_text_dict)
                else:
                    pass
        return para_bbox_text_list
    except Exception as e:
        print("Exception occured while extracting paragraphs list along with bbox")
        return None




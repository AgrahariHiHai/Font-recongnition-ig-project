import io
import os
import fitz
import pandas as pd
from openai import OpenAI
from config import ENV_CONFIG
from utils.artifacts import *

# OpenAI client setup
client = OpenAI(api_key=ENV_CONFIG["openai_api_key"])

def extract_images_from_file(uploaded_file):
    #TODO
    # convert pdf or docx to images for each page
    if uploaded_file.endswith("docx"):
        pass
    elif uploaded_file.endswith("pdf"):
        pass

    else:
        raise Exception("Unsupported file type")

    frames = []
    return frames

# Function to process the uploaded file and extract status and summary
def process_text(extracted_text, question):
    try:
        response = client.chat.completions.create(
            model=ENV_CONFIG["openai_model"],
            messages=[
            {"role": "user", "content": f"{extracted_text} is textual information for IG Trading. Intially answer with a 'Yes, No or Unsure' sentence. Then explain why in the following sentence: {question}."}
        ]
        )

        # print(response.choices[0].message.content)
        response_content = response.choices[0].message.content

        # Initialize status and summary
        answer = "Unsure"
        explanation = ""

        # Check for keywords and separate the response
        for keyword in ["Yes", "No", "Unsure"]:
            if response_content.startswith(keyword):
                # Split the response at the keyword
                answer = keyword
                explanation = response_content[len(keyword):].lstrip("., \n")

                # Capitalize the first letter of the summary
                if explanation:
                    explanation = explanation[0].upper() + explanation[1:]

                break

        return {"answer": answer, "explanation": explanation}

    except Exception as e:
        print("An error occurred in answering question: {}: {}".format(question, e))
        traceback.print_exc()
        return {"answer": "Error", "explanation": f"An error occurred: {e}"}

def get_text_output(extracted_text, medium):

    try:
        compliance_list, cfd_list, risk_list = [], [], []
        df = pd.DataFrame()
        cols = ["title", "Question", "Answer", "text", "non_compliant_response"]

        # Process Compliance Questions
        for question_info in compliance_questions:
            question = question_info["question"]
            non_compliant_response = question_info["non_compliant_response"]
            result = process_text(extracted_text, question)
            compliance_list.append(["Compliance", question, result["answer"], result["explanation"], non_compliant_response])
            
        print("Compliance questions completed")

        # Process CFD Questions and create a table
        for question_info in cfd_questions:
            question = question_info["question"]
            non_compliant_response = question_info["non_compliant_response"]
            result = process_text(extracted_text, question)
            cfd_list.append(["CFD", question, result["answer"], result["explanation"], non_compliant_response])
        
        print("CFD questions completed")

        # Process risk statement and create table
        # Determine and process the relevant risk question(s)
        if medium == "IG Website":
            risk_questions = [question for question in risk_questions if question["id"] in [1, 2]]
            for question in risk_questions:
                question = question_info["question"]
                non_compliant_response = question_info["non_compliant_response"]
                result = process_text(extracted_text, question)
                risk_list.append(["RiskWarnings", question, result["answer"], result["explanation"], non_compliant_response])

        elif medium == "Email":
            question_info = risk_questions[3]
            question = question_info["question"]
            non_compliant_response = question_info["non_compliant_response"]
            result = process_text(extracted_text, question)
            risk_list.append(["RiskWarnings", question, result["answer"], result["explanation"], non_compliant_response])

        elif medium in ["Google PPC", "Paid Social"]:
            risk_questions = [question for question in risk_questions if question["id"] in [3, 4]]
            for question in risk_questions:
                question = question_info["question"]
                non_compliant_response = question_info["non_compliant_response"]
                result = process_text(extracted_text, question)
                risk_list.append(["RiskWarnings", question, result["answer"], result["explanation"], non_compliant_response])
        
        print("Risk questions completed")

        df = pd.concat([df, pd.DataFrame(compliance_list), pd.DataFrame(cfd_list), 
                        pd.DataFrame(risk_list)], axis=0)
        df.columns = cols

        non_compliance_df = df.query('Answer == non_compliant_response')
        non_compliance_df = non_compliance_df[["title", "text"]].reset_index(drop=True)
        summary = non_compliance_df["title"].value_counts().to_dict()
        
        titles = ["RiskWarnings", "CFD", "Compliance"]
        for title in titles:
            if title not in summary:
                summary[title] = 0
        
        non_compliance_list = non_compliance_df.to_dict(orient="records")
        print("Non compliance list identified: {}".format(len(non_compliance_list))) 

    except Exception:
        print("Error in text compliance identification")
        traceback.print_exc()
        non_compliance_list = []
        summary = {}

    return non_compliance_list, summary

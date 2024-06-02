import os
 
ENV_CONFIG = {
    "project_id": os.environ.get("project_id"),
    "bucket_name": os.environ.get("bucket_name"),
    "openai_api_key": os.environ.get("openai_api_key"),
    "openai_model": os.environ.get("openai_model"),
    "gemini_model": os.environ.get("gemini_model"),
    "location": os.environ.get("location"),
    "mime_type_image": "image/jpeg",
    "mime_type_text": "application/pdf",
    "processed_path": "processed",
    "buffer_path": "buffer",
    "endpoint_id": os.environ.get("endpoint_id"),
    "endpoint_project_id" : os.environ.get("endpoint_project_id"),
    "endpoint_network_name" : os.environ.get("endpoint_network_name")
    }


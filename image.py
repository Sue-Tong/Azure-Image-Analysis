# Setup
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
import pandas as pd
import numpy as np
import re
import csv

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType

# Keys and endpoint from Microsoft Azure
# Note: Not valid anymore
subscription_key_cv = "c908e35c94a04733aa62ff79e39b9377"
endpoint_cv = "https://inscv.cognitiveservices.azure.com/"

# Authentication
computervision_client = ComputerVisionClient(endpoint_cv, CognitiveServicesCredentials(subscription_key_cv))

# Save images to bolb
connect_str = 'DefaultEndpointsProtocol=https;AccountName=imageins;AccountKey=dJKdmy5acQdkDCK/NeBjGu5PCDeSeJoqHvzoflAJwGsUZGfzEvM4/dBEN9MmhLYymzHxPzDraGLs+ASt6m7iKg==;EndpointSuffix=core.windows.net'
container_name = 'image'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)

# endpoint_face = "https://aijins-face.cognitiveservices.azure.com/"
# subscription_key_face = "cb0b9aabf88f4409b25460851f989a6c"
# # Instantiate a FaceClient object using your Azure credentials
# face_client = FaceClient(endpoint_face, CognitiveServicesCredentials(subscription_key_face))


def azure_image_tag(read_image_url):
    '''Use Azure Computer Vision Image Tag to assign tags of an image
    Return a variable in dictionary format
    The key of the dictionary is the name of the image tag
    The value of the dictionary is the confidence score of corresponding image tag
    '''
    tags_result = computervision_client.tag_image(read_image_url)
    
    # Return empty dictionary if there is no image tag
    if (len(tags_result.tags) == 0):
        image_tags = {}
        
    # Get all the image tags with its confidence
    else:
        tag_name = []
        tag_confidence = []
        for tag in tags_result.tags:
            tag_name.append(tag.name)
            tag_confidence.append(tag.confidence)
        image_tags = {tag_name[i]: tag_confidence[i] for i in range(len(tag_name))}
    return image_tags

def text_presence_by_tag(image_tags):
    '''Identify if the image contains the text through image tag approach
    Return boolean True or False
    '''
    if "text" in image_tags:
        text_presence = True
    else:
        text_presence = False
    return text_presence

def find_unique_tag(image_tags):
    unique_tags = list(image_tags)
    return unique_tags

def azure_image_description(read_image_url):
    '''Use Azure Computer Vision Image Description to describe an image
    Return two variables in list format
    The description_text is the text description of the image
    The description_confidence is the confidence score of the description
    '''
    description_result = computervision_client.describe_image(read_image_url,language= 'en' , max_candidates=3)
    if (len(description_result.captions) == 0):
        description_text = []
        description_confidence = []
    else:
        description_text = []
        description_confidence = []
        for caption in description_result.captions:
            caption_text = caption.text
            description_text.append(caption_text)
            description_confidence.append(caption.confidence)
    return description_text, description_confidence

def azure_image_category(read_image_url):
    '''Use Azure Computer Vision Analyze Image to categorize an image
    Return two variables in list format
    The category_name is the category of the image
    The category_score is the confidence score of corresponding category
    '''
    analyze_result = computervision_client.analyze_image(read_image_url, visual_features= ['Categories'], language= 'en')
    if (len(analyze_result.categories) == 0):
        category_name = []
        category_score = []
    else:
        category_name = []
        category_score = []
        for category in analyze_result.categories:
            category_name.append(category.name)
            category_score.append(category.score)
    return category_name, category_score

def azure_ocr(read_image_url):
    '''Use Azure Computer Vision OCR to extract text from an image
    Return two variables, all in list format
    line_text is the actural text extracted
    line_bouding_box is the region of the text in the image
    '''
    # Call API with URL and raw response (allows you to get the operation location)
    read_response = computervision_client.read(read_image_url, raw=True)
    
    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers["Operation-Location"]
    
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]
    
    # Call the "GET" API and wait for it to retrieve the results
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)
    # Print the detected text, line by line
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            line_text = []
            line_bouding_box = []
            for line in text_result.lines:
                line_text.append(line.text)
                line_bouding_box.append(line.bounding_box)
    return line_text, line_bouding_box

def text_presence_by_ocr(line_text):
    '''Identify if the image contains the text through image OCR approach
    The text is present if the line_text is empty
    Return boolean True or False
    '''
    if len(line_text) != 0:
        text_presence = True
    else:
        text_presence = False
    return text_presence

def azure_detect_gender(read_image_url):
    '''
    Use Azure Cognitive Services to detect gender of a person in an image
    Return a string with the detected gender ('male' or 'female')
    If no face is detected, return None
    '''
    
    # Detect faces in the image
    detected_faces = face_client.face.detect_with_url(url=read_image_url)
    
    # If no faces are detected, return None
    if len(detected_faces) == 0:
        return None
    
    # If a face is detected, use the first detected face to determine gender
    detected_gender = detected_faces[0].face_attributes.gender
    
    # Return 'male' or 'female' based on the detected gender
    if detected_gender == 'male':
        return 'male'
    else:
        return 'female'

def create_image_analysis_df(image_url):
    '''This fuction create a dataframe that contains the image name, URL and the variables in interest
    azure_storage_path: Azure Storage Account path
    image_folder_item: A list containing all image names. e.g., "sample_image.jpg"
    Return a pandas dataframe
    '''
    
    with open('new.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])

        with open('full_invalid.csv', mode='a', newline='') as file2:
            invalid = csv.writer(file2)

            for i in range(0, len(image_url)):
                # Call all previous functions to extract variables in interest through Azure
                read_image_url = image_url[i]
                name = read_image_url.split("/")[-1]
                print(str(i) + ":" + read_image_url)

                # OCR part
                try:
                    line_text, line_bouding_box = azure_ocr(read_image_url = read_image_url)
                except:
                    invalid.writerow([i, read_image_url])
                    file2.flush()
                    continue

                try:
                    has_text_ocr = text_presence_by_ocr(line_text = line_text)
                except:
                    invalid.writerow([i, read_image_url])
                    file2.flush()
                    continue

                try:
                # Image tag part
                    image_tags = azure_image_tag(read_image_url = read_image_url)
                except:
                    invalid.writerow([i, read_image_url])
                    file2.flush()
                    continue
                
                try:
                    has_text_tag = text_presence_by_tag(image_tags)
                except:
                    invalid.writerow([i, read_image_url])
                    file2.flush()
                    continue

                try:
                    unique_tags = find_unique_tag(image_tags=image_tags)
                except:
                    invalid.writerow([i, read_image_url])
                    file2.flush()
                    continue
                
                # Image Description part
                try:
                    description_text, description_confidence = azure_image_description(read_image_url=read_image_url)
                except:
                    invalid.writerow([i, read_image_url])
                    file2.flush()
                    continue

                # Image Category part
                try:
                    category_name, category_score = azure_image_category(read_image_url=read_image_url)
                except:
                    invalid.writerow([i, read_image_url])
                    file2.flush()
                    continue

                # gender = azure_detect_gender(read_image_url=read_image_url)
                row = [i, name, read_image_url, line_text, line_bouding_box, has_text_ocr, image_tags, has_text_tag, unique_tags, description_text, description_confidence, category_name, category_score]
                print(row)
                writer.writerow(row)
                file.flush()
                print("write success")

    return "complete"

if __name__ == "__main__":
    image_urls = []
    image = pd.read_csv('new.csv', index_col=0)
    processed_urls = set(image['image_storage_URL'])
    blobs = container_client.list_blobs()
    for blob in blobs:
        image_url = f"https://imageins.blob.core.windows.net/{container_name}/{blob.name}"
        print(image_url)
        if image_url not in processed_urls:
            image_urls.append(image_url)

    create_image_analysis_df(image_urls)
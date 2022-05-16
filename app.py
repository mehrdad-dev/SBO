import streamlit as st
# import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np
import torch
import clip
import cv2
import os
# ================================================================================================

OBJDETECTIONREPO = 'ultralytics/yolov5'
DEVICE = 'cpu'
N = 5

def objectDetection(img_path:str, model) -> list:
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_name = os.path.basename(img_path)
    image_name = image_name.split('.')[0]

    result = model(image)
    result.crop(save_dir=image_name)
    detectedObjects = result.render()[0]
    path = image_name + '/crops/**/*.jpg'    

    listOfObjects = []
    for filename in glob(path):
        obj = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        listOfObjects.append(obj)     

    return listOfObjects, detectedObjects


def similarity_top(similarity_list:list, listOfObjects:list, N) -> tuple():
    results = zip(range(len(similarity_list)), similarity_list)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    images = []
    scores=[]
    for index, score in results[:N]:
        scores.append(score)
        images.append(listOfObjects[index])

    return scores, images


def findObjects(listOfObjects:list, query:str, model, preprocess, device:str, N) -> tuple():
    objects = torch.stack([preprocess(Image.fromarray(im)) for im in listOfObjects]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(objects)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = model.encode_text(clip.tokenize(query).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Retrieve the description vector and the photo vectors
    # @: https://docs.python.org/3/whatsnew/3.5.html?highlight=operator#whatsnew-pep-465
    similarity = (text_features.cpu().numpy() @ image_features.cpu().numpy().T) * 100
    similarity = similarity[0]
    scores, images = similarity_top(similarity, listOfObjects, N=N)     

    return scores, images


def pipeline(image, query):
    listOfObjects, detectedObjects = objectDetection(image, objectDetectorModel)
    scores, images = findObjects(listOfObjects, query, objectFinderModel, preProcess, DEVICE, N)
    detectedObjects = np.array(detectedObjects)
    st.image(detectedObjects, caption='Detected Objects', use_column_width=True)
    for index, img in enumerate(images):
        img = np.array(img)
        st.image(img, caption=scores[index])

# ================================================================================================
# image = Image.open('assets/BBC_News.png')
# st.image(image, caption=None)

st.title('Search Between the Objects - SBO')
st.markdown(
    'By [Mehrdad Mohammadian](https://mehrdad-dev.github.io)', unsafe_allow_html=True)

about = """
This demo provides a simple interface to search between the objects in a given image.
SBO Net is based on the [Yolo v5](https://github.com/ultralytics/yolov5) and the [Openai CLIP](https://github.com/openai/CLIP) models.
"""
st.markdown(about, unsafe_allow_html=True)


# ================================================================================================
uploaded_file = st.file_uploader("Upload an jpg image", type=["jpg"])
if uploaded_file is not None:
    image = uploaded_file.getvalue()
    
# ================================================================================================
OBJDETECTIONMODEL = st.selectbox(
     'Which model do you want to use for object detection?',
     ('yolov5x6', 'yolov5n', 'yolov5s', 'yolov5x'))

st.info('yolov5x6 is accurate.')
st.write('You selected:', OBJDETECTIONMODEL)

# ================================================================================================
FINDERMODEL = st.selectbox(
     'Which model do you want to use for object finder?',
     ('ViT-B/32', 'ViT-B/16'))
st.write('You selected:', FINDERMODEL)

# ================================================================================================
objectDetectorModel = torch.hub.load(OBJDETECTIONREPO, OBJDETECTIONMODEL)
objectFinderModel, preProcess = clip.load(FINDERMODEL, device=DEVICE)

# ================================================================================================

query = st.text_input('Search Query:')

left_column, right_column = st.columns(2)
pressed = left_column.button('Search!')
if pressed:
    pipeline(image, query)
    st.balloons()

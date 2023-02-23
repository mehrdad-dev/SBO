# Search Between the Objects - SBO

Search between the objects in an image, and cut the region of the detected object.

## About this project
CLIP model was proposed by the OpenAI company, to understand the semantic similarity between images and texts.
It's used for preform zero-shot learning tasks, to find objects in an image based on an input query.
![Mehrdad Mohammadian](https://raw.githubusercontent.com/mehrdad-dev/SBO/main/assets/clip.png)

CLIP pre-trains an image encoder and a text encoder to predict which images were paired with which texts in our dataset. We then use this behavior to turn CLIP into a zero-shot classifier. We convert all of a dataset’s classes into captions such as “a photo of a dog” and predict the class of the caption CLIP estimates best pairs with a given image.

Also, YOLOv5 was used in the first step of the method, to detect the location of the objects in an image.


## Demo
Demo is ready!

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/mehrdad-dev/sbo/main)

(Sometimes, the Streamlit website may crash! because models are heavy for it.)

## Notebook
Run this notebook on Google Colab and test on your images!
(It works both on CPU and GPU)

[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/mehrdad-dev/SBO/blob/master/notebooks/search_objects_on_images.ipynb)


[Open in nbviewer](https://nbviewer.org/github/mehrdad-dev/SBO/blob/main/notebooks/search_objects_on_images.ipynb#)

## Limitations
Obviously object detector model only can find object classes learned from the COCO dataset. So if your results are not related to your query, maybe the object you want is not in the COCO classes.

## Example
Sorted from left based on similarity.

**Query:** Clock

![Mehrdad Mohammadian](https://raw.githubusercontent.com/mehrdad-dev/SBO/main/test_images/ex1.png)

![Mehrdad Mohammadian](https://raw.githubusercontent.com/mehrdad-dev/SBO/main/test_images/ex1-1.png)

 
**Query:** wine glass

![Mehrdad Mohammadian](https://raw.githubusercontent.com/mehrdad-dev/SBO/main/test_images/ex2.png)

![Mehrdad Mohammadian](https://raw.githubusercontent.com/mehrdad-dev/SBO/main/test_images/ex2-1.png)


**Query:** woman with blue pants

![Mehrdad Mohammadian](https://raw.githubusercontent.com/mehrdad-dev/SBO/main/test_images/ex3.png)

![Mehrdad Mohammadian](https://raw.githubusercontent.com/mehrdad-dev/SBO/main/test_images/ex3-1.png)


## License

[MIT license](https://github.com/mehrdad-dev/SBO/blob/main/LICENSE)

## Based on
- [OpenAI CLIP](https://pytorch.org/hub/ultralytics_yolov5/)
- [YOLO v5](https://github.com/openai/CLIP)



<a href="https://www.buymeacoffee.com/mehrdaddev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

<a href="http://www.coffeete.ir/mehrdad-dev">
       <img src="http://www.coffeete.ir/images/buttons/lemonchiffon.png" style="width:260px;" />
</a>

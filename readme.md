# Search Between the Objects - SBO

Search between the objects in an image.

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
- [Openai CLIP](https://pytorch.org/hub/ultralytics_yolov5/)
- [YOLO v5](https://github.com/openai/CLIP)

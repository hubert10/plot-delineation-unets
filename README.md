# [keras-unet-collection] UNet and Attention-UNet models for Plot Delineation as an application of Instance Segmentation.

[Notice] : The original code came from [here](https://www.youtube.com/watch?v=L5iV5BHkMzM&t=1176s) and was adjusted to be work on Field Boundary Detection.


 The datasets come from Sentinel -2 images for Agriculture Rice growing fields in Senegal [here](https://drive.google.com/drive/folders/1t_aWfuQ_K-jWCehr0cz7Czk60T3UpceB?usp=sharing).


# Training

* [train_attention_unet_model.py](train_attention_unet_model.py) Run the above script to train your attention unet model.

It includes code to run object detection and instance segmentation on arbitrary images.


# Prediction

* [pred_attention_unet.py](pred_attention_unet.py) shows how to Run the above script to make predictons with the trained attention unet model.


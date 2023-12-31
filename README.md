# <div align="center">PPP</div>
Privacy preserving PropTech.  

## <div align="center">Poster</div>
![initial](https://github.com/darkenergy814/PPP/assets/79552567/4f89f8b5-7594-4223-9bcb-8e37af454544)  


## <div align="center">Test Code</div>
There are two versions. One is with yolov5 and LaMa Inpainting, and the other is with mask_RCNN with LaMa Inpainting. You could try it on Colab.  
<a style='display:inline' target="_blank" href="https://colab.research.google.com/drive/1RriMSIG31VYJoelEpTrrJGILWfjZTOvn?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>  

## <div align="center">How to use</div>
### <div>Yolov5+LaMa Inpainting Version</div>
1. Use code block named "settings for ppp" for setting.  
    - Git clone PPP  
    - Install packages for yolov5 and LaMa Inpainting
    - Install an LaMa model file
    ```
    PPP
    │ lama   
    │  └── big-lama   
    │      ├── config.yaml # model configs  
    │      └── models           
    │          └── best.ckpt # model weights     
    ```  
    - Make an directory for LaMa Inpainting prediction
    ```
    PPP
    │ lama
    │  └── data_for_prediction
    │      ├── sample1.png # example
    │      └── sample1_mask.png # example
    ```  

2. Use code block named "Change file path and start object detection and Inpainting" for object detection, mask generating, and mask inpainting.
    - By using maskGenerator.yolov5Mask, you could make an mask(black and white) with yolov5.
    - Make an Inpainted Image from the mask made by the maskGenerator.yolov5Mask.

### <div>MaskRCNN+LaMa Inpainting Version</div>
1. Use code block named "Change python version 3.10 -> 3.7" for change python version.
    - Python version change in google colab
2. Use code bloack named "Settings for ppp" for setting
    - Git clone PPP
    - Install packages for mask rcnn and LaMa Inpainting
    - Install LaMa model file
    ```
    .root
    │ lama   
    │  └── big-lama   
    │      ├── config.yaml # model configs     
    │      └── models           
    │          └── best.ckpt # model weights    
    ```
    - Make an directory for LaMa Inpainting prediction.
    ```
    PPP
    │ lama
    │  └── data_for_prediction
    │      ├── sample8.png # example
    │      └── sample8_mask.png # example
    ```
3. Use code block named "Change file path and start Inpainting" for object detection, mask generating, and mask inpainting.
    - By using maskGenerator.mrcnnMask, you could make an mask(black and white) with mask RCNN.
    - After making mask, reset gpu ram for LaMa Inpainting
    - Make an Inpainted Image from the mask made by the maskGenerator.mrcnnMask
  
## <div align="center">Requirements</div>
We changed a few codes of yolov5 and Mask RCNN for Running both models(yolov5+LaMa or mask RCNN+Lama).
### <div>Yolov5+LaMa Version</div>
python 3.10  torch 2.0.0  cu118  etc
### <div>Mask RCNN+LaMa Version</div>
python 3.7  torch 1.8.0  cu111  etc  

## <div align="center">License</div>
YOLOv5 is available under two different licenses:

- **AGPL-3.0 License**: See [LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) file for details.
- **Enterprise License**: Provides greater flexibility for commercial product development without the open-source requirements of AGPL-3.0. Typical use cases are embedding Ultralytics software and AI models in commercial products and applications. Request an Enterprise License at [Ultralytics Licensing](https://ultralytics.com/license).

## <div align="center">Citation</div>
Use this bibtex to cite this repository:
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```

###Download segmentation model SAM:

```wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth```


## Run detection and segmentation 

```python infer_seg.py```

## ToDo

1. Add config file that takes in detection and segmentation config.
2. If the objects are not part of MS-COCO, search for custom object detectors
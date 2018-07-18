# PyTorch-RetinaNet
Train _RetinaNet_ with _Focal Loss_ in PyTorch.

Reference:  
[1] [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)  


## Code Structure
1. *datagen.py* for ListDataset, based on torch.dataset, it uses the other files:
    - *transform.py*: data transforms [eg. random_flip, random_crop, resize, etc.]
    - *utils.py*: calculate box_iou, box_nms & some utils for displaying
    - *encoder.py*: encode data to 9 anchor boxes.

2. *train.py*:
    - *fpn.py*: Feature Pyramid Network definition & architecture. **TODO**: Review the code & refer to the paper
    - *loss.py*: Focal loss definition
    - *retinanet.py*: RetinaNet definition, also based on FPN50 inside *fpn.py*
3. *test.py*: test using the trained model & decode the results.

4. *script/get_state_dict.py*: use the pre-trained weights & generate FPN50 *feature_extractor* weights

### conclusion
    the code is good, TODO: refer to the paper & review the code, understanding all the code snippets, and train you own data



## Train-You-Own-Data



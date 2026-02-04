# 实现细节（针对基线方法和改进方法）	
batch size	8
epoches	72
优化器	AdamW
image size	640x640
lr	0.0008
gpu	一块A100

# 对比实验
所有方法使用mmdetection好Ultralytics中的默认实验设置，为了公平对比，所有模型使用640x640输入，且使用主干的8倍，16倍，32倍下采样特征用于后续任务管道。
对比方法：
YOLO11
YOLO26
CO-DETR
DETR
DINO
Dynamic_rcnn
Faster_rcnn
libra_rcnn
RetinaNet
RtmDET
VFNet
VitDet

# 实验设计与评估指标
## 非消融实验
### GWHD 2021 上的实验
1. 在GWHD 2021 的所有模型（基线，改进，对比）的COCO指标（AP，AP_50，AP_75，AP_s，AP_m 	  AP_l） 和参数量和计算量。疑问：需不需要放上（AR_1，AR_10，AR_100，AR_s，AR_m，AR_l）的一个或多个这些？因为不好计算对比方法的WDA，这里就不要方法WDA指标了。
2. 在GWHD上18个子域的基线和改进方法的指标，每个子域的TP，FP，FN，AI，以及合适的COCO指标。然后最后总结基线和改进的WDA。
3. 在GWHD四个生长阶段的基线和改进方法的指标，每个阶段的TP，FP，FN，AI，以及合适的COCO指标。然后最后总结基线和改进的WDA。
4. DRPD的整个数据集上（不区分三种高度）的所有模型（基线，改进，对比）的COCO指标和参数量和计算量。同GWHD 2021.
5. DRPD在三种高度上的基线和改进方法的指标，每个高度的TP，FP，FN，AI以及合适的COCO指标。然后最后总结基线和改进的WDA。
6. 是否还有其它需要做的实验，比如 1.对测试集按密度（每个图的实例个数）进行划分多个组，在多个组上的TP，FP，FN，AI以及合适的COCO指标。然后最后总结基线和改进的WDA。2. 这些都是检测相关的，需要弄一些与计数相关的吗。

## 消融实验
按照你的建议
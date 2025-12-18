# 实验七

------------------------------------------------------------------------

## 一、实验目的

-   掌握目标检测的核心评价指标，特别是交并比（IoU）的数学定义与实际应用。
-   理解 Faster R-CNN 模型的基本架构，熟悉
    RPN（区域生成网络）在候选框提取中的作用。
-   通过查阅官方文档，掌握 PyTorch 推理模式下的数据流转逻辑（CPU 与 GPU
    之间的数据交换）。
-   通过调试验证，掌握非极大值抑制（NMS）算法在消除冗余检测框、优化模型输出质量中的关键作用。

------------------------------------------------------------------------

## 二、实验内容与过程

### 2.1 环境配置与类别映射
实验首先配置了中文字体以支持可视化输出，并定义了 COCO 数据集的类别映射表。

```
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COCO_CLASSES = ['__background__', 'person', 'bicycle', 'car', ..., 'toothbrush']
```

### 2.2 模型加载与理论架构
采用 ResNet-50 作为骨干网络的 Faster R-CNN 模型。

```
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.to(device).eval()
```

### 2.3 核心预测逻辑

#### 2.3.1 推理代码实现

预测逻辑涵盖了从图像预处理到张量上载显存，再到模型前向计算的全过程。

```
def get_raw_prediction(image_path):
    # 图像预处理：标准化并转换为 Tensor
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device) # 数据移动至 GPU 参与计算
    
    with torch.no_grad():
        # 执行前向传播，获取原始预测结果
        prediction = model([img_tensor])[0]
    
    return img, prediction
```

#### 2.3.2 发现问题：同一个自行车出现多个检测框
在对实验集图片（尤其是包含自行车的场景）进行测试时，观察到明显的冗余现象：同一个自行车物理实体被多个相互重叠的预测框包围,尽管这些框的置信度评分（Scores）均超过了 0.7.

![image](/output_results/result_dog_bike_car.jpg)

#### 2.3.3 深度理论分析：冗余产生的根源

RPN 与锚框机制（Anchors）：Faster R-CNN 的 RPN 网络在特征图的每个像素点生成不同尺度（Scales）和长宽比（Aspect Ratios）的锚框。对于自行车这种具有细长结构的目标，会有多个不同形状的锚框同时与该物体产生较高的交并比（IoU）。在训练过程中，这些锚框都会被标记为正样本进行回归训练。

坐标回归的收敛性：模型在推理时，RPN 会生成约 2000 个候选区域（Proposals）。虽然这些框起始位置不同，但经过后端的边界框回归（Bounding Box Regression）网络修正后，多个高分的预测框往往会收敛到物体的同一个特征区域。

交并比（IoU）的度量作用：冗余的本质是模型对同一空间区域进行了多次重复预测。为了衡量这种重叠程度，引入 IoU 指标，定义如下：

$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

实验发现，自行车的多个预测框之间的 IoU 往往在 0.5 到 0.8 之间。原始输出逻辑缺乏有效的互斥机制，导致这些高度重叠的框被全数保留。

### 2.4 代码改进与 NMS 优化决策

针对上述问题，实验引入了 `torchvision.ops.nms`
算子进行手动干预。通过计算预测框之间的交并比（IoU），在特定类别内仅保留置信度最高的边界框。

``` python
from torchvision.ops import nms
import numpy as np

def post_process(prediction, score_thresh=0.7, nms_thresh=0.3):
    # 步骤 A：初步筛选置信度高于 0.7 的预测结果
    scores = prediction['scores']
    mask = scores > score_thresh
    boxes = prediction['boxes'][mask]
    labels = prediction['labels'][mask]
    scores = scores[mask]

    # 步骤 B：执行 NMS 算法，解决“自行车多框”问题
    # nms_thresh 设为 0.3，意味着重叠面积超过 30% 则视作重复检测
    keep_idx = nms(boxes, scores, nms_thresh)
    
    # 最终结果移回 CPU 进行可视化处理
    return boxes[keep_idx].cpu().numpy(), labels[keep_idx].cpu().numpy()
```

------------------------------------------------------------------------

## 三、实验结果与分析

### 3.1 实验结果描述

通过对比改进前后的输出图片，实验结果显示：在应用手动 NMS（阈值设定为
0.3）后，原先堆叠在自行车周围的 3--4
个冗余框被有效抑制，每个物理实体成功实现了"一物一框"的精准定位。模型在
GPU 加速下的单图推理耗时保持在 0.1 s 以内，满足实时处理的基本需求。

### 3.2 关键代码分析

-   **推理模式设置（`model.eval()`）**\
    此操作确保了 Dropout 和 Batch Normalization
    层在实验中保持固定，保证了同一图片在不同测试批次下输出结果的一致性。

-   **置信度阈值决策（`score_threshold = 0.7`）**\
    该参数作为第一道防线，过滤掉了 RPN
    产生的低置信度背景噪声。实验观察到，若此值过低，图像中会出现较多无关的散碎框。

-   **NMS 算子与 IoU 理论应用**\
    代码核心在于利用 IoU 判断重叠程度.


    针对自行车检测中的多框问题，将 `nms_threshold` 设为 0.3
    是基于调试验证后的最优决策。较小的阈值能够强制删除那些重叠度即便不高但属于同一目标的冗余框，从而显著改善了检测框的唯一性。


------------------------------------------------------------------------

## 四、实验小结

本次实验通过对 Faster R-CNN 模型的全流程实践，达到了预期的学习目标。

-   **理论验证**：通过代码复现，验证了 Faster R-CNN "区域生成 +
    分类回归"的二级结构在处理复杂目标时的有效性。
-   **问题解决**：实验中遇到的"多框冗余"问题是典型的工程挑战。通过查阅文档并引入手动
    NMS 逻辑，我掌握了利用算法参数（如 IoU
    阈值）调节模型性能的调试方法。
-   **框架熟悉度**：本次实验加深了对 PyTorch 生态中 `torchvision`
    工具包的理解。预训练权重的加载与 GPU
    加速的配置过程体现了深度学习框架在基础研究中的高效性与便利性。

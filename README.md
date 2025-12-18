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

### 2.1 实验环境初始化与模型加载

在实验开始阶段，首先进行环境配置。为了提升推理效率，通过 `torch.device`
接口检测并启用显卡加速。随后，采用 PyTorch 官方推荐的 `weights`
参数化方式加载预训练模型，以替代已弃用的 `pretrained` 参数。

``` python
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 Faster R-CNN 模型，采用 ResNet-50 作为骨干网络并加载默认预训练权重
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.to(device)
model.eval()  
```

### 2.2 数据预处理流程

为了使本地图像满足模型的输入规范，定义了如下预处理逻辑。通过
`T.ToTensor()` 将图像像素值由 `[0, 255]` 归一化至 `[0, 1]` 浮点区间。

``` python
from PIL import Image

def preprocess_image(img_path):
    # 读取本地图片并转换为 RGB 格式
    img = Image.open(img_path).convert("RGB")
    # 将 PIL 图像转换为模型所需的 Tensor 格式并上载至 GPU
    transform = T.Compose([T.ToTensor()])
    return img, transform(img).to(device)
```

### 2.3 实验中发现的问题：同一物体的多框冗余

在初步测试中，使用包含"自行车"的本地图片进行推理。观察原始输出发现：由于
RPN
网络会在同一目标周围生成多个候选锚框（Anchors），导致同一个自行车实体被多个边界框覆盖。虽然这些框的置信度均较高，但严重的视觉重叠降低了检测的准确性。
![image](/output_results/result_dog_bike_car.jpg)
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

## 三、实验结果与分析（重点）

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
    代码核心在于利用 IoU 判断重叠程度，其数学公式如下：

    $$
    \text{IoU} = \frac{\mathcal{A} \cap \mathcal{B}}{\mathcal{A} \cup \mathcal{B}}
    $$

    针对自行车检测中的多框问题，将 `nms_threshold` 设为 0.3
    是基于调试验证后的最优决策。较小的阈值能够强制删除那些重叠度即便不高但属于同一目标的冗余框，从而显著改善了检测框的唯一性。

-   **硬件数据同步（`.to(device)` 与 `.cpu()`）**\
    这是实验能够顺畅运行的技术前提。原始图像通过 Tensor
    形式上载至显存参与高速矩阵运算，而推理出的坐标信息需通过 `.cpu()`
    下载回内存，以便 `matplotlib` 进行后续的图像渲染与保存。

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

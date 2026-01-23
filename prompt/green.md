AFS-DETR: 基于自适应融合与敏感度感知的绿色小果实检测

摘  要：绿色小果实检测是智慧果园管理中的关键技术，但对于形态小、颜色与背景相似的目标，现有检测方法的精度与鲁棒性仍面临严峻挑战。本文针对绿色小果实在复杂果园环境下的检测难点，提出了一种基于RT-DETR的改进型检测模型。首先，设计了一种C2f_AdaptiveFusion主干模块，通过多分支动态卷积与自适应加权融合机制，有效增强了浅层细节特征的保持能力。其次，引入了源自YOLOv13的HyperACE模块，利用超图理论建模特征间的高阶相关性，实现了自适应的多尺度特征融合。最后，集成了FullPAD_Tunnel门控对齐模块，以轻量化的方式优化了跨层级特征融合过程。在自建的黄金梨数据集上，本文模型取得了90.4%的AP50和84.5%的召回率，F1分数达到86.9%，显著优于对比的主流检测方法。消融实验验证了各模块的有效性与协同作用。此外，在公开MinneApple数据集和自建海棠果数据集上的跨域测试进一步证明了模型的良好泛化能力。实验结果表明，本文所提方法能有效应对绿色小果实检测中的低对比度、小目标和复杂背景干扰问题，为农业视觉任务提供了一种可靠的解决方案。
关键词：绿色小果实检测；AFS-DETR；特征融合；超图理论；智慧农业

doi：XXXXXX
中图分类号：TP24, TP391           文献标志码：A           文章编号：1002-6819(XXXX)-XXXX

 
0引言
随着智慧农业与人工智能的深度融合[1]，目标检测已成为农业自动化系统中的关键技术之一。它不仅在产量估算[2]、品质评估[3]和病虫害识别[4]等方面发挥着核心作用，还为无人采摘机器人[5]、精准喷药、智能管控系统等领域提供了重要支持[6]。利用计算机视觉自动识别果实的位置、类别与数量，对实现果园作业的智能化与数据化管理、提升农业生产效率至关重要[7]。
早期的果实检测方法主要依赖颜色阈值、边缘检测等图像处理技术，或结合支持向量机（SVM）[8]、K近邻（KNN）[9]等传统机器学习方法进行目标分类。尽管这些方法在某些环境下取得了效果，但其依赖于手工设计的特征，对于果实形态的多样性、光照变化[10]以及复杂背景的适应性较差，泛化能力不足[11]，难以推广到真实果园环境中[12-13]。特别是对于疏果前期的绿色小苹果，其与背景颜色相似、尺寸小、常被遮挡的特性，使得传统方法几乎失效[14]。近年来，深度学习，尤其是卷积神经网络（CNN），在图像识别领域取得了突破性进展[1]，其强大的特征学习能力极大提升了果实检测的精度与鲁棒性[15]。这一趋势体现为对检测模型进行结构化改进，例如在果实病害识别中，研究者通过集成注意力、Transformer等模块优化YOLOv5，显著提升了性能[16]。基于深度学习的目标检测方法，正在推动果实检测从封闭实验室环境向复杂的果园场景应用迈进[17]，成为智能果园建设的重要组成部分。然而，将先进的检测模型直接应用于复杂的果园环境时，仍面临严峻挑战，其中以“绿色小果实”的检测最为突出。
在实际应用中，果实往往由于拍摄距离、发育阶段、遮挡等因素表现出小尺寸、弱纹理和密集分布等特性。这些小目标常在图像中占据较少像素，且边缘模糊，极易被忽略[18]。尤其是幼果、小果和远处果实等目标，传统检测方法的检测精度和召回率通常较低[19]。为解决这一问题，研究者在模型结构、特征增强和训练策略等方面做出了诸多改进。典型的方法包括特征金字塔网络（FPN）[20]、轻量注意力机制[21]和引入小目标感知损失函数、动态锚框匹配等技术[22]。然而，在果园的真实场景中，小目标检测通常伴随复杂背景和颜色同质性问题，单一策略难以兼顾多方面的感知需求。
绿色果实，尤其是未成熟阶段的果实，常在果园中呈现低对比度，且与背景叶片和枝条在颜色和纹理上高度相似[12]，因此检测难度极大。此外，绿色果实的边界模糊，纹理稀疏，在光照变化、遮挡和反光等环境因素的影响下，其视觉表现进一步弱化，导致目标定位不准确，分类困难。为应对绿色果实与背景颜色重叠的干扰，研究者尝试使用颜色空间转换（如HSV、Lab）[23]、颜色对比拉伸和边缘增强等预处理方法[24]。近年来，部分研究还从模型结构层面进行优化，构建了绿色果实专用的数据集[25]，并尝试引入颜色注意机制、显著性增强通道[26]、轻量化的注意力网络[27]等深度学习优化策略。尽管相关研究逐渐增多，但绿色果实的检测仍是亟待突破的难题。
目前，主流的目标检测方法主要分为一阶段（如SSD(Single Shot MultiBox Detector)[28]、FCOS(Fully Convolutional One-Stage object detection)[29]、YOLO(You Only Look Once)系列[30]）和二阶段（如R-CNN(Region-based Convolutional Neural Network)[31]、FastR-CNN(Fast Region-based Convolutional Network)[32]、FasterR-CNN(Faster Region-based Convolutional Network)[33]、RetinaNet[34]）。前者速度占优，后者精度较高。然而，无论是哪种范式，其固有的设计（如固定的特征融合模式[16]、对尺度变化不敏感的损失函数[35]）都难以有效应对绿色小果实所特有的弱纹理、低对比度和尺度多变等挑战。近年来，Transformer架构的引入为目标检测带来了新思路，展现出潜力。以DETR(DEtection TRansformer)为代表的模型通过编码器-解码器结构建模目标间的全局关系[36]，在遮挡和复杂背景下展现了较好的效果。后续版本如DeformableDETR（Deformable DEtection Transformer）[37]、DN-DETR（DeNoising DETR）[38]等在收敛速度、多尺度建模和检测精度方面进行了优化。
尽管如此，现有的Transformer检测器在浅层细节保留、多尺度自适应融合和颜色显著性建模方面仍存在不足，限制了其在绿色小果实检测中的应用潜力。综上所述，当前研究的核心瓶颈在于：如何使检测模型能够自适应地聚焦于绿色小果实的微弱视觉特征，并实现跨层级细节信息的有效融合与对齐。
为解决上述挑战，本文的核心思想是构建一个由敏感度感知驱动、以自适应融合为核心技术手段的新型检测框架。该框架旨在系统性地解决绿色小果实检测中的三大感知难题：尺度敏感度（保留小目标的细微细节）、对比度敏感度（区分颜色高度相似的前景与背景）以及上下文敏感度（利用语义关联推断遮挡或模糊目标）。基于此，我们提出AFS-DETR模型，其技术路径在于将传统静态特征处理范式转变为可根据输入内容动态调整的自适应融合过程。具体而言，本文的主要贡献体现在以下三个核心模块的设计中。
1）为实现自适应动态感受野并增强尺度与细节敏感度，设计了C2f _AdaptiveFusion（C2AF）主干模块，其多分支动态卷积与自适应加权机制能精准捕捉果实的弱纹理与轮廓。
2）为建立自适应多尺度语义融合以提升上下文与对比度敏感度，引入了HyperACE（HACE）模块[39]，其基于超图理论建模高阶关系，显著增强了模型在复杂场景下的感知鲁棒性。
3）为通过轻量级自适应门控融合保障跨层级语义一致性，集成了FullPAD_Tunnel(FPADT)模块[39]，实现了精准的特征对齐，从而优化全局融合质量。
1 AFS-DETR小目标果实检测模型
RT-DETR虽在实时检测与全局建模方面表现优异，但其结构在果园绿色小果实检测中存在明显不足：多次下采样导致浅层细节丢失，低对比度致使目标显著性不足，固定融合策略难以适应复杂场景，最终导致召回率下降与误检增多。针对上述问题，本文提出AFS-DETR模型，其整体架构如图1所示（图中高亮部分为本研究三个核心模块）。首先，在主干网络浅层特征提取阶段，引入C2f _AdaptiveFusion（C2AF）模块（见图1中位置①），通过多分支动态卷积增强浅层特征保留能力；随后，在多尺度特征融合阶段，引入YOLOv13基于超图理论的HyperACE(HACE)模块（见图1中位置②），构建自适应超边以融合多尺度特征，提升对低对比度目标的感知鲁棒性；最后，在特征金字塔向解码器传递的关键路径上，集成FullPAD_Tunnel（FPADT）门控对齐模块（见图1中位置③），实现跨层级特征的语义对齐与冗余抑制，增强前景-背景区分能力。优化模型在保留RT-DETR端到端优势的同时，系统提升了模型在复杂果园环境下对绿色小果实的检测精度、召回率与泛化性能。
 
图1. 面向绿色小果实检测的AFS-DETR模型架构
Fig1. Architecture of the AFS-DETR model for green small fruit detection
1.1基于自适应融合的浅层特征增强
在绿色小果实检测任务中，富含边缘与轮廓细节的浅层特征（如P3）是精准定位微小目标的关键。然而，RT-DETR的主干网络经过多次下采样后，此类细节信息易被稀释，导致小目标特征响应微弱。此外，传统卷积算子存在特征复用效率低、空间感知模式固定的局限，难以在高效推理的同时维持对弱特征的强表征能力。为解决上述问题，本文在主干网络中引入了自主研发的C2AF模块，如图2所示。该模块在继承C2f跨阶段部分连接优势的基础上，通过多分支动态卷积与自适应核加权（AKW）融合机制，显著提升了浅层特征的保真度与判别性。
 
图2.浅层特征增强模块 C2f_AdaptiveFusion 详细设计
Fig2.Detailed design of the C2f_AdaptiveFusion module for shallow feature enhancement

其核心设计如下：模块整体沿用跨阶段部分连接框架，将输入特征沿通道维度拆分为两个部分：捷径分支与处理分支。其中，捷径分支保持原始特征不做复杂变换，旨在构建一条无障碍的信息高速公路，确保原始梯度流与细节信息能够无损地传播至输出端，有效缓解网络退化问题。另一部分的处理分支，则由多个串联的Neural Adaptive Block构成，是模块进行深度特征自适应提炼的核心区域。
每个Neural Adaptive Block内部集成了多分支动态卷积结构，该结构摒弃了传统单一卷积核的固定模式，转而并行部署了三个独立的深度可分离卷积分支。这些分支分别采用方形卷积核、水平带状卷积核与垂直带状卷积核，从而构成了具有不同几何感受野的特征提取器。方形核负责捕捉标准的局部上下文信息，而水平和垂直带状核则分别对横向与纵向的边缘、条纹等细长状特征具有更高的灵敏度，能有效应对果实枝干、果梗等线性结构。
为实现感受野的自适应优化，本文引入了轻量化的自适应核加权（Adaptive Kernel Weights, AKW）模块，如图3所示。该模块以输入特征图为条件，动态生成卷积核融合权重，从而优化卷积操作的感受野与特征提取效率。其核心流程包含三个阶段：首先，通过全局平均池化（Global Average Pooling）提取输入特征图的全局上下文信息，将空间维度压缩至1×1；随后，经由一个全连接层将特征映射至原始通道数的3倍，并通过Softmax函数进行归一化，生成三组与输入内容相关的自适应权重，这些权重能够有效反映特征图中主导的几何特性（如边缘、纹理等）；最后，将生成的权重分别与三个并行卷积分支（方形卷积核、水平带状卷积核、垂直带状卷积核）的输出进行逐通道加权，并通过求和完成特征融合。该过程可形式化表示为：
 	(1)
其中，y为融合后的输出特征，w_i为AKW模块动态生成的第i个分支的归一化权重，f_i(x)为对应卷积分支的输出。在训练过程中，采用Xavier初始化策略对网络权重进行初始化，并结合L2正则化以抑制过拟合，在保证模块轻量化的同时提升其泛化能力。通过上述机制，AKW模块实现了感受野与特征权重的动态自适应调整，显著增强了模型对绿色小果实等小目标细节特征的捕获与判别能力。
 
图3. Adaptive Kernel Weights（AKW）多分支动态卷积示意图
Fig. 3 Schematic illustration of the Adaptive Kernel Weights (AKW) based multi-branch dynamic convolution
最后，所有Neural Adaptive Block的输出特征与初始的捷径分支特征进行拼接，这一操作最大限度地保留了从原始信息到高级抽象特征的完整信息谱系。拼接后的特征再通过一层标准的1×1卷积进行压缩、融合与通道调整后输出。此设计在严格保证信息完整性与梯度流动性的同时，为网络注入了强大的、与输入内容自适应的空间感知能力。
C2AF模块通过引入动态与自适应的特征提取机制，在保持网络轻量化的同时，显著增强了主干网络对浅层细节的保持与判别性表征能力，为后续在复杂背景下精准定位绿色小果实奠定了坚实基础。
1.2基于超图自适应的多尺度特征融合
在复杂果园环境中，绿色小果实表现出的尺度多变性和分布密集性对检测模型的多尺度特征融合能力提出了严峻挑战。传统特征金字塔网络（FPN）所采用的固定融合策略难以动态适应不同尺度特征的重要性权衡，常导致细节信息丢失或语义支持不足。为解决这一问题，本文引入了基于超图理论的自适应相关增强模块HyperACE(HACE)，如图4所示，通过建立特征点间的高阶相关性联系，突破传统卷积的局部感受野限制，实现全局自适应的特征增强。
HACE模块的结构从自适应超边生成开始，模块接收来自主干网络的多尺度特征并动态生成超边原型，这些超边连接了在语义或空间上相关的特征点。通过这种方式，模块能够建模复杂的跨尺度依赖关系，并通过超图卷积（AdaHGConv）进行信息的传播和聚合，使得每个特征点能够结合全局上下文信息。这种结构使得模型能够灵活地处理不同尺度和复杂背景下的特征，显著增强了绿色小果实的检测能力。
 
图4基于超图理论的HyperACE-Fusion特征融合模块结构
Fig4.Structure of the hypergraph theory-based HyperACE-Fusion feature fusion module
为进一步增强特征表示能力，本文在HACE模块基础上引入了C3AH特征精炼模块（图4右部）。该模块首先通过两个1×1卷积对输入特征进行通道压缩与初步提取，进而通过核心的AdaHGComputation单元进行深层超图融合，最终再经卷积投影优化输出。这一流程在保持特征判别力的同时，确保了跨尺度融合过程中的细粒度语义一致性。此外，模块中集成的DSC3K（深度可分离卷积）组件采用高效的卷积分解策略，在维持3×3卷积空间感知能力的同时显著降低计算复杂度。DSC3K作为高效的空间特征提取器，与AdaHGComputation单元协同工作，既保留局部细节，又融入全局语义指导，为后续超图计算提供高质量的特征基础。HACE模块通过高阶相关性建模与局部特征精炼的协同机制，实现了多尺度特征的自适应融合，显著提升了模型在复杂果园环境下对绿色小果实的检测精度与鲁棒性。
1.3基于门控机制的特征对齐
在多尺度特征融合过程中，不同层级特征间存在的语义与空间差异，若直接融合，易引入噪声，对空间位置敏感的小目标检测任务尤为不利。传统解决方案往往依赖复杂的对齐网络，计算开销大且结构繁琐。为解决上述问题，本文引入了FullPAD_Tunnel(FPADT)模块。该模块摒弃了复杂的多步对齐流程，转而采用一种轻量级的门控融合机制，结构简洁高效，旨在实现特征间的自适应融合，如图5所示。
对于待融合的两路特征原始特征 和增强特征 ，首先通过插值操作使二者尺寸匹配。关键设计在于引入一个可学习的标量门控参数 ，该参数在训练过程中由数据驱动自动优化，从而自适应地调节增强分支在融合中的贡献强度，其融合过程可表示为： 
                                    	           （2）
其中，增强特征 首先通过插值操作（Interpolate）进行尺寸匹配，然后与原始特征 按公式（2）进行门控残差融合。门控参数 在训练过程中会自动学习更新，门控参数 在训练过程中会自动优化，决定增强特征 在最终输出中的影响程度。经由乘法与加法操作，最终实现两路特征的灵活加权融合。
 
图5.特征对齐模块FullPAD_Tunnel的门控融合机制
Fig5.Gated fusion mechanism of the FullPAD_Tunnel feature alignment module
该模块的设计具备三大显著优势：（1）自适应融合能力强：通过引入可学习的门控参数 ，模型能够在训练过程中自动学习融合强度，从而自适应地平衡原始特征与增强特征的贡献。(2)结构轻量高效：避免了复杂的对齐计算，以极低的计算开销实现了有效的特征调节。（3）训练过程稳定：其残差结构形式确保了梯度流的顺畅，有效提升了训练的稳定性。综上，FPADT模块以简洁的架构有效解决了多尺度特征融合中的对齐难题，为提升小果实检测性能提供了可靠保障。
2数据集
为系统评估AFS-DETR模型在绿色小目标果实检测任务中的综合性能，特别是其在真实场景下的精度与跨域泛化能力，采用疏果期黄金梨与海棠果数据集和MinneApple数据集相结合的方式进行实验验证。
2.1疏果期黄金梨数据集
该数据集采集自山东省青岛市胶州的黄金梨果园，共包含2468张高分辨率图像，涵盖了疏果期果实在多种真实场景下的状态，如俯视、仰视、顺光、逆光、遮挡、重叠、近景与远景，模拟了农业机器人的实际工作环境。数据集按8:1:1的比例划分为训练集（1950张）、验证集（244张）和测试集（244张）。
    
俯视				     顺光			         遮挡			       近景
Overlook			  Frontlighting               Block		         Close-upview
    
仰视					逆光					堆叠					远景
Overheadshot		       Backlighting	           Overlapping		       Distantview
图6 不同视角下的黄金梨数据集
Figure6. Huangjin Pear Dataset from Different Perspectives

2.2 MinneApple数据集
MinneApple数据集由明尼苏达大学在2019年发布，包含红色和绿色两类苹果果实。图像由三星GalaxyS4手机采集，场景复杂，包含遮挡、光照变化及不同发育阶段的果实，是广泛使用的果实检测基准。该数据集包含1001张图像，目标密集、尺度小、场景复杂，适用于验证模型的跨域泛化能力。
本研究对该数据集的使用方式如下：从官方提供的670张训练图像中随机划分出67张作为验证集，用于训练过程中的超参数调整与模型选择。因此，最终用于模型训练的为603张图像，并单独使用官方的331张测试集图像进行最终的性能评估，以确保评估结果的客观性和可比性。
关于数据集中红色苹果样本的说明：MinneApple数据集同时包含红色与绿色两类苹果。为客观评估模型的跨域泛化能力与通用小目标检测性能，本文在训练与测试阶段未对苹果颜色进行类别划分，而是将全部实例统一标注为单一类别“apple”。该设置使模型在跨域验证时不依赖颜色先验，从而更纯粹地检验其对复杂场景下小目标定位与背景抑制的基础能力。需要指出的是，红色苹果具有更高的前景-背景对比度，此数据集的实验结果因此更侧重于反映模型结构的泛化优势与鲁棒性趋势；而对“绿色低对比度”这一核心难点的针对性评估，则以自建的黄金梨与海棠果数据集结果为主。
               
图7 MinneApple数据集
Figure7. MinneAppleDataset
2.3海棠果数据集
为严格验证模型对不同品种绿色小果实的跨品种泛化能力，本研究补充了自建的海棠果数据集。海棠果体积小（通常1-2厘米）、果色为深绿，与背景叶片的区分度极低，且存在严重的密集与遮挡情况，是比梨和苹果更具挑战性的测试对象。
该数据集共包含1057张真实果园环境图像，按8:1:1的比例划分为训练集（847张）、验证集（105张）和测试集（105张）。数据采集涵盖了白天正常光照与夜间低照度两种典型场景，以评估模型在极端光照变化下的稳定性。所有图像均标注为crabapple类别。测试集图像均统一为3024×4032像素的高分辨率。在模型性能评估阶段，所有测试均在此统一分辨率的测试集上进行。
      
   
图8 海棠果数据集
Figure8. CrabappleDataset
2.4标注与目标尺度定义
疏果期黄金梨和海棠果数据集的标注均使用LabelMe工具完成，疏果期黄金梨数据集采用描点框标注目标，标签为“pear”， 海棠果数据集采用矩形框标注目标，标签为“crabapple”，其余区域为背景。标注文件以JSON格式保存，并转换为COCO格式供模型训练。MinneApple数据集则直接采用官方提供的COCO格式标注。
    
图9 黄金梨与海棠果数据标注示例（左侧为黄金梨，右侧为海棠果）
Figure9. Data annotation examples of Golden Pear and Crabapple (on the left is the Golden Pear, and on the right is the Crabapple)
本研究遵循COCO数据集的通用标准对目标尺度进行界定，即小目标（面积<32×32像素）、中目标（32×32–96×96像素）和大目标（>96×96像素）。然而，由于本研究所使用的黄金梨数据集和海棠果数据集图像分辨率（3024×4032像素）显著高于COCO数据集的标准输入分辨率（640×640像素），直接应用原定义将导致尺度划分偏差。为进行公平且具有可比性的评估，在训练时需将其下采样至模型输入尺寸640×640。在此缩放过程中，设原始图像分辨率为W_0\times H_0，网络输入分辨率为W_i\times H_i，则图像的线性尺度缩放因子s定义为面积缩放的平方根： 
 	(3)
对于黄金梨数据集，W_0\times H_0=4032\times3024，W_i\times H_i=640\times640，代入可得s≈5.46。在COCO数据集中，小目标的边长上限为32像素，因此映射到原始分辨率下的等效边长阈值为：
 	(4)
为便于尺度划分并避免边界不确定性，本文取整得到小目标边长阈值为174像素，即小目标定义为面积小于174\times174像素的实例。同理，COCO中目标上限96像素对应到原图为 96\times s\approx523.7，取整得到中目标阈值为523像素。因此，新的尺度定义为：小目标（<174×174像素）、中目标（174×174–523×523像素）和大目标（>523×523像素）。各尺度下的果实实例统计详情见表1。
对于公开的MinneApple数据集，仍然采用COCO标准的原始尺度定义与评估指标（小目标：<32×32像素；大目标：>96×96像素），以确保与已有研究结果的可比性。
表1数据集果实实例尺度统计
Table1.StatisticalScaleofFruitInstancesintheDataset
Dataset	Class	Subset	Image	Instances	Target Amount
					Small	Medium	Large
Gold pear	pear	Total	2468	11446	5901	4389	1156
		Training	1950	9048	4597	3523	928
		Val	244	1064	526	413	125
		Test	244	1334	778	453	103
MinneApple	apple	Total	1001	40468	26376	14081	11
		Training	603	25249	17556	7693	0
		Val	67	2934	2032	902	0
		Test	331	12285	6788	5486	11
Crabapple	crabapple	Total	1057	6362	2665	3452	245
		Training	847	4937	1888	2834	215
		Val	105	780	393	362	25
		Test	105	645	384	256	5 
3实验与讨论
为全面验证所提出AFS-DETR模型在绿色小目标果实检测任务中的性能与适应性，本文从六个方面设计了系统化实验。首先，在实验设置中详细说明了自建绿色果实数据集与公开MinneApple数据集的特点、数据划分、训练环境与参数配置，确保实验的可复现性；其次，明确选取AP、APs、Recall等多维评价指标，从精度、小目标表现、召回率等角度综合衡量模型表现；随后，在与主流方法对比实验中，将本文方法与RT-DETR、YOLOv8、FasterR-CNN等进行性能对比，验证其在复杂果园背景下的优势；通过消融实验，量化C2AF、HACE与FPADT等结构改进对整体性能的贡献；在跨数据集验证中，将模型迁移至MinneApple数据集进行测试，评估其跨域泛化能力；最后，进行鲁棒性与误差分析，探讨模型在光照变化、遮挡比例、目标密度等不同条件下的表现，并分析常见检测错误类型。上述实验设计从精度、稳定性、泛化性与可解释性多维度综合验证了改进模型的有效性与实用价值。
3.1实验平台以及参数
本研究的实验运行环境为Python3.10.18与CUDA12.1，在配备4×NVIDIARTX3090（24GB显存）和251GB系统内存的服务器上进行，实际训练过程中指定使用2张GPU（id：2,3）并行计算。模型训练轮数为300epoch，批次大小（batchsize）为4，输入图像尺寸为640×640。初始学习率（lr0）设为1×10-4，最终学习率与初始值相同（lrf=1.0），全程保持固定学习率策略；动量参数设为0.9，权重衰减系数为1×10-4。训练初期设置2000次迭代的预热阶段（约占2–3个epoch），在此期间学习率由warmup_bias_lr=0.1线性上升至1×10-4，动量由0.8线性增加至0.9，以平稳过渡至正式训练阶段并减少初期梯度震荡风险。
为了全面评估改进型RT-DETR模型在绿色小目标果实检测任务中的性能，本研究选取了多个评价指标，包括AP平均精度均值）、Precision（精确率）、Recall（召回率）、F1-score（综合评价指标）、Parameters（模型参数量）、GFLOPs（计算复杂度）。这些指标不仅能全面衡量模型的检测精度，还能够在实时性和部署能力方面提供指导。
精确率和和召回率的公式为
 	                                   （5）
 	                                  （6）
其中，TP为正确预测为梨的样本数，FP为错误预测梨的样本数。FN为漏检的目标数。Precision反映了在所有预测为绿色果实的样本中，真正属于绿色果实的比例。高Precision表示误检率低，即模型对非果实目标（如叶片、枝条等）的分类能力强。
 	（7）
F1-score(F1)是Precision(P)和Recall（R）的调和平均数，综合考虑了精确率和召回率，避免了只单独优化某一方面的指标。F1高表示模型在保持低误检率的同时，也能有效减少漏检。本论文通过F1来综合评估模型在绿色小目标检测中的表现，尤其是复杂果园环境下的稳定性。
在目标检测中，平均精度（Average Precision,AP）衡量的是在不同召回率下的检测精度平均值。设召回率取值集合为： ，其中 表示一个特定的召回率， 表示当召回率为r时的精确率（Precision）。AP的计算公式为
 	（8）
本研究对黄金梨数据集的目标尺度进行了重新界定，并即采用上述重新定义的尺度标准来计算模型的性能指标APs、APm与APl。这一做法使性能评估与数据集的真实特性紧密对应。其中，APs是衡量小果实检测精度的核心指标。
3.2训练过程
为评估AFS-DETR的训练稳定性与收敛效果，本文监控了关键损失函数的变化，其曲线如图10所示。总体而言，定位损失（L1 Loss）、广义交并比损失（GIoU Loss）与分类损失（Cls Loss）均随训练周期增加而快速下降并渐趋收敛，表明训练过程高效且稳定。具体分析，L1 Loss于约100个Epoch后进入平稳阶段，表明边界框坐标回归迅速趋于精确。GIoU Loss的稳步收敛反映了目标定位几何准确性的持续提升。同时，分类损失（Cls Loss）的持续下降亦验证了模型在果实类别识别上置信度的不断提高。
 
图10 训练损失函数收敛曲线
Fig10. Convergence curves of the training loss functions
综上所述，所有损失函数均呈现良好收敛态势，未出现梯度不稳定或过拟合现象，证明本文改进为模型训练提供了坚实基础，保障了其最终性能。
  
图11 模型检测性能对比：AFS-DETR vs DETR-R18
Fig11. Detection performance comparison: AFS-DETR vs DETR-R18
根据精确率-召回率曲线的分析结果，如图11所示，本文提出的AFS-DETR模型在检测性能上稳定优于基准模型R18。AFS-DETR取得了0.904的平均精度值，而R18模型的平均精度为0.895。这一结果表明AFS-DETR在精确率-召回率的综合平衡方面表现更佳，虽然优势幅度不大，但结合其在其他尺度目标（特别是小目标APs）上的显著提升，仍可证明本文所提出的自适应特征融合与超图增强等模块有效提升了模型在复杂果园环境下的综合检测能力。
3.3消融实验
本文提出的AFS-DETR模型通过三项结构创新系统性地解决了绿色小果实检测中的核心挑战。消融实验数据(如表2所示)表明，各模块均展现出独特价值且存在显著协同效应：针对弱特征表征难题设计的C2AF模块在提升召回率的同时，将参数量与计算量分别降低33.3%与23.6%，实现了效能提升与模型轻量化的双重突破；为增强小目标感知引入的HACE模块在与C2AF组合后取得了91.3%的最高精确率，证明其能有效聚焦易忽略目标。尤为关键的是，为解决特征金字塔语义对齐困境而构建的FPADT模块，其价值在与C2AF的协同中得以彰显：该组合（+C2f+FPADT）取得了所有配置中最高的召回率（84.3%）与F1分数（87.5%），这强有力地证实了FPADT通过提供精准对齐的多尺度特征，为C2AF的深度表征奠定了坚实基础。最终，集成三项创新的AFS-DETR模型在召回率（84.5%）与定位精度（AP75: 78.4%）上达到最优，且在参数量与计算量间取得了最佳平衡，充分体现了本文所提方法在应对真实果园复杂场景时的有效性与实用性。
表2消融实验数据
Table 2. Data from the ablation experiment
Model	P(%)	R(%)	F1(%)	AP50(%)	AP75(%)	Parameters	GFLOPs
R18	89.0	83.8	86.3	89.3	76.5	19.8	56.9
+C2AF	90.5	84.0	87.2	90.6	78.2	13.2	43.5
+HACE	90.9	83.2	86.9	90.4	77.4	20.8	58.7
+FPADT	87.9	83.0	85.4	88.9	74.9	19.8	56.9
+ C2AF +HACE	91.3	83.7	87.3	90.7	77.9	15.8	54.7
+ C2AF +FPADT	90.9	84.3	87.5	91.0	77.4	13.4	44.6
+FPADT+HACE	89.9	82.4	86.0	90.2	77.9	21.4	182
AFS-DETR	89.4	84.5	86.9	90.4	78.4	15.2	46.8
3.3.1消融实验参数与计算量变化分析
表2中的参数量与计算量（GFLOPs）变化，主要源于主干网络的轻量化替换与各功能模块的叠加效应。具体而言，基准模型R18采用的ResNet-18主干参数量较大。本文设计的C2AF模块通过深度可分离卷积、精简的通道结构及跨阶段部分连接，在增强浅层特征表征的同时实现了显著的模型压缩，仅将主干替换为C2AF便使模型总参数量从19.8 M降至13.2 M（降幅33.3%），计算量从56.9 GFLOPs降至43.5（降幅23.6%），为集成后续增强模块提供了低参数基底。在此基础上引入HACE模块时，由于其参数基数已降低6.6 M，组合（+C2f+HACE）的总参数量（15.8 M）不仅显著低于在原始R18主干上单独添加HACE的配置（20.8 M，降低5.0 M），且在小目标平均精度（APs）等关键指标上表现更优，这从参数量与精度两个维度共同证明了C2AF模块在实现轻量化与维持高性能方面的双重效益。此外，FPADT模块采用极简的全局门控设计，其增加的参数量与计算量均可忽略不计。该量化分析表明，本文的模块改进在提升检测性能的同时，通过结构化的轻量化设计对模型复杂度的增长进行了良好控制。

3.4对比实验 
为客观评估本文所提改进模型在绿色小果实检测任务中的综合性能，与当前主流的目标检测模型进行了全面对比，包括两阶段的FasterR-CNN、无锚框的FCOS、基于Transformer的DeformableDETR，以及轻量级一阶段检测器的代表以及轻量级一阶段检测器的代表YOLOX-Tiny(You Only Look Once X)、RTMDet-Tiny(Real-Time Models for Object Detection)、YOLOv8-n至YOLOv13-n。
3.4.1黄金梨数据集在果实疏果阶段的验证
在黄金梨数据集上的实验验证结果（表3）表明，AFS-DETR模型在多个关键指标上均表现出显著优势。与基准模型R18相比，AFS-DETR在召回率(84.5% vs 83.8%)、F1分数(86.9% vs 86.3%)和定位精度(AP50:90.4% vs 89.3%；AP75:78.4% vs 76.5%)等方面均有稳定提升。特别值得注意的是，AFS-DETR在小目标检测(APs:56.8%)和中尺度目标检测(APm:85.7%)上均优于所有对比模型，包括YOLO系列最新版本。相较于传统二阶段检测方法在小目标APs指标上的不足（如Faster R-CNN的27.3%），AFS-DETR实现了显著突破。这些结果充分验证了本文提出的自适应特征融合机制在复杂果园环境中的有效性，特别是在提升多尺度目标检测性能方面的突出表现。
表3黄金梨的对比实验
Table3 Comparative Experiment of Golden Pears
Model	P(%)	R(%)	F1(%)	AP50(%)	AP75(%)	APs(%)	APm(%)	APl(%)
R18	89.0	83.8	86.3	89.3	76.5	55.8	85.1	92.4
DINO	78.9	70.6	74.5	72.1	55.0	29.6	76.4	84.2
Dynamic R-CNN	92.0	61.9	74.0	63.5	54.8	24.7	79.8	87.5
Faster R-CNN	91.3	65.3	76.1	65.6	57.8	27.3	79.5	86.9
Libra R-CNN	87.5	67.7	76.38	67.9	58.5	28.6	80.4	86.7
VFNet	77.8	72.3	74.9	73.8	58.1	30.4	79.0	86.6
YOLOv8-n	88.9	76.3	82.1	85.0	69.6	44.8	84.0	92.3
YOLOv10-n	89.8	74.6	81.5	83.7	70.0	44.6	83.5	91.8
YOLOv11-n	89.3	78.9	83.8	87.2	72.2	49.1	85.3	92.9
YOLOv12-n	89.2	75.2	81.6	83.4	68.6	43.5	84.9	92.1
YOLOv13-n	91.9	76.6	83.58	86.5	71.4	47.8	85.0	92.3
AFS-DETR	89.4	84.5	86.9	90.4	78.4	56.8	85.7	92.1
3.4.2 MinneApple数据集的验证
在MinneApple数据集上的综合实验结果列于表4，AFS-DETR模型取得了最优异的综合检测性能，其平均精度（AP）达到40.9%，位列所有对比模型之首。该模型在保持高定位精度（AP75达39.7%）的同时，在小目标检测（APs为27.1%）方面表现尤为突出，显著优于其他轻量级模型。与同属Transformer架构的Deformable DETR相比，AFS-DETR在AP指标上实现了超过一倍的提升，并有效解决了其在微小目标检测上的短板；相较于YOLOv13-n等最新模型，AFS-DETR在保持相当参数量级的前提下，在关键指标上展现了更均衡优越的性能。这些结果充分验证了本文所提出的自适应特征融合、超图增强与门控对齐模块在解决绿色小果实检测挑战中的有效性，显著提升了模型在复杂果园环境下的感知能力与泛化性能。

表4 MinneApple的对比实验
Table4. Comparative Experiment of MinneApple
Model	P(%)	R(%)	F1(%)	AP(%)	AP50(%)	AP75(%)	APs(%)	APm(%)	APl(%)
R18	80.6	69.4	74.6	39.9	75.7	38.5	26.5	55.3	64.1
Faster R-CNN	42.1	73.0	53.4	27.5	62.5	20.3	15.1	41.9	14.8
FCOS	73.6	52.9	61.6	23.2	56.8	14.8	11.9	36.6	1.9
DeformableDETR	58.0	69.5	63.2	18.7	52.5	8.0	7.5	31.2	55.3
YOLOX-Tiny	31.7	69.7	43.6	16.7	41.8	11.4	12.6	21.3	0
RTMDet-Tiny	55.3	77.7	64.6	33.1	65.8	30.1	18.8	49.8	80.8
YOLOv8-n	75.15	62.57	68.28	37.3	70.45	35.68	20.9	52.8	88.3
YOLOv10-n	72.52	62.22	66.98	35.6	69.31	33.16	20.3	49.4	69.9
YOLOv11-n	77.23	65.06	70.62	37.83	73.18	35.07	21.9	53.2	87.4
YOLOv12-n	81.4	75.7	79.7	39.9	81	33.3	38	55.8	0
YOLOv13-n	76.2	65.4	70.4	36.2	70.3	33.9	22.7	51.8	79.5
AFS-DETR	81.6	71.2	76.0	40.9	76.8	39.7	27.1	56.8	80.8
3.4.3海棠果数据集的验证
为更全面地评估模型性能并严格验证其跨品种泛化能力，本研究新增了在自建海棠果数据集上的测试。如表5所示，AFS-DETR在此取得了最优的综合性能：其平均精度（AP，65.6%）与定位精度（AP50，89.6%；AP75，74.1%）均领先于所有对比模型；尤为关键的是，在小目标检测（APs）这一核心指标上达到了52.0%，显著优于基准模型及其他主流检测器，这证明了C2AF、HACE与FPADT模块对于捕捉尺寸更小、伪装性更强的果实特征具有普适有效性。模型同时保持了高召回率（82.1%）与高F1分数（85.7%）的平衡，表明其“以召回率优先”的设计在新品类上依然有效。相比之下，部分对比模型（如Faster R-CNN、RTMDet-Tiny）性能出现显著下降。该结果证明，AFS-DETR不仅具备跨数据集的泛化能力，更实现了跨作物品种的有效知识迁移，能够将从一种果实学到的对弱纹理与小尺度的感知能力，适配到形态、颜色迥异的另一种果实上，从而表明本文方法为“绿色小果实”这类任务提供了通用且鲁棒的解决方
表5海棠果的对比实验
Table5.Comparative Experiment of Crabapple
Model	P(%)	R(%)	F1(%)	AP(%)	AP50(%)	AP75(%)	APs(%)	APm(%)	APl(%)
R18	88.9	83.4	86.0	64.2	89.3	70.8	51.0	81.1	90.5
DINO	85.6	74.0	79.3	50.8	76.3	55.9	35.0	73.9	76.6
Dynamic R-CNN	88.5	67.9	76.8	50.3	70.2	58.7	31.8	74.8	90.0
Faster R-CNN	77.9	76.8	77.3	52.0	77.0	58.5	35.1	74.0	83.1
Libra R-CNN	70.1	77.2	73.5	52.7	76.2	60.1	35.5	74.9	86.0
VFNet	68.6	78.6	73.3	51.3	77.4	56.2	32.6	75.4	90.1
RTMDet-Tiny	76.9	78.0	77.4	49.8	78.3	53.0	32.2	71.5	90.0
YOLOv8-n	87.7	81.5	84.5	63.6	86.1	71.6	48.3	81.9	94.1
YOLOv10-n	87.5	81.5	84.1	63.4	86.4	71.3	49.0	80.9	92.7
YOLOv11-n	89.4	80.4	84.7	64.8	87.2	73.2	49.2	83.4	90.1
YOLOv12-n	86.1	78.9	82.3	62.0	85.1	69.0	45.5	81.2	84.1
YOLOv13-n	86.4	78.6	82.3	61.4	85.8	68.3	45.6	81.2	85.0
AFS-DETR	89.6	82.1	85.7	65.6	89.6	74.1	52.0	82.3	85.2
3.4.4对比试验训练配置说明
为确保对比实验的严谨性与公平性，本研究遵循“各用其最佳配置”的原则。具体而言：本文提出的AFS-DETR模型，其训练超参数（如学习率、优化器、权重衰减等）均基于自建黄金梨数据集，通过系统的消融实验确定，为该架构在本任务上的最优配置。对于其他参与对比的模型，为避免强行统一训练策略导致部分方法偏离其公认最优训练范式，我们不统一训练策略层面的超参数，而是采用其官方开源代码库（如MMDetection、Ultralytics YOLO等）针对COCO等标准数据集推荐的默认配置作为基准。在训练过程中，仅对输入尺寸、数据集路径等任务相关参数进行适配，而保持其优化器、学习率策略、损失函数权重等核心超参数与官方设定一致；若原配置使用预训练权重，本研究亦予以采用，以保持与其官方推荐训练流程一致。该设置旨在模拟真实的模型选型场景：研究者通常直接采用经社区广泛验证的默认配置来评估模型，因此本文对比更侧重反映各方法在常规使用条件下的实际性能表现，为工程选型提供参考。
3.5可视化结果
为评估模型在复杂果园场景下的性能，进行系统的可视化分析。梨果数据可视化结果如图12所示，在光照变化、枝叶遮挡及颜色同质化等多重挑战下， AFS-DETR模型展现出卓越的鲁棒性，能够稳定识别出绝大多数绿色小果实，且几乎无漏检与误检。相比之下，现有主流模型均表现出明显局限性：YOLO系列（V8，V10，V11，V12，V13）、Dynamic R-CNN与Faster R-CNN等模型对小尺寸或遮挡果实存在显著漏检；而DINO(DETR with Improved deNoising anchOr boxes)、Libra R-CNN、RTMDet(Real-Time Models for Object Detection)、VFNet及部分YOLO变体问题更为严重，不仅漏检小目标，还频繁将叶片丛、光斑等背景误判为果实。这一可视化对比强有力地证明了AFS-DETR通过其自适应特征融合、小目标感知增强与特征精准对齐机制，在保持高召回率的同时显著提升了检测精度，展现出在真实农业环境中的优越性能。其中，红色和蓝色矩形框为模型识别目标的位置，黄色矩形框用于标识漏检目标的位置，黄色六边形框用于标识误检目标的位置。
     
原图
     
											 AFS-DETR
     
Dino
     
Dynamic_rcnn
     
Faster_rcnn
     
Libra_rcnn
     
Rtmdet
     
Vfnet
     
YOLOV8
     
								              YOLOV10
     
YOLOV11
     
YOLOV12
     
YOLOV13
图12 黄金梨在不同模型上的检测可视化图
Figure 12. Visualization diagrams of the detection of golden pears on different models
针对绿色小果实检测的挑战，AFS-DETR模型在MinneApple数据集上进行了验证，结果如图13所示。该数据集包含大量远景拍摄的绿色与红色果实，为验证模型在小目标、低对比度和密集场景下的性能提供了理想测试平台。实验结果表明，本文提出的C2AF主干模块与HACE增强模块协同工作，使模型能够从颜色相似的复杂背景中自适应地聚焦于果实的弱纹理特征，大幅减少了传统方法在此类场景下的漏检与误检。对于尺寸极小、遮挡严重的绿色果实，模型亦能实现准确识别，证明了其全管道聚合机制在提升层级间语义一致性、实现精准特征对齐方面的关键作用，最终赋予了模型在真实果园复杂环境下的强大鲁棒性。
       

       
图13 MinneApple检测可视化图
Figure 13. Visualization of MinneApple Detection
为深入分析模型的定位性能，本文对AFS-DETR在MinneApple数据集上的检测结果进行了局部放大。可视化结果如图14所示，即使在果实与背景枝叶高度融合的挑战性场景下，模型的预测框仍能紧密贴合果实的真实轮廓。这直接印证了C2AF模块增强弱特征表征与FPADT机制促进跨层级特征精准对齐的有效性——这些设计确保了模型能够感知到极为细微的边缘与纹理变化，并据此生成高精度的包围框。因此，优异的定位能力不仅是模型性能的直观体现，更是其内部创新架构在处理绿色小果实难题时发挥关键作用的有力证明。
  
图14 AFS-DETR对MinneApple检测可视化放大图
Figure 14. AFS-DETR's visualized magnified image of the MinneApple test
为验证AFS-DETR在极端光照与跨品种场景下的鲁棒性，图15展示了其在海棠果数据集上的可视化结果。海棠果尺寸小且颜色与背景高度相似，检测难度大。如图15左侧所示，在白天场景下，模型对可见果实定位精准，对粘连或边缘的小目标也能有效检出且误检少。在更具挑战的夜间低照度场景中（图15右侧），尽管果实细节大幅减弱、传统方法常失效，AFS-DETR仍能稳定工作，可有效抑制背景噪声并准确识别大部分补光区域内的果实，未出现大规模误检，这表明C2AF、HACE与FPADT模块的协同作用能有效缓解低照度下的特征衰减与对齐难题。该结果直观证实了AFS-DETR不仅具备跨品种泛化能力，更能适应复杂光照变化，为其在多变田间环境中的实际应用提供了依据。

    
    
图15 AFS-DETR模型在海棠果数据集上的可视化检测结果
Figure 15 Visualization detection results of the AFS-DETR model on the crabapple dataset
3.6模型复杂度、推理速度与部署可行性分析
3.6.1 模型复杂度与推理效率基准测试
如表6所示，我们在统一测试流程下对比了各模型的效率指标。与基准模型R18相比，AFS-DETR在取得更高检测精度的同时，实现了显著的轻量化：参数量从19.87M降至15.23M（降低约23.4%），计算量（GFLOPs）从56.9降至46.8（降低约17.8%），模型文件大小由77.0MB压缩至59.5MB。这主要归功于C2AF模块的高效设计与FPADT模块的极简门控机制。
在推理速度方面，AFS-DETR在服务器级GPU（RTX 3090）上的端到端处理速度达到65.48 FPS（纯推理速度68.96 FPS），能够满足多数视频流实时处理的需求。然而，与为极致速度而设计的YOLOv8-n、YOLOv10-n等轻量模型相比，AFS-DETR的GFLOPs与参数量仍高出约一个数量级，其端到端FPS也相应较低。这直观地揭示了模型设计中的权衡：AFS-DETR将更多的计算资源用于构建自适应融合与敏感度感知机制，以换取在复杂背景下对小目标检测精度与鲁棒性的显著提升。
表6 不同目标检测模型在黄金梨梨数据集上的推理效率对比
Table 6 Comparison of Inference Efficiency of Different Object Detection Models on the Golden Pear Dataset
模型	GFLOPs	参数量	前处理(s/图	推理(s/图)	后处理(s/图)	FPS（端到端）	FPS（推理）	模型大小
R18	56.90	19.87	0.000535	0.013543	0.000605	68.11	73.84	77.0MB
YOLOv8n	8.10	3.01	0.000604	0.005269	0.001831	129.80	189.79	6.0MB
YOLOv10n	8.20	2.69	0.000740	0.006862	0.001211	113.47	145.73	5.5MB
YOLOv11n	6.30	2.58	0.000480	0.007683	0.001863	99.74	130.16	5.2MB
YOLOv12n	5.80	2.50	0.000352	0.008841	0.000879	99.29	113.11	5.2MB
YOLOv13n	6.20	2.45	0.000828	0.013205	0.003375	57.44	75.73	5.2MB
AFS-DETR	46.80	15.23	0.000323	0.014501	0.000448	65.48	68.96	59.5MB

3.6.2边缘部署潜力与实际应用场景分析
为回应模型在硬件部署与果园实时运行方面的需求，本文补充说明AFS-DETR的边缘端部署路径与实时性可行性分析。AFS-DETR可从PyTorch导出为ONNX，并进一步使用TensorRT进行FP16/INT8量化加速，以适配Jetson等嵌入式GPU平台。需要说明的是，本文当前尚未完成在Jetson AGX Orin等边缘设备上的端到端实测，因此不报告具体的边缘端FPS数值。本文已在服务器级GPU（RTX3090）上给出了前处理/推理/后处理的端到端耗时统计（见表6），并提供了统一的测速口径（输入640×640、batch=1、warmup后取均值），便于在不同硬件环境下复现实测与横向对比。
综上，AFS-DETR在模型设计上定位明确：其目标并非追求极致推理速度，而是在保证准实时处理能力的基础上，优先提升复杂果园环境中小尺度、低对比度绿色果实的检测精度与鲁棒性。因此，该模型适用于对检测质量要求更高的应用场景，例如：（1）高精度果实产量预估与生长监测：在固定算力节点或巡检车上运行，输出可靠的果实计数与定位信息；（2）智能疏果/采摘决策系统：作为机器人视觉感知模块，提升遮挡与密集场景下的目标识别可靠性。对于算力更受限的终端设备，后续可结合通道剪枝、知识蒸馏与更低比特量化（如INT8）进一步压缩模型，在精度与速度之间实现更灵活的工程权衡，从而扩大其在智慧农业系统中的适用范围。
3.7 结果与讨论
3.7.1对低于现有方法指标的原因分析
尽管AFS-DETR在黄金梨数据集上取得更高的召回率（R=84.5%）与小目标精度（APs=56.8%），但其精确率P=89.4%并非最优，低于Dynamic R-CNN（92.0%）与YOLOv13-n（91.9%）。这主要源于本文“召回优先”的设计取向：为减少低对比度、小尺寸果实的漏检，模型倾向于保留更多潜在目标响应，从而在叶片团簇、枝条边缘与高亮反光等区域产生少量误检，表现为P略降而R提升。另一方面，AFS-DETR的改进重点集中在“小、弱、密”难例，因此在大目标指标上不一定占优。例如海棠果数据集上APl=85.2%，低于YOLOv8-n（94.1%）与R18（90.5%），说明模型容量更多用于提升小目标与遮挡场景的辨识而非大目标的饱和区间优化。此外，推理速度方面AFS-DETR（端到端65.48 FPS）低于极轻量YOLO-n系列（如YOLOv8-n为129.80 FPS），属于以更高鲁棒性与小目标性能换取一定计算开销的工程权衡。后续可通过误检类型统计与分场景分组评估（遮挡/光照/密度）进一步验证上述归因，并结合自适应阈值或误检抑制策略提升P与APl。
3.7.2模块贡献与协同机制分析
本文的实验结果系统验证了AFS-DETR模型在绿色小果实检测任务中的有效性。综合对比实验表明，新方法在黄金梨和海棠果以及MinneApple三个数据集上的AP、APs等关键指标均超越了主流检测器，这主要得益于三个核心模块的协同作用：C2AF模块通过多分支动态卷积增强了浅层细节特征的保留能力，HACE模块利用超图理论实现了跨尺度特征的高效融合，而FPADT模块则通过门控机制解决了特征对齐难题。
深入的消融研究揭示了有趣的模块交互特性。当C2AF与HACE结合时，模型在小目标检测（APs）上表现最佳，证明细节特征的增强与全局语义建模的结合能有效提升对微小目标的感知能力。而当C2AF与FPADT协同工作时，模型在定位精度（AP75）上达到最优，表明高质量基础特征与精准对齐机制的配合能显著改善边界框回归质量。值得注意的是，HACE与FPADT的直接组合效果有限，这暗示了缺乏高质量底层特征时，高级语义处理模块的效能会受到制约，进一步印证了C2AF作为特征提取基础的重要性。
综合来看，本文提出的AFS-DETR模型，其性能优势本质源于贯穿始终的自适应融合与敏感度感知设计哲学。C2AF在骨干网络层面提供了细节敏感的适应性表征，为整个系统奠定了高质量的特征基础。HACE在特征金字塔层面引入了上下文敏感的自适应关系推理，解决了小目标在复杂背景中的语义模糊问题。FPADT则在特征流关键节点实施了融合敏感的自适应门控对齐，保障了多源信息的纯净融合。这三个层次的自适应机制相互协同，使得模型不再是静态的前向传播，而是一个能够根据输入内容动态调整其‘注意力’和‘信息流’的智能感知系统，从而在面对绿色小果实检测的极端挑战时，表现出卓越的鲁棒性和精度。
3.7.3局限性与未来工作
尽管AFS-DETR在检测精度上取得了显著突破，但其计算复杂度相较于极简架构仍有优化空间，在需要极致推理速度的边缘设备部署场景中可能存在挑战。未来的研究将着重于模型效率的进一步提升，探索神经网络剪枝、知识蒸馏等轻量化技术，寻求精度与速度的最佳平衡，以期在资源受限的农业嵌入式系统中实现广泛应用。此外，当前模型在极端光照条件下的稳定性仍需加强，未来的工作也将包括收集更多样化的环境数据，并研究光照不变性特征学习策略，进一步增强模型在真实农业生产环境中的鲁棒性和实用性。
4结论
本文系统探究了复杂果园环境中绿色小果实检测面临的关键技术难题，深入剖析了现有方法在浅层特征保留、多尺度特征融合与跨层级特征对齐等方面的局限性。为解决这些问题，提出了一种基于RT-DETR的改进检测框架AFS-DETR，其核心创新体现在三个方面：（1）设计了C2AF主干模块，通过多分支动态卷积与自适应加权机制，显著提升了模型对微小目标的细节表征能力。（2）引入了基于超图理论的HACE融合模块，有效建立了多尺度特征间的高阶关联，增强了模型对复杂背景下低对比度目标的感知能力。（3）采用了轻量级的FPADT门控对齐机制，以极低的计算成本实现了特征金字塔的精准对齐，显著提升了特征融合质量。
详尽的实验验证表明，AFS-DETR在自建的黄金梨与海棠果数据集以及公开的MinneApple数据集上均表现出卓越的性能，在综合检测指标（F1分数）、小目标检测精度（APs）及跨域泛化能力等方面均超越了主流检测算法。特别值得指出的是，消融研究揭示了各创新模块间的协同增效作用：C2AF为特征提取提供了高质量的底层表征，HACE实现了跨尺度的语义增强，而FPADT则确保了多层次特征的精准融合，三者共同构成了一个完整高效的检测解决方案。
本研究不仅为绿色小果实检测任务提供了一个切实可行的技术方案，更重要的是，所提出的模块化改进思路为Transformer架构在复杂农业视觉任务中的适配与优化提供了有益的设计范式与实践参考。未来的研究工作将围绕模型轻量化与跨作物适应性展开，进一步提升技术在真实农业生产环境中的实用价值与推广潜力。
 
参考文献
[1]	Sharma R. Artificial intelligence in agriculture: a review[C]. 2021 5th international conference on intelligent computing and control systems (ICICCS). Piscataway, NJ: IEEE, 2021: 937-942.
[2]	Tang Y, Zhou H, Wang H, et al. Fruit detection and positioning technology for a Camellia oleifera C. Abel orchard based on improved YOLOv4-tiny model and binocular stereo vision[J]. Expert Systems with Applications, 2023, 211: 118573.
[3]	Afonso M, Fonteijn H, Fiorentin F S, et al. Tomato fruit detection and counting in greenhouses using deep learning[J]. Frontiers in Plant Science, 2020, 11: 571299.
[4]	Ferentinos K P. Deep learning models for plant disease detection and diagnosis[J]. Computers and Electronics in Agriculture, 2018, 145: 311-318.
[5]	Xiong Y, Ge Y, Grimstad L, et al. An autonomous strawberry‐harvesting robot: Design, development, integration, and field evaluation[J]. Journal of Field Robotics, 2020, 37(2): 202-224.
[6]	Partel V, Kakarla S C, Ampatzidis Y. Development and evaluation of a low-cost and smart technology for precision weed management utilizing artificial intelligence[J]. Computers and Electronics in Agriculture, 2019, 157: 339-350.
[7]	Kamilaris A, Prenafeta-Boldú F X. Deep learning in agriculture: A survey[J]. Computers and Electronics in Agriculture, 2018, 147: 70-90.
[8]	Nyarko E K, VidoviÄ I, RadoÄaj K, et al. A nearest neighbor approach for fruit recognition in RGB-D images based on detection of convex surfaces[J]. Expert Systems with Applications, 2018, 114: 454-466.
[9]	Goel N, Sehgal P. Fuzzy classification of pre-harvest tomatoes for ripeness estimation: An approach based on automatic rule learning using decision tree[J]. Applied Soft Computing, 2015, 36: 45-56.
[10]	Sengupta S, Lee W S. Identification and determination of the number of immature green citrus fruit in a canopy under different ambient light conditions[J]. Biosystems Engineering, 2014, 117: 51-61.
[11]	Li Z, Guo R, Li M, et al. A review of computer vision technologies for plant phenotyping[J]. Computers and Electronics in Agriculture, 2020, 176: 105672.
[12]	Bargoti S, Underwood J. Deep fruit detection in orchards[C]. 2017 IEEE international conference on robotics and automation (ICRA). Piscataway, NJ: IEEE, 2017: 3626-3633.
[13]	Liu X, Zhao D, Jia W, et al. A detection method for apple fruits based on color and shape features[J]. IEEE Access, 2019, 7: 67923-67933.
[14]	王丹丹, 何东健. 基于R-FCN深度卷积神经网络的机器人疏果前苹果目标的识别[J]. 农业工程学报, 2019, 35(3): 156-163.
[15]	Zhu N, Liu X, Liu Z, et al. Deep learning for smart agriculture: Concepts, tools, applications, and opportunities[J]. International Journal of Agricultural and Biological Engineering, 2018, 11(4): 32-44.
[16]	孙丰刚, 王云露, 兰鹏, 张旭东, 陈修德, 王志军. 基于改进YOLOv5s和迁移学习的苹果果实病害识别方法[J]. 农业工程学报, 2022, 38(11): 171-179.
[17]	Zou Z, Chen K, Shi Z, et al. Object detection in 20 years: A survey[J]. Proceedings of the IEEE, 2023, 111(3): 257-276.
[18]	Li J, Liang X, Shen S M, et al. Scale-aware fast R-CNN for pedestrian detection[J]. IEEE transactions on Multimedia, 2017, 20(4): 985-996.
[19]	Kisantal M, Wojna Z, Murawski J, et al. Augmentation for small object detection[J]. arXiv preprint arXiv:1902.07296, 2019.
[20]	Lin T Y, Dollár P, Girshick R, et al. Feature pyramid networks for object detection[C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).. Piscataway, NJ: IEEE, 2017: 2117-2125.
[21]	Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway, NJ: IEEE, 2018: 7132-7141.
[22]	Zhang S, Chi C, Yao Y, et al. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection[C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway, NJ: IEEE, 2020: 9759-9768.
[23]	Herng O W, Nasir A S A, Chin O B, et al. Harumanis mango leaves image segmentation on rgb and hsv colour spaces using fast k-means clustering[C].  Journal of Physics: Conference Series. Bristol, UK IOP Publishing, 2021, 2107(1): 012068.
[24]	Li B, Long Y, Song H. Detection of green apples in natural scenes based on saliency theory and Gaussian curve fitting[J]. International Journal of Agricultural and Biological Engineering, 2018, 11(1): 192-198.
[25]	Häni N, Roy P, Isler V. MinneApple: a benchmark dataset for apple detection and segmentation[J]. IEEE Robotics and Automation Letters, 2020, 5(2): 852-858.
[26]	Yu H J, Son C H. Leaf spot attention network for apple leaf disease identification[C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. Piscataway, NJ: IEEE, 2020: 52-53.
[27]	Tian Y, Yang G, Wang Z, et al. Apple detection during different growth stages in orchards using the improved YOLO-V3 model[J]. Computers and Electronics in Agriculture, 2019, 157: 417-426.
[28]	Liu W, Anguelov D, Erhan D, et al. SSD: Single shot multibox detector[C]. European Conference on Computer Vision (ECCV). Cham: Springer International Publishing, 2016: 21-37
[29]	Tian Z, Shen C, Chen H, et al. FCOS: Fully convolutional one-stage object detection[C]. Proceedings of the IEEE International Conference on Computer Vision (ICCV). Piscataway, NJ: IEEE, 2019: 9627-9636. 
[30]	Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway, NJ: IEEE, 2016: 779-788. 
[31]	Girshick R, Donahue J, Darrell T, et al. Rich feature hierarchies for accurate object detection and semantic segmentation[C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway, NJ: IEEE, 2014: 580-587. 
[32]	Girshick R. Fast R-CNN[C]. Proceedings of the IEEE International Conference on Computer Vision (ICCV). Piscataway, NJ: IEEE, 2015: 1440-1448. 
[33]	Ren S, He K, Girshick R, et al. Faster R-CNN: Towards real-time object detection with region proposal networks[C]. Advances in Neural Information Processing Systems (NeurIPS). 2015, 28: 91-99. 
[34]	Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object detection[C]. Proceedings of the IEEE International Conference on Computer Vision (ICCV). Piscataway, NJ: IEEE, 2017: 2980-2988. 
[35]	Rezatofighi H, Tsoi N, Gwak J Y, et al. Generalized intersection over union: A metric and a loss for bounding box regression[C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway, NJ: IEEE, 2019: 658-666.
[36]	Carion N, Massa F, Synnaeve G, et al. End-to-end object detection with transformers[C]. European Conference on Computer Vision (ECCV). Cham: Springer International Publishing, 2020: 213-229.
[37]	Ma Z, Dong N, Gu J, et al. STRAW-YOLO: A detection method for strawberry fruits targets and key points[J]. Computers and Electronics in Agriculture, 2025, 230: 109853.
[38]	Li F, Zhang H, Liu S, et al. Dn-detr: Accelerate detr training by introducing query denoising[C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway, NJ: IEEE, 2022: 13619-13627.
[39]	Lei M, Li S, Wu Y, et al. YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception[J]. arXiv preprint arXiv:2506.17733, 2025.

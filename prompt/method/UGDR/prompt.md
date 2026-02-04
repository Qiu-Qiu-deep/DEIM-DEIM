UGDR是我设计的一个损失函数，启发自dfine和DHSA (ECCV 2024)。

1. UGDR代码见engine/paper_first/ugdr.py，engine/paper_first/criterion_with_ugdr.py
2. DFINE的论文你读过。DHSA (ECCV 2024)的代码见prompt/method/UGDR/49.0 (ECCV 2024) Dynamic-range Histogram Self-Attention (DHSA).py
3. 我之前对这个模块写过一些论文等内容，你可以参考，但是现在来看不全面了，只能参考。见prompt/method/UGDR/UGDR.tex，prompt/method/UGDR/UGDR_old.md
4. 配置见configs/yaml/my1_ugdr.yml
5. 论文里引入它的时候，包括各种改进需要有明确的实际动机（关于穗类的田间实际问题）

对于完成论文需要我补充的内容，请你在prompt/method/UGDR 下生成一个TODO.md文件，详细说明，包括相关的可视化；实验；消融等等
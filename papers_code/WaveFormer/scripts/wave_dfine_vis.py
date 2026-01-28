"""
Wave-DFINE 可视化工具
用于生成论文图表：频谱分析、检测对比、域泛化曲线
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def visualize_frequency_spectrum(model, image, save_path='freq_analysis.png'):
    """
    可视化Wave模块的频率域变换
    
    Args:
        model: Wave-DFINE模型
        image: 输入图像 [1, 3, H, W]
        save_path: 保存路径
    """
    model.eval()
    
    # 获取Wave模块
    wave_module = None
    for name, module in model.named_modules():
        if 'wave' in name.lower() and hasattr(module, 'dct2d'):
            wave_module = module
            break
    
    if wave_module is None:
        print("未找到Wave模块")
        return
    
    # Hook获取中间特征
    features = {}
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook
    
    # 注册hook
    wave_module.register_forward_hook(hook_fn('wave_output'))
    
    # 前向传播
    with torch.no_grad():
        _ = model(image)
    
    # 提取特征
    wave_feat = features.get('wave_output', None)
    if wave_feat is None:
        print("未能获取Wave特征")
        return
    
    # 计算频谱
    feat_freq = wave_module.dct2d(wave_feat[:, :1])  # 只取第一个通道
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    img_np = image[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # 空间域特征
    axes[0, 1].imshow(wave_feat[0, 0].cpu().numpy(), cmap='viridis')
    axes[0, 1].set_title('Spatial Domain Feature')
    axes[0, 1].axis('off')
    
    # 频率域特征（对数尺度）
    freq_vis = torch.log(torch.abs(feat_freq[0, 0]) + 1e-8).cpu().numpy()
    im1 = axes[0, 2].imshow(freq_vis, cmap='jet')
    axes[0, 2].set_title('Frequency Domain (DCT)')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2])
    
    # 频率能量分布
    freq_energy = torch.abs(feat_freq[0, 0]).flatten().cpu().numpy()
    axes[1, 0].hist(freq_energy, bins=50, color='blue', alpha=0.7)
    axes[1, 0].set_xlabel('Frequency Magnitude')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Frequency Energy Distribution')
    axes[1, 0].set_yscale('log')
    
    # 径向频率剖面
    h, w = feat_freq.shape[-2:]
    center_h, center_w = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_w)**2 + (y - center_h)**2).astype(int)
    r_max = min(center_h, center_w)
    
    freq_profile = []
    for i in range(r_max):
        mask = (r == i)
        if mask.sum() > 0:
            freq_profile.append(torch.abs(feat_freq[0, 0])[mask].mean().item())
    
    axes[1, 1].plot(freq_profile, linewidth=2)
    axes[1, 1].set_xlabel('Radial Frequency')
    axes[1, 1].set_ylabel('Average Magnitude')
    axes[1, 1].set_title('Radial Frequency Profile')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 阻尼效果
    alpha = wave_module.damping.item()
    t = np.linspace(0, 5, 100)
    damping_curve = np.exp(-alpha * t / 2)
    axes[1, 2].plot(t, damping_curve, linewidth=2, label=f'α={alpha:.2f}')
    axes[1, 2].set_xlabel('Propagation Time (t)')
    axes[1, 2].set_ylabel('Damping Factor')
    axes[1, 2].set_title('Wave Damping Effect')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"频谱分析图已保存: {save_path}")


def compare_detection_results(dfine_results, wave_dfine_results, images, save_dir='comparison'):
    """
    对比DFINE和Wave-DFINE的检测结果
    
    Args:
        dfine_results: DFINE的检测结果
        wave_dfine_results: Wave-DFINE的检测结果
        images: 测试图像列表
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 选择有代表性的案例
    cases = {
        'small_object': 'UQ_11',  # 小目标多的域
        'dense_scene': 'UQ_8',    # 密集场景
        'ood_domain': 'ARC_1',    # OOD域（苏丹）
    }
    
    for case_name, domain in cases.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原图
        axes[0].imshow(images[domain])
        axes[0].set_title(f'Original Image ({domain})')
        axes[0].axis('off')
        
        # DFINE结果
        # TODO: 绘制检测框
        axes[1].imshow(images[domain])
        axes[1].set_title(f'DFINE (AP={dfine_results[domain]["ap"]:.3f})')
        axes[1].axis('off')
        
        # Wave-DFINE结果
        axes[2].imshow(images[domain])
        axes[2].set_title(f'Wave-DFINE (AP={wave_dfine_results[domain]["ap"]:.3f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{case_name}_comparison.png', dpi=300)
    
    print(f"检测对比图已保存: {save_dir}")


def plot_domain_generalization_curve(results_dict, save_path='domain_gen_curve.png'):
    """
    绘制域泛化曲线
    
    Args:
        results_dict: {method_name: {domain: ap_value}}
        save_path: 保存路径
    """
    # 提取域和AP值
    domains = list(next(iter(results_dict.values())).keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：逐域AP对比
    x = np.arange(len(domains))
    width = 0.35
    
    for i, (method, results) in enumerate(results_dict.items()):
        aps = [results[d] for d in domains]
        axes[0].bar(x + i * width, aps, width, label=method, alpha=0.8)
    
    axes[0].set_xlabel('Test Domain')
    axes[0].set_ylabel('AP')
    axes[0].set_title('Per-Domain AP Comparison')
    axes[0].set_xticks(x + width / 2)
    axes[0].set_xticklabels(domains, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 右图：性能提升热图
    improvement = {}
    baseline = list(results_dict.keys())[0]
    for method in list(results_dict.keys())[1:]:
        improvement[method] = [
            (results_dict[method][d] - results_dict[baseline][d]) * 100
            for d in domains
        ]
    
    if improvement:
        sns.heatmap(
            np.array(list(improvement.values())),
            xticklabels=domains,
            yticklabels=list(improvement.keys()),
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            ax=axes[1],
            cbar_kws={'label': 'AP Improvement (%)'}
        )
        axes[1].set_title('AP Improvement over Baseline')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"域泛化曲线已保存: {save_path}")


def plot_ablation_results(ablation_results, save_path='ablation_results.png'):
    """
    绘制消融实验结果
    
    Args:
        ablation_results: {experiment_name: {metric: value}}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    experiments = list(ablation_results.keys())
    metrics = ['AP', 'AP_50', 'AP_75', 'AP_s']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        values = [ablation_results[exp][metric] for exp in experiments]
        colors = ['red' if 'baseline' in exp.lower() else 'blue' for exp in experiments]
        
        bars = ax.bar(range(len(experiments)), values, color=colors, alpha=0.7)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(range(len(experiments)))
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 标注数值
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"消融实验图已保存: {save_path}")


if __name__ == "__main__":
    print("Wave-DFINE 可视化工具")
    print("="*60)
    
    # 示例使用
    print("用法示例：")
    print("1. 频谱分析：")
    print("   visualize_frequency_spectrum(model, image, 'freq.png')")
    print("")
    print("2. 检测对比：")
    print("   compare_detection_results(dfine_res, wave_res, imgs, 'comp/')")
    print("")
    print("3. 域泛化曲线：")
    print("   plot_domain_generalization_curve(results, 'domain.png')")
    print("")
    print("4. 消融实验：")
    print("   plot_ablation_results(ablation_data, 'ablation.png')")
    
    visualize_frequency_spectrum('/root/DEIM-DEIM/outputs/my1_wave/best_stg1.pth', '/root/DEIM-DEIM/data/gwhd_2021/test/images/0a2ca7f12937ecf30083e59d60e6e385bf87f82290c58f15f1414eddf287d4b0.png')

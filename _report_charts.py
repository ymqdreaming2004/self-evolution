import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

dims = ['涉黄', '涉政', '涉暴血腥', '宗教', '纹身']
jundun_rates = [97.5, 96.3, 98.8, 97.5, 96.3]
aliyun_rates = [98.8, 97.5, 100.0, 98.8, 98.8]
jundun_miss = [1, 1, 1, 1, 2]  # 君盾相对漏报（阿里云多检出）

x = np.arange(len(dims))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('君盾 vs 阿里云图片审核检出率对比', fontsize=15, fontweight='bold', y=1.02)

ax1 = axes[0]
bars1 = ax1.bar(x - width/2, jundun_rates, width, label='君盾', color='#1890ff', edgecolor='white')
bars2 = ax1.bar(x + width/2, aliyun_rates, width, label='阿里云', color='#52c41a', edgecolor='white')
ax1.set_ylim(94, 101)
ax1.set_ylabel('检出率 (%)')
ax1.set_title('各维度检出率对比')
ax1.set_xticks(x)
ax1.set_xticklabels(dims)
ax1.legend(loc='lower right')
for bars in [bars1, bars2]:
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

ax2 = axes[1]
colors = ['#52c41a', '#ff4d4f', '#faad14', '#722ed1', '#1890ff']
bars3 = ax2.barh(dims, jundun_miss, color=colors, edgecolor='white')
ax2.set_xlabel('君盾相对漏报数（张）')
ax2.set_title('君盾相对漏报分布')
ax2.invert_yaxis()
for bar, val in zip(bars3, jundun_miss):
    pct = val / 80 * 100
    ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'{val} ({pct:.1f}%)', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('./report-detection-chart.png', dpi=160, bbox_inches='tight', facecolor='white')
plt.close()

fig2, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    jundun_miss, labels=dims, autopct='%1.1f%%', startangle=140,
    colors=colors, explode=(0, 0, 0, 0, 0.06),
    textprops={'fontsize': 11}
)
ax.set_title('君盾相对漏报构成（合计 6 张）', fontsize=14, fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig('./report-miss-pie.png', dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print('charts updated')

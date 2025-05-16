import matplotlib.pyplot as plt
import numpy as np

# Extended methods including DUET
methods_ext   = ['ICA Src1', 'ICA Src2', 'NMF Src1', 'NMF Src2', 'DUET Src1', 'DUET Src2']
delta_snr_ext = np.array([-21.70, -21.45,  3.83,  6.88,  7.31,  4.80])
mse_ext       = np.array([ 0.775216,  1.184401,  0.002172,  0.001740,  0.000210,  0.002581])
ssim_ext      = np.array([ 0.5483,    0.5283,    0.5168,    0.4636,    0.9384,    0.6618])

# Assign colors by method type
# First two: ICA, next two: NMF, last two: DUET
colors = ['#1f77b4']*2 + ['#ff7f0e']*2 + ['#2ca02c']*2

# Helper to annotate bars
def annotate_bars(ax, fmt="{:.2f}", offset=3):
    for bar in ax.patches:
        h = bar.get_height()
        ax.annotate(fmt.format(h),
                    xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

x = np.arange(len(methods_ext))

# 1) ΔSNR Comparison
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x, delta_snr_ext, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(methods_ext, rotation=30, ha='right')
ax.set_ylabel('ΔSNR (dB)')
ax.set_title('ΔSNR Comparison')
ax.grid(axis='y', linestyle='--', alpha=0.5)
annotate_bars(ax, fmt="{:.1f}")
fig.tight_layout()
fig.savefig('delta_snr_comparison_with_duet.png')
plt.show()

# 2) Time-domain MSE Comparison (log scale)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x, mse_ext, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(methods_ext, rotation=30, ha='right')
ax.set_yscale('log')
ax.set_ylabel('MSE')
ax.set_title('Time-domain MSE Comparison')
ax.grid(axis='y', linestyle='--', alpha=0.5)
annotate_bars(ax, fmt="{:.2e}")
fig.tight_layout()
fig.savefig('mse_comparison_with_duet.png')
plt.show()

# 3) Frequency-domain SSIM Comparison
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x, ssim_ext, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(methods_ext, rotation=30, ha='right')
ax.set_ylabel('SSIM')
ax.set_title('Frequency-domain SSIM Comparison')
ax.grid(axis='y', linestyle='--', alpha=0.5)
annotate_bars(ax, fmt="{:.3f}")
fig.tight_layout()
fig.savefig('ssim_comparison_with_duet.png')
plt.show()

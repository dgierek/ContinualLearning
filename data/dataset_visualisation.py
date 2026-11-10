import matplotlib.pyplot as plt

from data import load_substance
from analysis.plotting import set_standard_plot_params
from data import synthetic_k0_E0

data_path = synthetic_k0_E0
titles = ['A', 'B', 'C', 'D']

set_standard_plot_params()
fig, axes = plt.subplots(2, 2, figsize=(15, 7.5),
                         # sharey=True,
                         sharex=True
                         )
ax1, ax2, ax3, ax4 = axes.flat

ax1.set_ylabel('Current [$\mu$A]')
ax3.set_ylabel('Current [$\mu$A]')
ax3.set_xlabel('Voltage [V]')
ax4.set_xlabel('Voltage [V]')

print(list(data_path.rglob("*.pt")))

for file, ax, title in zip(data_path.rglob("*.pt"), axes.flat, titles):
  curves, labels, label_map = load_substance(file)
  for curve in curves:
    current = curve[1]
    ax.plot(curve[0], curve[1]*1e6)
    #ax.plot(curve[0], curve[1])
    ax.grid(True)
    ax.set_title(title, fontweight='bold', loc='left')

plt.subplots_adjust(wspace=0.07)
plt.show(block=True)
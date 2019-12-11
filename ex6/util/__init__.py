from ._hdi import hdi
from ._hdi import hdi_hist

import numpy as np

def mark_hdi(ax, y, hist_kwargs={}, line_kwargs={}, text_xoffset=0, text_yoffset=0, text_ha=['right', 'left'], text_va='bottom', text_rotation=0):
    hdi_lo, hdi_hi = hdi(y)
    hdi_mi = hdi_lo + 0.5*(hdi_hi - hdi_lo)

    h, bins = np.histogram(y, **hist_kwargs)
    bins = bins[:-1] + 0.5*(bins[1]-bins[0])

    # For debugging, plot estimated property. Make sure it agrees with
    # matplotlib std plotting.
    # plt.hist(y, linewidth=0.5, color='tab:black', histtype='step')
    # plt.step(bins, h, linewidth=0.5, color='tab:red')
    
    y_hdi_lo = h[np.digitize(hdi_lo, bins)]
    y_hdi_hi = h[np.digitize(hdi_hi, bins)]

    y_hdi = min(y_hdi_lo, y_hdi_hi)
    ax.hlines(y_hdi, xmin=hdi_lo, xmax=hdi_hi, **line_kwargs)
    
    text_color = line_kwargs['color'] if 'color' in line_kwargs else 'gray'
    text_ha = [text_ha, text_ha] if not isinstance(text_ha, list) else text_ha

    ax.annotate(f'hdi', (hdi_mi+text_xoffset, y_hdi+text_yoffset), ha=text_ha[0], color=text_color, va=text_va)
    ax.annotate(f'{hdi_lo:.3}', (hdi_lo+text_xoffset, y_hdi+text_yoffset), ha=text_ha[0], color=text_color, va=text_va, rotation=text_rotation)
    ax.annotate(f'{hdi_hi:.3}', (hdi_hi+text_xoffset, y_hdi+text_yoffset), ha=text_ha[1], color=text_color, va=text_va, rotation=text_rotation)


def mark_mean(ax, y, hist_kwargs={}, marker_kwargs=dict(marker='^'), text_color='gray', text_xoffset=0, text_yoffset=0, text_ha='left', text_va='bottom', text_rotation=0):
    mean = y.mean()
    h, bins = np.histogram(y, **hist_kwargs)
    bins = bins[:-1] + 0.5*(bins[1]-bins[0])
    
    y_mean = h[np.digitize(mean, bins)]

    annotation = f'mean: {mean:.3}' if mean < 10 else f'mean: {int(mean)}'

    ax.plot(mean, y_mean, color=text_color, **marker_kwargs)
    ax.annotate(annotation, (mean+text_xoffset, y_mean+text_yoffset), ha=text_ha, color=text_color, va=text_va, rotation=text_rotation)

def mark_mode(ax, y, hist_kwargs={}, marker_kwargs=dict(marker='^'), text_color='gray', text_xoffset=0, text_yoffset=0, text_ha='left', text_va='bottom', text_rotation=0):
    h, bins = np.histogram(y, **hist_kwargs)
    bins = bins[:-1] + 0.5*(bins[1]-bins[0])

    y_mode = h.max()
    mode = bins[np.argmax(h)]
    
    annotation = f'mode: {mode:.3}' if mode < 10 else f'mode: {int(mode)}'

    ax.plot(mode, y_mode, color=text_color, **marker_kwargs)
    ax.annotate(annotation, (mode+text_xoffset, y_mode+text_yoffset), ha=text_ha, color=text_color, va=text_va, rotation=text_rotation)

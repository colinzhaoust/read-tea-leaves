"""Draw assets/settings.png: three ways to read a run, as hand-drawn sketches.

    python assets/make_sketches.py        (needs matplotlib; the curves are made up)

matplotlib's xkcd mode looks for the fonts "xkcd Script" or "Humor Sans" and falls back to another
comic font with a warning when they are not installed; the committed image was drawn that way.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAVY, BLUE, ORANGE, GREEN, GREY = '#0d366b', '#2a78d6', '#e8743b', '#2e9e6b', '#8a8f98'
rng = np.random.default_rng(3)


def curve(x, floor, scale, rate, noise=0.012):
    return floor + scale * (x + 0.03) ** -rate + rng.normal(0, noise, len(x))


def frame(ax, title, ylabel='loss'):
    ax.set_title(title, fontsize=15, color=NAVY, pad=10)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.2, 1.0])
    ax.set_xticklabels(['0', '20%', 'end of run'], fontsize=10)
    ax.set_yticks([])
    ax.set_ylabel(ylabel, fontsize=11)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.axvline(0.2, color=GREY, lw=1, ls=':')


def card(ax, x, y, lines, color):
    ax.text(x, y, '\n'.join(lines), transform=ax.transAxes, fontsize=9.5, color=color, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=color, lw=1.3))


def main():
    seen = np.linspace(0.01, 0.2, 24)
    rest = np.linspace(0.2, 1.0, 80)
    with plt.xkcd(scale=0.8, length=120, randomness=2):
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 4.1))

        # 1. forecasting: one run, the rest of its curve
        frame(a, '1. FORECAST')
        a.plot(seen, curve(seen, 2.2, 0.9, 0.55), color=NAVY, lw=2.2)
        mid = 2.2 + 0.9 * (rest + 0.03) ** -0.55
        a.fill_between(rest, mid - 0.9 * (rest - 0.2), mid + 0.45 * (rest - 0.2), color=BLUE, alpha=0.18, lw=0)
        a.plot(rest, mid, color=BLUE, lw=2, ls='--')
        a.text(0.62, 0.78, '?', transform=a.transAxes, fontsize=40, color=BLUE)
        card(a, 0.30, 0.97, ['lr = 3e-4', 'layers = 12', '+ code'], NAVY)
        a.set_xlabel('you see 20% and the settings:\nwhere does the run end?', fontsize=11)

        # 2. matching: two finished curves, two settings
        frame(b, '2. MATCH')
        x = np.linspace(0.01, 1.0, 110)
        b.plot(x, curve(x, 2.05, 1.0, 0.75), color=ORANGE, lw=2.2)
        b.plot(x, curve(x, 2.45, 0.8, 0.38), color=GREEN, lw=2.2)
        b.text(0.86, 0.38, 'curve 1', transform=b.transAxes, fontsize=10, color=GREEN, ha='center')
        b.text(0.86, 0.10, 'curve 2', transform=b.transAxes, fontsize=10, color=ORANGE, ha='center')
        card(b, 0.33, 0.97, ['A: warmup 2k'], NAVY)
        card(b, 0.66, 0.97, ['B: no warmup'], NAVY)
        b.text(0.60, 0.66, 'which is which?', transform=b.transAxes, fontsize=13, color=NAVY, ha='center')
        b.set_xlabel('two curves, two settings:\nwhich setting drew which curve?', fontsize=11)

        # 3. contrast: runs that look the same so far, and a described change
        frame(c, '3. CONTRAST')
        base = curve(seen, 2.2, 0.9, 0.55, noise=0.008)
        c.plot(seen, base, color=NAVY, lw=2.2)
        c.plot(seen, base + 0.012, color=GREY, lw=2.2)
        tail = 2.2 + 0.9 * (rest + 0.03) ** -0.55
        c.plot(rest, tail, color=NAVY, lw=2, ls='--')
        c.plot(rest, tail + 0.22 * (rest - 0.2) ** 0.7, color=GREY, lw=2, ls='--')
        card(c, 0.30, 0.97, ['run A: as is'], NAVY)
        card(c, 0.62, 0.97, ['run B: freeze the', 'embeddings later'], GREY)
        c.text(0.64, 0.50, 'who ends lower?', transform=c.transAxes, fontsize=13, color=NAVY, ha='center')
        c.set_xlabel('same curve so far, one described change:\nwhich run ends lower?', fontsize=11)

        fig.tight_layout(w_pad=2.5)
        out = Path(__file__).with_name('settings.png')
        fig.savefig(out, dpi=130, facecolor='white')
        print('wrote', out)


if __name__ == '__main__':
    main()

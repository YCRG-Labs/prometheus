#!/usr/bin/env python3
"""Generate Graphical Abstract and System Architecture Diagram."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from pathlib import Path as FilePath


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'font.size': 10,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'mathtext.fontset': 'dejavusans',
    })


def box(ax, x, y, w, h, fc, alpha=1.0, lw=0, ec=None, rad=0.08):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                       facecolor=fc, alpha=alpha, edgecolor=ec or fc, linewidth=lw)
    ax.add_patch(p)


def arrow(ax, x1, y1, x2, y2, c='#888', lw=2.5):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                        mutation_scale=18, color=c, linewidth=lw)
    ax.add_patch(a)


def create_graphical_abstract(out):
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    BLU, RED, GRN, PUR = '#3498DB', '#E74C3C', '#27AE60', '#9B59B6'
    ORG, GRY, DRK = '#F39C12', '#7F8C8D', '#2C3E50'

    bw, bh, by, gap = 2.8, 4.5, 0.75, 0.6
    xs = [(14 - 4*bw - 3*gap)/2 + i*(bw+gap) for i in range(4)]
    cols = [BLU, RED, GRN, PUR]
    titles = ['Spin Data', 'VAE Model', 'Latent Space', 'Results']

    for x, c, t in zip(xs, cols, titles):
        box(ax, x, by, bw, bh, c, 0.1, 2.5, c, 0.15)
        ax.text(x + bw/2, by + bh - 0.35, t, fontsize=12, fontweight='bold', ha='center', color=c)
        ax.plot([x+0.2, x+bw-0.2], [by+bh-0.65]*2, color=c, alpha=0.4, lw=1.5)

    # Box 1: Spins
    x1 = xs[0]
    gx, gy, gw, gh = x1+0.3, by+0.7, bw-0.6, 2.8
    box(ax, gx, gy, gw, gh, 'white', 1, 1.5, GRY, 0.06)
    np.random.seed(42)
    for i in range(7):
        for j in range(7):
            cx, cy = gx + (i+0.5)*gw/7, gy + (j+0.5)*gh/7
            up = np.random.random() > 0.4
            ax.plot(cx, cy, '^' if up else 'v', ms=8, color=RED if up else BLU,
                    mec='white', mew=0.4)
    ax.text(x1+bw/2, by+0.35, 'Monte Carlo Samples', fontsize=9, ha='center', color=DRK)

    # Box 2: VAE
    x2 = xs[1]
    vae_cy = by + bh/2
    ew, eh = 0.7, 2.0
    pts1 = [(x2+0.25, vae_cy-eh/2), (x2+0.25, vae_cy+eh/2), 
            (x2+0.25+ew, vae_cy+eh/2-0.4), (x2+0.25+ew, vae_cy-eh/2+0.4)]
    ax.add_patch(Polygon(pts1, fc=BLU, alpha=0.85, ec='white', lw=2))
    ax.text(x2+0.25+ew/2, vae_cy, 'E', fontsize=16, ha='center', va='center', color='white', fontweight='bold')

    lx = x2 + bw/2
    ax.add_patch(Circle((lx, vae_cy), 0.4, fc=ORG, ec='white', lw=3))
    ax.text(lx, vae_cy, 'z', fontsize=18, ha='center', va='center', fontweight='bold', fontstyle='italic', color='white')

    dx = x2 + bw - 0.95
    pts2 = [(dx, vae_cy-eh/2+0.4), (dx, vae_cy+eh/2-0.4), (dx+ew, vae_cy+eh/2), (dx+ew, vae_cy-eh/2)]
    ax.add_patch(Polygon(pts2, fc=GRN, alpha=0.85, ec='white', lw=2))
    ax.text(dx+ew/2, vae_cy, 'D', fontsize=16, ha='center', va='center', color='white', fontweight='bold')

    ax.annotate('', xy=(lx-0.45, vae_cy), xytext=(x2+0.25+ew+0.08, vae_cy), arrowprops=dict(arrowstyle='->', color=GRY, lw=2))
    ax.annotate('', xy=(dx-0.08, vae_cy), xytext=(lx+0.45, vae_cy), arrowprops=dict(arrowstyle='->', color=GRY, lw=2))
    ax.text(x2+bw/2, by+0.35, 'Beta-VAE + Physics', fontsize=9, ha='center', color=DRK)

    # Box 3: Latent
    x3 = xs[2]
    px, py, pw, ph = x3+0.25, by+0.7, bw-0.5, 2.8
    box(ax, px, py, pw, ph, 'white', 1, 1.5, GRY, 0.06)
    np.random.seed(789)
    n = 35
    fx = np.clip(px + pw*0.72 + np.random.normal(0, pw*0.07, n), px+0.1, px+pw-0.1)
    fy = np.clip(py + ph*0.72 + np.random.normal(0, ph*0.07, n), py+0.1, py+ph-0.1)
    pmx = np.clip(px + pw*0.28 + np.random.normal(0, pw*0.08, n), px+0.1, px+pw-0.1)
    pmy = np.clip(py + ph*0.28 + np.random.normal(0, ph*0.08, n), py+0.1, py+ph-0.1)
    ax.scatter(fx, fy, c=RED, s=30, alpha=0.7, edgecolors='white', linewidths=0.5, zorder=3)
    ax.scatter(pmx, pmy, c=BLU, s=30, alpha=0.7, edgecolors='white', linewidths=0.5, zorder=3)
    ax.text(px+pw*0.78, py+ph*0.9, 'FM', fontsize=10, fontweight='bold', color=RED, ha='center')
    ax.text(px+pw*0.22, py+ph*0.1, 'PM', fontsize=10, fontweight='bold', color=BLU, ha='center')
    ax.text(x3+bw/2, by+0.35, 'Phase Separation', fontsize=9, ha='center', color=DRK)

    # Box 4: Results - use absolute positions to avoid overlap
    x4 = xs[3]
    
    # Fixed Y positions for each row (no calculation, just hardcoded)
    row_y_positions = [4.2, 3.2, 2.2]  # Top to bottom
    row_labels = ['beta = 0.326', 'nu = 0.630', 'gamma = 1.237']
    
    for ry, label in zip(row_y_positions, row_labels):
        # Small box for each exponent
        box(ax, x4 + 0.2, ry - 0.25, bw - 0.4, 0.5, '#F5F5F5', 1, 1, '#DDD', 0.05)
        ax.text(x4 + bw/2, ry, label, fontsize=10, ha='center', va='center', fontweight='bold', color=DRK)
    
    # Accuracy badge at very bottom
    box(ax, x4 + 0.3, by + 0.55, bw - 0.6, 0.45, GRN, 1, 0, None, 0.08)
    ax.text(x4 + bw/2, by + 0.78, '>=70%', fontsize=10, ha='center', va='center', color='white', fontweight='bold')

    # Arrows
    ay = by + bh/2
    for i in range(3):
        arrow(ax, xs[i]+bw+0.08, ay, xs[i+1]-0.08, ay, GRY, 3)

    out.mkdir(parents=True, exist_ok=True)
    for f in ['png', 'pdf', 'svg']:
        fig.savefig(out/f'graphical_abstract.{f}', format=f, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
        print(f"Saved: graphical_abstract.{f}")
    plt.close(fig)



def create_system_architecture_diagram(out):
    setup_style()
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    BLU, RED, GRN, PUR = '#3498DB', '#E74C3C', '#27AE60', '#9B59B6'
    ORG, GRY, DRK = '#F39C12', '#7F8C8D', '#2C3E50'

    # Layer 1: Data
    y1, h1 = 7.8, 1.6
    box(ax, 0.5, y1, 15, h1, BLU, 0.1, 2, BLU, 0.12)
    ax.text(8, y1+h1-0.25, 'DATA GENERATION', fontsize=12, fontweight='bold', ha='center', color=BLU)
    data = ['Monte Carlo', 'Temperature\nSampling', 'Equilibration', 'Data Loader']
    dx = [2.5, 5.5, 8.5, 11.5]
    for n, x in zip(data, dx):
        box(ax, x-1.2, y1+0.25, 2.4, 1.0, 'white', 1, 1.5, BLU, 0.06)
        ax.text(x, y1+0.75, n, fontsize=9, ha='center', va='center', fontweight='bold', color=DRK)
    for i in range(len(dx)-1):
        arrow(ax, dx[i]+1.25, y1+0.75, dx[i+1]-1.25, y1+0.75, BLU, 2)

    # Layer 2: Model
    y2, h2 = 5.0, 2.4
    box(ax, 0.5, y2, 15, h2, RED, 0.1, 2, RED, 0.12)
    ax.text(8, y2+h2-0.25, 'VAE MODEL', fontsize=12, fontweight='bold', ha='center', color=RED)

    box(ax, 1.0, y2+0.5, 2.0, 1.4, 'white', 1, 1.5, RED, 0.06)
    ax.text(2.0, y2+1.4, 'Input', fontsize=10, ha='center', fontweight='bold', color=DRK)
    ax.text(2.0, y2+0.9, 'Spin Config', fontsize=8, ha='center', color=GRY)

    box(ax, 3.5, y2+0.4, 2.8, 1.6, RED, 0.15, 2, RED, 0.08)
    ax.text(4.9, y2+1.7, 'ENCODER', fontsize=10, ha='center', fontweight='bold', color=RED)
    ax.text(4.9, y2+1.2, 'Conv3D layers', fontsize=8, ha='center', color=DRK)
    ax.text(4.9, y2+0.8, '32 -> 64 -> 128', fontsize=8, ha='center', color=GRY)

    ax.add_patch(Circle((8, y2+1.2), 0.55, fc=ORG, ec='white', lw=3))
    ax.text(8, y2+1.2, 'z', fontsize=20, ha='center', va='center', fontweight='bold', fontstyle='italic', color='white')
    ax.text(8, y2+0.4, 'Latent', fontsize=9, ha='center', color=DRK)

    box(ax, 9.2, y2+0.4, 2.8, 1.6, GRN, 0.15, 2, GRN, 0.08)
    ax.text(10.6, y2+1.7, 'DECODER', fontsize=10, ha='center', fontweight='bold', color=GRN)
    ax.text(10.6, y2+1.2, 'ConvT3D layers', fontsize=8, ha='center', color=DRK)
    ax.text(10.6, y2+0.8, '128 -> 64 -> 32', fontsize=8, ha='center', color=GRY)

    box(ax, 12.5, y2+0.4, 2.5, 1.6, 'white', 1, 1.5, RED, 0.06)
    ax.text(13.75, y2+1.7, 'LOSS', fontsize=10, ha='center', fontweight='bold', color=RED)
    ax.text(13.75, y2+1.2, 'Recon + KL', fontsize=9, ha='center', color=DRK)
    ax.text(13.75, y2+0.8, '+ Physics', fontsize=9, ha='center', color=GRY)

    arrow(ax, 3.05, y2+1.2, 3.45, y2+1.2, GRY, 2)
    arrow(ax, 6.35, y2+1.2, 7.4, y2+1.2, GRY, 2)
    arrow(ax, 8.6, y2+1.2, 9.15, y2+1.2, GRY, 2)
    arrow(ax, 12.05, y2+1.2, 12.45, y2+1.2, GRY, 2)

    # Layer 3: Analysis
    y3, h3 = 2.4, 2.2
    box(ax, 0.5, y3, 15, h3, GRN, 0.1, 2, GRN, 0.12)
    ax.text(8, y3+h3-0.25, 'ANALYSIS PIPELINE', fontsize=12, fontweight='bold', ha='center', color=GRN)
    analysis = ['Latent\nExtraction', 'Tc\nDetection', 'Beta\nExtraction', 'Finite-Size\nScaling', 'Ensemble']
    ax_pos = [2.0, 4.8, 7.6, 10.4, 13.2]
    for n, x in zip(analysis, ax_pos):
        box(ax, x-1.0, y3+0.35, 2.0, 1.4, 'white', 1, 1.5, GRN, 0.06)
        ax.text(x, y3+1.05, n, fontsize=9, ha='center', va='center', fontweight='bold', color=DRK)
    for i in range(len(ax_pos)-1):
        arrow(ax, ax_pos[i]+1.05, y3+1.05, ax_pos[i+1]-1.05, y3+1.05, GRN, 1.5)

    # Layer 4: Output
    y4, h4 = 0.4, 1.6
    box(ax, 0.5, y4, 15, h4, PUR, 0.1, 2, PUR, 0.12)
    ax.text(8, y4+h4-0.25, 'VALIDATION & OUTPUT', fontsize=12, fontweight='bold', ha='center', color=PUR)
    val = ['Bootstrap CI', 'ANOVA Tests', '2D Validation']
    vx = [2.5, 5.5, 8.5]
    for n, x in zip(val, vx):
        box(ax, x-1.2, y4+0.2, 2.4, 1.0, 'white', 1, 1.5, PUR, 0.06)
        ax.text(x, y4+0.7, n, fontsize=9, ha='center', fontweight='bold', color=DRK)
    box(ax, 10.3, y4+0.2, 2.4, 1.0, 'white', 1, 1.5, ORG, 0.06)
    ax.text(11.5, y4+0.7, 'Exponents', fontsize=9, ha='center', fontweight='bold', color=DRK)
    box(ax, 13.0, y4+0.2, 2.0, 1.0, GRN, 1, 0, None, 0.06)
    ax.text(14.0, y4+0.8, '>=70%', fontsize=12, ha='center', fontweight='bold', color='white')
    ax.text(14.0, y4+0.45, 'Accuracy', fontsize=8, ha='center', color='white')

    arrow(ax, 8, y1, 8, y1-0.35, GRY, 2.5)
    arrow(ax, 8, y2, 8, y2-0.35, GRY, 2.5)
    arrow(ax, 8, y3, 8, y3-0.35, GRY, 2.5)

    out.mkdir(parents=True, exist_ok=True)
    for f in ['png', 'pdf', 'svg']:
        fig.savefig(out/f'system_architecture.{f}', format=f, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
        print(f"Saved: system_architecture.{f}")
    plt.close(fig)


def main():
    out = FilePath('results/publication/figures')
    print("Generating figures...")
    create_graphical_abstract(out)
    create_system_architecture_diagram(out)
    print("Done!")


if __name__ == '__main__':
    main()

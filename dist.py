import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde


def create_comprehensive_heatmaps(hdf5_folder, grid_resolution=0.1, bin_multiplier=1,
                                  sample_size_kde=10000):

    hdf5_path = Path(hdf5_folder)
    if not hdf5_path.is_dir():
        print(f"Fehler: Ordner nicht gefunden unter {hdf5_folder}")
        return

    all_positions = []
    files = sorted([f for f in os.listdir(hdf5_folder) if f.endswith('.hdf5')])

    print(f"Scanne {len(files)} HDF5-Dateien für Ground Truth Positionen...")

    # 1. Sammeln aller Ground Truth Positionen
    for f_name in tqdm(files, desc="Processing files"):
        path = hdf5_path / f_name
        try:
            with h5py.File(path, 'r') as f:
                if 'ground_truth' in f:
                    positions = f['ground_truth'][:]
                    all_positions.append(positions)
        except Exception as e:
            print(f"Fehler beim Lesen von {f_name}: {e}")
            continue

    if not all_positions:
        print("Keine Ground Truth Daten gefunden.")
        return

    all_positions = np.concatenate(all_positions, axis=0)
    print(f"Gesamtanzahl gescannter Positionen: {len(all_positions)}")

    # 2. Berechnung der Grenzen
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()

    # Kleine Puffer hinzufügen (z.B. 5% der Spannweite oder minimal grid_resolution)
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding_x = max(grid_resolution, x_range * 0.02)
    padding_y = max(grid_resolution, y_range * 0.02)

    x_min_r = x_min - padding_x
    x_max_r = x_max + padding_x
    y_min_r = y_min - padding_y
    y_max_r = y_max + padding_y

    bins_x = int((x_max_r - x_min_r) / grid_resolution) * bin_multiplier
    bins_y = int((y_max_r - y_min_r) / grid_resolution) * bin_multiplier
    bins_x = max(1, bins_x)
    bins_y = max(1, bins_y)

    # 3. Histogramm-Daten
    heatmap_data, xedges, yedges = np.histogram2d(
        all_positions[:, 0],
        all_positions[:, 1],
        bins=[bins_x, bins_y],
        range=[[x_min_r, x_max_r], [y_min_r, y_max_r]]
    )
    heatmap_data = heatmap_data.T

    # 4. Erstelle Figure mit 6 Subplots
    fig = plt.figure(figsize=(18, 12))

    # ------------------------
    # Plot 1: Original Heatmap
    # ------------------------
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(heatmap_data,
                     extent=[x_min_r, x_max_r, y_min_r, y_max_r],
                     origin='lower',
                     cmap='inferno',
                     aspect='equal')
    plt.colorbar(im1, ax=ax1, label='Häufigkeit', fraction=0.046, pad=0.04)
    #ax1.set_xlabel('X Position (m)')
    #ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Heatmap')
    ax1.set_xlim(x_min_r, x_max_r)
    ax1.set_ylim(y_min_r, y_max_r)
    ax1.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.3)

    # ------------------------
    # Plot 2: Logarithmische Skalierung
    # ------------------------
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(np.log1p(heatmap_data), # np.log(heatmap_data) instabil ->Problem: log(0) = -∞ (undefiniert/NaN)
                     extent=[x_min_r, x_max_r, y_min_r, y_max_r],
                     origin='lower',
                     cmap='viridis',
                     aspect='equal')
    plt.colorbar(im2, ax=ax2, label='Log(Häufigkeit + 1)', fraction=0.046, pad=0.04, shrink=0.5)
    #ax2.set_xlabel('X Position (m)')
    #ax2.set_ylabel('Y Position (m)')
    #ax2.set_title('Heatmap (Logarithmisch)')
    ax2.set_xlim(x_min_r, x_max_r)
    ax2.set_ylim(y_min_r, y_max_r)
    ax2.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.3)

    # ------------------------
    # Plot 3: Hexbin Plot
    # ------------------------
    ax3 = plt.subplot(2, 3, 3)
    hexbin_gridsize = min(50, max(20, bins_x // 5))
    hb = ax3.hexbin(all_positions[:, 0], all_positions[:, 1],
                    gridsize=hexbin_gridsize, cmap='hot', mincnt=1)
    plt.colorbar(hb, ax=ax3, label='Anzahl')
    #ax3.set_xlabel('X Position (m)')
    #ax3.set_ylabel('Y Position (m)')
    ax3.set_title('Hexbin Plot')
    ax3.set_xlim(x_min_r, x_max_r)
    ax3.set_ylim(y_min_r, y_max_r)
    ax3.set_aspect('equal')

    # ------------------------
    # Plot 4: Konturplot mit Smoothing
    # ------------------------
    ax4 = plt.subplot(2, 3, 4)
    smoothed = gaussian_filter(heatmap_data, sigma=2)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    levels = 15
    cf = ax4.contourf(X, Y, smoothed, levels=levels, cmap='plasma')
    ax4.contour(X, Y, smoothed, levels=levels, colors='white',
                linewidths=0.5, alpha=0.4)
    plt.colorbar(cf, ax=ax4, label='Geglättete Dichte')
    #ax4.set_xlabel('X Position (m)')
    #ax4.set_ylabel('Y Position (m)')
    ax4.set_title('Konturplot (Gaussian Smoothing)')
    ax4.set_xlim(x_min_r, x_max_r)
    ax4.set_ylim(y_min_r, y_max_r)
    ax4.set_aspect('equal')

    # ------------------------
    # Plot 5: KDE (Kernel Density Estimation)
    # ------------------------
    ax5 = plt.subplot(2, 3, 5)

    # Sampling für Performance
    sample_size = min(sample_size_kde, len(all_positions))
    sample_indices = np.random.choice(len(all_positions), sample_size, replace=False)
    sample = all_positions[sample_indices]

    try:
        kde = gaussian_kde(sample.T)

        # Grid für KDE
        grid_points = 150
        xi = np.linspace(x_min_r, x_max_r, grid_points)
        yi = np.linspace(y_min_r, y_max_r, grid_points)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = kde(np.vstack([Xi.ravel(), Yi.ravel()]))
        zi = zi.reshape(Xi.shape)

        im5 = ax5.pcolormesh(Xi, Yi, zi, cmap='YlOrRd', shading='auto')
        plt.colorbar(im5, ax=ax5, label='Dichte (KDE)')
        ax5.set_title(f'5. KDE Plot (Sample: {sample_size})')
    except Exception as e:
        ax5.text(0.5, 0.5, f'KDE Fehler:\n{str(e)}',
                 ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('5. KDE Plot (Fehler)')

    #ax5.set_xlabel('X Position (m)')
    #ax5.set_ylabel('Y Position (m)')
    ax5.set_xlim(x_min_r, x_max_r)
    ax5.set_ylim(y_min_r, y_max_r)
    ax5.set_aspect('equal')

    # ------------------------
    # Plot 6: Scatter mit Transparenz
    # ------------------------
    ax6 = plt.subplot(2, 3, 6)

    # Für sehr große Datensätze: Sampling
    scatter_sample_size = min(50000, len(all_positions))
    scatter_indices = np.random.choice(len(all_positions), scatter_sample_size, replace=False)
    scatter_sample = all_positions[scatter_indices]

    ax6.scatter(scatter_sample[:, 0], scatter_sample[:, 1],
                alpha=0.05, s=1, c='navy', rasterized=True)
    #ax6.set_xlabel('X Position (m)')
    #ax6.set_ylabel('Y Position (m)')
    #ax6.set_title('Streudiagramm') # scatter plot
    ax6.set_xlim(x_min_r, x_max_r)
    ax6.set_ylim(y_min_r, y_max_r)
    ax6.set_aspect('equal')
    ax6.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.3)

    # Gesamttitel
    fig.suptitle('Ground Truth Positionsverteilung - Verschiedene Visualisierungen',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.show()

    # einzelplots
    # Berechne das Verhältnis der Achsen für konsistente Größe
    x_span = x_max_r - x_min_r
    y_span = y_max_r - y_min_r
    ratio = y_span / x_span

    # Erstelle Figure mit 2 Subplots nebeneinander
    base_width = 7  # Breite pro Subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(base_width * 2, base_width * ratio))

    # Plot 1: Logarithmische Heatmap
    im1 = ax1.imshow(np.log1p(heatmap_data),
                     extent=[x_min_r, x_max_r, y_min_r, y_max_r],
                     origin='lower',
                     cmap='viridis',
                     aspect='equal')
    plt.colorbar(im1, ax=ax1, label='Log sample density', fraction=0.046, pad=0.04)
    #ax1.set_xlabel('X Position (m)')
    #ax1.set_ylabel('Y Position (m)')
    ax1.set_xlim(x_min_r, x_max_r)
    ax1.set_ylim(y_min_r, y_max_r)
    ax1.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.3)

    # Plot 2: Scatter Plot
    ax2.scatter(scatter_sample[:, 0], scatter_sample[:, 1],
                alpha=0.05, s=1, c='navy', rasterized=True, edgecolors='none')
    #ax2.set_xlabel('X Position (m)')
    #ax2.set_ylabel('Y Position (m)')
    ax2.set_xlim(x_min_r, x_max_r)
    ax2.set_ylim(y_min_r, y_max_r)
    ax2.set_aspect('equal')
    ax2.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistiken ausgeben
    print("\n" + "=" * 60)
    print("STATISTIKEN")
    print("=" * 60)
    print(f"Anzahl Positionen: {len(all_positions):,}")
    print(f"X-Bereich: [{x_min:.2f}, {x_max:.2f}] m (Spannweite: {x_max - x_min:.2f} m)")
    print(f"Y-Bereich: [{y_min:.2f}, {y_max:.2f}] m (Spannweite: {y_max - y_min:.2f} m)")
    print(f"Heatmap Bins: {bins_x} x {bins_y}")
    print(f"Max. Häufigkeit pro Bin: {heatmap_data.max():.0f}")
    print(f"Min. Häufigkeit pro Bin: {heatmap_data.min():.0f}")
    print(f"Durchschn. Häufigkeit: {heatmap_data.mean():.2f}")
    print("=" * 60)


# --- Beispiel-Nutzung ---

# BITTE PASSEN SIE DIESEN PFAD AN IHRE DATENSTRUKTUR AN!
DATASET_FOLDER = "dataset/train_val_test"

# Führen Sie die Funktion aus
try:
    create_comprehensive_heatmaps(
        hdf5_folder=DATASET_FOLDER,
        grid_resolution=0.1,
        bin_multiplier=2,
        sample_size_kde=10000  # Reduzieren bei Performance-Problemen
    )
except Exception as e:
    print(f"\nEin Fehler ist aufgetreten: {e}")
    print("Stellen Sie sicher, dass der Pfad 'dataset/train_val_test' existiert und HDF5-Dateien enthält.")
    import traceback

    traceback.print_exc()
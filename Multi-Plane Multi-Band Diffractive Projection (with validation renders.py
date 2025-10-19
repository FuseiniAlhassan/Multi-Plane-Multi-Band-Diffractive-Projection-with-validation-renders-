

"""
Multi-Plane Multi-Band Diffractive Projection (with validation renders)
------------------------------------------------------------------------
This notebook designs a single diffractive phase element that projects different
target images at different planes/wavelengths. It now includes validation
visualizations: synthetic camera captures near the target planes, RGB composite
render, and PSNR/er
/- intensity_evolution_band*.gif
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import animation
from PIL import Image, ImageDraw, ImageFont # Import ImageDraw and ImageFont
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio as psnr

# Results directory
os.makedirs('results', exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def to_np(x):
    return x.detach().cpu().numpy()

# ----------------------------
# Parameters
# ----------------------------
N = 128                 # grid size
L = 4e-3                # physical aperture (4 mm)
dx = L / N
wavelengths = [450e-9, 550e-9, 650e-9]  # RGB, can add more
planes = [8e-3, 10e-3, 12e-3]           # target planes for each band (m), must match len(wavelengths)
reg_weight = 0.01       # Total variation regularization weight
n_material = 1.5        # Refractive index for height map computation (e.g., glass)

# ----------------------------
# Coordinate grids
# ----------------------------
x = torch.linspace(-L/2, L/2, N, device=device)
X, Y = torch.meshgrid(x, x, indexing='xy')
R2 = X**2 + Y**2

# ----------------------------
# Angular spectrum transfer functions (precompute per lambda and plane)
# ----------------------------
fx = torch.fft.fftfreq(N, d=dx, device=device)
FX, FY = torch.meshgrid(fx, fx, indexing='ij')

transfer = {}
for wl in wavelengths:
    transfer[wl] = {}
    k = 2 * np.pi / wl
    term = torch.clamp(1 - (wl * FX)**2 - (wl * FY)**2, min=0.0)
    for z in planes:
        H = torch.exp(1j * k * z * torch.sqrt(term))
        transfer[wl][z] = H.to(device)

# ----------------------------
# Target images (grayscale) - Now load from files or generate
# ----------------------------
def load_or_generate_target(filename, N, default_gen_func, *args):
    if os.path.exists(filename):
        img = Image.open(filename).convert('L')
        img = img.resize((N, N), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
    else:
        arr = default_gen_func(N, *args)
    return arr

# Note: Text rendering requires specific font files to be available in the environment.
# If font loading fails, a simple square is generated as a fallback.
def make_text(N, text='U', fontsize=96):
    try:
        img = Image.new('L', (N, N), color=0)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype('DejaVuSans-Bold.ttf', fontsize)
        except Exception:
            font = ImageFont.load_default()

        if font is None:
             raise IOError("Could not load any font.") # Raise error if font is still None

        w, h = draw.textbbox((0, 0), text, font=font)[2:]
        draw.text(((N-w)/2, (N-h)/2), text, fill=255, font=font)
        arr = np.array(img).astype(np.float32)/255.0
        return arr
    except IOError:
        print("Warning: Could not load font for text rendering. Generating a square instead.")
        # Fallback: Generate a simple square if font loading fails
        img = np.zeros((N,N), dtype=np.float32)
        square_size = N // 2
        start = (N - square_size) // 2
        img[start : start + square_size, start : start + square_size] = 1.0
        return img


def make_ring(N):
    xv, yv = np.meshgrid(np.linspace(-1,1,N), np.linspace(-1,1,N))
    r = np.sqrt(xv**2 + yv**2)
    return np.clip((0.4 < r) & (r < 0.6), 0, 1).astype(np.float32)

def make_bar(N):
    img = np.zeros((N,N), dtype=np.float32)
    img[N//4:3*N//4, N//2 - N//8:N//2 + N//8] = 1.0
    return np.array(Image.fromarray(img).rotate(20)).astype(np.float32)

# Load or generate targets (can provide custom files like 'target1.png')
target_files = ['target1.png', 'target2.png', 'target3.png']  # Custom files if exist
target_gens = [(make_text, 'M'), (make_ring,), (make_bar,)]
targets = [load_or_generate_target(f, N, gen[0], *gen[1:]) for f, gen in zip(target_files, target_gens)]
for i, t in enumerate(targets):
    plt.imsave(f'results/target_{i+1}.png', t, cmap='gray')

# ----------------------------
# Phase mask parameterization
# ----------------------------
phase = torch.nn.Parameter(2 * np.pi * torch.rand((N,N), device=device))
optimizer = torch.optim.Adam([phase], lr=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

# Input field: uniform amplitude
E0 = torch.ones((N,N), dtype=torch.complex64, device=device)

# ----------------------------
# Forward and loss
# ----------------------------
def propagate(Ein, H):
    return torch.fft.ifft2(torch.fft.fft2(Ein) * H)

def plane_intensity_for_phase(phase_mask, wl, z):
    E = E0 * torch.exp(1j * phase_mask)
    H = transfer[wl][z]
    Eout = propagate(E, H)
    I = torch.abs(Eout)**2
    I = I / (I.max() + 1e-12)
    return I

def total_variation(phase):
    dx = torch.mean(torch.abs(phase[1:, :] - phase[:-1, :]))
    dy = torch.mean(torch.abs(phase[:, 1:] - phase[:, :-1]))
    return dx + dy

# ----------------------------
# Optimization loop
# ----------------------------
steps = 500  # Increased steps
loss_history = []
phase_frames = []
intensity_frames = {i: [] for i in range(len(wavelengths))}  # Per band intensity evolution

for step in tqdm(range(steps)):
    optimizer.zero_grad()
    loss = 0.0
    for i, wl in enumerate(wavelengths):
        z = planes[i]
        I = plane_intensity_for_phase(phase, wl, z)
        tgt = torch.from_numpy(targets[i]).to(device)
        loss += F.mse_loss(I.float(), tgt.float())
    reg_loss = reg_weight * total_variation(phase)
    total_loss = loss + reg_loss
    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss)
    with torch.no_grad():
        phase[:] = (phase + np.pi) % (2 * np.pi) - np.pi

    loss_history.append(total_loss.item())
    if step % 10 == 0:
        phase_frames.append(to_np(phase))
        for i, wl in enumerate(wavelengths):
            z = planes[i]
            I = plane_intensity_for_phase(phase, wl, z)
            intensity_frames[i].append(to_np(I))

# ----------------------------
# Save final phase and loss
# ----------------------------
final_phase = to_np(phase)
np.save('results/final_phase_multiplane.npy', final_phase)

plt.figure(figsize=(6,5))
plt.imshow(final_phase % (2*np.pi), cmap='twilight')
plt.title('Optimized Phase Mask')
plt.colorbar(label='Phase [rad]')
plt.tight_layout()
plt.savefig('results/optimized_phase_multiplane.png', dpi=300)
plt.close()

plt.figure()
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Optimization Loss')
plt.grid(True)
plt.savefig('results/loss_multiplane.png', dpi=300)
plt.close()

# ----------------------------
# Render intensity maps for each wavelength/plane (final)
# ----------------------------
for i, wl in enumerate(wavelengths):
    z = planes[i]
    I = plane_intensity_for_phase(phase, wl, z)
    plt.figure(figsize=(6,5))
    plt.imshow(to_np(I), cmap='inferno')
    plt.title(f'Final Intensity: {int(wl*1e9)} nm at z={z*1e3:.1f} mm')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(f'results/final_intensity_{i+1}.png', dpi=300)
    plt.close()

# ----------------------------
# Animation of phase evolution
# ----------------------------
fig, ax = plt.subplots(figsize=(6,5))
ims = []
for fr in phase_frames:
    im = ax.imshow(fr % (2*np.pi), animated=True, cmap='twilight')
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
ani.save('results/phase_evolution_multiplane.gif', writer='pillow', fps=5)
plt.close()

# ----------------------------
# Animation of intensity evolution per band
# ----------------------------
for i in range(len(wavelengths)):
    fig, ax = plt.subplots(figsize=(6,5))
    ims = []
    for fr in intensity_frames[i]:
        im = ax.imshow(fr, animated=True, cmap='inferno')
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
    ani.save(f'results/intensity_evolution_band{i+1}.gif', writer='pillow', fps=5)
    plt.close()

# ----------------------------
# Compute height map for fabrication (phase / (k0 * (n-1)))
# ----------------------------
# Use central wavelength for reference
wl_ref = wavelengths[len(wavelengths)//2]
height = (final_phase % (2*np.pi)) / (2*np.pi / wl_ref * (n_material - 1))
plt.figure(figsize=(6,5))
plt.imshow(height * 1e6, cmap='viridis')  # in um
plt.title('Height Map (um)')
plt.colorbar(label='Height [um]')
plt.tight_layout()
plt.savefig('results/height_map.png', dpi=300)
plt.close()

# ----------------------------
# Validation visualizations
# ----------------------------
# 1) synthetic camera captures: propagate to small offsets around each target plane
offsets = [-0.5e-3, 0.0, 0.5e-3]  # offsets in meters relative to each plane
validation_captures = []

for i, wl in enumerate(wavelengths):
    z0 = planes[i]
    captures = []
    for d in offsets:
        ztest = z0 + d
        k = 2 * np.pi / wl
        term = torch.clamp(1 - (wl * FX)**2 - (wl * FY)**2, min=0.0)
        Htest = torch.exp(1j * k * ztest * torch.sqrt(term))
        E = E0 * torch.exp(1j * phase)
        Eout = propagate(E, Htest)
        I = torch.abs(Eout)**2
        I = I / (I.max() + 1e-12)
        captures.append(to_np(I))
    validation_captures.append(captures)

# Save validation captures images
for i, caps in enumerate(validation_captures):
    for j, cap in enumerate(caps):
        plt.imsave(f'results/validation_capture_band{i+1}_offset{j}.png', cap, cmap='inferno')

# 2) composite RGB render: use the capture at zero offset for each band and combine
R = validation_captures[2][1] if len(wavelengths) > 2 else validation_captures[0][1]  # Red longest wl
G = validation_captures[1][1] if len(wavelengths) > 1 else validation_captures[0][1]
B = validation_captures[0][1]  # Blue shortest wl
# normalize each channel
R = (R - R.min()) / (R.max() - R.min() + 1e-12)
G = (G - G.min()) / (G.max() - G.min() + 1e-12)
B = (B - B.min()) / (B.max() - B.min() + 1e-12)
RGB = np.stack([R, G, B], axis=-1)
plt.imsave('results/composite_rgb_capture.png', np.clip(RGB, 0, 1))

# 3) Compute PSNR and efficiency between each target and captured intensity at zero offset
metrics = {}
for i in range(len(wavelengths)):
    tgt = targets[i]
    cap = validation_captures[i][1]
    val_psnr = psnr(tgt, cap, data_range=1.0)
    # Efficiency: energy in target mask region
    mask = tgt > 0.1  # Threshold for target region
    energy_target = np.sum(cap[mask])
    energy_total = np.sum(cap)
    efficiency = energy_target / energy_total if energy_total > 0 else 0
    metrics[f'band_{i+1}_psnr'] = float(val_psnr)
    metrics[f'band_{i+1}_efficiency'] = float(efficiency)

# Save metrics
with open('results/validation_metrics.txt', 'w') as f:
    f.write('Validation PSNR and Efficiency metrics (target vs. capture at zero offset)\n')
    for k, v in metrics.items():
        if 'psnr' in k:
            f.write(f'{k}: {v:.2f} dB\n')
        else:
            f.write(f'{k}: {v:.2%}\n')

# 4) Save a comparison figure (target vs captured vs diff)
for i in range(len(wavelengths)):
    tgt = targets[i]
    cap = validation_captures[i][1]
    diff = np.abs(tgt - cap)
    fig, axes = plt.subplots(1,3, figsize=(9,3))
    axes[0].imshow(tgt, cmap='gray'); axes[0].set_title('Target'); axes[0].axis('off')
    axes[1].imshow(cap, cmap='inferno'); axes[1].set_title('Captured'); axes[1].axis('off')
    axes[2].imshow(diff, cmap='viridis'); axes[2].set_title('Abs Diff'); axes[2].axis('off')
    plt.suptitle(f'Band {i+1} ({int(wavelengths[i]*1e9)} nm)')
    plt.tight_layout()
    plt.savefig(f'results/compare_band_{i+1}.png', dpi=200)
    plt.close()

print('Saved validation captures, composite RGB, height map, intensity animations, and validation_metrics.txt in ./results')
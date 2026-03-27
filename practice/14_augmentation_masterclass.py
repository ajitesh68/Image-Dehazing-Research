"""
=============================================================================
PRACTICE 14: AUGMENTATION MASTERCLASS — Data Ko Artificially BADHAO!
=============================================================================

DATA AUGMENTATION KYA HAI?
==============================
Existing images ko TRANSFORM karke NAYE images banao!
1000 images → augmentation se → effectively 10,000+ images!

KYUN ZAROORI:
1. MORE DATA = BETTER MODEL (DL ka golden rule)
2. OVERFITTING rokta hai (same images baar baar nahi dekhta)
3. GENERALIZATION: model VARIATIONS handle karna seekhta hai
4. Cost effective: new data collect karne se SASTA!

TYPES:
  GEOMETRIC:    Shape/position change (flip, rotate, crop)
  PHOTOMETRIC:  Color/brightness change (ColorJitter, blur)
  NOISE-BASED:  Random noise add karo
  ADVANCED:     CutOut, MixUp, CutMix, Mosaic
  PAIRED:       Input-Target DONO pe SAME transform (image-to-image ke liye!)

⚠️ GOLDEN RULE: Augmentation SIRF TRAINING data pe! Test/Val pe KABHI NAHI!

=============================================================================
"""

import numpy as np               # 🔑 YAAD RAKHO: arrays
import os                        # ℹ️ COMMON: files
from PIL import Image, ImageFilter, ImageEnhance  # 🔑 YAAD RAKHO: PIL image processing
import matplotlib                # ℹ️ COMMON: plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # ℹ️ COMMON: plt
import random                    # ℹ️ COMMON: randomness
import math                      # ℹ️ COMMON: math

# PyTorch transforms
try:
    import torch                 # 🔑 YAAD RAKHO: PyTorch
    import torchvision.transforms as T  # 🔑 YAAD RAKHO: T = transforms shorthand (industry convention)
    # TF: tf.image module + tf.keras.layers (preprocessing layers)
    import torchvision.transforms.functional as TF_func  # ⭐ IMPORTANT: functional = manual control
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch not installed for transforms demo")


# =============================================================================
# SECTION 1: GEOMETRIC AUGMENTATIONS (Shape/Position Change)
# =============================================================================

def demonstrate_geometric():
    """
    GEOMETRIC = image ki shape, position, orientation change karo.
    Ye SABSE SAFE augmentations hain — almost hamesha improve karte hain!
    """
    print("=" * 60)
    print("SECTION 1: Geometric Augmentations")
    print("=" * 60)

    # Demo image (gradient pattern — transforms clearly dikhe)
    np.random.seed(42)
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:, :64] = [200, 100, 50]   # Left half orange
    img[:, 64:] = [50, 100, 200]   # Right half blue
    img[32:96, 32:96] = [255, 255, 0]  # Center yellow square
    pil_img = Image.fromarray(img)

    results = {'Original': pil_img}

    # 1. Horizontal Flip
    h_flip = pil_img.transpose(Image.FLIP_LEFT_RIGHT)  # ⭐ IMPORTANT: left-right mirror — SABSE COMMON augmentation!
    # PyTorch: T.RandomHorizontalFlip(p=0.5)
    # TF: tf.image.random_flip_left_right(img)
    results['H-Flip'] = h_flip
    print(f"\n  1. Horizontal Flip:")
    print(f"     PIL:     img.transpose(Image.FLIP_LEFT_RIGHT)")
    print(f"     PyTorch: T.RandomHorizontalFlip(p=0.5)")
    print(f"     TF:      tf.image.random_flip_left_right(img)")
    print(f"     ⚠️ WARNING: Medical images (X-ray) mein flip mat karo — left/right matters!")

    # 2. Vertical Flip
    v_flip = pil_img.transpose(Image.FLIP_TOP_BOTTOM)  # ℹ️ COMMON: top-bottom mirror
    # PyTorch: T.RandomVerticalFlip(p=0.5)
    # TF: tf.image.random_flip_up_down(img)
    results['V-Flip'] = v_flip
    print(f"\n  2. Vertical Flip:")
    print(f"     PyTorch: T.RandomVerticalFlip(p=0.5)")
    print(f"     TF:      tf.image.random_flip_up_down(img)")

    # 3. Rotation
    rotated = pil_img.rotate(30, fillcolor=(0,0,0))  # ⭐ IMPORTANT: arbitrary angle rotation
    # PyTorch: T.RandomRotation(degrees=30)
    # TF: tf.image.rot90(img, k=1) ya tfa.image.rotate(img, angle)
    results['Rotate 30°'] = rotated
    print(f"\n  3. Rotation:")
    print(f"     PIL:     img.rotate(30, fillcolor=(0,0,0))")
    print(f"     PyTorch: T.RandomRotation(degrees=(-30, 30))")
    print(f"     TF:      tfa.image.rotate(img, angle)")

    # 4. Random Crop
    w, h = pil_img.size
    left, top = random.randint(0, 30), random.randint(0, 30)
    cropped = pil_img.crop((left, top, left+96, top+96))  # ⭐ IMPORTANT: (left, top, right, bottom)
    # PyTorch: T.RandomCrop(96)
    # TF: tf.image.random_crop(img, [96, 96, 3])
    results['Random Crop'] = cropped.resize((128, 128))
    print(f"\n  4. Random Crop:")
    print(f"     PIL:     img.crop((left, top, right, bottom))")
    print(f"     PyTorch: T.RandomCrop(size)")
    print(f"     TF:      tf.image.random_crop(img, [H, W, C])")

    # 5. RandomResizedCrop (BEST for training!)
    print(f"\n  5. RandomResizedCrop (INDUSTRY STANDARD!):")
    print(f"     PyTorch: T.RandomResizedCrop(224, scale=(0.08, 1.0))")
    print(f"     TF:      tf.image.random_crop() + tf.image.resize()")
    print(f"     → Random area crop + resize = augmentation + resize EK SAATH!")
    print(f"     ⭐ ImageNet training mein HAMESHA use hota hai")

    # 6. Affine Transform
    print(f"\n  6. Affine Transform (translate + scale + shear):")
    print(f"     PyTorch: T.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.8,1.2))")
    print(f"     TF:      tfa.image.transform(img, transforms)")

    # Plot all
    fig, axes = plt.subplots(1, len(results), figsize=(3*len(results), 3))
    for ax, (name, im) in zip(axes, results.items()):
        ax.imshow(np.array(im)); ax.set_title(name, fontsize=9); ax.axis('off')
    plt.suptitle('Geometric Augmentations', fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs('practice/results', exist_ok=True)
    plt.savefig('practice/results/14_geometric.png', dpi=100)
    plt.close()
    print(f"\n  📊 Saved: practice/results/14_geometric.png")


# =============================================================================
# SECTION 2: PHOTOMETRIC AUGMENTATIONS (Color/Brightness Change)
# =============================================================================

def demonstrate_photometric():
    """
    PHOTOMETRIC = color, brightness, contrast change karo.
    Image ka CONTENT same rehta hai, bas APPEARANCE badal jaata hai.
    Real world mein lighting conditions vary karte hain — ye simulate karta hai!
    """
    print("\n" + "=" * 60)
    print("SECTION 2: Photometric Augmentations")
    print("=" * 60)

    np.random.seed(42)
    img = np.random.randint(60, 200, (128, 128, 3), dtype=np.uint8)
    img[30:100, 30:100] = [255, 150, 50]  # Orange patch
    pil_img = Image.fromarray(img)

    results = {'Original': pil_img}

    # 1. Brightness
    bright = ImageEnhance.Brightness(pil_img).enhance(1.5)  # ⭐ IMPORTANT: >1 = brighter, <1 = darker
    results['Bright ×1.5'] = bright
    # PyTorch: T.ColorJitter(brightness=0.5)  — range [1-0.5, 1+0.5] = [0.5, 1.5]
    # TF: tf.image.random_brightness(img, max_delta=0.3)
    print(f"\n  1. Brightness:")
    print(f"     PIL:     ImageEnhance.Brightness(img).enhance(factor)")
    print(f"     PyTorch: T.ColorJitter(brightness=0.5)")
    print(f"     TF:      tf.image.random_brightness(img, 0.3)")

    # 2. Contrast
    contrast = ImageEnhance.Contrast(pil_img).enhance(1.8)  # ⭐ IMPORTANT: contrast enhance
    results['Contrast ×1.8'] = contrast
    # PyTorch: T.ColorJitter(contrast=0.5)
    # TF: tf.image.random_contrast(img, 0.5, 1.5)
    print(f"\n  2. Contrast:")
    print(f"     PyTorch: T.ColorJitter(contrast=0.5)")
    print(f"     TF:      tf.image.random_contrast(img, lower, upper)")

    # 3. Saturation
    saturated = ImageEnhance.Color(pil_img).enhance(2.0)  # ⭐ IMPORTANT: color vibrancy
    results['Saturate ×2'] = saturated
    # PyTorch: T.ColorJitter(saturation=0.5)
    # TF: tf.image.random_saturation(img, 0.5, 1.5)
    print(f"\n  3. Saturation:")
    print(f"     PyTorch: T.ColorJitter(saturation=0.5)")
    print(f"     TF:      tf.image.random_saturation(img, lower, upper)")

    # 4. Hue
    print(f"\n  4. Hue shift:")
    print(f"     PyTorch: T.ColorJitter(hue=0.1)")
    print(f"     TF:      tf.image.random_hue(img, max_delta=0.1)")
    print(f"     ⚠️ WARNING: hue zyada change karo → images UNREALISTIC lagti hain")

    # 5. ALL TOGETHER — ColorJitter (MOST COMMON!)
    print(f"\n  5. ColorJitter (ALL IN ONE — INDUSTRY STANDARD!):")
    print(f"     PyTorch: T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)")
    print(f"     Ek line mein SAARI color augmentations!")

    # 6. Gaussian Blur
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=2))  # ℹ️ COMMON: blur
    results['Blur r=2'] = blurred
    # PyTorch: T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    # TF: tfa.image.gaussian_filter2d(img, sigma=2.0)
    print(f"\n  6. Gaussian Blur:")
    print(f"     PIL:     img.filter(ImageFilter.GaussianBlur(radius=2))")
    print(f"     PyTorch: T.GaussianBlur(kernel_size=5)")

    # 7. Grayscale
    gray = pil_img.convert('L').convert('RGB')  # ℹ️ COMMON: gray → back to RGB (channels same)
    results['Grayscale'] = gray
    # PyTorch: T.RandomGrayscale(p=0.1)
    # TF: tf.image.rgb_to_grayscale()
    print(f"\n  7. Random Grayscale:")
    print(f"     PyTorch: T.RandomGrayscale(p=0.1)")
    print(f"     p=0.1 → 10% chance se grayscale → color pe depend mat karo!")

    # Plot
    fig, axes = plt.subplots(1, len(results), figsize=(3*len(results), 3))
    for ax, (name, im) in zip(axes, results.items()):
        ax.imshow(np.array(im)); ax.set_title(name, fontsize=8); ax.axis('off')
    plt.suptitle('Photometric Augmentations', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('practice/results/14_photometric.png', dpi=100)
    plt.close()
    print(f"\n  📊 Saved: practice/results/14_photometric.png")


# =============================================================================
# SECTION 3: NOISE-BASED AUGMENTATIONS
# =============================================================================

def demonstrate_noise():
    """
    Real world mein images mein NOISE hota hai (camera sensor, low light).
    Noise add karke model ko ROBUST banao!
    """
    print("\n" + "=" * 60)
    print("SECTION 3: Noise-Based Augmentations")
    print("=" * 60)

    np.random.seed(42)
    img = np.random.randint(60, 200, (128, 128, 3), dtype=np.uint8)
    img[30:100, 30:100] = [200, 100, 50]

    results = {'Original': img}

    # 1. Gaussian Noise
    noise = np.random.normal(0, 25, img.shape)  # 🔑 YAAD RAKHO: normal(mean, std, shape)
    # TF: tf.random.normal(shape, mean=0, stddev=25)
    gaussian = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # ⭐ IMPORTANT: clip(0,255) ZAROORI — warna invalid pixel values!
    results['Gaussian\nσ=25'] = gaussian
    print(f"\n  1. Gaussian Noise:")
    print(f"     noise = np.random.normal(0, std, img.shape)")
    print(f"     noisy = np.clip(img + noise, 0, 255)")
    print(f"     TF: img + tf.random.normal(shape, stddev=25)")

    # 2. Salt & Pepper Noise
    sp = img.copy()
    num_salt = int(0.02 * img.size / 3)  # ℹ️ COMMON: 2% pixels
    num_pepper = int(0.02 * img.size / 3)
    # Salt (white dots)
    for _ in range(num_salt):
        y, x = random.randint(0, 127), random.randint(0, 127)
        sp[y, x] = [255, 255, 255]       # ℹ️ COMMON: white pixel
    # Pepper (black dots)
    for _ in range(num_pepper):
        y, x = random.randint(0, 127), random.randint(0, 127)
        sp[y, x] = [0, 0, 0]             # ℹ️ COMMON: black pixel
    results['Salt &\nPepper'] = sp
    print(f"\n  2. Salt & Pepper Noise:")
    print(f"     Random pixels ko white (salt) ya black (pepper) karo")
    print(f"     Camera sensor defects simulate karta hai")

    # 3. Speckle Noise
    speckle = np.random.randn(*img.shape) * 0.2  # ℹ️ COMMON: multiplicative noise
    speckled = np.clip(img.astype(np.float32) * (1 + speckle), 0, 255).astype(np.uint8)
    # ⭐ IMPORTANT: speckle = MULTIPLICATIVE (gaussian = ADDITIVE) — fark samjho!
    results['Speckle'] = speckled
    print(f"\n  3. Speckle Noise (Multiplicative):")
    print(f"     noisy = img * (1 + random_noise)")
    print(f"     Radar/ultrasound images mein common hai")

    # Plot
    fig, axes = plt.subplots(1, len(results), figsize=(3*len(results), 3))
    for ax, (name, im) in zip(axes, results.items()):
        ax.imshow(im); ax.set_title(name, fontsize=9); ax.axis('off')
    plt.suptitle('Noise Augmentations', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('practice/results/14_noise.png', dpi=100)
    plt.close()
    print(f"\n  📊 Saved: practice/results/14_noise.png")


# =============================================================================
# SECTION 4: ADVANCED AUGMENTATIONS (CutOut, MixUp, CutMix)
# =============================================================================

def demonstrate_advanced():
    """
    MODERN augmentation techniques — research papers mein use hote hain!
    Ye simple techniques se ZYADA effective hain.
    """
    print("\n" + "=" * 60)
    print("SECTION 4: Advanced Augmentations (CutOut, MixUp, CutMix)")
    print("=" * 60)

    np.random.seed(42)
    img1 = np.zeros((128, 128, 3), dtype=np.uint8)
    img1[:, :] = [200, 100, 50]  # Orange
    img1[30:100, 30:100] = [255, 255, 0]  # Yellow center

    img2 = np.zeros((128, 128, 3), dtype=np.uint8)
    img2[:, :] = [50, 100, 200]  # Blue
    img2[20:80, 40:110] = [0, 255, 0]  # Green rectangle

    results = {'Image A': img1, 'Image B': img2}

    # 1. CutOut / Random Erasing
    cutout = img1.copy()
    cx, cy = random.randint(20, 100), random.randint(20, 100)
    size = 40                    # ℹ️ COMMON: erasing patch size
    cutout[cy:cy+size, cx:cx+size] = 0  # ⭐ IMPORTANT: ek random patch ko BLACK (0) karo
    # PyTorch: T.RandomErasing(p=0.5, scale=(0.02, 0.33))
    # TF: tfa.image.random_cutout(img, mask_size)
    results['CutOut'] = cutout
    print(f"\n  1. CutOut / Random Erasing:")
    print(f"     Random patch ko zero karo → model PARTIAL info se seekhe!")
    print(f"     PyTorch: T.RandomErasing(p=0.5, scale=(0.02, 0.33))")
    print(f"     TF:      tfa.image.random_cutout(img)")
    print(f"     Paper: 'Improved Regularization of CNNs with Cutout' (2017)")

    # 2. MixUp
    alpha = 0.4                  # ⭐ IMPORTANT: mixing ratio
    lam = np.random.beta(alpha, alpha)  # 🔑 YAAD RAKHO: Beta distribution se lambda sample karo
    mixup = (lam * img1.astype(np.float32) + (1-lam) * img2.astype(np.float32)).astype(np.uint8)
    # ⭐ IMPORTANT: MixUp = 2 images ka WEIGHTED AVERAGE (labels bhi mix hote hain!)
    # label_mixed = lam * label1 + (1-lam) * label2
    results[f'MixUp\nλ={lam:.2f}'] = mixup
    print(f"\n  2. MixUp:")
    print(f"     mixed = λ × img1 + (1-λ) × img2")
    print(f"     label = λ × label1 + (1-λ) × label2")
    print(f"     λ ~ Beta(α, α), α=0.4 typical")
    print(f"     Paper: 'mixup: Beyond Empirical Risk Minimization' (2018)")

    # 3. CutMix
    cutmix = img1.copy()
    # Random box coordinates
    cx, cy = 64, 64
    bw, bh = 60, 60
    x1, y1 = cx - bw//2, cy - bh//2
    x2, y2 = cx + bw//2, cy + bh//2
    cutmix[y1:y2, x1:x2] = img2[y1:y2, x1:x2]  # ⭐ IMPORTANT: img1 ka ek patch img2 se REPLACE karo!
    # Label adjustment: area ratio se label mix karo
    results['CutMix'] = cutmix
    print(f"\n  3. CutMix (CutOut + MixUp combined — BEST!):")
    print(f"     img1 ka random patch img2 se replace karo")
    print(f"     label = area_ratio × label1 + (1-area_ratio) × label2")
    print(f"     Paper: 'CutMix: Regularization Strategy' (2019)")
    print(f"     ⭐ ImageNet SOTA mein use hota hai!")

    # 4. Mosaic (YOLO style)
    h, w = 64, 64
    mosaic = np.zeros((128, 128, 3), dtype=np.uint8)
    mosaic[:h, :w] = np.array(Image.fromarray(img1).resize((w, h)))
    mosaic[:h, w:] = np.array(Image.fromarray(img2).resize((w, h)))
    mosaic[h:, :w] = np.array(Image.fromarray(img2[::-1]).resize((w, h)))
    mosaic[h:, w:] = np.array(Image.fromarray(img1[:, ::-1]).resize((w, h)))
    results['Mosaic\n(4 images)'] = mosaic
    # ⭐ IMPORTANT: 4 images ko EK image mein combine karo — YOLO v4/v5 mein use hota hai!
    print(f"\n  4. Mosaic (YOLOv4 style):")
    print(f"     4 images ko 1 mein combine karo (2×2 grid)")
    print(f"     Object detection mein BAHUT effective!")
    print(f"     Paper: 'YOLOv4: Optimal Speed and Accuracy' (2020)")

    # Plot
    fig, axes = plt.subplots(1, len(results), figsize=(3*len(results), 3))
    for ax, (name, im) in zip(axes, results.items()):
        ax.imshow(im); ax.set_title(name, fontsize=8); ax.axis('off')
    plt.suptitle('Advanced Augmentations', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('practice/results/14_advanced.png', dpi=100)
    plt.close()
    print(f"\n  📊 Saved: practice/results/14_advanced.png")


# =============================================================================
# SECTION 5: PAIRED AUGMENTATION (Image-to-Image Tasks — DEHAZING!)
# =============================================================================

def demonstrate_paired_augmentation():
    """
    IMAGE-TO-IMAGE tasks (dehazing, super-resolution, colorization) mein:
    Input (hazy) aur Target (clean) DONO pe SAME augmentation lagani chahiye!

    ⚠️ BAHUT IMPORTANT:
    Agar hazy image flip kari par clean nahi → model GALAT pairing seekhega!
    Dono pe EXACTLY SAME transform → consistent pairs!

    TRICK: Same random seed use karo!
    """
    print("\n" + "=" * 60)
    print("SECTION 5: Paired Augmentation (Image-to-Image)")
    print("=" * 60)

    np.random.seed(42)
    hazy = np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)
    clean = np.random.randint(50, 250, (128, 128, 3), dtype=np.uint8)

    print(f"""
  ⚠️ GALAT TARIKA (DONO pe ALAG transform):
  ─────────────────────────────────────────
  hazy_flip  = T.RandomHorizontalFlip()(hazy)   # Maybe flip
  clean_flip = T.RandomHorizontalFlip()(clean)  # Maybe NOT flip
  # ❌ WRONG! Hazy flipped par clean nahi → MISMATCH!

  ✅ SAHI TARIKA (SEED MATCHING):
  ───────────────────────────────
  # Method 1: Functional transforms with same params
  if random.random() > 0.5:
      hazy  = TF_func.hflip(hazy)
      clean = TF_func.hflip(clean)   # ✅ DONO flip!

  # Method 2: Same seed trick
  seed = torch.randint(0, 2**32, (1,)).item()
  torch.manual_seed(seed)
  hazy_aug  = transform(hazy)
  torch.manual_seed(seed)              # ⭐ SAME seed!
  clean_aug = transform(clean)         # ✅ SAME transform!

  # Method 3: Custom paired transform class
  class PairedTransform:
      def __call__(self, img1, img2):
          if random.random() > 0.5:
              img1 = TF_func.hflip(img1)
              img2 = TF_func.hflip(img2)
          angle = random.uniform(-15, 15)
          img1 = TF_func.rotate(img1, angle)
          img2 = TF_func.rotate(img2, angle)    # ⭐ SAME angle!
          return img1, img2
    """)

    print(f"  PAIRED AUGMENTATIONS (dono pe lagao):")
    print(f"  ✅ Horizontal Flip")
    print(f"  ✅ Vertical Flip")
    print(f"  ✅ Random Rotation")
    print(f"  ✅ Random Crop (same position se!)")
    print(f"  ✅ Resize")
    print(f"")
    print(f"  SIRF INPUT PE LAGAO (target pe NAHI!):")
    print(f"  ✅ Gaussian Noise (hazy mein noise realistic hai)")
    print(f"  ✅ Blur (hazy mein blur natural hai)")
    print(f"  ❌ ColorJitter — ye dono pe nahi lagana, artifacts aa sakte hain")


# =============================================================================
# SECTION 6: TORCHVISION TRANSFORMS COMPOSE (Complete Pipeline)
# =============================================================================

def demonstrate_compose_pipeline():
    """
    transforms.Compose — SABKUCH ek chain mein!
    Real project mein aisi pipeline banate hain.
    """
    print("\n" + "=" * 60)
    print("SECTION 6: Complete Augmentation Pipeline")
    print("=" * 60)

    print(f"""
  TRAINING PIPELINE (Classification):
  ─────────────────────────────────────
  train_transform = T.Compose([
      T.RandomResizedCrop(224, scale=(0.8, 1.0)),  # ⭐ Resize + crop
      T.RandomHorizontalFlip(p=0.5),                # ⭐ 50% flip
      T.ColorJitter(0.3, 0.3, 0.3, 0.1),            # ⭐ Color variation
      T.RandomGrayscale(p=0.05),                     # ℹ️ 5% grayscale
      T.GaussianBlur(kernel_size=3, sigma=(0.1,2)),  # ℹ️ Random blur
      T.ToTensor(),                                  # 🔑 PIL→Tensor, /255
      T.Normalize([0.485,0.456,0.406],               # 🔑 ImageNet norm
                  [0.229,0.224,0.225]),
      T.RandomErasing(p=0.2),                        # ⭐ CutOut (after tensor!)
  ])

  TEST PIPELINE:
  ──────────────
  test_transform = T.Compose([
      T.Resize(256),                                # ℹ️ Fixed resize
      T.CenterCrop(224),                            # ℹ️ Deterministic crop
      T.ToTensor(),                                 # 🔑 PIL→Tensor
      T.Normalize([0.485,0.456,0.406],              # 🔑 SAME norm as train!
                  [0.229,0.224,0.225]),
      # ⚠️ NO augmentation here! Test = consistent!
  ])

  DEHAZING PIPELINE (Paired Image-to-Image):
  ────────────────────────────────────────────
  # Shared transforms (dono pe lagao):
  shared_transform = T.Compose([
      T.Resize((256, 256)),
      T.RandomHorizontalFlip(p=0.5),  # ⚠️ SAME flip using seed!
      T.RandomVerticalFlip(p=0.5),
      T.ToTensor(),
  ])

  TF EQUIVALENT:
  ──────────────
  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_ds = train_ds.map(lambda x, y: augment(x, y))
  train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    """)

    print(f"  🔑 PIPELINE RULES:")
    print(f"  1. Resize PEHLE, augmentation BAAD mein")
    print(f"  2. ToTensor() ke PEHLE PIL operations, BAAD mein Tensor operations")
    print(f"  3. Normalize HAMESHA LAST (ToTensor ke baad)")
    print(f"  4. RandomErasing tensor pe kaam karta hai (ToTensor ke baad)")
    print(f"  5. Train/Test mein Normalize SAME values use karo!")


# =============================================================================
# SECTION 7: AUGMENTATION STRENGTH GUIDE
# =============================================================================

def augmentation_guide():
    """
    KITNI augmentation lagayen? Zyada ya kam?
    """
    print("\n" + "=" * 60)
    print("SECTION 7: Augmentation Strength Guide")
    print("=" * 60)

    print(f"""
  ┌──────────────────┬──────────────┬──────────────┬──────────────┐
  │ Augmentation     │ Light        │ Medium       │ Heavy        │
  ├──────────────────┼──────────────┼──────────────┼──────────────┤
  │ H-Flip           │ p=0.5 ✅     │ p=0.5 ✅     │ p=0.5 ✅     │
  │ V-Flip           │ ❌           │ p=0.3        │ p=0.5        │
  │ Rotation         │ ±10°         │ ±20°         │ ±45°         │
  │ ColorJitter      │ 0.1          │ 0.3          │ 0.5          │
  │ Blur             │ ❌           │ σ=0.5        │ σ=2.0        │
  │ CutOut           │ ❌           │ 10%          │ 33%          │
  │ MixUp            │ ❌           │ α=0.2        │ α=1.0        │
  │ CutMix           │ ❌           │ α=0.5        │ α=1.0        │
  └──────────────────┴──────────────┴──────────────┴──────────────┘

  KAISE DECIDE KAREIN?
  ─────────────────────
  Small dataset (<1K images)  → HEAVY augmentation (data kam hai!)
  Medium dataset (1K-10K)     → MEDIUM augmentation
  Large dataset (>10K)        → LIGHT augmentation (data enough hai)
  
  DEHAZING ke liye:           → MEDIUM + Paired augmentation
  Classification ke liye:     → HEAVY (CutMix, MixUp, etc.)
  Medical imaging ke liye:    → LIGHT (realistic rehna zaroori!)
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":       # 🔑 YAAD RAKHO: main guard
    print("=" * 60)
    print("PRACTICE 14: Augmentation Masterclass")
    print("=" * 60)

    demonstrate_geometric()                # Section 1
    demonstrate_photometric()              # Section 2
    demonstrate_noise()                    # Section 3
    demonstrate_advanced()                 # Section 4
    demonstrate_paired_augmentation()      # Section 5
    demonstrate_compose_pipeline()         # Section 6
    augmentation_guide()                   # Section 7

    print(f"\n{'='*60}")
    print(f"PRACTICE 14 COMPLETE!")
    print(f"{'='*60}")
    print(f"\nKYA SEEKHA:")
    print(f"  1. Geometric — Flip, Rotate, Crop, Affine")
    print(f"  2. Photometric — Color, Brightness, Contrast, Blur")
    print(f"  3. Noise — Gaussian, Salt&Pepper, Speckle")
    print(f"  4. Advanced — CutOut, MixUp, CutMix, Mosaic")
    print(f"  5. Paired — same transform on input+target (SEED matching!)")
    print(f"  6. Pipeline — Compose([...]) for train vs test")
    print(f"  7. Strength guide — small data=heavy, large data=light")
    print(f"\nNEXT: python practice/15_custom_dataset_dataloader.py")

"""
=============================================================================
PRACTICE 13: DATA CLEANING & PREPROCESSING — Data Ko Model-Ready Banao!
=============================================================================

CLEANING vs PREPROCESSING — FARK SAMJHO!
==========================================
CLEANING = GALAT data hatao (corrupt, duplicates, wrong format)
          → "Kachra saaf karo" 🧹

PREPROCESSING = SAHI data ko MODEL-FRIENDLY format mein laao
              → "Ingredients ko recipe ke hisaab se kaat lo" 🔪

DONO ZAROORI HAI — agar data saaf nahi toh model GALAT seekhega!

IS FILE MEIN SEEKHOGE:
  ✅ Corrupt image removal & verification
  ✅ Duplicate detection (hash-based)
  ✅ Empty/blank image detection
  ✅ Normalization: MinMax [0,1], Z-score, ImageNet
  ✅ Resizing strategies: resize, crop, pad
  ✅ Color space conversion: RGB, BGR, HSV, LAB, Gray
  ✅ Histogram equalization & CLAHE
  ✅ Channel ordering: CHW vs HWC
  ✅ Dtype conversion: uint8 vs float32

=============================================================================
"""

import numpy as np               # 🔑 YAAD RAKHO: numerical operations
import os                        # ℹ️ COMMON: file operations
from PIL import Image, ImageFilter  # 🔑 YAAD RAKHO: PIL image processing
# TF: tf.image module for most operations
import matplotlib                # ℹ️ COMMON: plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # ℹ️ COMMON: plt
import hashlib                   # ℹ️ COMMON: hashing for duplicates

# OpenCV optional hai — install nahi hai toh bhi chalega
try:
    import cv2                   # ⭐ IMPORTANT: OpenCV — computer vision ki STANDARD library
    # TF: OpenCV TF mein bhi use hota hai — framework independent
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ OpenCV not installed. Install: pip install opencv-python")


# =============================================================================
# SECTION 1: CORRUPT IMAGE CLEANING
# =============================================================================

def clean_corrupt_images(image_dir, dry_run=True):
    """
    Corrupt/broken images detect aur optionally remove karo.

    3 LEVELS OF CHECK:
    Level 1: PIL.verify()  → file structure check (FAST)
    Level 2: PIL.load()    → actually pixels load karo (THOROUGH)
    Level 3: Size check    → suspiciously small files

    dry_run=True: sirf REPORT karo, delete mat karo (SAFE!)
    dry_run=False: actually delete karo (DANGEROUS — backup lo pehle!)
    """
    print("=" * 60)
    print("SECTION 1: Corrupt Image Cleaning")
    print("=" * 60)

    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    results = {'valid': 0, 'corrupt_verify': 0, 'corrupt_load': 0, 'too_small': 0}

    # Demo mode
    if not os.path.exists(image_dir):
        print(f"\n  Demo mode (directory nahi mili: {image_dir})")
        _demo_cleaning()
        return results

    for root, dirs, files in os.walk(image_dir):  # 🔑 YAAD RAKHO: os.walk recursive
        for f in files:
            ext = os.path.splitext(f)[1].lower()  # ℹ️ COMMON: extension check
            if ext not in valid_ext:
                continue

            path = os.path.join(root, f)  # ℹ️ COMMON: full path

            # Level 1: Size check
            size_kb = os.path.getsize(path) / 1024  # ℹ️ COMMON: file size in KB
            if size_kb < 1:                          # ⚠️ WARNING: 1KB se choti image almost definitely corrupt
                results['too_small'] += 1
                if not dry_run:
                    os.remove(path)
                continue

            # Level 2: PIL verify
            try:
                img = Image.open(path)   # 🔑 YAAD RAKHO: open
                img.verify()             # ⭐ IMPORTANT: file integrity check
            except Exception:
                results['corrupt_verify'] += 1
                if not dry_run:
                    os.remove(path)
                continue

            # Level 3: Actually load pixels
            try:
                img = Image.open(path)   # ℹ️ COMMON: reopen (verify ke baad close ho jaata hai)
                img.load()               # ⭐ IMPORTANT: force pixel loading — thorough check
                # TF: tf.image.decode_image() mein same check hota hai automatically
                results['valid'] += 1
            except Exception:
                results['corrupt_load'] += 1
                if not dry_run:
                    os.remove(path)

    print(f"\n  Results:")
    print(f"  ✅ Valid:          {results['valid']}")
    print(f"  ❌ Corrupt (verify): {results['corrupt_verify']}")
    print(f"  ❌ Corrupt (load):   {results['corrupt_load']}")
    print(f"  ❌ Too small (<1KB): {results['too_small']}")
    if dry_run:
        print(f"\n  ℹ️ DRY RUN — kuch delete nahi hua. dry_run=False karo delete ke liye.")
    return results


def _demo_cleaning():
    """Demo cleaning concepts"""
    print("\n  Cleaning Methods:")
    print("  ─────────────────")
    print("  # Method 1: PIL verify (FAST)")
    print("  img = Image.open(path)")
    print("  img.verify()  # ⭐ File structure check")
    print()
    print("  # Method 2: PIL load (THOROUGH)")
    print("  img = Image.open(path)")
    print("  img.load()    # ⭐ Actually pixel data load")
    print()
    print("  # Method 3: Size check")
    print("  if os.path.getsize(path) < 1024:  # < 1KB")
    print("      print('Suspiciously small!')")


# =============================================================================
# SECTION 2: NORMALIZATION TECHNIQUES
# =============================================================================

def demonstrate_normalization():
    """
    NORMALIZATION KYA HAI?
    Raw pixel values (0-255) ko chote range mein laana.

    KYUN ZAROORI:
    1. Bade values → bade gradients → UNSTABLE training
    2. Different features alag scales pe hain → bias create
    3. Activation functions (sigmoid/tanh) small range mein kaam karte hain
    4. Optimizer (Adam/SGD) normalized data pe BETTER kaam karte hain

    3 TYPES:
    1. MinMax [0,1]:     x_norm = x / 255
    2. MinMax [-1,1]:    x_norm = (x / 255) * 2 - 1  (GANs ke liye)
    3. Z-score:          x_norm = (x - mean) / std   (ImageNet style)
    """
    print("\n" + "=" * 60)
    print("SECTION 2: Normalization Techniques")
    print("=" * 60)

    # Demo image (random pixels)
    np.random.seed(42)
    img = np.random.randint(50, 200, (8, 8), dtype=np.uint8)  # ℹ️ COMMON: dummy image
    print(f"\n  Original image (0-255):")
    print(f"  Range: [{img.min()}, {img.max()}], Mean: {img.mean():.1f}")

    # --- Method 1: MinMax [0, 1] ---
    img_01 = img.astype(np.float32) / 255.0  # ⭐ IMPORTANT: (0-255) → (0-1) — SABSE COMMON method!
    # 🔑 YAAD RAKHO: .astype(float32) PEHLE karo, warna integer division hoga → sab 0!
    # TF: tf.image.convert_image_dtype(img, tf.float32) — automatically 0-1 mein
    print(f"\n  MinMax [0,1]:   x / 255")
    print(f"  Range: [{img_01.min():.2f}, {img_01.max():.2f}], Mean: {img_01.mean():.3f}")
    print(f"  USE CASE: Image reconstruction, segmentation, dehazing (HAMARA PROJECT!)")

    # --- Method 2: MinMax [-1, 1] ---
    img_11 = (img.astype(np.float32) / 255.0) * 2 - 1  # ⭐ IMPORTANT: (0-1) → (-1,1) — GANs ke liye!
    # TF: tf.image.convert_image_dtype() phir * 2 - 1
    print(f"\n  MinMax [-1,1]:  (x / 255) * 2 - 1")
    print(f"  Range: [{img_11.min():.2f}, {img_11.max():.2f}], Mean: {img_11.mean():.3f}")
    print(f"  USE CASE: GANs (Tanh output ke saath match karna)")

    # --- Method 3: Z-score (StandardScaler) ---
    mean, std = img.mean(), img.std()              # ℹ️ COMMON: stats compute
    img_zscore = (img.astype(np.float32) - mean) / std  # ⭐ IMPORTANT: Z-score = (x-μ)/σ — mean=0, std=1
    # TF: tf.image.per_image_standardization(img)
    print(f"\n  Z-score:        (x - mean) / std")
    print(f"  Range: [{img_zscore.min():.2f}, {img_zscore.max():.2f}], Mean: {img_zscore.mean():.6f}")
    print(f"  USE CASE: Classification with pre-trained models (ImageNet normalize)")

    # --- Method 4: ImageNet Normalization ---
    imagenet_mean = np.array([0.485, 0.456, 0.406])  # ⭐ IMPORTANT: ImageNet ka OFFICIAL mean — ye ratt lo!
    imagenet_std = np.array([0.229, 0.224, 0.225])    # ⭐ IMPORTANT: ImageNet ka OFFICIAL std
    # 🔑 YAAD RAKHO: Pre-trained models (ResNet, VGG, etc.) ne YE distribution seekhi hai
    # Agar tum alag normalize karoge → model ka performance BARBAAD!
    print(f"\n  ImageNet Norm:  (x - [0.485,0.456,0.406]) / [0.229,0.224,0.225]")
    print(f"  USE CASE: Transfer learning with pretrained models")
    print(f"  PyTorch: transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])")
    print(f"  TF:      tf.keras.applications.resnet.preprocess_input(img)")

    # Comparison plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    titles = ['Original\n(0-255)', 'MinMax\n[0,1]', 'MinMax\n[-1,1]', 'Z-score\n(μ=0,σ=1)']
    data = [img, img_01, img_11, img_zscore]
    for ax, d, t in zip(axes, data, titles):
        ax.imshow(d, cmap='gray'); ax.set_title(t, fontsize=10)
        ax.axis('off')
    plt.suptitle('Normalization Methods Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs('practice/results', exist_ok=True)
    plt.savefig('practice/results/13_normalization.png', dpi=100)
    plt.close()
    print(f"\n  📊 Saved: practice/results/13_normalization.png")


# =============================================================================
# SECTION 3: RESIZING STRATEGIES
# =============================================================================

def demonstrate_resizing():
    """
    Neural networks ko FIXED SIZE input chahiye!
    Resize KAISE karein? Multiple strategies hain:

    1. SIMPLE RESIZE:     Seedha size change karo (distortion ho sakti hai)
    2. CENTER CROP:       Center se ek portion crop karo (corners ka info lose)
    3. PAD + RESIZE:      Padding lagao phir resize (no distortion!)
    4. RANDOM CROP:       Random position se crop (training augmentation!)
    5. ASPECT PRESERVE:   Aspect ratio rakho, pad karo baaki

    KAUN SA USE KAREIN?
    - Training:  RandomResizedCrop (augmentation + resizing ek saath!)
    - Testing:   Resize + CenterCrop (consistent results ke liye)
    """
    print("\n" + "=" * 60)
    print("SECTION 3: Resizing Strategies")
    print("=" * 60)

    # Demo image
    np.random.seed(42)
    # Imagine a 480×640 image (typical camera photo)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    target_size = (256, 256)     # ℹ️ COMMON: target size for model

    print(f"\n  Original:  {img.shape[:2]} (H×W)")
    print(f"  Target:    {target_size}")

    # Method 1: Simple resize with PIL
    pil_img = Image.fromarray(img)           # 🔑 YAAD RAKHO: numpy → PIL
    # TF: tf.image.resize(img, [256, 256])

    resized = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    # ⚠️ WARNING: PIL.resize() = (WIDTH, HEIGHT) — ULTA hai numpy se!
    # 🔑 YAAD RAKHO: PIL = (W,H), numpy/torch = (H,W) — HAMESHA confuse hota hai!
    print(f"\n  1. Simple Resize:   {np.array(resized).shape[:2]}")
    print(f"     PIL:     img.resize((W, H), Image.BILINEAR)")
    print(f"     PyTorch: transforms.Resize((H, W))")
    print(f"     TF:      tf.image.resize(img, [H, W])")

    # Method 2: Center crop
    h, w = img.shape[:2]
    crop_size = min(h, w)                    # ⭐ IMPORTANT: chota dimension le lo → square crop
    start_h = (h - crop_size) // 2           # ℹ️ COMMON: center find karo
    start_w = (w - crop_size) // 2
    cropped = img[start_h:start_h+crop_size, start_w:start_w+crop_size]
    # TF: tf.image.central_crop(img, central_fraction=0.75)
    print(f"\n  2. Center Crop:     {cropped.shape[:2]} → then resize to {target_size}")
    print(f"     PyTorch: transforms.CenterCrop(min(H,W))")
    print(f"     TF:      tf.image.central_crop(img, fraction)")

    # Method 3: Pad to square, then resize
    pad_h = max(0, w - h)                    # ℹ️ COMMON: padding calculate
    pad_w = max(0, h - w)
    padded = np.pad(img, ((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2), (0, 0)))
    # ⭐ IMPORTANT: np.pad = border pe zeros (ya mirror/reflect) add karo
    # TF: tf.pad(img, paddings)
    print(f"\n  3. Pad + Resize:    Pad to {padded.shape[:2]} → resize to {target_size}")
    print(f"     → NO distortion! Aspect ratio preserved!")
    print(f"     PyTorch: F.pad(img, [left,right,top,bottom])")
    print(f"     TF:      tf.pad(img, paddings)")

    # Method 4: RandomResizedCrop (TRAINING ke liye BEST!)
    print(f"\n  4. RandomResizedCrop (BEST for training!):")
    print(f"     PyTorch: transforms.RandomResizedCrop(256, scale=(0.8, 1.0))")
    print(f"     TF:      tf.image.random_crop(img, [256, 256, 3])")
    print(f"     → Random position + random scale = augmentation + resize ek saath!")

    # Interpolation methods
    print(f"\n  💡 INTERPOLATION METHODS:")
    print(f"     NEAREST:   Fast, par pixelated (blocky)")
    print(f"     BILINEAR:  Good balance (DEFAULT in most frameworks)")
    print(f"     BICUBIC:   Smoother (better quality, slower)")
    print(f"     LANCZOS:   Best quality (sabse slow)")
    print(f"  ⭐ RECOMMENDATION: BILINEAR for training, BICUBIC for final results")


# =============================================================================
# SECTION 4: COLOR SPACE CONVERSION
# =============================================================================

def demonstrate_color_spaces():
    """
    COLOR SPACES KYA HAIN?
    Ek image ko ALAG tarike se represent karo:

    RGB:  Red, Green, Blue — standard (camera output)
    BGR:  Blue, Green, Red — OpenCV ka default (⚠️ TRAP!)
    HSV:  Hue, Saturation, Value — color operations ke liye BEST
    LAB:  Lightness, A, B — perceptually uniform (human aankh ke liye)
    Gray: Single channel — simplest

    KYUN IMPORTANT:
    - RGB achi hai par color operations (brightness, contrast) mein MUSHKIL
    - HSV mein brightness change karna EASY (sirf V channel change karo!)
    - LAB mein lightness alag hai — dehazing mein bahut useful
    - BGR vs RGB confuse mat ho — OpenCV HAMESHA BGR deta hai!
    """
    print("\n" + "=" * 60)
    print("SECTION 4: Color Space Conversion")
    print("=" * 60)

    # Demo image
    np.random.seed(42)
    img_rgb = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)

    print(f"\n  Color Space Conversions:")
    print(f"  ─────────────────────────")

    # RGB → Grayscale
    gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    # ⭐ IMPORTANT: Grayscale = 0.299R + 0.587G + 0.114B (ITU standard)
    # Ye numbers human eye ki sensitivity reflect karte hain — Green sabse zyada!
    print(f"\n  1. RGB → Grayscale:")
    print(f"     Formula: 0.299*R + 0.587*G + 0.114*B")
    print(f"     PIL:     img.convert('L')")
    print(f"     PyTorch: transforms.Grayscale()")
    print(f"     TF:      tf.image.rgb_to_grayscale(img)")
    print(f"     OpenCV:  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")

    if HAS_CV2:
        # RGB → BGR (OpenCV format)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # 🔑 YAAD RAKHO: cvtColor = color convert
        print(f"\n  2. RGB ↔ BGR:")
        print(f"     ⚠️ WARNING: OpenCV HAMESHA BGR format mein kaam karta hai!")
        print(f"     cv2.imread() → BGR (NOT RGB!)")
        print(f"     Fix: img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)")
        print(f"     Ya:  img_rgb = img_bgr[:, :, ::-1]  # channels reverse")

        # RGB → HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)  # ⭐ IMPORTANT: HSV for color operations
        print(f"\n  3. RGB → HSV (Hue, Saturation, Value):")
        print(f"     H: Color type (0-180 in OpenCV, 0-360 normally)")
        print(f"     S: Kitna vibrant (0=gray, 255=pure color)")
        print(f"     V: Kitna bright (0=dark, 255=bright)")
        print(f"     USE CASE: Brightness fix → sirf V channel change karo!")
        print(f"     TF: tf.image.rgb_to_hsv() — but range [0,1] mein")

        # RGB → LAB
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)  # ⭐ IMPORTANT: LAB for perceptual
        print(f"\n  4. RGB → LAB (CIE Lab):")
        print(f"     L: Lightness (0=black, 100=white) — BRIGHTNESS ALAG!")
        print(f"     A: Green-Red axis")
        print(f"     B: Blue-Yellow axis")
        print(f"     USE CASE: Dehazing mein L channel pe CLAHE lagao!")
        print(f"     TF: tfio.experimental.color.rgb_to_lab()")
    else:
        print(f"\n  ℹ️ OpenCV not installed — HSV/LAB conversion skip")
        print(f"     Install: pip install opencv-python")

    # Channel ordering
    print(f"\n  5. Channel Ordering:")
    print(f"     ⚠️ BAHUT COMMON BUG — framework confusion!")
    print(f"     PyTorch: [C, H, W] — channels FIRST  (3, 256, 256)")
    print(f"     TF:      [H, W, C] — channels LAST   (256, 256, 3)")
    print(f"     NumPy:   [H, W, C] — same as TF")
    print(f"     OpenCV:  [H, W, C] — same as TF (but BGR!)")
    print(f"")
    print(f"     PyTorch mein convert:")
    print(f"       CHW → HWC: img.permute(1, 2, 0)   # display ke liye")
    print(f"       HWC → CHW: img.permute(2, 0, 1)   # model ke liye")
    print(f"       Ya: transforms.ToTensor() automatically HWC → CHW karta hai!")
    print(f"     TF: tf.transpose(img, [2, 0, 1]) for CHW")


# =============================================================================
# SECTION 5: HISTOGRAM EQUALIZATION & CLAHE
# =============================================================================

def demonstrate_histogram_equalization():
    """
    HISTOGRAM EQUALIZATION KYA HAI?
    Low contrast image ka contrast BADHAO by spreading pixel values.

    CLAHE (Contrast Limited Adaptive Histogram Equalization):
    Normal equalization se BETTER — locally apply hota hai, over-enhancement nahi!

    DEHAZING MEIN BAHUT USEFUL:
    Hazy images ka contrast LOW hota hai → CLAHE se fix karo → phir model ko do!
    """
    print("\n" + "=" * 60)
    print("SECTION 5: Histogram Equalization & CLAHE")
    print("=" * 60)

    # Demo
    np.random.seed(42)
    # Low contrast image (narrow histogram)
    low_contrast = np.random.normal(128, 15, (64, 64)).clip(0, 255).astype(np.uint8)
    # ⭐ IMPORTANT: std=15 = very narrow = low contrast (hazy image jaisi!)

    # Method 1: Simple histogram equalization
    if HAS_CV2:
        equalized = cv2.equalizeHist(low_contrast)  # ⭐ IMPORTANT: histogram equalization
        # TF: tfa.image.equalize(img) (tensorflow_addons mein)
        print(f"\n  1. Simple Histogram Equalization:")
        print(f"     Before: std={low_contrast.std():.1f} (narrow)")
        print(f"     After:  std={equalized.std():.1f} (spread out!)")
        print(f"     cv2.equalizeHist(gray_img)")
        print(f"     ⚠️ Problem: globally apply → over-enhancement ho sakta hai")

        # Method 2: CLAHE (BETTER!)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # ⭐ IMPORTANT: CLAHE object banao
        # 🔑 YAAD RAKHO: clipLimit = kitna enhance karna hai (2.0 good default)
        # 🔑 YAAD RAKHO: tileGridSize = kitni locally apply karna hai (8×8 tiles)
        clahe_result = clahe.apply(low_contrast)  # ⭐ IMPORTANT: CLAHE apply karo
        print(f"\n  2. CLAHE (Adaptive — BETTER!):")
        print(f"     After: std={clahe_result.std():.1f}")
        print(f"     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))")
        print(f"     result = clahe.apply(gray_img)")
        print(f"     ✅ Locally adaptive → no over-enhancement!")

        # Color image pe CLAHE
        print(f"\n  3. COLOR image pe CLAHE (RGB pe seedha mat lagao!):")
        print(f"     Step 1: RGB → LAB convert")
        print(f"     Step 2: SIRF L channel pe CLAHE lagao")
        print(f"     Step 3: LAB → RGB wapas convert")
        print(f"     ⚠️ WARNING: RGB pe seedha CLAHE → colors KHARAB ho jayenge!")

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        axes[0].imshow(low_contrast, cmap='gray'); axes[0].set_title('Low Contrast\n(Original)')
        axes[1].imshow(equalized, cmap='gray'); axes[1].set_title('Histogram Eq\n(Global)')
        axes[2].imshow(clahe_result, cmap='gray'); axes[2].set_title('CLAHE\n(Adaptive — BEST)')
        for ax in axes: ax.axis('off')
        plt.suptitle('Histogram Equalization Comparison', fontsize=12)
        plt.tight_layout()
        plt.savefig('practice/results/13_histogram_eq.png', dpi=100)
        plt.close()
        print(f"\n  📊 Saved: practice/results/13_histogram_eq.png")
    else:
        print(f"\n  ℹ️ OpenCV required for CLAHE demo")
        print(f"     pip install opencv-python")


# =============================================================================
# SECTION 6: DTYPE CONVERSION
# =============================================================================

def demonstrate_dtype_conversion():
    """
    Image data types — BAHUT COMMON BUG SOURCE!

    uint8:   0-255 integers (standard image format — PIL, OpenCV default)
    float32: 0.0-1.0 floats (model ke liye — PyTorch/TF default)
    float64: 0.0-1.0 doubles (unnecessary — memory waste)

    COMMON BUGS:
    1. uint8 image ko float mein divide nahi kiya → values 0-255 → model CONFUSED!
    2. float image ko plt.imshow() diya bina clipping → WARNING/wrong display
    3. uint8 mein math kiya → OVERFLOW! (200 + 100 = 44, NOT 300!)
    """
    print("\n" + "=" * 60)
    print("SECTION 6: Data Type Conversion")
    print("=" * 60)

    # Demo
    img_uint8 = np.array([[200, 100], [50, 250]], dtype=np.uint8)

    print(f"\n  Original (uint8): {img_uint8.flatten()}")
    print(f"  dtype: {img_uint8.dtype}, range: [{img_uint8.min()}, {img_uint8.max()}]")

    # uint8 → float32 (CORRECTLY)
    img_float = img_uint8.astype(np.float32) / 255.0  # 🔑 YAAD RAKHO: PEHLE float banao, PHIR divide!
    print(f"\n  Correct conversion:")
    print(f"  img.astype(np.float32) / 255.0")
    print(f"  Result: {img_float.flatten()}")

    # WRONG way (integer division)
    img_wrong = img_uint8 / 255  # ⚠️ WARNING: numpy mein ye FLOAT deta hai (OK), par PyTorch mein nahi!
    print(f"\n  ⚠️ Potential issue:")
    print(f"  img / 255 → numpy mein OK, par explicit .astype(float32) BEST PRACTICE hai")

    # uint8 overflow demo
    a = np.uint8(200)           # ℹ️ COMMON: uint8 value
    b = np.uint8(100)
    print(f"\n  ⚠️ uint8 OVERFLOW BUG:")
    print(f"  200 + 100 = {np.uint8(200) + np.uint8(100)} (expected 300, got {(200+100) % 256}!)")
    print(f"  KYUN? uint8 max = 255, 300 wrap around → 300 - 256 = 44")
    print(f"  FIX: pehle float mein convert karo, phir math karo!")

    # float32 → uint8 (display ke liye)
    back_to_uint8 = (img_float * 255).clip(0, 255).astype(np.uint8)
    # 🔑 YAAD RAKHO: .clip(0, 255) ZAROORI hai — warna overflow!
    # TF: tf.cast(img * 255, tf.uint8)
    print(f"\n  float→uint8 (display ke liye):")
    print(f"  (img * 255).clip(0, 255).astype(np.uint8)")


# =============================================================================
# SECTION 7: COMPLETE PREPROCESSING PIPELINE
# =============================================================================

def demonstrate_preprocessing_pipeline():
    """
    Ek COMPLETE preprocessing pipeline jaise real project mein hota hai.

    TYPICAL PIPELINE:
    Raw Image → Load → Verify → Resize → Color Fix → Normalize → Tensor → Model

    PyTorch:  transforms.Compose([...]) — chain of transforms
    TF:       tf.keras.Sequential([...]) ya tf.data.Dataset.map()
    """
    print("\n" + "=" * 60)
    print("SECTION 7: Complete Preprocessing Pipeline")
    print("=" * 60)

    print(f"""
  PYTORCH PIPELINE:
  ─────────────────
  transform_train = transforms.Compose([
      transforms.Resize((256, 256)),           # ℹ️ Fixed size
      transforms.RandomHorizontalFlip(p=0.5),  # ⭐ Augmentation (training only!)
      transforms.ColorJitter(0.2, 0.2, 0.2),   # ⭐ Color augmentation
      transforms.ToTensor(),                    # 🔑 HWC→CHW, uint8→float32, /255
      transforms.Normalize(                     # 🔑 Dataset-specific normalize
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225]
      )
  ])

  transform_test = transforms.Compose([
      transforms.Resize((256, 256)),            # ℹ️ Same resize
      transforms.ToTensor(),                    # 🔑 ToTensor
      transforms.Normalize(                     # 🔑 SAME normalize as train!
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225]
      )
      # ⚠️ WARNING: Test mein augmentation NAHI! Consistent results chahiye!
  ])

  TENSORFLOW PIPELINE:
  ─────────────────────
  def preprocess_train(image, label):
      image = tf.image.resize(image, [256, 256])
      image = tf.image.random_flip_left_right(image)
      image = tf.image.random_brightness(image, 0.2)
      image = tf.cast(image, tf.float32) / 255.0
      image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
      return image, label

  ds = tf.data.Dataset.from_tensor_slices((images, labels))
  ds = ds.map(preprocess_train).batch(32).prefetch(tf.data.AUTOTUNE)
    """)

    print(f"  🔑 KEY RULES:")
    print(f"  1. Augmentation SIRF training mein — test/val mein NAHI!")
    print(f"  2. Normalize train aur test DONO mein SAME values se!")
    print(f"  3. ToTensor() AUTOMATICALLY /255 karta hai — dobara mat karo!")
    print(f"  4. Resize PEHLE karo, augmentation BAAD mein")
    print(f"  5. Normalize LAST step hona chahiye (transforms ke end mein)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":       # 🔑 YAAD RAKHO: main guard
    print("=" * 60)
    print("PRACTICE 13: Data Cleaning & Preprocessing")
    print("=" * 60)

    DATA_DIR = '../RESIDE-6K'

    clean_corrupt_images(DATA_DIR, dry_run=True)   # Section 1
    demonstrate_normalization()                     # Section 2
    demonstrate_resizing()                          # Section 3
    demonstrate_color_spaces()                      # Section 4
    demonstrate_histogram_equalization()            # Section 5
    demonstrate_dtype_conversion()                  # Section 6
    demonstrate_preprocessing_pipeline()            # Section 7

    print(f"\n{'='*60}")
    print(f"PRACTICE 13 COMPLETE!")
    print(f"{'='*60}")
    print(f"\nKYA SEEKHA:")
    print(f"  1. Corrupt detection — PIL verify() + load()")
    print(f"  2. Normalization — MinMax [0,1], [-1,1], Z-score, ImageNet")
    print(f"  3. Resizing — Resize, CenterCrop, Pad, RandomResizedCrop")
    print(f"  4. Color spaces — RGB, BGR, HSV, LAB, Grayscale")
    print(f"  5. CLAHE — adaptive contrast enhancement")
    print(f"  6. Dtype — uint8 vs float32 (overflow bug!)")
    print(f"  7. Pipeline — Compose([...]) complete chain")
    print(f"\nNEXT: python practice/14_augmentation_masterclass.py")

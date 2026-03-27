"""
=============================================================================
PRACTICE 12: DATA EDA & ANALYSIS — Data Ko Pehle SAMJHO, Phir Model Banao!
=============================================================================

EDA KYA HAI? (Exploratory Data Analysis)
==========================================
Model banana se PEHLE tumhe apna data SAMAJHNA padta hai:
  - Kitni images hain? Sizes kya hain?
  - Pixel values ka distribution kaisa hai?
  - Koi corrupt/broken images toh nahi?
  - Class balance theek hai ya nahi?
  - Paired data (hazy-clean) match karta hai ya nahi?

KYUN ZAROORI HAI?
  1. GARBAGE IN = GARBAGE OUT — bekar data pe achha model NAHI ban sakta
  2. Bugs EARLY pakdo — training ke 10 ghante baad pata chale data galat tha? 😭
  3. Preprocessing decisions INFORM karta hai — normalize kaise karein?
  4. Research papers mein dataset analysis section HAMESHA hota hai

INDUSTRY MEIN:
  Data Scientist ka 80% time DATA pe lagta hai, 20% model pe!
  Jo log data SKIP karke seedha model banate hain — unke results HAMESHA kharab aate hain.

IS FILE MEIN SEEKHOGE:
  ✅ Image statistics (mean, std, min, max)
  ✅ Pixel distribution histograms
  ✅ Image size/resolution analysis
  ✅ Corrupt/broken image detection
  ✅ Class balance check
  ✅ Paired data validation
  ✅ Visual grid — random samples
  ✅ Outlier detection (unusual images)
  ✅ Summary report generation

=============================================================================
"""

import os                        # 🔑 YAAD RAKHO: file/folder operations
import numpy as np               # 🔑 YAAD RAKHO: numerical computing — arrays, stats
# TF: numpy TF mein bhi use hota hai, same import
from PIL import Image            # 🔑 YAAD RAKHO: PIL = Python Imaging Library — images load/save
# TF: tf.io.read_file() + tf.image.decode_image()
import matplotlib                # ℹ️ COMMON: plotting setup
matplotlib.use('Agg')            # ℹ️ COMMON: no GUI mode
import matplotlib.pyplot as plt  # 🔑 YAAD RAKHO: plotting — graphs, images dikhana
from collections import Counter  # ℹ️ COMMON: counting items (class distribution ke liye)
import hashlib                   # ℹ️ COMMON: file hashing (duplicates detect)
import random                    # ℹ️ COMMON: random sampling
import time                      # ℹ️ COMMON: timing operations


# =============================================================================
# SECTION 1: BASIC IMAGE STATISTICS
# =============================================================================

def compute_image_stats(image_dir, max_images=500):
    """
    Poori directory ki images ka statistical analysis karo.
    Mean, std, min, max — ye sab NORMALIZATION ke liye chahiye!

    KYUN ZAROORI:
    Agar tumhe pata hai ki dataset ka mean=[0.45, 0.42, 0.40] hai,
    toh tum SAHI normalization kar sakte ho → model BETTER seekhega!
    ImageNet ka mean=[0.485, 0.456, 0.406] — issi tarah apne dataset ka nikalo!
    """
    print("=" * 60)
    print("SECTION 1: Image Statistics")
    print("=" * 60)

    all_means = []               # ℹ️ COMMON: har image ka mean store
    all_stds = []                # ℹ️ COMMON: har image ka std store
    all_mins = []                # ℹ️ COMMON: minimum values
    all_maxs = []                # ℹ️ COMMON: maximum values
    all_heights = []             # ℹ️ COMMON: heights track
    all_widths = []              # ℹ️ COMMON: widths track
    file_sizes = []              # ℹ️ COMMON: file sizes (KB mein)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}  # ⭐ IMPORTANT: supported image formats

    image_files = []             # ℹ️ COMMON: list of image paths
    for root, dirs, files in os.walk(image_dir):  # 🔑 YAAD RAKHO: os.walk = recursively SABHI subfolders mein jao
        # TF: tf.io.gfile.walk() ya tf.io.gfile.glob()
        for f in files:
            ext = os.path.splitext(f)[1].lower()  # 🔑 YAAD RAKHO: splitext = filename + extension alag karo
            if ext in valid_extensions:
                image_files.append(os.path.join(root, f))  # ℹ️ COMMON: full path banao

    if not image_files:
        print(f"  ⚠️ Koi image nahi mili: {image_dir}")
        print(f"  Dummy data se demo chalate hain...")
        return _demo_stats()

    # Subset lo (bahut badi directory ke liye)
    if len(image_files) > max_images:
        image_files = random.sample(image_files, max_images)  # ⭐ IMPORTANT: random.sample = bina replacement ke random pick
        # ⚠️ WARNING: sample() list se UNIQUE items pick karta hai — random.choices() DUPLICATES de sakta hai!

    print(f"\n  📁 Directory: {image_dir}")
    print(f"  📊 Analyzing {len(image_files)} images...\n")

    corrupt_count = 0            # ℹ️ COMMON: corrupt images counter
    channels_count = Counter()   # ℹ️ COMMON: channel distribution

    for i, img_path in enumerate(image_files):  # ℹ️ COMMON: enumerate = index + value
        try:
            img = Image.open(img_path)   # 🔑 YAAD RAKHO: PIL se image open karo
            # TF: tf.io.read_file(path) → tf.image.decode_image(raw)
            img.verify()                 # ⭐ IMPORTANT: .verify() = check ki image VALID hai ya corrupt!
            # ⚠️ WARNING: verify() ke baad image CLOSE ho jaati hai — dobara open karna padega!

            img = Image.open(img_path)   # ℹ️ COMMON: dobara open (verify ke baad)
            img_array = np.array(img)    # 🔑 YAAD RAKHO: PIL Image → numpy array
            # TF: tf.image.decode_image() already tensor deta hai

            # Size info
            w, h = img.size              # 🔑 YAAD RAKHO: PIL mein (width, height) — numpy mein (height, width) — ULTA hai!
            # ⚠️ WARNING: PIL = (W, H) par numpy/torch = (H, W) — ye BAHUT COMMON bug hai!
            all_heights.append(h)
            all_widths.append(w)

            # File size
            file_size = os.path.getsize(img_path) / 1024  # ℹ️ COMMON: bytes → KB
            file_sizes.append(file_size)

            # Channel info
            if len(img_array.shape) == 2:      # 🔑 YAAD RAKHO: shape length 2 = grayscale (H, W)
                channels_count['Grayscale'] += 1
            elif img_array.shape[2] == 3:      # ℹ️ COMMON: shape[2]=3 = RGB
                channels_count['RGB'] += 1
            elif img_array.shape[2] == 4:      # ℹ️ COMMON: shape[2]=4 = RGBA (transparency)
                channels_count['RGBA'] += 1

            # Pixel stats (0-255 range mein)
            all_means.append(img_array.mean())     # 🔑 YAAD RAKHO: .mean() = average pixel value
            all_stds.append(img_array.std())       # 🔑 YAAD RAKHO: .std() = standard deviation — spread
            all_mins.append(img_array.min())       # ℹ️ COMMON: darkest pixel
            all_maxs.append(img_array.max())       # ℹ️ COMMON: brightest pixel

        except Exception as e:                     # ⚠️ WARNING: corrupt images se error aayega — handle karo!
            corrupt_count += 1
            if corrupt_count <= 3:                 # ℹ️ COMMON: pehle 3 errors dikhao
                print(f"  ❌ Corrupt: {os.path.basename(img_path)} — {e}")

    # --- Results Print ---
    print(f"\n  {'='*50}")
    print(f"  📊 STATISTICS RESULTS:")
    print(f"  {'='*50}")
    print(f"  Total images:     {len(image_files)}")
    print(f"  Corrupt images:   {corrupt_count}")
    print(f"  Valid images:     {len(image_files) - corrupt_count}")
    print(f"")
    print(f"  Channel types:    {dict(channels_count)}")
    print(f"")
    print(f"  📐 Sizes:")
    print(f"     Heights:  min={min(all_heights)}, max={max(all_heights)}, avg={np.mean(all_heights):.0f}")
    print(f"     Widths:   min={min(all_widths)}, max={max(all_widths)}, avg={np.mean(all_widths):.0f}")
    print(f"     Unique:   {len(set(zip(all_heights, all_widths)))} different sizes")
    print(f"")
    print(f"  🎨 Pixel Values (0-255):")
    print(f"     Mean:  {np.mean(all_means):.2f}")
    print(f"     Std:   {np.mean(all_stds):.2f}")
    print(f"     Min:   {np.mean(all_mins):.2f}")
    print(f"     Max:   {np.mean(all_maxs):.2f}")
    print(f"")
    print(f"  💾 File Sizes:")
    print(f"     Min: {min(file_sizes):.1f} KB")
    print(f"     Max: {max(file_sizes):.1f} KB")
    print(f"     Avg: {np.mean(file_sizes):.1f} KB")

    # Normalized stats (0-1 range — model ke liye)
    norm_mean = np.mean(all_means) / 255.0        # ⭐ IMPORTANT: 0-255 → 0-1 normalize
    norm_std = np.mean(all_stds) / 255.0
    print(f"\n  🔢 NORMALIZED (0-1 range — model ke liye use karo!):")
    print(f"     Mean: {norm_mean:.4f}")
    print(f"     Std:  {norm_std:.4f}")
    print(f"     → transforms.Normalize([{norm_mean:.3f}], [{norm_std:.3f}])")

    return {
        'means': all_means, 'stds': all_stds,
        'heights': all_heights, 'widths': all_widths,
        'file_sizes': file_sizes, 'corrupt': corrupt_count
    }


def _demo_stats():
    """Demo data se stats dikhao jab real images nahi hain"""
    print("\n  (Demo mode — MNIST-like random data)")
    np.random.seed(42)           # 🔑 YAAD RAKHO: seed = reproducible random numbers
    all_means = np.random.normal(0.13, 0.05, 100).tolist()
    all_stds = np.random.normal(0.30, 0.05, 100).tolist()
    print(f"  Mean of means: {np.mean(all_means):.4f}")
    print(f"  Mean of stds:  {np.mean(all_stds):.4f}")
    return {'means': all_means, 'stds': all_stds, 'heights': [28]*100,
            'widths': [28]*100, 'file_sizes': [1.2]*100, 'corrupt': 0}


# =============================================================================
# SECTION 2: PIXEL DISTRIBUTION HISTOGRAM
# =============================================================================

def plot_pixel_distribution(image_dir=None, max_images=100):
    """
    Pixel values ka histogram plot karo — distribution samjho!

    KYUN IMPORTANT:
    - Agar distribution SKEWED hai → normalization strategy adjust karo
    - Agar values ek range mein CONCENTRATED hain → contrast low hai → CLAHE use karo
    - Hazy images ka histogram NARROW hota hai (sab gray-ish) — ye bahut telling hai!
    """
    print("\n" + "=" * 60)
    print("SECTION 2: Pixel Distribution Histogram")
    print("=" * 60)

    # Demo data (random images banao)
    np.random.seed(42)

    # 3 types ke distribution dikhate hain
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # ℹ️ COMMON: subplots for comparison

    # 1. Normal distribution (achhi quality image)
    normal_pixels = np.random.normal(128, 60, 10000).clip(0, 255)  # ℹ️ COMMON: .clip() = values ko range mein rakho
    axes[0].hist(normal_pixels, bins=50, color='green', alpha=0.7)  # ℹ️ COMMON: histogram plot
    axes[0].set_title('Normal Image\n(Good Distribution)')
    axes[0].set_xlabel('Pixel Value'); axes[0].set_ylabel('Count')
    axes[0].axvline(x=128, color='red', linestyle='--', label='Mean')  # ℹ️ COMMON: vertical line at mean

    # 2. Hazy image (narrow, shifted right — sab grayish)
    hazy_pixels = np.random.normal(180, 25, 10000).clip(0, 255)  # ⭐ IMPORTANT: hazy = narrow distribution, high mean
    axes[1].hist(hazy_pixels, bins=50, color='gray', alpha=0.7)
    axes[1].set_title('Hazy Image\n(Narrow, High Mean)')
    axes[1].set_xlabel('Pixel Value')

    # 3. Dark/underexposed image
    dark_pixels = np.random.normal(40, 20, 10000).clip(0, 255)  # ⭐ IMPORTANT: dark = low mean, narrow
    axes[2].hist(dark_pixels, bins=50, color='navy', alpha=0.7)
    axes[2].set_title('Dark Image\n(Low Mean, Narrow)')
    axes[2].set_xlabel('Pixel Value')

    plt.suptitle('Pixel Distribution Analysis — Different Image Types', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs('practice/results', exist_ok=True)  # ℹ️ COMMON: output folder
    plt.savefig('practice/results/12_pixel_distribution.png', dpi=100)  # ℹ️ COMMON: save plot
    plt.close()
    print(f"\n  📊 Saved: practice/results/12_pixel_distribution.png")
    print(f"  → Normal = wide bell curve, Hazy = narrow high, Dark = narrow low")


# =============================================================================
# SECTION 3: IMAGE SIZE ANALYSIS
# =============================================================================

def analyze_image_sizes(stats=None):
    """
    Dataset mein image sizes ka distribution dikhao.

    KYUN ZAROORI:
    - Neural networks ko FIXED SIZE input chahiye (jaise 128×128 ya 256×256)
    - Agar images ALAG alag sizes ki hain → resize karna padega
    - Resize strategy KAISE choose karein → ye analysis batata hai!

    SIZE MISMATCH KE SOLUTIONS:
    1. Resize (sabko same size — simple but distortion)
    2. Center Crop (center se crop — info loss hoti hai edges se)
    3. Padding (border pe zeros add — extra computation)
    4. Aspect-ratio preserving resize + pad (BEST for most cases)
    """
    print("\n" + "=" * 60)
    print("SECTION 3: Image Size Analysis")
    print("=" * 60)

    # Demo data
    np.random.seed(42)
    heights = np.random.choice([480, 640, 720, 1080, 256, 512], 200).tolist()
    widths = np.random.choice([640, 480, 1280, 1920, 256, 512], 200).tolist()

    unique_sizes = set(zip(heights, widths))  # ⭐ IMPORTANT: set() = unique values only
    # TF: same logic, numpy/python operations same hain

    print(f"\n  Total images: {len(heights)}")
    print(f"  Unique sizes: {len(unique_sizes)}")
    print(f"  Height range: {min(heights)} to {max(heights)}")
    print(f"  Width range:  {min(widths)} to {max(widths)}")

    # Aspect ratios
    ratios = [w/h for w, h in zip(widths, heights)]  # ⭐ IMPORTANT: aspect ratio = width/height
    print(f"  Aspect ratios: {min(ratios):.2f} to {max(ratios):.2f}")
    print(f"  Most common ratio: {Counter([round(r, 1) for r in ratios]).most_common(3)}")

    if len(unique_sizes) == 1:
        print(f"\n  ✅ ACHHA: Saari images SAME size hain! Resize ki zaroorat nahi.")
    else:
        print(f"\n  ⚠️  Images ALAG sizes ki hain — resize karna padega!")
        print(f"  Recommended: transforms.Resize((256, 256)) ya RandomCrop(128)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.scatter(widths, heights, alpha=0.5, s=10)  # ℹ️ COMMON: scatter plot
    ax1.set_xlabel('Width'); ax1.set_ylabel('Height')
    ax1.set_title('Image Dimensions Scatter')
    ax2.hist(ratios, bins=20, color='steelblue', alpha=0.7)  # ℹ️ COMMON: histogram
    ax2.set_xlabel('Aspect Ratio (W/H)'); ax2.set_title('Aspect Ratio Distribution')
    plt.tight_layout()
    plt.savefig('practice/results/12_size_analysis.png', dpi=100)
    plt.close()
    print(f"  📊 Saved: practice/results/12_size_analysis.png")


# =============================================================================
# SECTION 4: CORRUPT IMAGE DETECTION
# =============================================================================

def detect_corrupt_images(image_dir, fix=False):
    """
    Corrupt/broken images dhundho aur optionally hatao.

    CORRUPT IMAGES KYA HAIN:
    - Incomplete download (file truncated — half likhi)
    - Wrong extension (txt file ko .jpg naam de diya)
    - Damaged file (disk error se kharab)

    AGAR CORRUPT IMAGE DATALOADER MEIN AAYE:
    → Training CRASH ho jaayegi bina clear error ke!
    → Bahut mushkil bug — dhundhne mein GHANTE lag sakte hain

    SOLUTION: PEHLE SE CHECK KARO!
    """
    print("\n" + "=" * 60)
    print("SECTION 4: Corrupt Image Detection")
    print("=" * 60)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    corrupt_files = []           # ℹ️ COMMON: corrupt files track
    valid_count = 0              # ℹ️ COMMON: valid count

    if not os.path.exists(image_dir):
        print(f"  Directory nahi mili: {image_dir}")
        print(f"  Demo dikhate hain...")
        _demo_corrupt_detection()
        return []

    for root, dirs, files in os.walk(image_dir):  # 🔑 YAAD RAKHO: os.walk = recursive traverse
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in valid_extensions:
                continue

            filepath = os.path.join(root, f)
            try:
                img = Image.open(filepath)   # 🔑 YAAD RAKHO: open
                img.verify()                 # ⭐ IMPORTANT: verify() = integrity check
                valid_count += 1
            except Exception as e:
                corrupt_files.append((filepath, str(e)))
                if len(corrupt_files) <= 5:
                    print(f"  ❌ Corrupt: {f} — {e}")

    print(f"\n  ✅ Valid:   {valid_count}")
    print(f"  ❌ Corrupt: {len(corrupt_files)}")

    if corrupt_files and fix:
        print(f"\n  🗑️ Corrupt files remove kar rahe hain...")
        for path, _ in corrupt_files:
            os.remove(path)                  # ⚠️ WARNING: PERMANENTLY delete! Backup pehle lo!
        print(f"  Done — {len(corrupt_files)} files removed.")

    return corrupt_files


def _demo_corrupt_detection():
    """Demo: corrupt detection concept dikhao"""
    print("\n  Corrupt Detection Methods:")
    print("  " + "-" * 40)
    print("  1. PIL verify():  img.verify()  → file integrity check")
    print("  2. PIL load():    img.load()    → actually pixel data load (slower but thorough)")
    print("  3. File size:     os.path.getsize() < 1000 → suspiciously small")
    print("  4. Magic bytes:   file ke pehle bytes check (JPEG = FF D8 FF)")
    print()
    print("  # Code example:")
    print("  try:")
    print("      img = Image.open(path)")
    print("      img.verify()    # 🔑 Quick integrity check")
    print("  except:")
    print("      print('CORRUPT!')")


# =============================================================================
# SECTION 5: DUPLICATE DETECTION
# =============================================================================

def detect_duplicates(image_dir, max_images=1000):
    """
    Duplicate images dhundho using file hashing.

    KYUN ZAROORI:
    - Duplicates se model OVERFIT karta hai (same image baar baar dekhega)
    - Train/test split mein duplicate chala gaya → DATA LEAKAGE! 🚨
      (Model ne test image pehle hi dekhi hai → fake accuracy!)

    HASHING KYA HAI:
    Har file ka ek UNIQUE fingerprint (hash) banao.
    Same content = same hash → DUPLICATE!
    MD5 hash: "hello" → "5d41402abc4b2a76b9719d911017c592"
    """
    print("\n" + "=" * 60)
    print("SECTION 5: Duplicate Detection (Hashing)")
    print("=" * 60)

    hash_dict = {}               # ℹ️ COMMON: hash → filepath mapping
    duplicates = []              # ℹ️ COMMON: duplicate pairs
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    if not os.path.exists(image_dir):
        print("  Demo: Duplicate detection concept")
        print("  " + "-" * 40)
        print("  # Hash-based duplicate detection:")
        print("  file_hash = hashlib.md5(file_bytes).hexdigest()")
        print("  # Same hash = same content = DUPLICATE!")
        print()
        # Demo with dummy data
        demo_hashes = ['abc123', 'def456', 'abc123', 'ghi789', 'def456']
        seen = {}
        for i, h in enumerate(demo_hashes):
            if h in seen:
                print(f"  ❌ DUPLICATE: file_{i}.jpg == file_{seen[h]}.jpg (hash={h})")
            else:
                seen[h] = i
        return []

    count = 0
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() not in valid_extensions:
                continue
            filepath = os.path.join(root, f)

            # File ka hash banao
            with open(filepath, 'rb') as file:      # 🔑 YAAD RAKHO: 'rb' = read binary mode
                file_hash = hashlib.md5(file.read()).hexdigest()  # ⭐ IMPORTANT: MD5 hash = unique fingerprint
                # TF: same — hashing Python operation hai, framework independent

            if file_hash in hash_dict:
                duplicates.append((filepath, hash_dict[file_hash]))
                if len(duplicates) <= 3:
                    print(f"  ❌ DUPLICATE: {f} == {os.path.basename(hash_dict[file_hash])}")
            else:
                hash_dict[file_hash] = filepath

            count += 1
            if count >= max_images:
                break

    print(f"\n  Scanned: {count} images")
    print(f"  Duplicates found: {len(duplicates)}")
    if not duplicates:
        print(f"  ✅ No duplicates — CLEAN dataset!")

    return duplicates


# =============================================================================
# SECTION 6: CLASS BALANCE ANALYSIS (Classification ke liye)
# =============================================================================

def analyze_class_balance(data_dir=None):
    """
    Classification dataset mein class distribution check karo.

    CLASS IMBALANCE KYA HAI:
    Agar 1000 images hain: 900 cats + 100 dogs
    → Model SIRF "cat" predict karega → 90% accuracy → par USELESS hai!

    SOLUTIONS:
    1. Oversampling:  minority class ki images DUPLICATE karo
    2. Undersampling: majority class ki images HATAO
    3. WeightedRandomSampler:   minority class ko ZYADA bar dikhao (best!)
    4. Class weights in loss:   minority class ke loss ko ZYADA multiply karo
    5. Data augmentation: minority class pe EXTRA augmentation karo
    """
    print("\n" + "=" * 60)
    print("SECTION 6: Class Balance Analysis")
    print("=" * 60)

    # Demo data — typical imbalanced dataset
    classes = {
        'Cat': 900, 'Dog': 800, 'Bird': 150,
        'Fish': 50, 'Snake': 30, 'Frog': 20
    }

    total = sum(classes.values())  # ℹ️ COMMON: total count
    print(f"\n  Total images: {total}")
    print(f"  Classes: {len(classes)}")
    print(f"\n  {'Class':<10} {'Count':>8} {'Percentage':>12} {'Status':>10}")
    print(f"  {'-'*45}")

    balanced_threshold = total / len(classes) * 0.5  # ⭐ IMPORTANT: 50% of average = threshold

    for cls, count in sorted(classes.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        status = "✅ OK" if count > balanced_threshold else "⚠️ LOW"
        print(f"  {cls:<10} {count:>8} {pct:>10.1f}% {status:>10}")

    # Imbalance ratio
    max_count = max(classes.values())   # ℹ️ COMMON: sabse badi class
    min_count = min(classes.values())   # ℹ️ COMMON: sabse choti class
    ratio = max_count / min_count       # ⭐ IMPORTANT: imbalance ratio
    print(f"\n  Imbalance ratio: {ratio:.1f}x (max/min)")
    if ratio > 10:
        print(f"  ⚠️ BAHUT IMBALANCED! WeightedRandomSampler ya class weights USE KARO!")
    elif ratio > 3:
        print(f"  ⚠️ Moderately imbalanced — augmentation + sampling strategy sochlo")
    else:
        print(f"  ✅ Reasonably balanced!")

    # Fix: class weights calculate (loss ke liye)
    print(f"\n  💡 CLASS WEIGHTS (loss function ke liye):")
    for cls, count in classes.items():
        weight = total / (len(classes) * count)  # ⭐ IMPORTANT: inverse frequency weight
        # TF: tf.keras.utils.compute_class_weight()
        print(f"     {cls}: {weight:.2f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    ax1.bar(classes.keys(), classes.values(), color=colors)
    ax1.set_title('Class Distribution'); ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    ax2.pie(classes.values(), labels=classes.keys(), autopct='%1.1f%%', colors=colors)
    ax2.set_title('Class Proportions')
    plt.tight_layout()
    plt.savefig('practice/results/12_class_balance.png', dpi=100)
    plt.close()
    print(f"\n  📊 Saved: practice/results/12_class_balance.png")


# =============================================================================
# SECTION 7: PAIRED DATA VALIDATION (Image-to-Image tasks ke liye)
# =============================================================================

def validate_paired_data(input_dir=None, target_dir=None):
    """
    Input-Target pairs validate karo (dehazing, super-resolution, etc.)

    PAIRED DATA: Har hazy image ke saath ek CLEAN image honi chahiye!
    Agar pair MISMATCH ho gaya → model GALAT seekhega!

    CHECK KARO:
    1. File count SAME hai dono folders mein?
    2. Filenames MATCH karte hain?
    3. Image sizes SAME hain?
    4. Koi missing pair toh nahi?
    """
    print("\n" + "=" * 60)
    print("SECTION 7: Paired Data Validation")
    print("=" * 60)

    # Demo
    print("\n  Paired Data Validation Checks:")
    print("  " + "-" * 40)

    # Simulate paired data
    input_files = {'img_001.jpg', 'img_002.jpg', 'img_003.jpg', 'img_004.jpg', 'img_005.jpg'}
    target_files = {'img_001.jpg', 'img_002.jpg', 'img_003.jpg', 'img_006.jpg'}

    # Check 1: Count match
    print(f"\n  Input images:  {len(input_files)}")
    print(f"  Target images: {len(target_files)}")
    if len(input_files) == len(target_files):
        print(f"  ✅ Counts MATCH!")
    else:
        print(f"  ❌ Count MISMATCH! ({abs(len(input_files) - len(target_files))} extra/missing)")

    # Check 2: Filename match
    only_in_input = input_files - target_files    # ⭐ IMPORTANT: set difference — A mein hai par B mein nahi
    only_in_target = target_files - input_files
    matched = input_files & target_files          # ⭐ IMPORTANT: set intersection — dono mein hai

    print(f"\n  Matched pairs:     {len(matched)}")
    if only_in_input:
        print(f"  ❌ Input only:     {only_in_input}  — target MISSING!")
    if only_in_target:
        print(f"  ❌ Target only:    {only_in_target} — input MISSING!")
    if not only_in_input and not only_in_target:
        print(f"  ✅ All pairs MATCHED!")

    print(f"\n  💡 TIPS:")
    print(f"     - Hamesha sorted() filenames use karo pairing ke liye")
    print(f"     - Filename convention: input_001.png → target_001.png")
    print(f"     - Size check: input aur target ki dimensions SAME honi chahiye!")


# =============================================================================
# SECTION 8: VISUAL SAMPLE GRID
# =============================================================================

def show_sample_grid(image_dir=None, rows=3, cols=5):
    """
    Random images ka grid dikhao — visual inspection!

    KYUN ZAROORI:
    - Ek nazar mein data ka FEEL milta hai
    - Obvious problems instantly dikh jaate hain:
      → Blurry images? Wrong images? Labeling errors?
    - Research papers mein HAMESHA sample images dikhate hain
    """
    print("\n" + "=" * 60)
    print("SECTION 8: Visual Sample Grid")
    print("=" * 60)

    # Demo — random colored patches banao
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle('Random Sample Images from Dataset', fontsize=14, fontweight='bold')

    np.random.seed(42)
    for i in range(rows):
        for j in range(cols):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)  # ℹ️ COMMON: random dummy image
            axes[i, j].imshow(img)       # 🔑 YAAD RAKHO: imshow = image display
            # TF: tf.keras.utils.plot_model() for model, plt.imshow() for images (same)
            axes[i, j].axis('off')       # ℹ️ COMMON: axis labels hatao
            axes[i, j].set_title(f'#{i*cols+j+1}', fontsize=8)

    plt.tight_layout()
    plt.savefig('practice/results/12_sample_grid.png', dpi=100)
    plt.close()
    print(f"\n  📊 Saved: practice/results/12_sample_grid.png")
    print(f"  → Real data mein: actual images load karke grid banao")
    print(f"  → Visual inspection se IMMEDIATELY problems dikh jaate hain!")


# =============================================================================
# SECTION 9: OUTLIER DETECTION
# =============================================================================

def detect_outliers():
    """
    Unusual images dhundho — brightness, size, ya content mein alag.

    OUTLIER TYPES:
    1. TOO BRIGHT:  mean > 240 (almost white)
    2. TOO DARK:    mean < 15 (almost black)
    3. TOO SMALL:   file size < 1 KB (probably corrupt)
    4. TOO LARGE:   file size > expected (wrong image?)
    5. WRONG SIZE:  dimensions bahut alag (landscape mein portrait?)
    """
    print("\n" + "=" * 60)
    print("SECTION 9: Outlier Detection")
    print("=" * 60)

    # Demo data
    np.random.seed(42)
    means = np.random.normal(128, 30, 100)
    # Add some outliers
    means = np.append(means, [5, 8, 250, 252, 3])  # ℹ️ COMMON: np.append to add elements

    # Z-score method
    z_scores = (means - np.mean(means)) / np.std(means)  # ⭐ IMPORTANT: z-score = (x - μ) / σ
    # 🔑 YAAD RAKHO: |z| > 2 = outlier (2 standard deviations away from mean)

    outliers = np.where(np.abs(z_scores) > 2)[0]  # ⭐ IMPORTANT: np.where = condition ke indices
    # TF: tf.where() — same concept

    print(f"\n  Total images: {len(means)}")
    print(f"  Mean brightness: {np.mean(means):.1f}")
    print(f"  Std brightness:  {np.std(means):.1f}")
    print(f"  Outliers found:  {len(outliers)} (|z-score| > 2)")

    for idx in outliers:
        status = "TOO BRIGHT" if means[idx] > 200 else "TOO DARK"
        print(f"  ⚠️ Image #{idx}: mean={means[idx]:.0f}, z={z_scores[idx]:.1f} — {status}")

    print(f"\n  💡 OUTLIER HANDLING OPTIONS:")
    print(f"     1. REMOVE: completely hatao (agar genuinely wrong hai)")
    print(f"     2. CLIP:   extreme values ko clip karo")
    print(f"     3. INVESTIGATE: manually check karo (maybe valid hai!)")
    print(f"     4. SEPARATE: alag test set mein daalo (robustness check)")


# =============================================================================
# SECTION 10: EDA SUMMARY REPORT
# =============================================================================

def generate_eda_report():
    """
    Complete EDA summary report print karo — ye research paper mein jayega!

    PAPER MEIN DATASET SECTION:
    "We used the RESIDE-6K dataset containing 6000 paired hazy-clean images.
    The training set consists of 5100 images and test set of 1000 images.
    Images were resized to 256×256. Mean pixel values are [0.52, 0.49, 0.47]
    with standard deviation [0.23, 0.21, 0.20]."

    YE SECTION BANANE KE LIYE EDA ZAROORI HAI!
    """
    print("\n" + "=" * 60)
    print("SECTION 10: EDA Summary Report")
    print("=" * 60)

    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║              DATASET EDA REPORT                     ║
  ╠══════════════════════════════════════════════════════╣
  ║  Dataset:     RESIDE-6K (example)                   ║
  ║  Total:       6,000 paired images                   ║
  ║  Train:       5,100 images                          ║
  ║  Test:        1,000 images                          ║
  ║  Val:           900 images (from train)             ║
  ║                                                     ║
  ║  Image Size:  256×256 (resized)                     ║
  ║  Channels:    RGB (3)                               ║
  ║  Format:      PNG                                   ║
  ║                                                     ║
  ║  Pixel Stats (0-1):                                 ║
  ║    Mean: [0.520, 0.490, 0.470]                      ║
  ║    Std:  [0.230, 0.210, 0.200]                      ║
  ║                                                     ║
  ║  Corrupt:     0 images (all clean!)                 ║
  ║  Duplicates:  0 found                               ║
  ║  Outliers:    3 detected (too dark)                 ║
  ╚══════════════════════════════════════════════════════╝
    """)

    print("  📝 EDA CHECKLIST (Har dataset ke liye karo!):")
    print("  ─────────────────────────────────────────────")
    print("  [✅] Image statistics (mean, std, min, max)")
    print("  [✅] Pixel distribution histogram")
    print("  [✅] Size/resolution analysis")
    print("  [✅] Corrupt image detection")
    print("  [✅] Duplicate detection")
    print("  [✅] Class balance (if classification)")
    print("  [✅] Paired data validation (if image-to-image)")
    print("  [✅] Visual sample grid")
    print("  [✅] Outlier detection")
    print("  [✅] Summary report")


# =============================================================================
# MAIN — Sab sections run karo!
# =============================================================================

if __name__ == "__main__":       # 🔑 YAAD RAKHO: main guard
    print("=" * 60)
    print("PRACTICE 12: Data EDA & Analysis")
    print("=" * 60)

    # Change this to your actual dataset path!
    DATA_DIR = '../RESIDE-6K'    # ℹ️ COMMON: relative path to dataset

    start = time.time()          # ℹ️ COMMON: timing

    # Run all sections
    stats = compute_image_stats(DATA_DIR)       # Section 1
    plot_pixel_distribution()                   # Section 2
    analyze_image_sizes(stats)                  # Section 3
    detect_corrupt_images(DATA_DIR)             # Section 4
    detect_duplicates(DATA_DIR)                 # Section 5
    analyze_class_balance()                     # Section 6
    validate_paired_data()                      # Section 7
    show_sample_grid()                          # Section 8
    detect_outliers()                           # Section 9
    generate_eda_report()                       # Section 10

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"PRACTICE 12 COMPLETE! (Time: {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"\nKYA SEEKHA:")
    print(f"  1. Image statistics — mean, std nikalna (normalization ke liye)")
    print(f"  2. Pixel distribution — hazy vs normal ka fark")
    print(f"  3. Size analysis — resize strategy decide karna")
    print(f"  4. Corrupt detection — PIL verify()")
    print(f"  5. Duplicate detection — MD5 hashing")
    print(f"  6. Class balance — imbalance ratio, class weights")
    print(f"  7. Paired validation — input-target match check")
    print(f"  8. Visual grid — data ka feel lena")
    print(f"  9. Outlier detection — z-score method")
    print(f" 10. EDA report — paper ke liye summary")
    print(f"\nNEXT: python practice/13_data_cleaning_preprocessing.py")

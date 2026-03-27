"""
=============================================================================
PRACTICE 15: CUSTOM DATASET & DATALOADER — Data Pipeline Master!
=============================================================================

DATASET & DATALOADER KYA HAI?
==============================
DATASET:    Data ko ORGANIZE karo (images load karo, transforms lagao)
DATALOADER: Data ko BATCHES mein model ko FEED karo (efficiently!)

Analogy:
  Dataset = KITCHEN (ingredients ready karo, recipe follow karo)
  DataLoader = WAITER (plates mein serve karo, table pe le jao)
  Model = CUSTOMER (khana khaata hai — training!)

KYUN ZAROORI:
  1. Custom data ke liye CUSTOM Dataset class banana padta hai
  2. DataLoader efficiently batches banata hai (parallel!), shuffle karta hai
  3. Ye SABSE BASIC skill hai — BINA iske koi project NAHI ban sakta!
  4. Industry mein 90% time custom datasets hote hain, built-in nahi

IS FILE MEIN SEEKHOGE:
  ✅ Custom Dataset class (__init__, __len__, __getitem__)
  ✅ Classification Dataset
  ✅ Paired Dataset (hazy-clean dehazing ke liye!)
  ✅ DataLoader (batch, shuffle, num_workers, pin_memory)
  ✅ Samplers (Random, Weighted, Subset)
  ✅ Transforms pipeline (train vs test)
  ✅ Error handling (corrupt images)
  ✅ Performance optimization tips

=============================================================================
"""

import os                        # ℹ️ COMMON: file operations
import numpy as np               # 🔑 YAAD RAKHO: numpy
from PIL import Image            # 🔑 YAAD RAKHO: image loading
import random                    # ℹ️ COMMON: random ops
import time                      # ℹ️ COMMON: timing
import matplotlib                # ℹ️ COMMON: plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # ℹ️ COMMON: plt

# PyTorch imports
try:
    import torch                                # 🔑 YAAD RAKHO: PyTorch
    from torch.utils.data import (              # 🔑 YAAD RAKHO: data utilities
        Dataset,                                # ⭐ IMPORTANT: base class for custom datasets
        DataLoader,                             # ⭐ IMPORTANT: batch loading + shuffling
        random_split,                           # ℹ️ COMMON: train/val split
        Subset,                                 # ℹ️ COMMON: dataset ka subset
        WeightedRandomSampler,                  # ⭐ IMPORTANT: class imbalance fix
        SequentialSampler,                      # ℹ️ COMMON: sequential order
        RandomSampler,                          # ℹ️ COMMON: random order
    )
    import torchvision.transforms as T          # 🔑 YAAD RAKHO: transforms
    # TF: tf.data.Dataset, tf.data.DataLoader equivalent nahi, tf.data pipeline use karo
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch not installed!")


# =============================================================================
# SECTION 1: BASIC CUSTOM DATASET (The 3 Required Methods)
# =============================================================================

class SimpleDataset:
    """
    SABSE BASIC custom dataset — concept samjhne ke liye.

    CUSTOM DATASET KE 3 MANDATORY METHODS:
    ─────────────────────────────────────────
    __init__():     → Setup (paths load, transforms set)    → "Kitchen setup"
    __len__():      → Total items kitne hain                → "Menu mein kitne items"
    __getitem__(i): → i-th item return karo                 → "Order #i dedo"

    TF EQUIVALENT:
    tf.data.Dataset.from_tensor_slices(data)
    ya tf.data.Dataset.from_generator(generator_fn)
    """

    def __init__(self, size=100):
        """Setup: data generate/load karo"""
        print("  [SimpleDataset] __init__ called — data setup ho raha hai")
        self.size = size                     # ℹ️ COMMON: total items
        np.random.seed(42)
        self.data = np.random.randn(size, 10).astype(np.float32)   # 🔑 YAAD RAKHO: float32 — DL standard
        self.labels = np.random.randint(0, 3, size)                 # ℹ️ COMMON: 3 classes

    def __len__(self):
        """Total items kitne hain — DataLoader ko batana ZAROORI hai!"""
        return self.size                     # 🔑 YAAD RAKHO: len(dataset) = total items

    def __getitem__(self, idx):
        """I-th item return karo — DataLoader YAHI repeatedly call karta hai!"""
        # ⚠️ WARNING: __getitem__ mein HEAVY computation mat karo — ye BAAR BAAR call hota hai!
        return self.data[idx], self.labels[idx]  # 🔑 YAAD RAKHO: (data, label) tuple return


if HAS_TORCH:
    # ==========================================================================
    # SECTION 2: IMAGE CLASSIFICATION DATASET
    # ==========================================================================

    class ClassificationDataset(Dataset):  # 🔑 YAAD RAKHO: torch Dataset se inherit!
        """
        Image classification ke liye custom dataset.
        Folder structure expected:
          data/
            class_0/
              img1.jpg, img2.jpg, ...
            class_1/
              img1.jpg, img2.jpg, ...
        """

        def __init__(self, root_dir, transform=None):
            super().__init__()               # 🔑 YAAD RAKHO: super()
            self.root_dir = root_dir         # ℹ️ COMMON: data directory
            self.transform = transform       # ℹ️ COMMON: transforms store (baad mein __getitem__ mein use)

            self.image_paths = []            # ℹ️ COMMON: sabhi image paths
            self.labels = []                 # ℹ️ COMMON: sabhi labels
            self.classes = []                # ℹ️ COMMON: class names

            if os.path.exists(root_dir):
                self.classes = sorted(os.listdir(root_dir))  # ⭐ IMPORTANT: sorted() = consistent ordering!
                # ⚠️ WARNING: sorted() ZAROORI hai — warna different OS pe different order → bugs!

                for class_idx, class_name in enumerate(self.classes):
                    class_dir = os.path.join(root_dir, class_name)
                    if not os.path.isdir(class_dir):
                        continue
                    for img_name in os.listdir(class_dir):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)  # ℹ️ COMMON: class index as label

        def __len__(self):
            return len(self.image_paths)     # 🔑 YAAD RAKHO: total images

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]  # ℹ️ COMMON: path get
            label = self.labels[idx]          # ℹ️ COMMON: label get

            try:
                image = Image.open(img_path).convert('RGB')  # 🔑 YAAD RAKHO: .convert('RGB') = HAMESHA RGB force karo!
                # ⚠️ WARNING: kuch images grayscale ya RGBA hoti hain — convert('RGB') se consistent!
                # TF: tf.image.decode_image() + tf.image.grayscale_to_rgb()
            except Exception as e:
                # ⭐ IMPORTANT: corrupt image handling — skip karke blank image do!
                print(f"  ⚠️ Error loading {img_path}: {e}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))  # Black placeholder
                # ⚠️ WARNING: ye TRAINING mein noise add karta hai — better to REMOVE corrupt from dataset!

            if self.transform:               # 🔑 YAAD RAKHO: transform lagao (agar available hai)
                image = self.transform(image)

            return image, label              # 🔑 YAAD RAKHO: (image_tensor, label) return


    # ==========================================================================
    # SECTION 3: PAIRED DATASET (Dehazing — Hazy + Clean pairs)
    # ==========================================================================

    class PairedDataset(Dataset):  # ⭐ IMPORTANT: dehazing/super-res ke liye PAIRED dataset!
        """
        Image-to-Image tasks ke liye PAIRED dataset.
        Structure:
          data/train/hazy/  → input images
          data/train/GT/    → ground truth (clean) images

        KYUN ALAG HAI Classification se?
        Classification: 1 image + 1 label (number)
        Paired:         1 input image + 1 target IMAGE
        """

        def __init__(self, input_dir, target_dir, transform=None,
                     input_transform=None, target_transform=None, image_size=256):
            super().__init__()
            self.input_dir = input_dir       # ℹ️ COMMON: hazy images folder
            self.target_dir = target_dir     # ℹ️ COMMON: clean images folder
            self.transform = transform       # ⭐ IMPORTANT: SHARED transform (dono pe apply)
            self.input_transform = input_transform    # ℹ️ COMMON: input-only transform
            self.target_transform = target_transform  # ℹ️ COMMON: target-only transform
            self.image_size = image_size     # ℹ️ COMMON: target size

            # Paired files dhundho
            self.pairs = []                  # ℹ️ COMMON: (input_path, target_path) pairs

            if os.path.exists(input_dir) and os.path.exists(target_dir):
                input_files = set(os.listdir(input_dir))   # ⭐ IMPORTANT: set for fast lookup O(1)
                target_files = set(os.listdir(target_dir))
                common = sorted(input_files & target_files)  # ⭐ IMPORTANT: & = intersection (dono mein common files!)
                # ⚠️ WARNING: set intersection = SIRF matching filenames! Unpaired files AUTOMATICALLY skip

                for fname in common:
                    self.pairs.append((
                        os.path.join(input_dir, fname),
                        os.path.join(target_dir, fname)
                    ))

            if not self.pairs:
                print("  ⚠️ No paired images found — using dummy data")
                self._create_dummy_pairs()

        def _create_dummy_pairs(self):
            """Dummy pairs for demo"""
            os.makedirs('practice/temp_data/hazy', exist_ok=True)
            os.makedirs('practice/temp_data/clean', exist_ok=True)
            for i in range(20):
                # Hazy = lighter (fog effect)
                hazy = np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(hazy).save(f'practice/temp_data/hazy/{i:04d}.png')
                # Clean = full range
                clean = np.random.randint(20, 240, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(clean).save(f'practice/temp_data/clean/{i:04d}.png')
                self.pairs.append((
                    f'practice/temp_data/hazy/{i:04d}.png',
                    f'practice/temp_data/clean/{i:04d}.png'
                ))

        def __len__(self):
            return len(self.pairs)           # 🔑 YAAD RAKHO: total pairs

        def __getitem__(self, idx):
            input_path, target_path = self.pairs[idx]  # ℹ️ COMMON: paths unpack

            input_img = Image.open(input_path).convert('RGB')   # 🔑 YAAD RAKHO: load + RGB
            target_img = Image.open(target_path).convert('RGB')

            # Resize
            input_img = input_img.resize((self.image_size, self.image_size), Image.BILINEAR)   # ℹ️ COMMON: resize
            target_img = target_img.resize((self.image_size, self.image_size), Image.BILINEAR)

            # Shared augmentation (SAME transform on both!)
            if self.transform:
                seed = random.randint(0, 2**32)  # ⭐ IMPORTANT: same seed for both!
                random.seed(seed)
                torch.manual_seed(seed)          # 🔑 YAAD RAKHO: manual_seed = reproducible random
                input_img = self.transform(input_img)
                random.seed(seed)                # ⭐ IMPORTANT: SAME seed dobara!
                torch.manual_seed(seed)
                target_img = self.transform(target_img)

            # Individual transforms
            if self.input_transform:
                input_img = self.input_transform(input_img)    # ℹ️ COMMON: input-specific
            if self.target_transform:
                target_img = self.target_transform(target_img) # ℹ️ COMMON: target-specific

            # Ensure tensors
            if not isinstance(input_img, torch.Tensor):
                input_img = T.ToTensor()(input_img)   # 🔑 YAAD RAKHO: PIL→Tensor
            if not isinstance(target_img, torch.Tensor):
                target_img = T.ToTensor()(target_img)

            return input_img, target_img     # ⭐ IMPORTANT: (hazy_tensor, clean_tensor)


    # ==========================================================================
    # SECTION 4: DATALOADER — Batches Mein Feed Karo!
    # ==========================================================================

    def demonstrate_dataloader():
        """
        DataLoader KYA KARTA HAI:
        1. Dataset se items lata hai (__getitem__ call)
        2. BATCH banata hai (jaise 32 images ek saath)
        3. SHUFFLE karta hai (random order — overfitting rokta hai!)
        4. PARALLEL loading (num_workers > 0 → multiple CPU cores)
        5. GPU TRANSFER optimize (pin_memory=True)
        """
        print("\n" + "=" * 60)
        print("SECTION 4: DataLoader Configuration")
        print("=" * 60)

        # Simple dataset for demo
        dataset = SimpleDataset(size=100)

        # --- Basic DataLoader ---
        loader = DataLoader(
            dataset,                         # 🔑 YAAD RAKHO: dataset object
            batch_size=16,                   # ⭐ IMPORTANT: ek batch mein kitne items (16-128 typical)
            shuffle=True,                    # ⭐ IMPORTANT: random order — training mein HAMESHA True!
            # TF: ds.shuffle(buffer_size).batch(16)
        )
        print(f"\n  Basic DataLoader:")
        print(f"  Dataset size:  {len(dataset)}")
        print(f"  Batch size:    16")
        print(f"  Total batches: {len(loader)} (100/16 = 6.25 → 7 batches)")

        # Ek batch dekhte hain
        data_batch, label_batch = next(iter(loader))  # 🔑 YAAD RAKHO: next(iter(loader)) = pehla batch
        print(f"  Batch shape:   data={data_batch.shape}, labels={label_batch.shape}")

        # --- Advanced DataLoader ---
        print(f"\n  Advanced DataLoader Parameters:")
        print(f"  ─────────────────────────────────")

        advanced_loader = DataLoader(
            dataset,
            batch_size=32,                   # ⭐ IMPORTANT: batch_size — GPU memory ke hisaab se choose karo
            shuffle=True,                    # ⭐ IMPORTANT: training=True, test=False
            num_workers=0,                   # ⭐ IMPORTANT: parallel data loading threads
            # 🔑 YAAD RAKHO: num_workers guide:
            #   0 = main process mein load (debug ke liye)
            #   2-4 = typical for most setups
            #   CPU cores / 2 = good rule of thumb
            # ⚠️ WARNING: Windows pe num_workers>0 → if __name__=='__main__' ZAROORI!
            # TF: ds.prefetch(tf.data.AUTOTUNE) — auto-optimized
            pin_memory=False,                # ⭐ IMPORTANT: True = CPU→GPU transfer FASTER (CUDA only)
            # 🔑 YAAD RAKHO: pin_memory=True sirf GPU training mein karo
            # TF: automatic — no equivalent needed
            drop_last=False,                 # ℹ️ COMMON: True = incomplete last batch DROP karo
            # ⚠️ WARNING: BatchNorm ke liye batch_size=1 → error! drop_last=True se bachao
        )

        print(f"  batch_size=32:    Ek batch mein 32 items")
        print(f"  shuffle=True:     Random order (training)")
        print(f"  num_workers=0:    Main thread (safe for Windows)")
        print(f"  pin_memory=False: CPU→GPU transfer (True for GPU)")
        print(f"  drop_last=False:  Incomplete batch keep karo")


    # ==========================================================================
    # SECTION 5: SAMPLERS (Custom Ordering)
    # ==========================================================================

    def demonstrate_samplers():
        """
        SAMPLER = DataLoader ko batao KIS ORDER mein items lene hain.

        DEFAULT: RandomSampler (training) / SequentialSampler (test)
        CUSTOM: WeightedRandomSampler → class imbalance FIX!
        """
        print("\n" + "=" * 60)
        print("SECTION 5: Samplers (Class Imbalance Fix)")
        print("=" * 60)

        # Imbalanced dataset demo
        np.random.seed(42)
        n = 1000
        labels = np.array([0]*800 + [1]*150 + [2]*50)  # ⚠️ Very imbalanced!
        data = np.random.randn(n, 5).astype(np.float32)

        print(f"\n  Imbalanced Dataset:")
        print(f"  Class 0: 800 (80%)")
        print(f"  Class 1: 150 (15%)")
        print(f"  Class 2:  50 (5%)")

        # WeightedRandomSampler — SOLUTION!
        class_counts = np.bincount(labels)   # ⭐ IMPORTANT: np.bincount = har class kitne items
        # TF: tf.math.bincount(labels)
        class_weights = 1.0 / class_counts   # ⭐ IMPORTANT: inverse frequency = rare class ko ZYADA weight
        sample_weights = class_weights[labels]  # ⭐ IMPORTANT: har item ko uski class ka weight do

        sampler = WeightedRandomSampler(
            weights=sample_weights,          # ⭐ IMPORTANT: har sample ka weight
            num_samples=len(labels),         # ℹ️ COMMON: kitne samples per epoch
            replacement=True,               # ⭐ IMPORTANT: True = ek item baar baar aa sakta hai
            # ⚠️ WARNING: replacement=True ZAROORI hai — warna minority class zyada nahi aa paayegi!
        )

        print(f"\n  WeightedRandomSampler:")
        print(f"  Class weights: {class_weights}")
        print(f"  → Rare class (2) ko {class_weights[2]/class_weights[0]:.0f}x zyada dikhaayega!")
        print(f"\n  Code:")
        print(f"  sampler = WeightedRandomSampler(weights, num_samples, replacement=True)")
        print(f"  loader = DataLoader(dataset, batch_size=32, sampler=sampler)")
        print(f"  # ⚠️ WARNING: sampler use karo toh shuffle=False karo! Dono saath error dete hain")

        # Subset sampler (train/val split ke liye)
        print(f"\n  Train/Val Split Methods:")
        print(f"  ────────────────────────")
        print(f"  # Method 1: random_split (EASY!)")
        print(f"  train_set, val_set = random_split(dataset, [800, 200])")
        print(f"  # TF: tf.keras.utils.split_dataset(dataset, left_size=0.8)")
        print(f"")
        print(f"  # Method 2: Subset with indices (MORE CONTROL)")
        print(f"  indices = list(range(len(dataset)))")
        print(f"  train_idx, val_idx = indices[:800], indices[800:]")
        print(f"  train_set = Subset(dataset, train_idx)")
        print(f"  val_set = Subset(dataset, val_idx)")


    # ==========================================================================
    # SECTION 6: TRANSFORMS PIPELINE (Train vs Test)
    # ==========================================================================

    def demonstrate_transforms_pipeline():
        """
        TRAIN aur TEST ke transforms ALAG hone chahiye!
        Training: augmentation + normalize
        Testing:  sirf normalize (NO augmentation!)
        """
        print("\n" + "=" * 60)
        print("SECTION 6: Train vs Test Transforms")
        print("=" * 60)

        # Training transforms
        train_transform = T.Compose([       # 🔑 YAAD RAKHO: Compose = chain
            T.Resize((256, 256)),            # ℹ️ COMMON: fixed size
            T.RandomHorizontalFlip(p=0.5),   # ⭐ IMPORTANT: augmentation (training ONLY!)
            T.RandomVerticalFlip(p=0.3),     # ℹ️ COMMON: vertical flip
            T.ColorJitter(0.2, 0.2, 0.2),    # ⭐ IMPORTANT: color augmentation
            T.ToTensor(),                    # 🔑 YAAD RAKHO: PIL→Tensor, HWC→CHW, /255
            # TF: tf.image.random_flip_left_right(), tf.image.random_brightness()
        ])

        # Test transforms (NO augmentation!)
        test_transform = T.Compose([        # ⚠️ WARNING: test mein augmentation NAHI!
            T.Resize((256, 256)),            # ℹ️ COMMON: same resize
            T.ToTensor(),                    # 🔑 YAAD RAKHO: ToTensor
            # TF: tf.image.resize() + tf.cast(img, tf.float32) / 255.0
        ])

        print(f"\n  Train Transform: Resize + Flip + ColorJitter + ToTensor")
        print(f"  Test Transform:  Resize + ToTensor (NO augmentation!)")
        print(f"")
        print(f"  ⚠️ RULES:")
        print(f"  1. Augmentation SIRF train mein — test/val mein KABHI NAHI!")
        print(f"  2. Normalize train aur test mein SAME values se!")
        print(f"  3. ToTensor() automatically /255 karta hai — dobara mat karo!")
        print(f"  4. Test mein deterministic hona chahiye (same input → same output)")
        print(f"")

        # Custom transform class
        print(f"  Custom Transform Class:")
        print(f"  ───────────────────────")
        print(f"  class AddGaussianNoise:")
        print(f"      def __init__(self, mean=0, std=0.1):")
        print(f"          self.mean = mean")
        print(f"          self.std = std")
        print(f"      def __call__(self, tensor):              # 🔑 __call__ making it callable")
        print(f"          noise = torch.randn_like(tensor) * self.std + self.mean")
        print(f"          return torch.clamp(tensor + noise, 0, 1)")
        print(f"  # Usage: T.Compose([..., AddGaussianNoise(0, 0.05)])")


    # ==========================================================================
    # SECTION 7: PERFORMANCE OPTIMIZATION
    # ==========================================================================

    def demonstrate_performance():
        """
        DataLoader ko FAST kaise banayein?
        GPU ko STARVE mat hone do — data FAST aana chahiye!
        """
        print("\n" + "=" * 60)
        print("SECTION 7: DataLoader Performance Optimization")
        print("=" * 60)

        print(f"""
  BOTTLENECK KAHAN HOTA HAI?
  ──────────────────────────
  GPU is FAST, but data loading SLOW ho sakta hai:
    GPU:  100 batches/sec process kar sakta hai
    CPU:  10 batches/sec load kar sakta hai
    → GPU 90% time IDLE! (bahut WASTED compute!)

  SOLUTIONS:
  ──────────

  1. num_workers (Parallel Loading):
     loader = DataLoader(ds, num_workers=4)    # ⭐ 4 parallel threads!
     Rule: num_workers = CPU_cores // 2
     ⚠️ Windows: if __name__=='__main__' block ZAROORI!
     TF: ds.prefetch(tf.data.AUTOTUNE) — automatically optimize

  2. pin_memory (Faster CPU→GPU Transfer):
     loader = DataLoader(ds, pin_memory=True)  # ⭐ GPU ke liye FAST!
     Pinned memory = CPU memory jo GPU ke liye "reserved" hai
     Transfer ASYNC hota hai — overlap with computation!
     TF: automatic — GPU pipeline optimized by default

  3. persistent_workers (Workers Reuse):
     loader = DataLoader(ds, num_workers=4, persistent_workers=True)
     Workers KILL nahi hote epoch ke baad — startup cost SAVE!
     ⚠️ Memory zyada lagti hai (RAM check karo)

  4. prefetch_factor (Advance Loading):
     loader = DataLoader(ds, prefetch_factor=2)  # ℹ️ 2 batches advance mein load
     Default = 2 (usually sufficient)
     TF: ds.prefetch(buffer_size)

  5. Efficient Data Format:
     Raw images (JPEG/PNG) → SLOW (decompress karna padta hai)
     HDF5 / LMDB / TFRecord → FAST (pre-processed, sequential read)
     TF: tf.data.TFRecordDataset() — TFRecord format FAST!
     PyTorch: webdataset library ya custom binary format

  TYPICAL OPTIMAL CONFIG:
  ────────────────────────
  loader = DataLoader(
      dataset,
      batch_size=32,
      shuffle=True,
      num_workers=4,           # ⭐ Parallel loading
      pin_memory=True,         # ⭐ Fast GPU transfer
      persistent_workers=True, # ⭐ Workers reuse
      prefetch_factor=2,       # ℹ️ Advance loading
      drop_last=True,          # ℹ️ Clean batches
  )
        """)


    # ==========================================================================
    # SECTION 8: COMPLETE EXAMPLE — Putting It All Together
    # ==========================================================================

    def complete_example():
        """
        POORA pipeline — dataset se lekar training loop tak!
        """
        print("\n" + "=" * 60)
        print("SECTION 8: Complete Pipeline Example")
        print("=" * 60)

        # Create dummy paired dataset
        print("\n  Creating dummy paired dataset...")
        paired_ds = PairedDataset(
            input_dir='practice/temp_data/hazy',
            target_dir='practice/temp_data/clean',
            image_size=64
        )
        print(f"  Dataset size: {len(paired_ds)} pairs")

        # Train/Val split
        train_size = int(0.8 * len(paired_ds))      # ⭐ IMPORTANT: 80-20 split
        val_size = len(paired_ds) - train_size
        train_ds, val_ds = random_split(paired_ds, [train_size, val_size])
        # TF: tf.keras.utils.split_dataset(ds, left_size=0.8)
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        # DataLoaders
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)    # ⭐ IMPORTANT: shuffle=True for training
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)       # ⭐ IMPORTANT: shuffle=False for validation

        # Iterate through batches
        print(f"\n  Iterating through train loader:")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(f"    Batch {batch_idx}: inputs={inputs.shape}, targets={targets.shape}")
            if batch_idx >= 2:
                print(f"    ... ({len(train_loader)} total batches)")
                break

        # Visualize a batch
        inputs, targets = next(iter(train_loader))
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle('Batch Visualization: Input (top) vs Target (bottom)', fontsize=12)
        for i in range(min(4, inputs.shape[0])):
            inp = inputs[i].permute(1, 2, 0).numpy()    # 🔑 YAAD RAKHO: CHW→HWC for display
            # TF: no permute needed (already HWC)
            tgt = targets[i].permute(1, 2, 0).numpy()
            inp = np.clip(inp, 0, 1)         # ℹ️ COMMON: clip for valid display
            tgt = np.clip(tgt, 0, 1)
            axes[0, i].imshow(inp); axes[0, i].axis('off'); axes[0, i].set_title(f'Input {i}')
            axes[1, i].imshow(tgt); axes[1, i].axis('off'); axes[1, i].set_title(f'Target {i}')
        plt.tight_layout()
        os.makedirs('practice/results', exist_ok=True)
        plt.savefig('practice/results/15_batch_visualization.png', dpi=100)
        plt.close()
        print(f"\n  📊 Saved: practice/results/15_batch_visualization.png")

        # Cleanup temp data
        import shutil
        if os.path.exists('practice/temp_data'):
            shutil.rmtree('practice/temp_data')  # ℹ️ COMMON: temp folder delete
            print(f"  🧹 Temp data cleaned up")

        # Summary
        print(f"\n  COMPLETE PIPELINE SUMMARY:")
        print(f"  ─────────────────────────")
        print(f"  1. Custom Dataset class (init, len, getitem)")
        print(f"  2. Transforms pipeline (train vs test)")
        print(f"  3. Train/Val split (random_split)")
        print(f"  4. DataLoader (batch, shuffle, workers)")
        print(f"  5. Training loop (iterate batches)")
        print(f"  6. Visualize (batch check)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":       # 🔑 YAAD RAKHO: main guard (WINDOWS mein num_workers ke liye ZAROORI!)
    print("=" * 60)
    print("PRACTICE 15: Custom Dataset & DataLoader")
    print("=" * 60)

    # Section 1: Basic concept
    print("\n" + "=" * 60)
    print("SECTION 1: Basic Custom Dataset")
    print("=" * 60)
    ds = SimpleDataset(size=50)
    print(f"  Length: {len(ds)}")
    print(f"  Item 0: data shape={ds[0][0].shape}, label={ds[0][1]}")

    if HAS_TORCH:
        # Section 2: Classification Dataset
        print(f"\n  (Classification Dataset — use with actual image folders)")

        # Section 3: Paired Dataset
        print(f"\n  (Paired Dataset — use for dehazing, super-res, etc.)")

        # Section 4-8
        demonstrate_dataloader()             # Section 4
        demonstrate_samplers()               # Section 5
        demonstrate_transforms_pipeline()    # Section 6
        demonstrate_performance()            # Section 7
        complete_example()                   # Section 8
    else:
        print("\n  ⚠️ PyTorch not installed — sections 2-8 skipped")

    print(f"\n{'='*60}")
    print(f"PRACTICE 15 COMPLETE!")
    print(f"{'='*60}")
    print(f"\nKYA SEEKHA:")
    print(f"  1. Custom Dataset — __init__, __len__, __getitem__")
    print(f"  2. Classification Dataset — folder-based labels")
    print(f"  3. Paired Dataset — hazy-clean pairs with SEED matching")
    print(f"  4. DataLoader — batch, shuffle, num_workers, pin_memory")
    print(f"  5. Samplers — WeightedRandomSampler (class imbalance fix!)")
    print(f"  6. Train vs Test transforms (augmentation sirf train!)")
    print(f"  7. Performance — parallel loading, pinned memory")
    print(f"  8. Complete pipeline — data → loader → batch → train")
    print(f"\n🎉 DATA SKILLS PRACTICE COMPLETE!")
    print(f"   Ab tumhe data handle karna ACHHE se aata hai!")
    print(f"   NEXT: Research improvements pe kaam karte hain! 🚀")

"""Course-specific lightweight feature extractors for the Week 12 practical."""

from typing import List, Optional

import numpy as np


class LightweightSAD:
    """Patch-normalised grayscale appearance descriptor using Pillow/NumPy."""

    def __init__(self, n_pixels: int = 2048, patch_length: int = 8):
        self.n_pixels = n_pixels
        self.patch_length = patch_length
        self.image_shape = None

    def _choose_image_shape(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        # Preserve the aspect ratio while keeping the descriptor near the
        # requested size. Multiples of patch_length avoid partial patches.
        scale = np.sqrt(self.n_pixels / (height * width))
        height = int(np.ceil(scale * height))
        width = int(np.ceil(scale * width))
        height -= height % self.patch_length
        width -= width % self.patch_length
        self.image_shape = (max(height, self.patch_length), max(width, self.patch_length))

    def _patch_normalise(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        # Normalising each spatial patch independently reduces sensitivity to
        # local brightness and contrast changes between day and night.
        for row in range(0, image.shape[0], self.patch_length):
            for column in range(0, image.shape[1], self.patch_length):
                patch = image[
                    row:row + self.patch_length,
                    column:column + self.patch_length,
                ]
                value_range = patch.max() - patch.min()
                if value_range > 0:
                    patch[:] = 255 * (patch - patch.min()) / value_range
                else:
                    # A constant patch contains no contrast information.
                    patch[:] = 0
        return image

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        from PIL import Image

        if not imgs:
            return np.empty((0, 0), dtype=np.float32)
        if self.image_shape is None:
            self._choose_image_shape(imgs[0])

        height, width = self.image_shape
        # Pillow moved the resampling constants under Image.Resampling in v9.1.
        bilinear = getattr(Image, "Resampling", Image).BILINEAR
        descriptors = []
        for image in imgs:
            resized = Image.fromarray(image).convert("L").resize(
                (width, height),
                bilinear,
            )
            descriptors.append(self._patch_normalise(np.asarray(resized)).reshape(-1))
        return np.stack(descriptors).astype(np.float32)


class DINOv2FeatureExtractor:
    """Extract global DINOv2 or AnyLoc-inspired DINOv2+VLAD descriptors.

    VLAD mode learns a visual vocabulary from the first image set passed to
    ``compute_features`` (the database in the practical), then reuses it for
    subsequent query sets.
    """

    def __init__(
        self,
        aggregation: str = "global",
        num_clusters: int = 16,
        max_kmeans_descriptors: int = 20_000,
        kmeans_iterations: int = 10,
        batch_size: int = 8,
    ):
        import torch
        from torchvision import transforms

        if aggregation not in {"global", "vlad"}:
            raise ValueError("aggregation must be 'global' or 'vlad'")

        self.aggregation = aggregation
        self.num_clusters = num_clusters
        self.max_kmeans_descriptors = max_kmeans_descriptors
        self.kmeans_iterations = kmeans_iterations
        self.batch_size = batch_size
        self.cluster_centres: Optional[np.ndarray] = None

        # Prefer an accelerator when present, but retain a CPU-only path for
        # students without a CUDA-capable machine.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        print("Loading DINOv2 ViT-S/14 (weights are downloaded on first use)")
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            trust_repo=True,
        ).to(self.device)
        self.model.eval()

        # DINOv2 uses the usual ImageNet channel statistics. The 224x224 crop
        # produces 16x16 = 256 spatial tokens with a ViT-S/14 backbone.
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _extract_tokens(self, imgs: List[np.ndarray]):
        import torch

        class_tokens = []
        patch_tokens = []
        # inference_mode avoids retaining gradients and substantially reduces
        # memory use while extracting frozen features.
        with torch.inference_mode():
            for start in range(0, len(imgs), self.batch_size):
                batch = torch.stack([
                    self.preprocess(image)
                    for image in imgs[start:start + self.batch_size]
                ]).to(self.device)
                features = self.model.forward_features(batch)
                # The class token is the global descriptor. VLAD instead uses
                # all normalised patch tokens to retain local visual evidence.
                class_tokens.append(features["x_norm_clstoken"].cpu().numpy())
                if self.aggregation == "vlad":
                    patch_tokens.append(features["x_norm_patchtokens"].cpu().numpy())

        classes = np.concatenate(class_tokens).astype(np.float32)
        patches = np.concatenate(patch_tokens).astype(np.float32) if patch_tokens else None
        return classes, patches

    @staticmethod
    def _squared_distances(descriptors: np.ndarray, centres: np.ndarray) -> np.ndarray:
        # Pairwise ||x-c||^2 without allocating an (N,K,D) residual tensor.
        return (
            np.sum(descriptors * descriptors, axis=1, keepdims=True)
            + np.sum(centres * centres, axis=1)[None, :]
            - 2 * descriptors @ centres.T
        )

    def _fit_vocabulary(self, patches: np.ndarray) -> None:
        # Pool patches from all database images into one visual-word training
        # set. Sampling caps the cost for larger databases.
        descriptors = patches.reshape(-1, patches.shape[-1])
        # A fixed seed makes practical results repeatable across runs.
        rng = np.random.default_rng(0)
        if len(descriptors) > self.max_kmeans_descriptors:
            indices = rng.choice(len(descriptors), self.max_kmeans_descriptors, replace=False)
            descriptors = descriptors[indices]
        if len(descriptors) < self.num_clusters:
            raise ValueError("Not enough patch descriptors to fit the VLAD vocabulary")

        centres = descriptors[
            rng.choice(len(descriptors), self.num_clusters, replace=False)
        ].copy()
        # A small, dependency-free implementation of Lloyd's k-means update.
        for _ in range(self.kmeans_iterations):
            assignments = np.argmin(self._squared_distances(descriptors, centres), axis=1)
            for cluster in range(self.num_clusters):
                members = descriptors[assignments == cluster]
                if len(members):
                    centres[cluster] = members.mean(axis=0)
        self.cluster_centres = centres.astype(np.float32)

    def _aggregate_vlad(self, patches: np.ndarray) -> np.ndarray:
        if self.cluster_centres is None:
            raise RuntimeError("Fit the VLAD vocabulary before aggregation")

        descriptors = []
        for image_patches in patches:
            # Assign each spatial patch to its closest visual word.
            assignments = np.argmin(
                self._squared_distances(image_patches, self.cluster_centres),
                axis=1,
            )
            residuals = np.zeros_like(self.cluster_centres)
            for cluster in range(self.num_clusters):
                members = image_patches[assignments == cluster]
                if len(members):
                    # VLAD records how the assigned patches differ from their
                    # visual word, rather than merely counting assignments.
                    residuals[cluster] = (
                        members - self.cluster_centres[cluster]
                    ).sum(axis=0)

            # Intra-normalisation prevents one frequently occurring visual
            # word from dominating; final L2 normalisation enables cosine use.
            cluster_norms = np.linalg.norm(residuals, axis=1, keepdims=True)
            residuals = residuals / np.maximum(cluster_norms, 1e-12)
            descriptor = residuals.reshape(-1)
            descriptor = descriptor / max(np.linalg.norm(descriptor), 1e-12)
            descriptors.append(descriptor)
        return np.stack(descriptors).astype(np.float32)

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        class_tokens, patch_tokens = self._extract_tokens(imgs)
        if self.aggregation == "global":
            return class_tokens
        # The practical always calls the database first, so the first VLAD
        # call learns the vocabulary and later query calls reuse it.
        if self.cluster_centres is None:
            print("Learning a VLAD vocabulary from database patch descriptors")
            self._fit_vocabulary(patch_tokens)
        return self._aggregate_vlad(patch_tokens)

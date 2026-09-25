# Weight and sample-data provenance and hosting

- Upstream: `facebook/convnextv2-tiny-1k-224`
- Revision: `f4db009e63145e02b3c075aa64d90ce41bcca4b1`, pinned 2026-09-25 by `python tools/pin_snapshot.py`, which resolved the Hub's `main` to this commit, downloaded every manifest-listed file at it, cross-checked each LFS file against the SHA-256 the Hub records, recorded the Hub's LFS SHA-256 of the two reference files, and wrote the commit and digests into the manifest and `src/convnextv2_classification_pipeline/pipeline.py`.
- Executed artifact: `model.safetensors` (114,565,784 bytes, SHA-256 `ac05396e9e8b222e43be9fff080b464c3cccaa54800923d04a7d20182d2fdf5f`).
- Hosted, not executed: `pytorch_model.bin` (114,608,874 bytes), a pickle of the same weights, and `tf_model.h5` (114,787,096 bytes), the TensorFlow checkpoint. Both are listed under `referenceFiles`; the pin tool records their Hub LFS SHA-256 without downloading them.
- Manifest: `weights/convnextv2-tiny-1k-224/dimer-base-manifest.json` (4 staged files: `README.md`, `config.json`, `preprocessor_config.json`, `model.safetensors`; `totalBytes` 114639230; plus the two reference files). Every entry carries its SHA-256.
- Committed copy: `preprocessor_config.json` (352 bytes) is committed as the Hub serves it at the pinned commit, so a test can check the pipeline's torchvision transform against the Transformers processor built from it. The pin tool replaced it with the bytes downloaded at the pinned commit before hashing (they were byte-identical), so its digest describes the pinned bytes.
- Upstream weight licence: **CC-BY-NC-4.0.** The upstream `facebookresearch/ConvNeXt-V2` repository states: "This project is released under the MIT license except ImageNet pre-trained and fine-tuned models which are licensed under a CC-BY-NC", and its LICENSE file carries the MIT licence followed by the Creative Commons Attribution-NonCommercial 4.0 International licence. The Hugging Face card for the converted checkpoint is tagged `apache-2.0`. The converted weights derive from the upstream models, so this repository applies the more restrictive licence, including to adapters fine-tuned from them.
- Hosting: the Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored), and nothing here redistributes the weights. CC-BY-NC-4.0 requires attribution and forbids commercial use.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files that are absent, at the pinned revision; `verify_snapshot()` then checks every file before any load. `weights/**` is marked `-text` in `.gitattributes`, so Windows `core.autocrlf` cannot rewrite the committed files and break their digests.
- Loader trust boundary: `ConvNextV2Config.from_pretrained(<verified dir>, local_files_only=True)` and `ConvNextV2ForImageClassification(config)` build the architecture without contacting the Hub or running repository code; `load_state_dict(..., strict=True)` loads the verified SafeTensors file and refuses a missing, unexpected or mis-shaped tensor; and the loader refuses a config whose labels at the zero-shot group indices are not the expected ImageNet classes.

## Tutorial sample data

- Dataset: `Cleanlab/cifar-10-subset`, file `CIFAR-10-subset.zip`, at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75`.
- Size and digest: 986,707 bytes, SHA-256 `66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19`. `fetch_sample_archive` refuses any other bytes and has no fallback.
- Licence: MIT (the dataset card). The images are CIFAR-10 images (Krizhevsky, 2009); the tutorial keeps the `frog` and `truck` folders.
- Hosting: the archive is downloaded at runtime into a working directory (`data/`, git-ignored) and is not redistributed by this repository.

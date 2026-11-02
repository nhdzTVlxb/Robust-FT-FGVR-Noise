# Tara

This package contains training, validation, inference, and environment setup. Dataset files and model files stay outside the container image and are mounted from the host.

## Training

Mount the labeled class-folder tree at `/mnt/aic/train` and the approved OpenAI CLIP ViT-B/32 checkpoint at `/mnt/aic/open_clip_model.pth`. Then run:

```bash
conda env create -f environment.yml
conda activate tara
./run.sh
```

The run uses the fixed recipe in `recipe.toml`, reserves five images from each class with at least 15 images for validation, uses seed 42 and a three-epoch warmup, and runs for at most 40 epochs. It writes validation history to `reports/epochs.csv` and updates only `weights/best.pth`. Early stopping starts after epoch 20 and stops after three consecutive non-improving validation epochs. Set `CUDA_VISIBLE_DEVICES` to select a GPU. `TARA_TRAIN_TREE`, `TARA_CLIP`, and `TARA_OUTPUT_DIR` can point to host-mounted paths.

## Inference

Mount the class-folder tree and test images at `/mnt/aic/train` and `/mnt/aic/test`. Put the matching `best.pth` next to `predict.py` or under `weights/best.pth`, then run `python predict.py`. The fixed inference recipe uses 448/480/512/544/576-pixel scales, seven crop positions with separate original and horizontal-flip passes at each scale, batch size 704, `torch.compile`, Co3 prototype propagation, and soft class balancing at strength 0.28 at every balancing step. It writes and validates `submission.zip` in the current directory. `predict.py` and `tool.py` contain all inference code and do not import the training modules.

The Dockerfile installs only the runtime environment. It does not copy dataset, model, or application files into the image. Mount those paths at runtime. The inference checkpoint is in the separate model-files directory.

See `PACKING.md` for container mount commands.

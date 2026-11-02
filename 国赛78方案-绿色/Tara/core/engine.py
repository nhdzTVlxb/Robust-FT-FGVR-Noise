import csv
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .encoder import PATCH, TaraClassifier
from .images import AugmentedPhotoDataset, ClipPixelNormalizer, EvaluationPhotoDataset, inspect_class_folders, quiet_worker
from .optim import NoiseAwareObjective, layerwise_groups, warm_cosine
from .plan import InteractionAgenda
from .seed import initialise_readout
from .validation import photo_loader, validate_sides

INIT_SIDE = 224
WEIGHT_NAME = 'best.pth'


@torch.no_grad()
def estimate_sample_trust(embeddings, targets, class_count, guard):
    vectors = F.normalize(embeddings.float(), dim=1)
    labels = torch.as_tensor(targets, device=vectors.device, dtype=torch.long)
    sums = torch.zeros(class_count, vectors.shape[1], device=vectors.device).index_add_(0, labels, vectors)
    counts = torch.bincount(labels, minlength=class_count).float()
    own_centres = F.normalize(sums[labels] - vectors, dim=1)
    similarities = vectors @ F.normalize(sums, dim=1).T
    own_similarity = (vectors * own_centres).sum(1)
    similarities[torch.arange(len(labels), device=vectors.device), labels] = -float('inf')
    disagreement = (similarities.max(1).values > own_similarity + 0.05) & (counts[labels] >= 3)
    trust = torch.ones(len(labels), device=vectors.device)
    tail = counts[labels] < guard.tail_count_limit
    rare = counts[labels] < guard.rare_count_limit
    trust[disagreement & ~tail] = guard.head_suspect_weight
    trust[disagreement & tail] = guard.tail_suspect_weight
    trust[disagreement & rare] = guard.rare_suspect_weight
    boost = (guard.tail_count_limit / counts[labels].clamp_min(1)).sqrt().clamp(min=1.0, max=guard.tail_boost_cap)
    trust.mul_(boost)
    summary = {
        'classes_below_tail_limit': int((counts < guard.tail_count_limit).sum()),
        'classes_below_rare_limit': int((counts < guard.rare_count_limit).sum()),
        'head_disagreements': int((disagreement & ~tail).sum()),
        'tail_disagreements': int((disagreement & tail).sum()),
        'trust_min': float(trust.min()),
        'trust_max': float(trust.max()),
    }
    return trust.cpu(), summary


def validation_indices(labels: np.ndarray, split, seed: int) -> np.ndarray:
    rng = np.random.default_rng((seed, 11))
    selected = [rng.choice(np.flatnonzero(labels == code), split.per_class, replace=False)
                for code in np.unique(labels) if (labels == code).sum() >= split.class_floor]
    return np.sort(np.concatenate(selected)).astype(np.int64) if selected else np.zeros(0, np.int64)


@torch.no_grad()
def encode_reference(model: TaraClassifier, paths, side: int, edge_limit: int, batch: int, workers: int,
                     normalizer: ClipPixelNormalizer) -> torch.Tensor:
    model.eval()
    encoded = []
    loader = photo_loader(EvaluationPhotoDataset(paths, side, edge_limit), batch, workers)
    for pixels in loader:
        with torch.autocast('cuda', torch.bfloat16):
            encoded.append(model.embed(normalizer(pixels)).float())
    return torch.cat(encoded)


@torch.no_grad()
def score_validation(model: TaraClassifier, paths, labels: np.ndarray, cfg, workers: int) -> float:
    probabilities, _ = validate_sides(
        model,
        paths,
        cfg.picture.side,
        cfg.readout.batch_size,
        workers,
        cfg.picture.source_edge_cap,
    )
    model.train()
    return float((probabilities.astype(np.float32).argmax(1) == labels).mean())


def _save_inference_state(model: TaraClassifier, destination: Path) -> None:
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if name != 'init_temperature'
    }
    torch.save(state, destination)


def train(tree: Path, clip_file: Path, output_dir: Path, cfg, workers: int = 12,
          report: Callable[[str], None] = print) -> Path:
    picture_cfg = cfg.picture
    optimizer_cfg = cfg.optim
    seed = cfg.holdout.seed
    picture_cfg.color = cfg.color
    output_dir = Path(output_dir)
    weight_dir = output_dir / 'weights'
    report_dir = output_dir / 'reports'
    weight_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    best_path = weight_dir / WEIGHT_NAME
    if best_path.exists():
        raise FileExistsError('best.pth already exists; choose a clean output directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this training run')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    gpu_ids = [0]

    paths, labels, class_codes = inspect_class_folders(tree)
    held = validation_indices(labels, cfg.holdout, seed)
    train_ids = np.setdiff1d(np.arange(len(paths)), held)
    steps_per_epoch = len(train_ids) // optimizer_cfg.batch_size
    if steps_per_epoch < 1:
        raise ValueError('training split is smaller than one global batch')
    total_steps = steps_per_epoch * optimizer_cfg.epochs
    warm_steps = max(1, int(round(optimizer_cfg.warm_epochs * steps_per_epoch)))
    patch_grid = picture_cfg.side // PATCH
    report(
        '[data] classes=%d total=%d train=%d validation=%d steps=%d epochs=%d GPUs=%s global_batch=%d' %
        (len(class_codes), len(paths), len(train_ids), len(held), steps_per_epoch,
         optimizer_cfg.epochs, gpu_ids, optimizer_cfg.batch_size)
    )

    model = TaraClassifier(len(class_codes), picture_cfg.side, clip_file).to(device)
    model.class_codes.copy_(torch.as_tensor(class_codes, device=device))
    normalizer = ClipPixelNormalizer(str(device))
    sample_trust = torch.ones(len(paths))
    frozen = TaraClassifier(len(class_codes), INIT_SIDE, clip_file).to(device)
    start = time.time()
    reference = encode_reference(
        frozen,
        [paths[index] for index in train_ids],
        INIT_SIDE,
        picture_cfg.source_edge_cap,
        optimizer_cfg.batch_size,
        workers,
        normalizer,
    )
    trust_values, trust_summary = estimate_sample_trust(
        reference, labels[train_ids], len(class_codes), cfg.tail_guard
    )
    sample_trust[torch.as_tensor(train_ids)] = trust_values
    report('[tailguard] ' + json.dumps(trust_summary, sort_keys=True))
    del frozen

    seed_weight, seed_bias, temperature = initialise_readout(
        reference,
        torch.as_tensor(labels[train_ids], device=device),
        len(class_codes),
        seed,
    )
    with torch.no_grad():
        model.readout.weight.copy_(seed_weight)
        model.readout.bias.copy_(seed_bias)
        model.init_temperature.fill_(temperature)
    del reference
    report('[initialisation] prototype head ready in %.1fs at temperature %.2f' % (time.time() - start, temperature))

    parameter_groups = layerwise_groups(model, optimizer_cfg)
    optimizer = torch.optim.AdamW(parameter_groups, betas=(0.9, 0.999), fused=True)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, warm_cosine(total_steps, warm_steps))
    objective = NoiseAwareObjective(
        len(class_codes),
        optimizer_cfg.label_smoothing,
        optimizer_cfg.gce_fraction,
        optimizer_cfg.gce_power,
    )
    depth_count = cfg.token_swap.candidate_blocks
    swap_mask = torch.zeros(depth_count, 1, 1 + patch_grid * patch_grid, 1, dtype=torch.bool, device=device)
    dataset = AugmentedPhotoDataset(paths, labels, picture_cfg, seed)
    validation_paths = [paths[index] for index in held]
    validation_labels = labels[held]
    forward_model = torch.compile(model)
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    metrics_file = report_dir / 'epochs.csv'
    with metrics_file.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'validation_accuracy', 'best_accuracy', 'seconds'])
        writer.writeheader()

        for epoch_index in range(optimizer_cfg.epochs):
            plan = InteractionAgenda(cfg, seed, epoch_index, train_ids, steps_per_epoch, patch_grid, picture_cfg.side)
            dataset.epoch = epoch_index
            sampler = plan.order[:steps_per_epoch * optimizer_cfg.batch_size].tolist()
            loader_args = {
                'batch_size': optimizer_cfg.batch_size,
                'sampler': sampler,
                'drop_last': True,
                'num_workers': workers,
                'pin_memory': True,
                'worker_init_fn': quiet_worker,
            }
            if workers:
                loader_args['prefetch_factor'] = 4
            loader = DataLoader(dataset, **loader_args)
            lambda_values = torch.as_tensor(plan.lam, dtype=torch.float32, device=device)
            loss_total = torch.zeros((), device=device)
            epoch_start = time.time()

            for step_index, (pixels, labels_batch, sample_ids) in enumerate(loader):
                inputs = plan.blend_pixels(normalizer(pixels), step_index)
                labels_batch = labels_batch.to(device, non_blocking=True)
                swap_mask.copy_(plan.swap[step_index])
                with torch.autocast('cuda', torch.bfloat16):
                    logits, _ = forward_model(inputs, swap_mask)
                loss = objective(
                    logits,
                    labels_batch,
                    lambda_values[step_index],
                    sample_trust[sample_ids].to(device),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                schedule.step()
                loss_total += loss.detach()
                if step_index == 0 or (step_index + 1) % 200 == 0:
                    report('[epoch %d step %d/%d] loss=%.4f' %
                           (epoch_index + 1, step_index + 1, steps_per_epoch, loss.item()))

            validation_accuracy = score_validation(model, validation_paths, validation_labels, cfg, workers)
            elapsed = time.time() - epoch_start
            improved = validation_accuracy > best_score
            if improved:
                best_score = validation_accuracy
                best_epoch = epoch_index + 1
                stale_epochs = 0
                _save_inference_state(model, best_path)
            elif epoch_index + 1 >= cfg.early_stop.minimum_epochs:
                stale_epochs += 1

            mean_loss = float(loss_total.item() / steps_per_epoch)
            writer.writerow({
                'epoch': epoch_index + 1,
                'train_loss': '%.7f' % mean_loss,
                'validation_accuracy': '%.7f' % validation_accuracy,
                'best_accuracy': '%.7f' % best_score,
                'seconds': '%.1f' % elapsed,
            })
            stream.flush()
            report('[epoch %d] loss=%.4f val=%.5f best=%.5f@%d stale=%d mix=%s seconds=%.1f' %
                   (epoch_index + 1, mean_loss, validation_accuracy, best_score, best_epoch,
                    stale_epochs, plan.tally(), elapsed))

            if epoch_index + 1 >= cfg.early_stop.minimum_epochs and stale_epochs >= cfg.early_stop.patience_epochs:
                report('[early-stop] best epoch %d, validation %.5f' % (best_epoch, best_score))
                break

    summary = {
        'best_epoch': best_epoch,
        'best_validation_accuracy': best_score,
        'epochs_completed': epoch_index + 1,
        'validation_images': len(held),
        'training_images': len(train_ids),
        'saved_weight': str(best_path),
        'weight_keys': 'model inference state only; no optimizer, schedule, EMA or training metadata',
    }
    (report_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    report('[complete] ' + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return best_path

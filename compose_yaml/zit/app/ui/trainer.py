"""LoRA training for Z-Image-Turbo — kohya-style (no PEFT).

Trains LoRA adapters using direct A/B matrix decomposition, matching
the ostris/ai-toolkit approach. Forward hooks inject LoRA output during
training, and saved weights are directly compatible with inference hooks.

Training runs in a **separate process** so that stopping it via
process.kill() instantly frees all GPU memory.
"""

import gc
import logging
import math
import multiprocessing as _mp
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("zit-ui")

_ctx = _mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Dataset: images + captions from folder
# ---------------------------------------------------------------------------
class ImageCaptionDataset(Dataset):
    """Load images with paired .txt caption files.

    Structure:
        dataset_dir/
        ├── image1.jpg
        ├── image1.txt   → "trigger_word, portrait, front view"
        ├── image2.png
        └── image2.txt   → "trigger_word, portrait, side view"
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, dataset_dir: str, resolution: int = 512):
        self.dataset_dir = Path(dataset_dir)
        self.resolution = resolution

        # Find all image files with matching .txt caption
        self.samples = []
        for f in sorted(self.dataset_dir.iterdir()):
            if f.suffix.lower() in self.EXTENSIONS:
                txt = f.with_suffix(".txt")
                caption = txt.read_text().strip() if txt.exists() else ""
                self.samples.append((str(f), caption))

        if not self.samples:
            raise ValueError(f"No images found in {dataset_dir}")

        logger.info("Dataset: %d images from %s", len(self.samples), dataset_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # Resize to resolution (center crop to square)
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)

        # To tensor: [0, 255] → [0, 1] (ostris/ToTensor convention)
        import numpy as np
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW

        return tensor, caption


# ---------------------------------------------------------------------------
# Kohya-style LoRA module
# ---------------------------------------------------------------------------
class LoRAModule(nn.Module):
    """Single LoRA adapter: down (in→rank) + up (rank→out).

    Matches ostris/kohya convention:
      - down init: kaiming uniform
      - up init: zeros
      - alpha stored as buffer (saved in state_dict)
      - scale = alpha / rank
    """

    def __init__(self, orig_module: nn.Linear, rank: int, alpha: float,
                 module_dropout: float = 0.0, rank_dropout: float = 0.0):
        super().__init__()
        in_features = orig_module.in_features
        out_features = orig_module.out_features

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Kaiming for down, zeros for up (ostris convention)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.register_buffer("alpha", torch.tensor(alpha))
        self.scale = alpha / rank
        self.rank = rank
        self.module_dropout = module_dropout
        self.rank_dropout = rank_dropout

    def forward(self, x):
        # Module dropout: skip entire LoRA (ostris convention)
        if self.module_dropout > 0 and self.training:
            if torch.rand(1).item() < self.module_dropout:
                # Return zeros — hook adds this to output, so net effect = no LoRA
                return torch.zeros(
                    *x.shape[:-1], self.lora_up.out_features,
                    device=x.device, dtype=x.dtype,
                )

        lx = self.lora_down(x)

        # Rank dropout: zero out random rank dimensions (ostris convention)
        if self.rank_dropout > 0 and self.training:
            mask = torch.rand(lx.shape[-1], device=lx.device) > self.rank_dropout
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        return self.lora_up(lx) * scale


def apply_lora_modules(
    transformer: nn.Module,
    rank: int,
    alpha: float,
    target_names: list[str],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    prefix_filter: str | None = None,
    module_dropout: float = 0.0,
    rank_dropout: float = 0.0,
) -> list[tuple[str, LoRAModule]]:
    """Attach LoRA modules to target Linear layers via forward hooks.

    Args:
        prefix_filter: If set, only apply to modules whose path starts with
                       this prefix (e.g. "layers." to skip noise_refiner/context_refiner).

    Returns list of (module_path, lora_module) for optimizer and saving.
    """
    lora_modules = []
    hooks = []

    for name, module in transformer.named_modules():
        # Prefix filter (e.g. only "layers.*")
        if prefix_filter and not name.startswith(prefix_filter):
            continue
        # Check if this module's name ends with any target name
        # Support multi-part targets like "adaLN_modulation.0"
        matched = False
        for t in target_names:
            if name == t or name.endswith("." + t):
                matched = True
                break
        if not matched:
            continue
        if not isinstance(module, nn.Linear):
            continue

        lora = LoRAModule(module, rank, alpha,
                          module_dropout=module_dropout,
                          rank_dropout=rank_dropout).to(device=device, dtype=dtype)
        lora.train()

        # Forward hook: output += lora(input)
        def _make_hook(lora_mod):
            def hook(mod, input, output):
                x = input[0] if isinstance(input, tuple) else input
                return output + lora_mod(x.to(lora_mod.lora_down.weight.dtype))
            return hook

        handle = module.register_forward_hook(_make_hook(lora))
        hooks.append(handle)
        lora_modules.append((name, lora))

    return lora_modules, hooks


def save_kohya_lora(
    lora_modules: list[tuple[str, LoRAModule]],
    output_path: str,
    alpha: float,
    rank: int,
):
    """Save LoRA weights in format compatible with inference hooks.

    Keys: "{module_path}.lora_A.weight", "{module_path}.lora_B.weight"
    Metadata: lora_alpha, rank
    """
    from safetensors.torch import save_file

    state_dict = {}
    for module_path, lora in lora_modules:
        # A = down weight (rank, in_features) — matches inference hook expectation
        state_dict[f"{module_path}.lora_A.weight"] = lora.lora_down.weight.data.cpu()
        # B = up weight (out_features, rank)
        state_dict[f"{module_path}.lora_B.weight"] = lora.lora_up.weight.data.cpu()

    metadata = {
        "lora_alpha": str(int(alpha)),
        "rank": str(rank),
    }

    save_file(state_dict, output_path, metadata=metadata)
    size_mb = os.path.getsize(output_path) / 1024**2
    logger.info("LoRA saved: %s (%.1f MB, %d pairs)", output_path, size_mb, len(lora_modules))


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------
def _train_process_entry(progress_queue, result_queue, model_dir, dataset_dir,
                         output_name, train_kwargs):
    """Runs in a child process. All GPU memory is freed when this process exits."""
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _logger = _logging.getLogger("zit-ui")

    def _send(msg):
        try:
            progress_queue.put_nowait(msg)
        except Exception:
            pass

    try:
        trainer = LoRATrainer(model_dir, dataset_dir, output_name)

        # Status callback — fires when loading stage changes
        def on_status(msg):
            _send({"type": "status", "message": msg})

        trainer.status_callback = on_status

        # Progress callback — fires every training step, throttled to ~1/sec
        _last_send = [0.0]

        def on_progress(step, total, loss):
            now = time.time()
            if now - _last_send[0] >= 1.0 or step == 1 or step == total:
                _last_send[0] = now
                _send({
                    "type": "progress",
                    "step": step,
                    "total": total,
                    "loss": loss,
                    "elapsed": trainer.elapsed,
                    "eta": trainer.eta,
                })

        trainer.progress_callback = on_progress

        path = trainer.train(**train_kwargs)
        result_queue.put({"status": "done", "path": path})

    except Exception as e:
        import traceback
        result_queue.put({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# TrainProcessManager — used by Gradio main process
# ---------------------------------------------------------------------------
class TrainProcessManager:
    """Spawns/kills the training subprocess. Kill = instant GPU memory release."""

    def __init__(self):
        self._process = None
        self._progress_queue = None
        self._result_queue = None

    def start(self, model_dir: str, dataset_dir: str, output_name: str,
              **train_kwargs):
        if self.is_alive():
            raise RuntimeError("Training already in progress")
        self._progress_queue = _ctx.Queue()
        self._result_queue = _ctx.Queue()
        self._process = _ctx.Process(
            target=_train_process_entry,
            args=(self._progress_queue, self._result_queue,
                  model_dir, dataset_dir, output_name, train_kwargs),
            daemon=True,
            name="zit-trainer",
        )
        self._process.start()
        logger.info("Training process started (pid=%d)", self._process.pid)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def kill(self) -> str:
        """Kill training process — all GPU memory freed instantly."""
        if self._process is None or not self._process.is_alive():
            return "No training in progress"
        pid = self._process.pid
        self._process.kill()
        self._process.join(timeout=5)
        logger.info("Training process killed (pid=%d)", pid)
        self._cleanup()
        return f"Training stopped (pid={pid})"

    def poll_progress(self) -> list[dict]:
        """Drain all pending progress messages from the queue."""
        messages = []
        if self._progress_queue is None:
            return messages
        while True:
            try:
                messages.append(self._progress_queue.get_nowait())
            except Exception:
                break
        return messages

    def get_result(self) -> dict | None:
        """Check if training finished. Returns result dict or None."""
        if self._result_queue is None:
            return None
        try:
            return self._result_queue.get_nowait()
        except Exception:
            return None

    def _cleanup(self):
        self._process = None
        self._progress_queue = None
        self._result_queue = None


# ---------------------------------------------------------------------------
# LoRA Trainer (runs inside subprocess)
# ---------------------------------------------------------------------------
class LoRATrainer:
    """Train LoRA adapters on Z-Image-Turbo transformer (kohya-style)."""

    def __init__(
        self,
        model_dir: str,
        dataset_dir: str,
        output_name: str = "my_lora",
        device: str = "cuda",
    ):
        self.model_dir = Path(model_dir)
        self.dataset_dir = dataset_dir
        self.output_name = output_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # State
        self._stop_requested = False
        self._training = False
        self.current_step = 0
        self.total_steps = 0
        self.current_loss = 0.0
        self.elapsed = 0.0
        self.eta = 0.0
        self.status_message = ""
        self.progress_callback = None   # fn(step, total, loss)
        self.status_callback = None     # fn(message)

    def _set_status(self, msg: str):
        self.status_message = msg
        if self.status_callback:
            self.status_callback(msg)

    def stop(self):
        """Request training to stop after current step."""
        self._stop_requested = True

    @property
    def is_training(self):
        return self._training

    @staticmethod
    def _sample_timesteps_sigmoid(batch_size: int, device: torch.device) -> torch.Tensor:
        """Sigmoid timestep sampling (ostris style) — biases toward middle timesteps."""
        t = torch.sigmoid(torch.randn(batch_size, device=device))
        return t

    def train(
        self,
        steps: int = 2000,
        lr: float = 1e-4,
        rank: int = 16,
        lora_alpha: int | None = None,
        batch_size: int = 1,
        resolution: int = 512,
        gradient_accumulation: int = 1,
        save_every: int = 500,
        target_modules: list[str] | None = None,
        prefix_filter: str | None = "layers.",
        use_deturbo: bool = False,
        caption_dropout: float = 0.0,
        timestep_sampling: str = "sigmoid",
        noise_offset: float = 0.0,
        module_dropout: float = 0.0,
        rank_dropout: float = 0.0,
        differential_guidance: float = 0.0,
    ) -> str:
        """Run LoRA training. Returns path to saved LoRA."""
        if lora_alpha is None:
            lora_alpha = rank
        from diffusers import AutoencoderKL
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from zit_config import ZIMAGE_TURBO_DIR, LORAS_DIR

        self._stop_requested = False
        self._training = True
        self.current_step = 0
        self.total_steps = steps
        self.current_loss = 0.0

        if target_modules is None:
            target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",
                "w1", "w2", "w3",
                "adaLN_modulation.0",
            ]

        model_path = self.model_dir / ZIMAGE_TURBO_DIR
        output_dir = self.model_dir / LORAS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.output_name}.safetensors"

        try:
            # --- Load dataset ---
            self._set_status("Loading dataset...")
            logger.info("Loading dataset from %s...", self.dataset_dir)
            dataset = ImageCaptionDataset(self.dataset_dir, resolution=resolution)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True,
            )

            # --- Load VAE (frozen, float32 for precision) ---
            self._set_status("Loading VAE...")
            logger.info("Loading VAE...")
            vae = AutoencoderKL.from_pretrained(
                str(model_path / "vae"), torch_dtype=torch.float32,
            ).to(self.device)
            vae.eval()
            vae.requires_grad_(False)
            vae_scale = vae.config.scaling_factor

            # --- Load text encoder (frozen) ---
            self._set_status("Loading text encoder...")
            logger.info("Loading text encoder...")
            text_encoder = AutoModelForCausalLM.from_pretrained(
                str(model_path / "text_encoder"), torch_dtype=torch.bfloat16,
            ).to(self.device)
            text_encoder.eval()
            text_encoder.requires_grad_(False)

            tokenizer_dir = model_path / "tokenizer"
            if not tokenizer_dir.exists():
                tokenizer_dir = model_path / "text_encoder"
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

            # --- Load transformer (BF16, base model without ControlNet) ---
            from videox_models.z_image_transformer2d import ZImageTransformer2DModel

            deturbo_path = self.model_dir / "Z-Image-De-Turbo" / "transformer"
            if use_deturbo and deturbo_path.exists():
                self._set_status("Loading De-Turbo transformer...")
                logger.info("Loading De-Turbo transformer (no adapter needed)...")
                transformer = ZImageTransformer2DModel.from_pretrained(
                    str(deturbo_path),
                    torch_dtype=torch.bfloat16,
                ).to(self.device)
            else:
                self._set_status("Loading transformer...")
                logger.info("Loading transformer (BF16 for training)...")
                transformer = ZImageTransformer2DModel.from_pretrained(
                    str(model_path / "transformer"),
                    torch_dtype=torch.bfloat16,
                ).to(self.device)

            # Freeze transformer weights but keep train mode (ostris convention)
            # train() keeps Dropout active; requires_grad_(False) prevents base weight updates
            transformer.train()
            transformer.requires_grad_(False)

            # --- Merge training adapter (de-distillation) — skip if using De-Turbo ---
            adapter_dir = self.model_dir / "training_adapter"
            if use_deturbo and deturbo_path.exists():
                logger.info("Using De-Turbo model — skipping adapter merge")
                adapter_dir = None  # skip adapter
            adapter_file = adapter_dir / "zimage_turbo_training_adapter_v2.safetensors" if adapter_dir else None
            if adapter_file is not None and not adapter_file.exists():
                adapter_file = adapter_dir / "zimage_turbo_training_adapter_v1.safetensors"
            if adapter_file is not None and adapter_file.exists():
                self._set_status("Merging training adapter (de-distill)...")
                logger.info("Loading training adapter: %s", adapter_file.name)
                from safetensors.torch import load_file as safe_load
                adapter_sd = safe_load(str(adapter_file), device=str(self.device))
                # Adapter is a LoRA — merge A*B into base weights
                lora_pairs = {}
                for key, tensor in adapter_sd.items():
                    clean_key = key.removeprefix("diffusion_model.")
                    if ".lora_A." in clean_key:
                        module_path = clean_key.split(".lora_A.")[0]
                        lora_pairs.setdefault(module_path, {})["A"] = tensor.to(torch.bfloat16)
                    elif ".lora_B." in clean_key:
                        module_path = clean_key.split(".lora_B.")[0]
                        lora_pairs.setdefault(module_path, {})["B"] = tensor.to(torch.bfloat16)
                merged = 0
                for module_path, pair in lora_pairs.items():
                    if "A" not in pair or "B" not in pair:
                        continue
                    try:
                        target = transformer
                        for part in module_path.split("."):
                            target = getattr(target, part)
                        with torch.no_grad():
                            target.weight.data += (pair["B"] @ pair["A"]).to(target.weight.dtype)
                        merged += 1
                    except AttributeError:
                        logger.warning("Adapter module not found: %s", module_path)
                del adapter_sd
                logger.info("Training adapter merged: %d layers", merged)
            elif adapter_dir is not None:
                logger.warning("No training adapter found at %s — training without de-distillation", adapter_dir)

            # --- Apply kohya-style LoRA modules ---
            self._set_status("Applying LoRA (kohya-style)...")
            logger.info("Applying LoRA (kohya-style, rank=%d, alpha=%d, targets=%s)...",
                        rank, lora_alpha, target_modules)

            lora_modules, lora_hooks = apply_lora_modules(
                transformer, rank=rank, alpha=lora_alpha,
                target_names=target_modules,
                device=self.device, dtype=torch.bfloat16,
                prefix_filter=prefix_filter,
                module_dropout=module_dropout,
                rank_dropout=rank_dropout,
            )

            trainable = sum(
                p.numel() for _, lora in lora_modules
                for p in lora.parameters()
            )
            total_params = sum(p.numel() for p in transformer.parameters())
            logger.info("LoRA modules: %d, Trainable: %d / %d (%.2f%%)",
                        len(lora_modules), trainable, total_params,
                        100 * trainable / total_params)

            # --- Optimizer (only LoRA params) ---
            lora_params = []
            for _, lora in lora_modules:
                lora_params.extend(lora.parameters())

            optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-2, eps=1e-6)

            # --- Pre-encode captions if dataset is small (saves time) ---
            self._set_status("Pre-encoding captions...")
            logger.info("Pre-encoding %d captions...", len(dataset.samples))
            caption_cache = {}
            all_captions = list(set(cap for _, cap in dataset.samples))
            for cap in all_captions:
                with torch.no_grad():
                    embeds = self._encode_text([cap], tokenizer, text_encoder, self.device)
                    caption_cache[cap] = embeds[0].detach()

            # Also encode empty caption for dropout
            if caption_dropout > 0:
                with torch.no_grad():
                    empty_embeds = self._encode_text([""], tokenizer, text_encoder, self.device)
                    caption_cache[""] = empty_embeds[0].detach()

            # Free text encoder after pre-encoding
            del text_encoder
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Text encoder freed (captions pre-encoded)")

            # --- Training loop ---
            self._set_status("Training...")
            logger.info("Starting training: %d steps, lr=%.1e, rank=%d, alpha=%d, res=%d, ts=%s",
                        steps, lr, rank, lora_alpha, resolution, timestep_sampling)
            start_time = time.time()
            _prev_log_time, _prev_log_step = start_time, 0
            _ema_step_time = 0.0
            data_iter = iter(dataloader)
            running_loss = 0.0

            import random

            for step in range(1, steps + 1):
                if self._stop_requested:
                    logger.info("Training stopped by user at step %d", step)
                    break

                # Get batch (loop over dataset)
                try:
                    images, captions = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    images, captions = next(data_iter)

                images = images.to(self.device, dtype=torch.float32)

                # --- Encode image to latents (B,C,H,W) ---
                with torch.no_grad():
                    raw_latents = vae.encode(images).latent_dist.sample()
                    # ostris formula: scaling_factor * (latents - shift_factor)
                    vae_shift = getattr(vae.config, 'shift_factor', 0.0) or 0.0
                    latents = vae_scale * (raw_latents - vae_shift)
                    latents = latents.to(torch.bfloat16)

                # --- Get text embeddings (from cache, with optional dropout) ---
                prompt_embeds = []
                for cap in captions:
                    if caption_dropout > 0 and random.random() < caption_dropout:
                        prompt_embeds.append(caption_cache[""])
                    else:
                        prompt_embeds.append(caption_cache[cap])

                # --- Flow matching noise (B,C,H,W) — no frame dim yet (ostris order) ---
                noise = torch.randn_like(latents)
                if noise_offset > 0:
                    # Per-channel noise offset (ostris convention)
                    noise += noise_offset * torch.randn(
                        latents.shape[0], latents.shape[1], 1, 1,
                        device=self.device, dtype=latents.dtype,
                    )

                if timestep_sampling == "sigmoid":
                    t = self._sample_timesteps_sigmoid(latents.shape[0], self.device)
                else:
                    t = torch.rand(latents.shape[0], device=self.device)
                t = t.to(torch.bfloat16)

                # Timestep: ostris uses (1-t)*1000 scale
                timestep_1000 = (1 - t) * 1000.0

                # add_noise: t_01 = timestep/1000, noisy = (1-t_01)*x + t_01*noise
                t_01 = (timestep_1000 / 1000.0).view(-1, 1, 1, 1)
                noisy_latents = (1.0 - t_01) * latents + t_01 * noise

                # Loss target (B,C,H,W)
                target = (noise - latents).detach()

                # --- Forward pass ---
                # Add frame dim AFTER add_noise (ostris order)
                noisy_latents = noisy_latents.unsqueeze(2)  # (B,C,H,W) → (B,C,1,H,W)

                # Model timestep: ostris inverts: (1000 - timestep) / 1000
                timestep_model = (1000.0 - timestep_1000) / 1000.0

                # Match inference format: List of individual tensors
                x_list = list(noisy_latents.unbind(dim=0))  # List[(C,1,H,W)]
                cap_list = prompt_embeds  # List[Tensor(seq_len, dim)]

                # Full forward with LoRA hooks active
                output = transformer(
                    x=x_list,
                    t=timestep_model,
                    cap_feats=cap_list,
                )
                model_pred = output[0] if isinstance(output, (tuple, list)) else output
                model_pred = model_pred.squeeze(2)  # Remove frame dim: (B,C,1,H,W) → (B,C,H,W)

                # Negate output (ostris convention for ZImage flow matching)
                model_pred = -model_pred

                # Differential guidance: adjust target toward model prediction
                # target = noise_pred + scale * (target - noise_pred)
                if differential_guidance > 0:
                    with torch.no_grad():
                        target = model_pred.detach() + differential_guidance * (
                            target - model_pred.detach()
                        )

                # --- Loss ---
                loss = F.mse_loss(model_pred.float(), target.float())

                if not torch.isfinite(loss):
                    logger.warning("Step %d: NaN/inf loss detected, skipping", step)
                    optimizer.zero_grad()
                    continue

                loss = loss / gradient_accumulation
                loss.backward()

                if step % gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                    has_nan = False
                    for p in lora_params:
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            has_nan = True
                            break
                    if has_nan:
                        logger.warning("Step %d: NaN gradient detected, skipping optimizer step", step)
                        optimizer.zero_grad()
                        continue
                    optimizer.step()
                    optimizer.zero_grad()

                # Tracking
                running_loss = 0.9 * running_loss + 0.1 * loss.item() * gradient_accumulation
                self.current_step = step
                self.current_loss = running_loss
                self.elapsed = time.time() - start_time

                # EMA-smoothed step time for stable ETA
                now = time.time()
                step_time = self.elapsed / step if step == 1 else (now - _prev_log_time) / max(step - _prev_log_step, 1)
                _ema_step_time = step_time if step == 1 else 0.3 * step_time + 0.7 * _ema_step_time
                self.eta = _ema_step_time * (steps - step)

                if self.progress_callback:
                    self.progress_callback(step, steps, running_loss)

                if step % 50 == 0 or step == 1:
                    _prev_log_time, _prev_log_step = now, step
                    elapsed = self.elapsed
                    eta = self.eta
                    total = elapsed + eta
                    e_m, e_s = int(elapsed) // 60, int(elapsed) % 60
                    eta_m, eta_s = int(eta) // 60, int(eta) % 60
                    t_m, t_s = int(total) // 60, int(total) % 60
                    logger.info("Step %d/%d  loss=%.4f  elapsed=%dm%02ds  eta=%dm%02ds  total=%dm%02ds",
                                step, steps, running_loss, e_m, e_s, eta_m, eta_s, t_m, t_s)

                # Save checkpoint
                if save_every > 0 and step % save_every == 0 and step < steps:
                    ckpt_path = output_dir / f"{self.output_name}_step{step}.safetensors"
                    save_kohya_lora(lora_modules, str(ckpt_path),
                                    alpha=lora_alpha, rank=rank)
                    logger.info("Checkpoint saved: %s", ckpt_path.name)

            # --- Save final LoRA ---
            self._set_status("Saving LoRA...")
            save_kohya_lora(lora_modules, str(output_path),
                            alpha=lora_alpha, rank=rank)
            elapsed = time.time() - start_time
            e_m, e_s = int(elapsed) // 60, int(elapsed) % 60
            logger.info("Training complete: %d steps in %dm%02ds → %s", step, e_m, e_s, output_path.name)

            return str(output_path)

        except Exception as e:
            logger.error("Training failed: %s", e)
            import traceback
            traceback.print_exc()
            raise
        finally:
            self._training = False
            logger.info("Training process will exit — GPU memory released automatically")

    @staticmethod
    def _encode_text(captions, tokenizer, text_encoder, device, max_length=512):
        """Encode captions to text embeddings (matching pipeline's method)."""
        formatted = []
        for caption in captions:
            messages = [{"role": "user", "content": caption}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            formatted.append(text)

        tokens = tokenizer(
            formatted, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt",
        )

        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device).bool()

        outputs = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-2]

        embeds_list = []
        for i in range(len(hidden)):
            embeds_list.append(hidden[i][attention_mask[i]])

        return embeds_list

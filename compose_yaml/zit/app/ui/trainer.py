"""LoRA training for Z-Image-Turbo — flow matching objective.

Trains LoRA adapters on the transformer using peft. Supports any use case
(face, style, object, etc.) — just provide images + captions.

Training runs in a **separate process** (like the inference worker) so that
stopping it via process.kill() instantly frees all GPU memory.
"""

import gc
import logging
import multiprocessing as _mp
import os
import time
from pathlib import Path

import torch
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

        # To tensor: [0, 255] → [-1, 1]
        import numpy as np
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW

        return tensor, caption


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
    """Train LoRA adapters on Z-Image-Turbo transformer."""

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
    ) -> str:
        """Run LoRA training. Returns path to saved LoRA."""
        if lora_alpha is None:
            lora_alpha = 1
        from peft import LoraConfig, get_peft_model
        from diffusers import AutoencoderKL
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from zit_config import ZIMAGE_TURBO_DIR, LORAS_DIR

        self._stop_requested = False
        self._training = True
        self.current_step = 0
        self.total_steps = steps
        self.current_loss = 0.0

        if target_modules is None:
            target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

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
            self._set_status("Loading transformer...")
            logger.info("Loading transformer (BF16 for training)...")
            from videox_models.z_image_transformer2d import ZImageTransformer2DModel

            transformer = ZImageTransformer2DModel.from_pretrained(
                str(model_path / "transformer"),
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            transformer.eval()

            # --- Apply LoRA ---
            self._set_status("Applying LoRA...")
            logger.info("Applying LoRA (rank=%d, targets=%s)...", rank, target_modules)
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.0,
                bias="none",
            )
            transformer = get_peft_model(transformer, lora_config)
            transformer.train()

            trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in transformer.parameters())
            logger.info("Trainable: %d / %d (%.2f%%)", trainable, total_params, 100 * trainable / total_params)

            # --- Optimizer ---
            optimizer = torch.optim.AdamW(
                [p for p in transformer.parameters() if p.requires_grad],
                lr=lr, weight_decay=1e-2,
            )

            # --- Training loop ---
            self._set_status("Training...")
            logger.info("Starting training: %d steps, lr=%.1e, rank=%d, res=%d",
                        steps, lr, rank, resolution)
            start_time = time.time()
            _prev_log_time, _prev_log_step = start_time, 0
            _ema_step_time = 0.0
            data_iter = iter(dataloader)
            running_loss = 0.0

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

                # --- Encode image to latents ---
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample() * vae_scale
                    latents = latents.to(torch.bfloat16)

                # --- Encode text ---
                with torch.no_grad():
                    prompt_embeds = self._encode_text(
                        captions, tokenizer, text_encoder, self.device,
                    )

                # --- Add frame dimension: (B,C,H,W) → (B,C,1,H,W) ---
                latents = latents.unsqueeze(2)

                # --- Flow matching noise ---
                noise = torch.randn_like(latents)
                t = torch.rand(latents.shape[0], device=self.device, dtype=torch.bfloat16)

                t_expanded = t.view(-1, 1, 1, 1, 1)
                noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
                target = noise - latents

                # --- Forward pass ---
                timestep = t * 1000.0

                max_len = max(e.shape[0] for e in prompt_embeds)
                cap_dim = prompt_embeds[0].shape[-1]
                cap_feats = torch.zeros(
                    len(prompt_embeds), max_len, cap_dim,
                    device=self.device, dtype=torch.bfloat16,
                )
                for i, emb in enumerate(prompt_embeds):
                    cap_feats[i, :emb.shape[0]] = emb

                output = transformer(
                    x=noisy_latents,
                    t=timestep,
                    cap_feats=cap_feats,
                )
                model_pred = output[0] if isinstance(output, (tuple, list)) else output

                # --- Loss ---
                loss = F.mse_loss(model_pred.float(), target.float())

                if not torch.isfinite(loss):
                    logger.warning("Step %d: NaN/inf loss detected, skipping", step)
                    optimizer.zero_grad()
                    continue

                loss = loss / gradient_accumulation
                loss.backward()

                if step % gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in transformer.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                    has_nan = False
                    for p in transformer.parameters():
                        if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
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
                    self._save_lora(transformer, str(ckpt_path),
                                    lora_alpha=lora_alpha, rank=rank)
                    logger.info("Checkpoint saved: %s", ckpt_path.name)

            # --- Save final LoRA ---
            self._set_status("Saving LoRA...")
            self._save_lora(transformer, str(output_path),
                            lora_alpha=lora_alpha, rank=rank)
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
            # Process exit will free all GPU memory — minimal cleanup only
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

    @staticmethod
    def _save_lora(peft_model, output_path: str, lora_alpha: int | None = None,
                   rank: int | None = None):
        """Save only LoRA weights as safetensors with metadata."""
        from safetensors.torch import save_file

        state_dict = {}
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if clean_name.startswith("base_model.model."):
                    clean_name = clean_name[len("base_model.model."):]
                state_dict[clean_name] = param.data.cpu()

        metadata = {}
        if lora_alpha is not None:
            metadata["lora_alpha"] = str(lora_alpha)
        if rank is not None:
            metadata["rank"] = str(rank)

        save_file(state_dict, output_path, metadata=metadata)
        size_mb = os.path.getsize(output_path) / 1024**2
        logger.info("LoRA saved: %s (%.1f MB, %d tensors)", output_path, size_mb, len(state_dict))

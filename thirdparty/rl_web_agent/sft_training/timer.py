import time
from typing import Any, Optional

from lightning import Callback, LightningModule, Trainer

try:
    import torch
except Exception:
    torch = None


class LogPerformanceCallback(Callback):
    def __init__(self):
        super().__init__()
        self._batch_t0: Optional[float] = None
        self._bwd_t0: Optional[float] = None
        self._step_t0: Optional[float] = None
        self._val_t0: Optional[float] = None
        self._fwd_logged: bool = False
        # optional GPU-timing events for the optimizer step
        self._step_ev_start = None
        self._step_ev_end = None

    # ---- utils ----
    def _sync(self, pl_module: LightningModule) -> None:
        if torch is None:
            return
        dev = getattr(pl_module, "device", None)
        if dev is not None and torch.cuda.is_available() and getattr(dev, "type", "") == "cuda":
            torch.cuda.synchronize(dev)

    # -------------------- TRAIN --------------------
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int) -> None:
        print("on_train_batch_start")
        self._batch_t0 = time.perf_counter()
        self._fwd_logged = False
        self._bwd_t0 = None
        self._step_t0 = None
        self._step_ev_start = None
        self._step_ev_end = None

    def on_before_backward(self, trainer: Trainer, pl_module: LightningModule, loss) -> None:
        print("on_before_backward")
        # forward time: from batch start to right before backward
        if self._batch_t0 is not None and not self._fwd_logged:
            self._sync(pl_module)
            fwd_time = time.perf_counter() - self._batch_t0
            pl_module.log("train/forward_time_seconds", fwd_time, on_step=True, on_epoch=False, rank_zero_only=True)
            self._fwd_logged = True
        self._bwd_t0 = time.perf_counter()

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print("on_after_backward")
        if self._bwd_t0 is not None:
            self._sync(pl_module)
            bwd_time = time.perf_counter() - self._bwd_t0
            pl_module.log("train/backward_time_seconds", bwd_time, on_step=True, on_epoch=False, rank_zero_only=True)
            self._bwd_t0 = None

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Any) -> None:
        print("on_before_optimizer_step")
        # wall-clock timer (CPU)
        self._step_t0 = time.perf_counter()
        # optional GPU timer (more accurate for CUDA work)
        if torch is not None and torch.cuda.is_available():
            self._step_ev_start = torch.cuda.Event(enable_timing=True)
            self._step_ev_end = torch.cuda.Event(enable_timing=True)
            self._step_ev_start.record()

    def on_before_zero_grad(self, trainer: Trainer, pl_module: LightningModule, optimizer: Any) -> None:
        print("on_before_zero_grad")
        # NeMo/Lightning call this after optimizer.step(); close the step timer here
        if self._step_t0 is not None:
            # prefer GPU timing if available
            if self._step_ev_start is not None and self._step_ev_end is not None:
                self._step_ev_end.record()
                torch.cuda.synchronize()  # ensure end is recorded
                step_s = self._step_ev_start.elapsed_time(self._step_ev_end) / 1000.0
            else:
                self._sync(pl_module)
                step_s = time.perf_counter() - self._step_t0

            pl_module.log("train/step_time_seconds", step_s, on_step=True, on_epoch=False, rank_zero_only=True)
            self._step_t0 = None
            self._step_ev_start = None
            self._step_ev_end = None

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        print("on_train_batch_end")
        # full batch time
        if self._batch_t0 is not None:
            self._sync(pl_module)
            batch_time = time.perf_counter() - self._batch_t0
            pl_module.log("train/batch_time_seconds", batch_time, on_step=True, on_epoch=False, rank_zero_only=True)
        # reset
        self._batch_t0 = None
        self._fwd_logged = False

    # -------------------- VALIDATION --------------------
    def on_validation_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        print("on_validation_batch_start")
        self._val_t0 = time.perf_counter()

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        print("on_validation_batch_end")
        if self._val_t0 is not None:
            self._sync(pl_module)
            val_time = time.perf_counter() - self._val_t0
            pl_module.log("validation/batch_time_seconds", val_time, on_step=True, on_epoch=False, rank_zero_only=True)
            self._val_t0 = None

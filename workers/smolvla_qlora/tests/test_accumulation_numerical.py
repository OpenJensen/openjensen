"""Opt-in CPU optimizer tests; no pretrained model, dataset download or CUDA claim."""

import copy
import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FIREBIRD_TEST_ACCUMULATION_CPU") != "1",
    reason="Requires the existing Torch/Accelerate CPU environment and explicit opt-in",
)


@pytest.fixture
def runtime():
    import torch
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, GradientState
    from accelerate.utils import GradientAccumulationPlugin

    torch.set_num_threads(1)
    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=2, sync_with_dataloader=False
        ),
        step_scheduler_with_optimizer=False,
    )
    yield torch, accelerator
    AcceleratorState._reset_state(True)
    GradientState._reset_state()


def policy(torch):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, batch):
            loss = (self.linear(batch["state"]) - batch["action"]).square().mean()
            return loss, {"mse": loss.detach()}

    return Policy()


def samples(torch):
    return {
        "state": torch.tensor([[0.1, 0.5], [0.3, -0.4], [0.9, 0.2], [-0.2, 0.7], [0.8, -0.6]]),
        "action": torch.tensor([[1.0], [-1.0], [0.3], [2.0], [-0.8]]),
    }


def split(batch):
    return [{key: value[a:b] for key, value in batch.items()} for a, b in [(0, 3), (3, 5)]]


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
def test_actual_accelerate_update_matches_large_batch_with_short_tail(runtime, optimizer_name):
    from firebird_vla.accumulation import NativeAccumulation, make_contract

    torch, accelerator = runtime
    torch.manual_seed(81)
    accumulated = policy(torch)
    reference = copy.deepcopy(accumulated)
    optimizer_cls = getattr(torch.optim, optimizer_name)
    kwargs = {"lr": 0.03, "weight_decay": 0.01}
    optimizer = optimizer_cls(accumulated.parameters(), **kwargs)
    reference_optimizer = optimizer_cls(reference.parameters(), **kwargs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, 1, gamma=0.9)
    accumulated, optimizer, scheduler = accelerator.prepare(accumulated, optimizer, scheduler)
    cursor = NativeAccumulation(make_contract(3, 2), 5)
    batch = samples(torch)
    clip_calls = []
    original_clip = accelerator.clip_grad_norm_

    def clip(*args, **kw):
        clip_calls.append(cursor.consumed)
        return original_clip(*args, **kw)

    accelerator.clip_grad_norm_ = clip
    for update in range(3):
        before = copy.deepcopy(accelerator.unwrap_model(accumulated).state_dict())
        before_lr = scheduler.get_last_lr()
        metrics = SimpleNamespace()
        for i, microbatch in enumerate(split(batch)):
            cursor.update(
                metrics,
                accumulated,
                microbatch,
                optimizer,
                0.4,
                accelerator=accelerator,
                lr_scheduler=scheduler,
            )
            if i == 0:
                for name, value in accumulated.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                assert scheduler.get_last_lr() == before_lr
                assert cursor.updates == update
                with pytest.raises(ValueError, match="complete update window"):
                    cursor.record()
        loss, _ = reference(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.4)
        reference_optimizer.step()
        reference_optimizer.zero_grad()
        reference_scheduler.step()
        for actual, expected in zip(accumulated.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        assert scheduler.get_last_lr() == reference_scheduler.get_last_lr()
        assert metrics.loss == pytest.approx(float(loss.detach()), rel=2e-6)
    assert clip_calls == [2, 4, 6]
    assert cursor.record()["consumed_examples"] == 15


def test_actual_checkpoint_state_restores_next_weighted_update(runtime, tmp_path):
    from firebird_vla.accumulation import NativeAccumulation, make_contract, validate_progress

    torch, accelerator = runtime
    torch.manual_seed(17)
    model = policy(torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    contract = make_contract(3, 2)
    cursor = NativeAccumulation(contract, 5)
    for batch in split(samples(torch)):
        cursor.update(
            SimpleNamespace(),
            model,
            batch,
            optimizer,
            1,
            accelerator=accelerator,
            lr_scheduler=scheduler,
        )
    path = tmp_path / "state.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "progress": cursor.record(),
            "rng": torch.get_rng_state(),
        },
        path,
    )
    saved = torch.load(path, weights_only=True)
    for batch in split(samples(torch)):
        cursor.update(
            SimpleNamespace(),
            model,
            batch,
            optimizer,
            1,
            accelerator=accelerator,
            lr_scheduler=scheduler,
        )
    expected = copy.deepcopy(model.state_dict())
    # New model and optimizer objects; no reference to the first model's tensors.
    restored = policy(torch)
    restored.load_state_dict(saved["model"])
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.02)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_opt, 1, gamma=0.8)
    restored_opt.load_state_dict(saved["optimizer"])
    restored_scheduler.load_state_dict(saved["scheduler"])
    restored, restored_opt, restored_scheduler = accelerator.prepare(
        restored, restored_opt, restored_scheduler
    )
    validate_progress(saved["progress"], contract, frames=5, updates=1, consumed=2)
    resumed = NativeAccumulation(contract, 5, updates=1, consumed=2)
    torch.set_rng_state(saved["rng"])
    for batch in split(samples(torch)):
        resumed.update(
            SimpleNamespace(),
            restored,
            batch,
            restored_opt,
            1,
            accelerator=accelerator,
            lr_scheduler=restored_scheduler,
        )
    for name, actual in restored.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert resumed.record() == cursor.record()


def test_nonfinite_native_gradient_never_steps_scheduler_or_publishes(runtime):
    from firebird_vla.accumulation import NativeAccumulation, make_contract

    torch, accelerator = runtime
    model = policy(torch)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    before = copy.deepcopy(model.state_dict())
    scheduler = SimpleNamespace(step=lambda: pytest.fail("scheduler advanced on rejected update"))
    cursor = NativeAccumulation(make_contract(3, 2), 5)
    first, second = split(samples(torch))
    cursor.update(
        SimpleNamespace(),
        model,
        first,
        optimizer,
        1,
        accelerator=accelerator,
        lr_scheduler=scheduler,
    )
    for p in model.parameters():
        p.grad.fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="Non-finite accumulated"):
        cursor.update(
            SimpleNamespace(),
            model,
            second,
            optimizer,
            1,
            accelerator=accelerator,
            lr_scheduler=scheduler,
        )
    assert cursor.updates == 0
    with pytest.raises(ValueError, match="complete update window"):
        cursor.record()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_actual_cpu_grad_scaler_rejects_nonfinite_smol_window(runtime):
    from firebird_vla.train import optimizer_step

    torch, _ = runtime
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    scaler.scale(parameter.square().mean()).backward()
    parameter.grad.fill_(float("inf"))
    before = parameter.detach().clone()
    _, updated = optimizer_step([parameter], optimizer, scaler, 1)
    assert not updated
    assert scaler.get_scale() == 4
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)


def test_native_sampler_resume_keeps_actual_order_across_epochs(runtime):
    from itertools import islice

    from lerobot.datasets.sampler import EpisodeAwareSampler
    from lerobot.utils.utils import cycle
    from torch.utils.data import DataLoader

    torch, accelerator = runtime

    def loader(start=None):
        sampler = EpisodeAwareSampler([0], [5], shuffle=True, seed=42)
        if start:
            sampler.load_state_dict(start)
        return DataLoader(
            list(range(5)),
            sampler=sampler,
            batch_size=3,
            generator=torch.Generator().manual_seed(42),
        )

    original = accelerator.prepare_data_loader(loader())
    expected = [batch.tolist() for batch in islice(cycle(original), 10)]
    resumed = loader({"epoch": 1, "start_index": 3})
    # After one A=3 window, resume inside epoch 1 and cross several more epochs.
    from firebird_vla.accumulation import bind_resumed_sampler_epoch

    bind_resumed_sampler_epoch(resumed.sampler, consumed=3, frames=5, batch_size=3)
    resumed = accelerator.prepare_data_loader(resumed)
    actual = [batch.tolist() for batch in islice(cycle(resumed), 7)]
    assert actual == expected[3:]

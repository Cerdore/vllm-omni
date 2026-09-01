# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sliding-window shape math and history-block packing for MiniMax H3."""

from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# --------------------------------------------------------------------------- #
# Overlap / window frame math
# --------------------------------------------------------------------------- #
def _frames_from_latent_t(out_t: int) -> int:
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        MINIMAX_H3_SHAPE_PLANNER,
    )

    return MINIMAX_H3_SHAPE_PLANNER.frame_count_from_video_latent_t(out_t)


def test_windowing_plan_30s_is_two_windows_with_overlap_drop():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    plan = _resolve_minimax_h3_windowing(
        duration=30.0,
        fps=24,
        num_segments=None,  # auto-activates because duration > 15
        overlap_frames=None,
        window_duration=None,
    )
    assert plan is not None
    assert plan.is_active
    # Default window is the native ceiling (15 s -> 362 frames = 17*21 + 5).
    assert plan.window_num_frames == 362
    # Default overlap is the smallest span on the latent grid: 2 latents =
    # 5 frames. overlap_latent_t must satisfy (wt - overlap_latent_t) % 15 == 0
    # so the concatenated latent stays on the VAE's 5n+2 grid AND each
    # continuation window contributes an integral number of frames and
    # audio latents.
    assert plan.overlap_latent_t == 2
    assert plan.overlap_frames == 5
    # Audio overlap is derived from the same wall-clock span as the video
    # contribution, not converted independently from overlap_frames.
    assert plan.overlap_audio_t == 8
    # 30 s rounds to two windows.
    assert plan.num_windows == 2
    # total_num_frames is what the concatenated latent actually decodes to:
    # frames(107 + 105) = 719 = 362 + 357.
    assert plan.total_num_frames == 719


def test_windowing_contribution_is_av_exact():
    """Every continuation window must add the same wall-clock span of video
    and audio, or the A/V desync accumulates per window."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    for window_duration in (8.0, 12.0, 15.0):
        plan = _resolve_minimax_h3_windowing(
            duration=45.0,
            fps=24,
            num_segments=3,
            overlap_frames=None,
            window_duration=window_duration,
        )
        wt = _video_latent_t(plan.window_num_frames)
        contributed_frames = _frames_from_latent_t(wt + (wt - plan.overlap_latent_t)) - plan.window_num_frames
        video_seconds = contributed_frames / 24.0
        audio_seconds = (plan.window_audio_t - plan.overlap_audio_t) / 40.0
        assert video_seconds == audio_seconds, window_duration
        # And the plan's total matches what the latent decodes to.
        total_t = wt + (plan.num_windows - 1) * (wt - plan.overlap_latent_t)
        assert plan.total_num_frames == _frames_from_latent_t(total_t)


def test_windowing_overlap_snaps_to_latent_grid():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    # 100 requested frames -> 27 latents, which is off the wt=107 grid
    # (27 % 15 != 107 % 15); nearest valid is 32.
    plan = _resolve_minimax_h3_windowing(
        duration=30.0,
        fps=24,
        num_segments=2,
        overlap_frames=100,
        window_duration=None,
    )
    assert plan.overlap_latent_t == 32
    # An 8 s window (192 frames, wt=57) needs overlap_latent_t ≡ 57
    # (mod 15) = 12; the default 5-frame request (2 latents) is below the
    # lowest valid value and clamps up to 12.
    plan = _resolve_minimax_h3_windowing(
        duration=20.0,
        fps=24,
        num_segments=2,
        overlap_frames=None,
        window_duration=8.0,
    )
    assert plan.window_num_frames == 192
    assert plan.overlap_latent_t == 12
    assert plan.overlap_audio_t == 65
    assert plan.total_num_frames == 192 + 153


def test_windowing_explicit_num_segments():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    plan = _resolve_minimax_h3_windowing(
        duration=45.0,
        fps=24,
        num_segments=3,
        overlap_frames=58,
        window_duration=None,
    )
    assert plan.num_windows == 3
    # Window 0 contributes 362; windows 1-2 each contribute 306 frames
    # (90 latents); frames(107 + 180) = 974.
    assert plan.total_num_frames == 974


def test_windowing_inactive_for_single_window():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    # Within the native contract and no num_segments -> single window.
    assert (
        _resolve_minimax_h3_windowing(
            duration=8.0,
            fps=24,
            num_segments=None,
            overlap_frames=None,
            window_duration=None,
        )
        is None
    )
    # num_segments=1 is an explicit single-window request (no windowing).
    assert (
        _resolve_minimax_h3_windowing(
            duration=8.0,
            fps=24,
            num_segments=1,
            overlap_frames=None,
            window_duration=None,
        )
        is None
    )


def test_windowing_rejects_overlap_larger_than_window():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError):
        _resolve_minimax_h3_windowing(
            duration=30.0,
            fps=24,
            num_segments=2,
            overlap_frames=400,
            window_duration=None,
        )


def _video_latent_t(frame_count: int) -> int:
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        MINIMAX_H3_SHAPE_PLANNER,
    )

    return MINIMAX_H3_SHAPE_PLANNER.video_latent_t(frame_count)


def _audio_latent_t(duration_seconds: float) -> int:
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        MINIMAX_H3_SHAPE_PLANNER,
    )

    return MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(duration_seconds)


# --------------------------------------------------------------------------- #
# History-block packing (video_audio ref block)
# --------------------------------------------------------------------------- #
def test_history_video_audio_block_emits_frozen_rows():
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    latent_h, latent_w = 48, 80  # 768x1280-ish canvas at /16
    overlap_latent_t = 17
    overlap_audio_t = 93
    window_latent_t = 107
    window_audio_t = 603
    history_block = {
        "kind": "video_audio",
        "ref_audio_t": overlap_audio_t,
        "latent_t": overlap_latent_t,
        "latent_h": latent_h,
        "latent_w": latent_w,
    }
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=128,
        latent_t=window_latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=window_audio_t,
        ref_blocks=[history_block],
    )
    frame_rows = (latent_h // 2) * (latent_w // 2)
    ref_visual_rows = overlap_latent_t * frame_rows
    ref_audio_rows = overlap_audio_t * 2

    # Visual: history rows frozen, target rows updated.
    assert int(packed["update_mask"][:ref_visual_rows].sum()) == 0
    assert bool(packed["update_mask"][ref_visual_rows:].all())
    # Audio: history rows frozen, target rows updated.
    assert int(packed["audio_update_mask"][:ref_audio_rows].sum()) == 0
    assert bool(packed["audio_update_mask"][ref_audio_rows:].all())
    # The history block is advertised as a reference span.
    roles = [span["role"] for span in packed["video_spans"]]
    assert "reference" in roles
    assert roles[-1] == "target"


def test_audio_history_rows_round_trip_through_latent_tail():
    """Continuation windows pack the previous audio tail via pack_audio_latent."""
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_pack_audio_latent,
        minimax_h3_unpack_audio_tokens,
    )

    audio_latent = torch.arange(2 * 32 * 602).reshape(2, 32, 602).float()
    overlap_audio_t = 93
    tail = audio_latent[:, :, -overlap_audio_t:]
    rows = minimax_h3_pack_audio_latent(tail)
    assert rows.shape == (overlap_audio_t * 2, 32)
    restored = minimax_h3_unpack_audio_tokens(rows, audio_t=overlap_audio_t * 2, audio_channel=2)
    torch.testing.assert_close(restored, tail)


def test_video_history_rows_round_trip_through_latent_tail():
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
        minimax_h3_unpatchify_video_tokens,
    )

    latent = torch.arange(1 * 24 * 107 * 48 * 80).reshape(1, 24, 107, 48, 80).float()
    overlap_latent_t = 17
    tail = latent[:, :, -overlap_latent_t:, :, :]
    rows = minimax_h3_patchify_video_latent(tail, patch_size=(1, 2, 2))
    frame_rows = (48 // 2) * (80 // 2)
    assert rows.shape == (overlap_latent_t * frame_rows, 96)
    restored = minimax_h3_unpatchify_video_tokens(
        rows, latent_shape=(overlap_latent_t, 24, 40, 24), patch_size=(1, 2, 2)
    )
    torch.testing.assert_close(restored, tail)


def test_identity_anchor_plus_history_block_frozen_row_split():
    """Ref2VA continuation layout: [image identity-anchor, video_audio history] + target.

    Mirrors the ref_blocks the ref2va window loop assembles for a continuation
    window: a 1-frame image anchor (window 0's first frame) followed by the
    video+audio history block (17 latents / 93 audio latents when the overlap
    is 58 frames), then the target. t2va/fl2va windows take the first-frame
    path instead (see ``test_continuation_window_is_a_first_frame_request``).
    """
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    latent_h, latent_w = 48, 80
    frame_rows = (latent_h // 2) * (latent_w // 2)
    window_latent_t = 107
    window_audio_t = 603
    overlap_latent_t = 17
    overlap_audio_t = 93
    ref_blocks = [
        {"kind": "image", "latent_h": latent_h, "latent_w": latent_w},
        {
            "kind": "video_audio",
            "ref_audio_t": overlap_audio_t,
            "latent_t": overlap_latent_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
        },
    ]
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=128,
        latent_t=window_latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=window_audio_t,
        ref_blocks=ref_blocks,
    )
    # Anchor image (1 frame) + history video (overlap_latent_t frames) are the
    # frozen visual rows; the target window is the rest.
    anchor_rows = 1 * frame_rows
    history_video_rows = overlap_latent_t * frame_rows
    ref_visual_rows = anchor_rows + history_video_rows
    assert int(packed["update_mask"][:ref_visual_rows].sum()) == 0
    assert bool(packed["update_mask"][ref_visual_rows:].all())
    # Both reference spans are advertised; the target is last.
    roles = [span["role"] for span in packed["video_spans"]]
    assert roles.count("reference") == 1
    assert roles[-1] == "target"


# --------------------------------------------------------------------------- #
# Window-level bookkeeping for the first-frame continuation path
# --------------------------------------------------------------------------- #
def test_continuation_window_is_a_first_frame_request():
    """A continuation window is conditioned like a user fl2va request whose
    first frame is the handoff still, plus the request's own last frame when
    this is the final window."""
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        MINIMAX_H3_IMGVID_COND_ID,
        minimax_h3_packed_sequence,
    )
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _continuation_keyframes,
    )

    assert _continuation_keyframes(None) == [0]
    assert _continuation_keyframes([-1]) == [0, -1]
    # The layout of such a window is byte-identical to a standalone
    # first-frame (or first+last) fl2va request of the same size.
    for keyframes in ([0], [0, -1]):
        window = minimax_h3_packed_sequence(
            text_len=128,
            latent_t=107,
            latent_h=30,
            latent_w=52,
            audio_t=603,
            include_keyframe_cond=True,
            keyframe_frame_indices=_continuation_keyframes(keyframes[1:] or None),
            frame_count=362,
        )
        standalone = minimax_h3_packed_sequence(
            text_len=128,
            latent_t=107,
            latent_h=30,
            latent_w=52,
            audio_t=603,
            include_keyframe_cond=True,
            keyframe_frame_indices=keyframes,
            frame_count=362,
        )
        for key in ("input_ids", "update_mask", "img_position_ids", "token_tags", "cu_seqlens"):
            torch.testing.assert_close(window[key], standalone[key], rtol=0, atol=0)
        frame_rows = (30 // 2) * (52 // 2)
        cond_ids = window["input_ids"][128 : 128 + len(keyframes) * frame_rows]
        assert bool((cond_ids == MINIMAX_H3_IMGVID_COND_ID).all())
        # Every target video row is generated; only the stills are frozen.
        assert int(window["update_mask"].sum()) == 107 * frame_rows
        assert int((~window["update_mask"]).sum()) == len(keyframes) * frame_rows


def test_window_keyframes_split_first_and_last_across_windows():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _window_keyframe_indices,
    )

    # fl2va with both anchors: first frame pins window 0, last frame pins the
    # final window, middle windows carry only the handoff still.
    assert _window_keyframe_indices([0, -1], window_index=0, num_windows=3) == [0]
    assert _window_keyframe_indices([0, -1], window_index=1, num_windows=3) is None
    assert _window_keyframe_indices([0, -1], window_index=2, num_windows=3) == [-1]
    assert _window_keyframe_indices([-1], window_index=0, num_windows=2) is None
    assert _window_keyframe_indices([-1], window_index=1, num_windows=2) == [-1]
    assert _window_keyframe_indices([0], window_index=1, num_windows=2) is None
    # t2va has no keyframes anywhere.
    assert _window_keyframe_indices(None, window_index=0, num_windows=2) is None
    # A single window is untouched.
    assert _window_keyframe_indices([0, -1], window_index=0, num_windows=1) == [0, -1]


def test_window_trim_matches_plan_contribution():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
        _window_trim,
    )

    plan = _resolve_minimax_h3_windowing(
        duration=30.0, fps=24, num_segments=None, overlap_frames=None, window_duration=None
    )
    trim_frames, trim_samples = _window_trim(plan, sample_rate=32000)
    # Decoding a full continuation window and dropping the shared span must
    # leave exactly the planned contribution: 362 - 5 = 357 frames, 14.875 s
    # of audio (8 latents = 6400 samples dropped).
    assert plan.window_num_frames - trim_frames == 357
    assert trim_frames == 5
    assert trim_samples == plan.overlap_audio_t * 800 == 6400
    assert (plan.window_audio_t - plan.overlap_audio_t) * 800 == 14.875 * 32000
    # The handoff still is the frame the next window's frame 0 reproduces.
    assert plan.window_num_frames - trim_frames == plan.total_num_frames - plan.window_num_frames


def test_pin_overlap_audio_rows_freezes_both_channels_channel_major():
    """Audio rows are packed channel-major ([ch0 t0..T, ch1 t0..T]); pinning the
    first ``overlap_a`` steps must freeze those steps in BOTH channels, and the
    anchor must line up with the ascending ``~update`` order."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_pack_audio_latent
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _pin_overlap_audio_rows

    wa, overlap_a = 603, 93
    branch = SimpleNamespace(
        audio_update_mask_dev=torch.ones(2 * wa, dtype=torch.bool),
        audio_update_mask=torch.ones(2 * wa, dtype=torch.bool),
    )
    tail = torch.arange(2 * 32 * overlap_a, dtype=torch.float32).reshape(2, 32, overlap_a)
    tail_rows = minimax_h3_pack_audio_latent(tail)
    inputs = {"branch": branch, "audio_anchor": None}
    _pin_overlap_audio_rows(inputs, overlap_audio_rows=tail_rows, overlap_audio_steps=overlap_a)

    frozen = ~branch.audio_update_mask_dev
    expected = torch.zeros(2 * wa, dtype=torch.bool)
    expected[:overlap_a] = True  # channel 0, steps 0..overlap_a-1
    expected[wa : wa + overlap_a] = True  # channel 1, steps 0..overlap_a-1
    assert torch.equal(frozen, expected)
    # Anchor rows follow the frozen rows' ascending order: ch0 tail then ch1 tail.
    torch.testing.assert_close(inputs["audio_anchor"], tail_rows)


def test_pin_overlap_rows_ref2va_is_channel_major_for_audio():
    """The ref2va helper pins whole leading video frames and, for audio, the
    leading steps of BOTH channel blocks."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_pack_audio_latent
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _pin_overlap_rows

    frame_rows, latent_t, wa = 4, 10, 6
    overlap_frames, overlap_a = 2, 2
    branch = SimpleNamespace(
        update_mask_dev=torch.ones(latent_t * frame_rows, dtype=torch.bool),
        update_mask=torch.ones(latent_t * frame_rows, dtype=torch.bool),
        audio_update_mask_dev=torch.ones(2 * wa, dtype=torch.bool),
        audio_update_mask=torch.ones(2 * wa, dtype=torch.bool),
        static_kwargs={},
    )
    video_rows = torch.zeros(overlap_frames * frame_rows, 96)
    audio_rows = minimax_h3_pack_audio_latent(torch.zeros(2, 32, overlap_a))
    inputs: dict[str, Any] = {"branch": branch, "cond_anchor": None, "audio_anchor": None}
    _pin_overlap_rows(
        inputs,
        overlap_video_rows=video_rows,
        overlap_audio_rows=audio_rows,
        overlap_video_frames=overlap_frames,
        overlap_audio_steps=overlap_a,
        frame_rows=frame_rows,
    )
    frozen_video = ~branch.update_mask_dev
    assert int(frozen_video.sum()) == overlap_frames * frame_rows
    assert bool(frozen_video[: overlap_frames * frame_rows].all())
    frozen_audio = ~branch.audio_update_mask_dev
    expected = torch.zeros(2 * wa, dtype=torch.bool)
    expected[:overlap_a] = True
    expected[wa : wa + overlap_a] = True
    assert torch.equal(frozen_audio, expected)
    assert inputs["cond_anchor"].shape[0] == overlap_frames * frame_rows
    assert inputs["audio_anchor"].shape[0] == 2 * overlap_a


# --------------------------------------------------------------------------- #
# _generate_windowed plumbing on a fake pipeline (no model, no GPU)
# --------------------------------------------------------------------------- #
# A 64x32 canvas: latent 4x2, one token row per latent frame after (1, 2, 2) patching.
_FAKE_HEIGHT, _FAKE_WIDTH = 32, 64
_FAKE_LATENT_H, _FAKE_LATENT_W = _FAKE_HEIGHT // 16, _FAKE_WIDTH // 16
_FAKE_FRAME_ROWS = (_FAKE_LATENT_H // 2) * (_FAKE_LATENT_W // 2)


def _fake_image(value: int):
    from PIL import Image

    return Image.new("RGB", (4, 4), (value, value, value))


def _run_fake_windowed(*, task: str, keyframes: list[int] | None, image_values: list[int], text_encoder=object()):
    """Drive MiniMaxH3Pipeline._generate_windowed with stubbed model calls.

    Decoded frame ``t`` has constant pixel ``t / 1000`` so the handoff still is
    recognisable, and every visual-condition row carries its source image's
    pixel value so condition-row order is checkable.
    """
    from contextlib import contextmanager
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MiniMaxH3Pipeline,
        _resolve_minimax_h3_windowing,
    )
    from vllm_omni.diffusion.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER

    plan = _resolve_minimax_h3_windowing(
        duration=30.0, fps=24, num_segments=None, overlap_frames=None, window_duration=None
    )
    calls: dict[str, list] = {"encode_prompt": [], "build": [], "encode_image": []}

    def encode_prompt(*, task, prompt, images=None):
        calls["encode_prompt"].append((task, [img.getpixel((0, 0))[0] for img in (images or [])]))
        n = 24 + 100 * len(images or [])
        return torch.zeros(n, 8), torch.ones(n, dtype=torch.long)

    def build(**kw):
        branch = SimpleNamespace(
            audio_update_mask_dev=torch.ones(2 * kw["audio_t"], dtype=torch.bool),
            audio_update_mask=torch.ones(2 * kw["audio_t"], dtype=torch.bool),
        )
        inputs: dict[str, Any] = {"branch": branch, "audio_anchor": None}
        calls["build"].append((kw, inputs))
        return inputs

    def run_window_denoise(*, inputs, transformer, latent_t, latent_h, latent_w, audio_t, on_step=None):
        # Unpacked video latent is (B, 24, T, latent_h, latent_w); audio latent is (channels=2, 32, T).
        return torch.zeros(1, 24, latent_t, latent_h, latent_w), torch.zeros(2, 32, audio_t)

    def decode(video_latent, audio_latent, *, height, width):
        # Decoded video is (B, C, T, height, width) in [0, 1], cropped to the request canvas.
        frames = MINIMAX_H3_SHAPE_PLANNER.frame_count_from_video_latent_t(int(video_latent.shape[2]))
        video = torch.arange(frames, dtype=torch.float32).div(1000).view(1, 1, frames, 1, 1)
        return video.expand(1, 3, frames, height, width).clone(), torch.zeros(1, 2, int(audio_latent.shape[2]) * 800)

    def encode_image(image):
        # The handoff still must be a canvas-sized picture (PIL size is (width, height)).
        calls["encode_image"].append((image.getpixel((0, 0))[0], image.size))
        return torch.full((_FAKE_FRAME_ROWS, 96), float(image.getpixel((0, 0))[0]))

    @contextmanager
    def ctx(*args, **kwargs):
        yield SimpleNamespace(update=lambda: None)

    transformer = object()
    fake = SimpleNamespace(
        text_encoder=text_encoder,
        transformer=transformer,
        _transformer_for_task=lambda task: transformer,
        progress_bar=ctx,
        _resident_dit_layers_on_device=ctx,
        _component_on_device=ctx,
        _build_denoise_inputs=build,
        _run_window_denoise=run_window_denoise,
        decode=decode,
        video_vae=SimpleNamespace(encode_image=encode_image),
        encode_prompt=encode_prompt,
        device=torch.device("cpu"),
    )
    images = [_fake_image(v) for v in image_values]
    visual_condition = (
        torch.cat([torch.full((_FAKE_FRAME_ROWS, 96), float(v)) for v in image_values]) if image_values else None
    )
    request_text = (
        torch.zeros(24 + 100 * len(image_values), 8),
        torch.ones(24 + 100 * len(image_values), dtype=torch.long),
    )
    video, audio = MiniMaxH3Pipeline._generate_windowed(
        fake,
        task=task,
        text_embeddings=request_text[0],
        text_tags=request_text[1],
        seed=7,
        latent_t=plan.window_latent_t,
        latent_h=_FAKE_LATENT_H,
        latent_w=_FAKE_LATENT_W,
        audio_t=plan.window_audio_t,
        num_frames=plan.window_num_frames,
        num_steps=4,
        video_shift=5.0,
        audio_shift=3.0,
        base_schedule=None,
        visual_condition=visual_condition,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
        ref_blocks=None,
        visual_condition_shapes=[(1, _FAKE_LATENT_H, _FAKE_LATENT_W)] * len(image_values) or None,
        audio_condition_lengths=None,
        keyframe_frame_indices=keyframes,
        windowing=plan,
        prompt="a coast",
        images=images,
        height=_FAKE_HEIGHT,
        width=_FAKE_WIDTH,
    )
    return plan, calls, video, audio


def _text_len(kw) -> int:
    assert kw["text_embeddings"].shape[0] == kw["text_tags"].shape[0]
    return int(kw["text_embeddings"].shape[0])


def test_generate_windowed_t2va_hands_off_frame_357_as_a_first_frame_request():
    plan, calls, video, audio = _run_fake_windowed(task="t2va", keyframes=None, image_values=[])
    # Window 0 keeps the request text (no re-encode); window 1 is a first-frame
    # fl2va request around the decoded handoff frame 357 (pixel 0.357 -> 91).
    assert calls["encode_prompt"] == [("fl2va", [91])]
    assert calls["encode_image"] == [(91, (_FAKE_WIDTH, _FAKE_HEIGHT))]
    kw0, _ = calls["build"][0]
    kw1, inputs1 = calls["build"][1]
    assert kw0["keyframe_frame_indices"] is None and kw0["visual_condition"] is None
    assert kw0["seed"] == 7 and kw1["seed"] == 8
    # The text each window denoises with is the text encoded for that window:
    # the request's 24 tokens for window 0, the one-picture fl2va encoding after.
    assert _text_len(kw0) == 24 and _text_len(kw1) == 124
    assert kw1["keyframe_frame_indices"] == [0]
    assert kw1["visual_condition_shapes"] == [(1, _FAKE_LATENT_H, _FAKE_LATENT_W)]
    assert bool((kw1["visual_condition"] == 91.0).all())
    assert kw1["num_frames"] == plan.window_num_frames and kw1["latent_t"] == plan.window_latent_t
    # The previous audio tail is pinned into window 1 (both channels).
    assert int((~inputs1["branch"].audio_update_mask_dev).sum()) == 2 * plan.overlap_audio_t
    assert inputs1["audio_anchor"].shape[0] == 2 * plan.overlap_audio_t
    # Output: 362 + 357 frames; audio 15.075 s + (15.075 - 0.2) s.
    assert video.shape[2] == plan.total_num_frames == 719
    assert audio.shape[-1] == (2 * plan.window_audio_t - plan.overlap_audio_t) * 800
    # Frame 362 of the output is window 1's frame 5, i.e. the frame after the
    # handoff span; frames 0..361 are window 0 untouched.
    torch.testing.assert_close(video[0, 0, :362, 0, 0], torch.arange(362, dtype=torch.float32) / 1000)
    assert round(float(video[0, 0, 362, 0, 0]) * 1000) == 5


def test_generate_windowed_fl2va_keeps_user_keyframes_paired_with_their_text():
    # [0, -1]: window 0 anchors image 0 only (re-encoded with only that
    # picture); the final window anchors [handoff, image -1] in that order.
    _, calls, _, _ = _run_fake_windowed(task="fl2va", keyframes=[0, -1], image_values=[10, 20])
    assert calls["encode_prompt"] == [("fl2va", [10]), ("fl2va", [91, 20])]
    kw0, _ = calls["build"][0]
    kw1, _ = calls["build"][1]
    assert _text_len(kw0) == 124 and _text_len(kw1) == 224
    assert kw0["keyframe_frame_indices"] == [0]
    assert bool((kw0["visual_condition"] == 10.0).all()) and kw0["visual_condition"].shape[0] == _FAKE_FRAME_ROWS
    assert kw1["keyframe_frame_indices"] == [0, -1]
    assert kw1["visual_condition_shapes"] == [(1, _FAKE_LATENT_H, _FAKE_LATENT_W)] * 2
    assert bool((kw1["visual_condition"][:_FAKE_FRAME_ROWS] == 91.0).all())
    assert bool((kw1["visual_condition"][_FAKE_FRAME_ROWS:] == 20.0).all())

    # [-1] only: window 0 becomes a plain t2va window; the last frame moves to
    # the final window behind the handoff still.
    _, calls, _, _ = _run_fake_windowed(task="fl2va", keyframes=[-1], image_values=[20])
    assert calls["encode_prompt"] == [("t2va", []), ("fl2va", [91, 20])]
    kw0, _ = calls["build"][0]
    kw1, _ = calls["build"][1]
    assert kw0["keyframe_frame_indices"] is None and kw0["visual_condition"] is None
    assert _text_len(kw0) == 24 and _text_len(kw1) == 224

    # [0] only: window 0 is exactly the request; window 1 anchors the handoff.
    _, calls, _, _ = _run_fake_windowed(task="fl2va", keyframes=[0], image_values=[10])
    assert calls["encode_prompt"] == [("fl2va", [91])]
    kw0, _ = calls["build"][0]
    kw1, _ = calls["build"][1]
    assert kw0["keyframe_frame_indices"] == [0] and bool((kw0["visual_condition"] == 10.0).all())
    assert _text_len(kw0) == 124 and _text_len(kw1) == 124


def test_build_denoise_inputs_keyframe_segment_follows_the_indices_not_the_task():
    """A t2va request's continuation window carries a [0] keyframe (the handoff
    still); the packed layout must reserve its condition rows or the denoise
    loop rejects the anchor (``keyframe_cond_rows != layout cond rows``)."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    latent_t, latent_h, latent_w, audio_t, num_frames = 17, 2, 4, 93, 56
    frame_rows = (latent_h // 2) * (latent_w // 2)
    fake = SimpleNamespace(
        device=torch.device("cpu"),
        _initial_noise=lambda **kw: MiniMaxH3Pipeline._initial_noise(None, **kw),
    )

    def build(task, indices):
        return MiniMaxH3Pipeline._build_denoise_inputs(
            fake,
            task=task,
            text_embeddings=torch.zeros(24, 8),
            text_tags=torch.ones(24, dtype=torch.long),
            seed=3,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            num_frames=num_frames,
            num_steps=4,
            video_shift=5.0,
            audio_shift=3.0,
            base_schedule=None,
            visual_condition=None if indices is None else torch.zeros(len(indices) * frame_rows, 96),
            visual_condition_shape=None,
            audio_condition=None,
            ref_audio_t=None,
            visual_condition_shapes=None if indices is None else [(1, latent_h, latent_w)] * len(indices),
            keyframe_frame_indices=indices,
        )

    continuation = build("t2va", [0])
    frozen = int((~continuation["branch"].update_mask).sum())
    assert continuation["cond_anchor"].shape[0] == frame_rows
    assert frozen == frame_rows, "the layout must reserve rows for the handoff still"
    # ...and it is the same layout a first-frame fl2va request gets.
    first_frame = build("fl2va", [0])
    torch.testing.assert_close(first_frame["branch"].update_mask, continuation["branch"].update_mask)
    # Single-window t2va (no indices) is unchanged: no condition rows at all.
    plain = build("t2va", None)
    assert plain["cond_anchor"] is None
    assert int((~plain["branch"].update_mask).sum()) == 0


def test_generate_windowed_requires_a_local_text_encoder():
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError):
        _run_fake_windowed(task="t2va", keyframes=None, image_values=[], text_encoder=None)

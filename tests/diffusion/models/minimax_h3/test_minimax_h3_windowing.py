# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sliding-window shape math and history-block packing for MiniMax H3."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# --------------------------------------------------------------------------- #
# Overlap / window frame math
# --------------------------------------------------------------------------- #
def test_align_overlap_frames_snaps_to_17n_plus_1_grid():
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        minimax_h3_align_overlap_frames,
    )

    # 18 is already on the 17n+1 grid (17*1 + 1).
    assert minimax_h3_align_overlap_frames(18) == 18
    # 1 is the minimum (17*0 + 1).
    assert minimax_h3_align_overlap_frames(1) == 1
    assert minimax_h3_align_overlap_frames(0) == 1
    # 35 = 17*2 + 1.
    assert minimax_h3_align_overlap_frames(35) == 35
    # 19 snaps to the nearest of {18, 35} -> 18.
    assert minimax_h3_align_overlap_frames(19) == 18
    # 30 snaps to the nearest of {18, 35} -> 35.
    assert minimax_h3_align_overlap_frames(30) == 35


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
    # Default overlap 18 frames (on the 17n+1 grid).
    assert plan.overlap_frames == 18
    # 30 s rounds to two windows.
    assert plan.num_windows == 2
    # Window 0 contributes all 362 frames; window 1 contributes
    # 362 - 18 = 344 new frames (overlap is regenerated then dropped).
    assert plan.total_num_frames == 362 + (362 - 18)
    assert plan.overlap_latent_t == _video_latent_t(18)
    assert plan.overlap_audio_t == _audio_latent_t(18 / 24)


def test_windowing_explicit_num_segments():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_windowing,
    )

    plan = _resolve_minimax_h3_windowing(
        duration=45.0,
        fps=24,
        num_segments=3,
        overlap_frames=18,
        window_duration=None,
    )
    assert plan.num_windows == 3
    # Window 0 contributes 362; windows 1-2 each contribute 362 - 18 = 344.
    assert plan.total_num_frames == 362 + 2 * (362 - 18)


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
    overlap_latent_t = 2
    overlap_audio_t = 30
    window_latent_t = 107
    window_audio_t = 602
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
    overlap_audio_t = 30
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
    overlap_latent_t = 2
    tail = latent[:, :, -overlap_latent_t:, :, :]
    rows = minimax_h3_patchify_video_latent(tail, patch_size=(1, 2, 2))
    frame_rows = (48 // 2) * (80 // 2)
    assert rows.shape == (overlap_latent_t * frame_rows, 96)
    restored = minimax_h3_unpatchify_video_tokens(
        rows, latent_shape=(overlap_latent_t, 24, 40, 24), patch_size=(1, 2, 2)
    )
    torch.testing.assert_close(restored, tail)


def test_identity_anchor_plus_history_block_frozen_row_split():
    """Continuation layout: [image identity-anchor, video_audio history] + target.

    Mirrors the ref_blocks the window loop assembles for a t2va/fl2va
    continuation window: a 1-frame image anchor (window 0's first frame)
    followed by the multi-frame video+audio history, then the target.
    """
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    latent_h, latent_w = 48, 80
    frame_rows = (latent_h // 2) * (latent_w // 2)
    window_latent_t = 107
    window_audio_t = 603
    overlap_latent_t = 2
    overlap_audio_t = 30
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

from reconbot.active_guidance import guidance_code_from_stats, guidance_text


def test_guidance_rejects_blurry_frames_first():
    assert guidance_code_from_stats(1000, 0.9, 40.0, blur_score=12.0) == "slow_down"


def test_guidance_requests_baseline_for_nearly_identical_views():
    assert guidance_code_from_stats(1000, 0.8, 4.0, blur_score=100.0) == "add_baseline"


def test_reference_frame_respects_arc_direction():
    message = guidance_text("reference_frame", "left")
    assert "left" in message
    assert "Reference frame" in message

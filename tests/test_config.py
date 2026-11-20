from indic_rl.config import load_config


def test_smoke_overrides():
    c = load_config("configs/base.yaml", "configs/smoke.yaml")
    assert c["steps"] == 20 and c["lr"] == 5e-6


def test_arms_differ_only_in_arm():
    a = load_config("configs/base.yaml", "configs/arm_en.yaml")
    b = load_config("configs/base.yaml", "configs/arm_indic.yaml")
    assert {k for k in a if a[k] != b[k]} == {"arm"}


def test_t4_batch_divisible():
    c = load_config("configs/base.yaml", "configs/arm_en.yaml", "configs/t4.yaml")
    assert (c["per_device_batch"] * c["grad_accum"]) % c["num_generations"] == 0

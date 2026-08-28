import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"


def test_g1_package_keeps_required_self_contained_assets():
    required = (
        "assets/robots/hands/.g1-deployment-placeholder",
        "custom_assets/inspire_g2_hand/inspire_g2_hand_left.urdf",
        "custom_assets/inspire_g2_hand/inspire_g2_hand_right.urdf",
        "src/dex_retargeting/configs/teleop/inspire_g2_hand_left_dexpilot.yml",
        "src/dex_retargeting/configs/teleop/inspire_g2_hand_right_dexpilot.yml",
        "scripts/inspire_g2_left_mapping.yaml",
        "scripts/inspire_g2_right_mapping.yaml",
        "requirements-g1.txt",
        "MANIFEST-G1.txt",
        "config/g1_runtime.env.example",
        "docs/G1_MIGRATION_HANDOFF_ZH.md",
    )
    assert all((PROJECT_ROOT / path).is_file() for path in required)


def test_g1_live_launchers_require_operator_confirmation():
    for side in ("left", "right", "both"):
        launcher = SCRIPTS / f"run_g1_{side}_live.sh"
        text = launcher.read_text(encoding="utf-8")
        assert "RUN_LIVE_RETARGETING" in text


def test_g1_config_defaults_to_two_udp_ports_and_conservative_step():
    config = (PROJECT_ROOT / "config/g1_runtime.env.example").read_text(
        encoding="utf-8"
    )
    assert "LEFT_UDP_PORT=5005" in config
    assert "RIGHT_UDP_PORT=5006" in config
    assert "MAX_STEP_UNITS=2" in config
    assert "OMP_NUM_THREADS=1" in config


def test_g1_shell_scripts_have_valid_bash_syntax():
    scripts = (
        "g1_runtime_common.sh",
        "setup_g1_env.sh",
        "list_g1_io.sh",
        "run_g1_left_live.sh",
        "run_g1_right_live.sh",
        "run_g1_both_live.sh",
        "run_g1_left_dry.sh",
        "run_g1_right_dry.sh",
        "run_g1_udp_check.sh",
    )
    for name in scripts:
        subprocess.run(
            ["bash", "-n", str(SCRIPTS / name)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_g1_python_helpers_compile():
    for name in (
        "g1_preflight.py",
        "inspire_g2_vive_headless.py",
        "verify_g1_package.py",
    ):
        subprocess.run(
            ["python3", "-m", "py_compile", str(SCRIPTS / name)],
            check=True,
            capture_output=True,
            text=True,
        )

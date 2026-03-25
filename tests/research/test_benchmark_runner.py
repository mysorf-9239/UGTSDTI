import json

from omegaconf import OmegaConf

from ugtsdti.benchmark import _build_report_rows, _load_experiment_cfg, run_benchmark_matrix
from ugtsdti.data.protocols.probe import summarize_split_frames


def test_load_experiment_cfg_sets_model_data_loss_and_output(tmp_path):
    cfg = _load_experiment_cfg(
        model_name={
            "name": "hybrid_gcn_baseline_ug",
            "model": "hybrid",
            "teacher": "gcn",
            "student": "baseline",
            "fusion": "ug",
        },
        data_name="tdc_davis_s2",
        trainer_name="default_trainer",
        loss_name="kd",
        seed=7,
        output_dir=tmp_path / "run",
        wandb_enabled=False,
    )

    assert cfg.model_name == "hybrid_gcn_baseline_ug"
    assert cfg.data_name == "tdc_davis_s2"
    assert cfg.seed == 7
    assert cfg.trainer.loss.name == "kd"
    assert cfg.logging.wandb_enabled is False
    assert str(tmp_path / "run") == cfg.trainer.params.output_dir


def test_load_experiment_cfg_supports_slot_based_experiment_spec(tmp_path):
    cfg = _load_experiment_cfg(
        model_name={
            "name": "hybrid_gcn_baseline_ug",
            "model": "hybrid",
            "teacher": "gcn",
            "student": "baseline",
            "fusion": "ug",
        },
        data_name="tdc_davis_s2",
        trainer_name="default_trainer",
        loss_name="kd",
        seed=7,
        output_dir=tmp_path / "run",
        wandb_enabled=False,
    )

    assert cfg.model_name == "hybrid_gcn_baseline_ug"
    assert cfg.model.alias == "hybrid"
    assert cfg.teacher.alias == "gcn"
    assert cfg.student.alias == "baseline"
    assert cfg.fusion.alias == "ug"
    assert cfg.loss.name == "kd"


def test_run_benchmark_matrix_writes_summary_artifacts(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "output_root": str(tmp_path / "bench"),
            "trainer_config": "default_trainer",
            "experiments": [
                {
                    "name": "hybrid_gcn_baseline_ug",
                    "model": "hybrid",
                    "teacher": "gcn",
                    "student": "baseline",
                    "fusion": "ug",
                }
            ],
            "losses": ["bce", "kd"],
            "data_configs": ["tdc_davis_s1"],
            "seeds": [1, 2],
            "probe_protocols": True,
            "logging": {"wandb_enabled": False},
        }
    )

    def fake_run_experiment(experiment_cfg):
        return {
            "run_name": experiment_cfg.run_name,
            "seed": experiment_cfg.seed,
            "model_config": experiment_cfg.model_name,
            "data_config": experiment_cfg.data_name,
            "loss_name": experiment_cfg.trainer.loss.name,
            "val_best/auroc": 0.8,
            "val_best/student_auroc": 0.7,
            "val_best/teacher_auroc": 0.9,
            "test/auroc": 0.9,
            "test/student_auroc": 0.8,
            "test/teacher_auroc": 0.95,
        }

    monkeypatch.setattr("ugtsdti.experiment.benchmarking.run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        "ugtsdti.experiment.benchmarking.probe_tdc_split",
        lambda **_: {
            "dataset_name": "DAVIS",
            "scenario_name": "s1",
            "test_drug_overlap_ratio": 1.0,
            "test_target_overlap_ratio": 1.0,
        },
    )

    rows = run_benchmark_matrix(cfg)

    assert len(rows) == 4
    summary_json = tmp_path / "bench" / "summary.json"
    summary_csv = tmp_path / "bench" / "summary.csv"
    aggregate_json = tmp_path / "bench" / "aggregate" / "summary.json"
    report_json = tmp_path / "bench" / "report" / "summary.json"
    report_aggregate_json = tmp_path / "bench" / "report_aggregate" / "summary.json"
    protocol_json = tmp_path / "bench" / "protocol" / "summary.json"
    manifest_json = tmp_path / "bench" / "manifest.json"
    assert summary_json.exists()
    assert summary_csv.exists()
    assert aggregate_json.exists()
    assert report_json.exists()
    assert report_aggregate_json.exists()
    assert protocol_json.exists()
    assert manifest_json.exists()

    with summary_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert len(payload) == 4

    with aggregate_json.open("r", encoding="utf-8") as f:
        aggregate_payload = json.load(f)
    assert len(aggregate_payload) == 2

    with report_json.open("r", encoding="utf-8") as f:
        report_payload = json.load(f)
    assert len(report_payload) == 24

    with report_aggregate_json.open("r", encoding="utf-8") as f:
        report_aggregate_payload = json.load(f)
    assert len(report_aggregate_payload) == 12

    with protocol_json.open("r", encoding="utf-8") as f:
        protocol_payload = json.load(f)
    assert len(protocol_payload) == 1

    with manifest_json.open("r", encoding="utf-8") as f:
        manifest_payload = json.load(f)
    assert "benchmark_config" in manifest_payload
    assert "protocol_summary" in manifest_payload


def test_run_benchmark_matrix_supports_slot_based_experiments(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "output_root": str(tmp_path / "bench"),
            "trainer_config": "default_trainer",
            "experiments": [
                {
                    "name": "hybrid_gcn_baseline_ug",
                    "model": "hybrid",
                    "teacher": "gcn",
                    "student": "baseline",
                    "fusion": "ug",
                }
            ],
            "losses": ["bce"],
            "data_configs": ["tdc_davis_s4"],
            "seeds": [1],
            "probe_protocols": False,
            "logging": {"wandb_enabled": False},
        }
    )

    def fake_run_experiment(experiment_cfg):
        assert experiment_cfg.model.alias == "hybrid"
        assert experiment_cfg.teacher.alias == "gcn"
        assert experiment_cfg.student.alias == "baseline"
        assert experiment_cfg.fusion.alias == "ug"
        assert experiment_cfg.loss.name == "bce"
        return {
            "run_name": experiment_cfg.run_name,
            "seed": experiment_cfg.seed,
            "model_config": experiment_cfg.model_name,
            "data_config": experiment_cfg.data_name,
            "loss_name": experiment_cfg.loss.name,
            "val_best/auroc": 0.8,
            "test/auroc": 0.9,
        }

    monkeypatch.setattr("ugtsdti.experiment.benchmarking.run_experiment", fake_run_experiment)

    rows = run_benchmark_matrix(cfg)

    assert len(rows) == 1
    assert rows[0]["model_config"] == "hybrid_gcn_baseline_ug"


def test_summarize_split_frames_matches_realistic_overlap_counts():
    split_dict = {
        "train": {"Drug": ["d1", "d2"], "Target": ["t1", "t2"]},
        "valid": {"Drug": ["d2", "d3"], "Target": ["t2", "t3"]},
        "test": {"Drug": ["d4"], "Target": ["t1"]},
    }
    split_dict = {key: __import__("pandas").DataFrame(value) for key, value in split_dict.items()}

    summary = summarize_split_frames(split_dict, scenario_name="custom", dataset_name="DAVIS")

    assert summary["valid_drug_overlap_count"] == 1
    assert summary["valid_target_overlap_count"] == 1
    assert summary["test_drug_overlap_count"] == 0
    assert summary["test_target_overlap_count"] == 1


def test_load_experiment_cfg_uses_tdc_compatible_s1_method():
    cfg = _load_experiment_cfg(
        model_name={
            "name": "hybrid_gcn_baseline_ug",
            "model": "hybrid",
            "teacher": "gcn",
            "student": "baseline",
            "fusion": "ug",
        },
        data_name="tdc_davis_s1",
        trainer_name="default_trainer",
        loss_name="bce",
        seed=1,
        output_dir=__import__("pathlib").Path("/tmp/ugtsdti_s1_probe"),
        wandb_enabled=False,
    )

    assert cfg.data.train.params.split_type == "random"


def test_load_experiment_cfg_uses_real_s4_multi_column_protocol():
    cfg = _load_experiment_cfg(
        model_name={
            "name": "hybrid_gcn_baseline_ug",
            "model": "hybrid",
            "teacher": "gcn",
            "student": "baseline",
            "fusion": "ug",
        },
        data_name="tdc_davis_s4",
        trainer_name="default_trainer",
        loss_name="bce",
        seed=1,
        output_dir=__import__("pathlib").Path("/tmp/ugtsdti_s4_probe"),
        wandb_enabled=False,
    )

    assert list(cfg.data.train.params.column_name) == ["Drug", "Target"]


def test_build_report_rows_produces_tidy_branch_summary():
    rows = [
        {
            "model_config": "hybrid_baseline_baseline_ug",
            "data_config": "tdc_davis_s4",
            "loss_name": "kd",
            "seed": 42,
            "val_best/auroc": 0.6,
            "val_best/student_auroc": 0.5,
            "val_best/teacher_auroc": 0.7,
            "test/auroc": 0.55,
            "test/student_auroc": 0.52,
            "test/teacher_auroc": 0.62,
        }
    ]

    report_rows = _build_report_rows(rows)

    assert len(report_rows) == 6
    assert {row["branch"] for row in report_rows} == {"fused", "student", "teacher"}
    assert {row["split"] for row in report_rows} == {"val_best", "test"}
    assert {row["scenario_name"] for row in report_rows} == {"s4"}

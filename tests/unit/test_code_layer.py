"""code/: license gating, static repo mapping, env resolution, recipe drafting.

Nothing here executes repository code -- that is the invariant under test as
much as it is the test's setup.
"""

from __future__ import annotations

import pytest

from ml_research_agent.code.analyze import analyze_repo, file_excerpts
from ml_research_agent.code.discover import fidelity_score, parse_repo_url, to_repo
from ml_research_agent.code.env import install_commands, is_reproducible, pin_report, resolve_env
from ml_research_agent.code.license import (
    detect_from_repo_dir,
    detect_license,
    from_spdx,
    license_gate,
)
from ml_research_agent.code.recipe import draft_recipe, extract_reference_numbers, recipe_gaps
from ml_research_agent.config import CodeConfig, Config
from ml_research_agent.errors import MRAError
from ml_research_agent.types import CodeRepo, LicenseCategory, LicenseInfo, Paper

MIT = "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy"
APACHE = "Apache License, Version 2.0, January 2004\nhttp://www.apache.org/licenses/"
GPL3 = "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007"
AGPL = "GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3"
NC = "MIT License\n\nPermission is hereby granted... This software is for research purposes only."


@pytest.fixture
def config(tmp_path):
    return Config(paths={"workspace": tmp_path / "ws"}).ensure_paths()


@pytest.fixture
def repo_dir(tmp_path):
    """A miniature but realistic training repo."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "LICENSE").write_text(MIT)
    (root / "requirements.txt").write_text("torch==2.3.0\nnumpy>=1.26\n# a comment\n")
    (root / "configs" / "base.yaml").write_text("lr: 0.001\n")
    (root / "train.py").write_text(
        "import torch\nimport numpy as np\n\n"
        "def train():\n    model.cuda()\n\n"
        'if __name__ == "__main__":\n    train()\n'
    )
    (root / "src" / "eval.py").write_text("import torch\n\ndef evaluate():\n    pass\n")
    (root / "src" / "model.py").write_text(
        "import torch.nn as nn\n\nclass Net(nn.Module):\n    pass\n"
    )
    (root / "README.md").write_text(
        "# Toy Repo\n\nOur method achieves 76.1 top-1 accuracy on ImageNet.\n\n"
        "## Reproducing results\n\nSee requirements.txt. Trained on 8xA100.\n"
    )
    return root


class TestLicenseDetection:
    @pytest.mark.parametrize(
        ("text", "spdx", "category"),
        [
            (MIT, "MIT", LicenseCategory.PERMISSIVE),
            (APACHE, "Apache-2.0", LicenseCategory.PERMISSIVE),
            (GPL3, "GPL-3.0", LicenseCategory.COPYLEFT),
            (AGPL, "AGPL-3.0", LicenseCategory.COPYLEFT),
        ],
    )
    def test_common_licenses_are_identified(self, text, spdx, category):
        info = detect_license(text)
        assert info.spdx_id == spdx
        assert info.category is category

    def test_agpl_is_not_mistaken_for_gpl(self):
        assert detect_license(AGPL).spdx_id == "AGPL-3.0"

    def test_a_restrictive_clause_overrides_a_permissive_id(self):
        # "MIT, but research only" is not MIT, and treating it as permissive is
        # exactly the mistake the gate exists to prevent.
        info = detect_license(NC)
        assert info.category is LicenseCategory.NONCOMMERCIAL
        assert not info.permits_adaptation

    def test_no_license_text_is_unknown_not_permissive(self):
        assert detect_license("").category is LicenseCategory.UNKNOWN

    def test_license_is_read_from_a_fetched_repo(self, repo_dir):
        assert detect_from_repo_dir(repo_dir).spdx_id == "MIT"

    def test_a_repo_without_a_license_file_is_unknown(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert detect_from_repo_dir(tmp_path / "empty").category is LicenseCategory.UNKNOWN

    def test_noassertion_from_a_host_is_unknown(self):
        assert from_spdx("NOASSERTION").category is LicenseCategory.UNKNOWN


class TestLicenseGate:
    def _repo(self, license_info: LicenseInfo) -> CodeRepo:
        return CodeRepo(url="https://github.com/o/r", name="r", license=license_info)

    def test_permissive_allowed_license_passes(self):
        allowed, reason = license_gate(self._repo(from_spdx("MIT")), CodeConfig())
        assert allowed and "MIT" in reason

    def test_unknown_license_is_refused_not_waved_through(self):
        allowed, reason = license_gate(self._repo(LicenseInfo()), CodeConfig())
        assert not allowed and "could not be determined" in reason

    def test_a_license_outside_the_allow_list_is_refused(self):
        allowed, reason = license_gate(self._repo(from_spdx("GPL-3.0")), CodeConfig())
        assert not allowed and "allow-list" in reason

    def test_a_noncommercial_license_is_refused(self):
        allowed, reason = license_gate(self._repo(detect_license(NC)), CodeConfig())
        assert not allowed

    def test_the_allow_list_is_configurable(self):
        config = CodeConfig(allowed_licenses=["GPL-3.0"])
        info = from_spdx("GPL-3.0").model_copy(update={"permits_adaptation": True})
        allowed, _ = license_gate(self._repo(info), config)
        assert allowed


class TestStaticAnalysis:
    def _repo(self, repo_dir) -> CodeRepo:
        return CodeRepo(
            url="https://github.com/o/r",
            name="r",
            local_path=str(repo_dir),
            readme=(repo_dir / "README.md").read_text(),
        )

    def test_entrypoints_and_scripts_are_found(self, repo_dir, config):
        mapping = analyze_repo(self._repo(repo_dir), config)
        assert "train.py" in mapping.entrypoints
        assert "train.py" in mapping.train_scripts
        assert "src/eval.py" in mapping.eval_scripts
        assert "src/model.py" in mapping.model_files

    def test_third_party_imports_are_detected_and_stdlib_excluded(self, repo_dir, config):
        mapping = analyze_repo(self._repo(repo_dir), config)
        assert "torch" in mapping.dependencies
        assert "numpy" in mapping.dependencies
        assert "os" not in mapping.dependencies

    def test_hardware_assumptions_are_surfaced(self, repo_dir, config):
        mapping = analyze_repo(self._repo(repo_dir), config)
        assert any("CUDA" in h or "accelerator" in h for h in mapping.hardware_assumptions)

    def test_config_files_are_collected(self, repo_dir, config):
        mapping = analyze_repo(self._repo(repo_dir), config)
        assert any("base.yaml" in c for c in mapping.config_files)

    def test_analyzing_an_unfetched_repo_is_an_error(self, config):
        with pytest.raises(MRAError):
            analyze_repo(CodeRepo(url="https://github.com/o/r", name="r"), config)

    def test_excerpts_come_from_the_files_that_matter(self, repo_dir, config):
        repo = self._repo(repo_dir)
        excerpts = file_excerpts(repo, analyze_repo(repo, config))
        assert "train.py" in excerpts
        assert "def train" in excerpts["train.py"]


class TestEnvironmentResolution:
    def _repo(self, repo_dir) -> CodeRepo:
        return CodeRepo(
            url="https://github.com/o/r",
            name="r",
            local_path=str(repo_dir),
            readme=(repo_dir / "README.md").read_text(),
        )

    def test_packages_are_read_from_requirements(self, repo_dir, config):
        repo = self._repo(repo_dir)
        env = resolve_env(repo, analyze_repo(repo, config))
        assert "torch==2.3.0" in env.packages
        assert not any(p.startswith("#") for p in env.packages)

    def test_unpinned_dependencies_are_reported_as_a_reproducibility_gap(self, repo_dir, config):
        repo = self._repo(repo_dir)
        env = resolve_env(repo, analyze_repo(repo, config))
        assert any("unpinned" in problem for problem in pin_report(env))

    def test_a_requirements_file_alone_is_not_reproducible(self, repo_dir, config):
        repo = self._repo(repo_dir)
        assert not is_reproducible(resolve_env(repo, analyze_repo(repo, config)))

    def test_a_container_image_counts_as_reproducible(self, repo_dir, config):
        (repo_dir / "Dockerfile").write_text("FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime\n")
        repo = self._repo(repo_dir)
        env = resolve_env(repo, analyze_repo(repo, config))
        assert env.container_image and is_reproducible(env)
        assert install_commands(env)[0].startswith("docker pull")

    def test_env_hash_is_stable_and_sensitive(self, repo_dir, config):
        repo = self._repo(repo_dir)
        env = resolve_env(repo, analyze_repo(repo, config))
        assert env.env_hash == resolve_env(repo, analyze_repo(repo, config)).env_hash
        assert env.env_hash != env.model_copy(update={"python_version": "3.12"}).env_hash


class TestRecipeDrafting:
    def _repo(self, repo_dir) -> CodeRepo:
        return CodeRepo(
            url="https://github.com/o/r",
            name="r",
            local_path=str(repo_dir),
            readme=(repo_dir / "README.md").read_text(),
        )

    def test_a_draft_always_has_a_smoke_path(self, repo_dir, config):
        # Without one the scale ladder could start at `small`, which is the
        # ordering the system refuses to allow.
        repo = self._repo(repo_dir)
        recipe = draft_recipe(repo, analyze_repo(repo, config), config)
        assert recipe.smoke and all(step.command for step in recipe.smoke)

    def test_published_numbers_are_extracted_with_provenance(self, repo_dir, config):
        numbers = extract_reference_numbers(self._repo(repo_dir))
        assert numbers
        assert numbers[0].value == 76.1
        assert numbers[0].provenance.source.endswith("#README")

    def test_a_draft_is_marked_as_unverified(self, repo_dir, config):
        repo = self._repo(repo_dir)
        recipe = draft_recipe(repo, analyze_repo(repo, config), config)
        assert recipe.confidence < 0.5
        assert any("draft" in gap for gap in recipe_gaps(recipe))

    def test_gaps_name_what_is_missing(self, repo_dir, config):
        repo = self._repo(repo_dir)
        recipe = draft_recipe(repo, analyze_repo(repo, config), config)

        # This draft found a smoke path, a published number and a dataset in the
        # README, so none of those may be reported as missing; being unverified
        # is the one thing that is actually still wrong with it.
        gaps = recipe_gaps(recipe)
        assert [gap for gap in gaps if "draft" in gap] == gaps
        assert not any("smoke path" in gap for gap in gaps)
        assert not any("reference numbers" in gap for gap in gaps)
        assert not any("dataset" in gap for gap in gaps)

        # and a recipe stripped of everything names every one of them
        stripped = recipe.model_copy(
            update={"smoke": [], "reference_numbers": [], "datasets": [], "confidence": 0.9}
        )
        missing = recipe_gaps(stripped)
        assert len(missing) >= 3
        assert any("smoke path" in gap for gap in missing)
        assert any("reference numbers" in gap for gap in missing)
        assert any("dataset" in gap for gap in missing)
        assert not any("draft" in gap for gap in missing)  # confidence was raised


class TestDiscovery:
    def test_repo_urls_are_parsed(self):
        assert parse_repo_url("https://github.com/facebookresearch/dino") == (
            "facebookresearch",
            "dino",
        )
        assert parse_repo_url("https://github.com/o/r.git") == ("o", "r")
        assert parse_repo_url("https://gitlab.com/o/r") is None

    def test_official_repos_outrank_reimplementations(self):
        paper = Paper(title="A Paper", arxiv_id="2401.00001")
        data = {
            "html_url": "https://github.com/o/r",
            "name": "r",
            "owner": {"login": "o"},
            "stargazers_count": 10,
            "license": {"spdx_id": "MIT"},
        }
        official = to_repo(data, official=True)
        community = to_repo({**data, "stargazers_count": 5000}, official=False)

        official_score, _ = fidelity_score(official, paper)
        community_score, _ = fidelity_score(community, paper)
        assert official_score > community_score

    def test_a_permissive_license_raises_the_score(self):
        paper = Paper(title="A Paper")
        base = {"html_url": "https://github.com/o/r", "name": "r", "owner": {"login": "o"}}
        permissive, _ = fidelity_score(to_repo({**base, "license": {"spdx_id": "MIT"}}), paper)
        unlicensed, _ = fidelity_score(to_repo(base), paper)
        assert permissive > unlicensed

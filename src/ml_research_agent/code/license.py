"""License and usage-terms extraction; flags repos/datasets we may not use or
redistribute before any code is adapted."""

from __future__ import annotations

import re
from pathlib import Path

from ..config import CodeConfig
from ..types import CodeRepo, LicenseCategory, LicenseInfo, Provenance

LICENSE_FILENAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "LICENCE", "LICENSE-MIT")

# Ordered most-specific first: AGPL must not match as GPL, and "Apache" appears
# inside several dual-license headers.
_SIGNATURES: tuple[tuple[str, re.Pattern[str], LicenseCategory], ...] = (
    ("AGPL-3.0", re.compile(r"GNU AFFERO GENERAL PUBLIC LICENSE", re.I), LicenseCategory.COPYLEFT),
    ("LGPL-3.0", re.compile(r"GNU LESSER GENERAL PUBLIC LICENSE", re.I), LicenseCategory.COPYLEFT),
    (
        "GPL-3.0",
        re.compile(r"GNU GENERAL PUBLIC LICENSE\s*\n?\s*Version 3", re.I),
        LicenseCategory.COPYLEFT,
    ),
    (
        "GPL-2.0",
        re.compile(r"GNU GENERAL PUBLIC LICENSE\s*\n?\s*Version 2", re.I),
        LicenseCategory.COPYLEFT,
    ),
    ("MPL-2.0", re.compile(r"Mozilla Public License Version 2\.0", re.I), LicenseCategory.COPYLEFT),
    (
        "Apache-2.0",
        re.compile(r"Apache License[\s,]*Version 2\.0", re.I),
        LicenseCategory.PERMISSIVE,
    ),
    (
        "BSD-3-Clause",
        re.compile(r"Redistributions of source code.*?3\.\s*Neither the name", re.I | re.S),
        LicenseCategory.PERMISSIVE,
    ),
    (
        "BSD-2-Clause",
        re.compile(r"Redistribution and use in source and binary forms", re.I),
        LicenseCategory.PERMISSIVE,
    ),
    (
        "MIT",
        re.compile(r"Permission is hereby granted, free of charge", re.I),
        LicenseCategory.PERMISSIVE,
    ),
    ("ISC", re.compile(r"ISC License", re.I), LicenseCategory.PERMISSIVE),
    (
        "Unlicense",
        re.compile(r"This is free and unencumbered software", re.I),
        LicenseCategory.PERMISSIVE,
    ),
    ("CC-BY-NC-4.0", re.compile(r"Attribution-NonCommercial", re.I), LicenseCategory.NONCOMMERCIAL),
    (
        "CC-BY-4.0",
        re.compile(r"Creative Commons Attribution 4\.0", re.I),
        LicenseCategory.PERMISSIVE,
    ),
)

# Terms that make a license non-free regardless of what the SPDX id claims.
_RESTRICTIVE_PHRASES = (
    "non-commercial",
    "noncommercial",
    "research purposes only",
    "academic use only",
    "evaluation purposes only",
    "may not be used for commercial",
)

CATEGORY_BY_SPDX: dict[str, LicenseCategory] = {
    "MIT": LicenseCategory.PERMISSIVE,
    "Apache-2.0": LicenseCategory.PERMISSIVE,
    "BSD-3-Clause": LicenseCategory.PERMISSIVE,
    "BSD-2-Clause": LicenseCategory.PERMISSIVE,
    "ISC": LicenseCategory.PERMISSIVE,
    "Unlicense": LicenseCategory.PERMISSIVE,
    "MPL-2.0": LicenseCategory.COPYLEFT,
    "GPL-2.0": LicenseCategory.COPYLEFT,
    "GPL-3.0": LicenseCategory.COPYLEFT,
    "LGPL-3.0": LicenseCategory.COPYLEFT,
    "AGPL-3.0": LicenseCategory.COPYLEFT,
    "CC-BY-4.0": LicenseCategory.PERMISSIVE,
    "CC-BY-NC-4.0": LicenseCategory.NONCOMMERCIAL,
}


def detect_license(text: str, *, source: str = "") -> LicenseInfo:
    """Identify a license from its text.

    Restrictive-use phrases override the detected id: a file that says "MIT" and
    then "research purposes only" is not MIT, and treating it as permissive is
    exactly the mistake this gate exists to prevent.
    """
    if not text.strip():
        return LicenseInfo(category=LicenseCategory.UNKNOWN, notes="no license text found")

    spdx: str | None = None
    category = LicenseCategory.UNKNOWN
    for candidate, pattern, candidate_category in _SIGNATURES:
        if pattern.search(text):
            spdx, category = candidate, candidate_category
            break

    lowered = text.lower()
    restrictions = [phrase for phrase in _RESTRICTIVE_PHRASES if phrase in lowered]
    if restrictions:
        category = LicenseCategory.NONCOMMERCIAL

    return LicenseInfo(
        spdx_id=spdx,
        category=category,
        permits_adaptation=category is LicenseCategory.PERMISSIVE and not restrictions,
        notes="; ".join(f"restrictive term: '{r}'" for r in restrictions) or None,
        provenance=Provenance(source=source or "license-text", quote=text[:280])
        if source
        else None,
    )


def detect_from_repo_dir(path: Path) -> LicenseInfo:
    """Read the license file from a fetched repo. Reading only -- nothing runs."""
    for name in LICENSE_FILENAMES:
        candidate = path / name
        if candidate.exists() and candidate.is_file():
            return detect_license(
                candidate.read_text(encoding="utf-8", errors="replace")[:20000],
                source=f"file:{candidate}",
            )
    return LicenseInfo(category=LicenseCategory.UNKNOWN, notes="no license file in the repository")


def from_spdx(spdx_id: str | None, *, source: str = "") -> LicenseInfo:
    """Build LicenseInfo from an API-reported SPDX id (e.g. GitHub metadata)."""
    if not spdx_id or spdx_id.upper() in ("NOASSERTION", "OTHER"):
        return LicenseInfo(
            category=LicenseCategory.UNKNOWN, notes="the host reported no clear license"
        )
    category = CATEGORY_BY_SPDX.get(spdx_id, LicenseCategory.UNKNOWN)
    return LicenseInfo(
        spdx_id=spdx_id,
        category=category,
        permits_adaptation=category is LicenseCategory.PERMISSIVE,
        provenance=Provenance(source=source) if source else None,
    )


def license_gate(repo: CodeRepo, config: CodeConfig) -> tuple[bool, str]:
    """The gate that precedes any adaptation. Returns ``(allowed, reason)``.

    Unknown is refused, not waved through. "We could not tell" is a reason to
    stop, because the cost of being wrong lands on whoever publishes the work.
    """
    info = repo.license
    if info.category is LicenseCategory.UNKNOWN or not info.spdx_id:
        return False, "license could not be determined; adaptation is not permitted"
    if info.spdx_id not in config.allowed_licenses:
        return False, f"license {info.spdx_id} is not in the configured allow-list"
    if info.category is LicenseCategory.NONCOMMERCIAL:
        return (
            False,
            f"license {info.spdx_id} restricts use: {info.notes or 'non-commercial terms'}",
        )
    if not info.permits_adaptation:
        return False, f"license {info.spdx_id} does not permit adaptation"
    return True, f"license {info.spdx_id} permits adaptation"


def dataset_gate(license_info: LicenseInfo | None, config: CodeConfig) -> tuple[bool, str]:
    """The same check for datasets, which is where non-commercial terms actually bite."""
    if license_info is None:
        return False, "dataset license is unknown"
    if license_info.category is LicenseCategory.NONCOMMERCIAL:
        return False, f"dataset license restricts use: {license_info.notes or 'non-commercial'}"
    return True, f"dataset license {license_info.spdx_id or 'unspecified'} permits use"


__all__ = [
    "CATEGORY_BY_SPDX",
    "LICENSE_FILENAMES",
    "dataset_gate",
    "detect_from_repo_dir",
    "detect_license",
    "from_spdx",
    "license_gate",
]

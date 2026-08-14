"""Repo discovery: paper -> official/community implementations, ranked by
fidelity, activity, license, and whether they publish reproducible results."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from ..config import Config
from ..errors import SourceError
from ..observability.logging import StructuredLogger, get_logger
from ..types import CodeRepo, Paper, Provenance
from ..utils.cache import DiskCache
from .license import from_spdx

GITHUB_API = "https://api.github.com"
_REPO_URL = re.compile(r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:[/#?].*)?$", re.I)

# Weights for the fidelity ranking. Officialness dominates: the authors' own
# code reproduces the paper's numbers far more often than a reimplementation,
# and when it does not, that is itself the finding.
WEIGHTS = {"official": 0.40, "activity": 0.20, "popularity": 0.15, "license": 0.15, "docs": 0.10}


def parse_repo_url(url: str) -> tuple[str, str] | None:
    match = _REPO_URL.match(url.strip())
    return (match.group(1), match.group(2)) if match else None


class GitHubClient:
    """Minimal GitHub read client.

    Deliberately local to this layer rather than reusing the literature source
    adapter: capability layers do not import each other, and this one needs only
    repository metadata.
    """

    def __init__(self, config: Config, *, logger: StructuredLogger | None = None) -> None:
        self.config = config
        self.logger = logger or get_logger("code.discover")
        self.cache = DiskCache(
            config.paths.cache, namespace="github", ttl_seconds=config.literature.cache_ttl_seconds
        )
        self.token = os.environ.get("GITHUB_TOKEN")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.config.literature.user_agent,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        key = f"{path}?{sorted((params or {}).items())}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self.config.offline:
            return None
        try:
            response = httpx.get(
                f"{GITHUB_API}{path}",
                params=params,
                headers=self._headers(),
                timeout=self.config.literature.request_timeout_seconds,
                follow_redirects=True,
            )
            if response.status_code == 404:
                return None
            if response.status_code in (403, 429):
                raise SourceError("github rate limit", status=response.status_code, path=path)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SourceError(f"github request failed: {exc}", path=path) from exc
        data = response.json()
        self.cache.set(key, data)
        return data

    def repo(self, owner: str, name: str) -> dict[str, Any] | None:
        result = self.get(f"/repos/{owner}/{name}")
        return result if isinstance(result, dict) else None

    def readme(self, owner: str, name: str) -> str:
        data = self.get(f"/repos/{owner}/{name}/readme")
        if not isinstance(data, dict) or "content" not in data:
            return ""
        import base64

        try:
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            return ""

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        data = self.get("/search/repositories", {"q": query, "sort": "stars", "per_page": limit})
        items = data.get("items", []) if isinstance(data, dict) else []
        return [item for item in items if isinstance(item, dict)]


def to_repo(
    data: dict[str, Any], *, paper_key: str | None = None, official: bool = False
) -> CodeRepo:
    pushed = data.get("pushed_at") or data.get("updated_at")
    last_commit = None
    if isinstance(pushed, str):
        try:
            last_commit = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
        except ValueError:
            last_commit = None
    license_data = data.get("license") or {}
    return CodeRepo(
        url=data.get("html_url", ""),
        name=data.get("name", ""),
        owner=(data.get("owner") or {}).get("login"),
        default_branch=data.get("default_branch"),
        stars=int(data.get("stargazers_count", 0) or 0),
        forks=int(data.get("forks_count", 0) or 0),
        last_commit_at=last_commit,
        language=data.get("language"),
        description=data.get("description"),
        license=from_spdx(license_data.get("spdx_id"), source=data.get("html_url", "")),
        paper_keys=[paper_key] if paper_key else [],
        is_official=official,
        provenance=[Provenance(source=data.get("html_url", "github"))],
    )


def fidelity_score(
    repo: CodeRepo, paper: Paper, *, now: datetime | None = None
) -> tuple[float, dict[str, float]]:
    """Rank by whether this repo is likely to reproduce the paper's numbers."""
    now = now or datetime.now(UTC)
    components: dict[str, float] = {}

    components["official"] = 1.0 if repo.is_official else 0.0
    if not repo.is_official and paper.authors:
        surnames = {a.name.split()[-1].lower() for a in paper.authors}
        if repo.owner and repo.owner.lower() in surnames:
            components["official"] = 0.7

    if repo.last_commit_at is not None:
        age_days = max((now - repo.last_commit_at).days, 0)
        components["activity"] = max(0.0, 1.0 - age_days / 1095)  # three years to zero
    else:
        components["activity"] = 0.2

    components["popularity"] = min(1.0, (repo.stars / 1000) ** 0.5) if repo.stars else 0.0
    components["license"] = (
        1.0 if repo.license.permits_adaptation else (0.4 if repo.license.spdx_id else 0.0)
    )

    readme = (repo.readme or "").lower()
    docs = 0.0
    for marker, weight in (
        ("reproduc", 0.4),
        ("results", 0.2),
        ("pretrained", 0.15),
        ("citation", 0.1),
        ("requirements", 0.15),
    ):
        if marker in readme:
            docs += weight
    components["docs"] = min(1.0, docs)

    score = sum(WEIGHTS[k] * v for k, v in components.items())
    return min(1.0, score), components


def discover_repos(
    paper: Paper,
    config: Config,
    *,
    logger: StructuredLogger | None = None,
    client: GitHubClient | None = None,
) -> list[CodeRepo]:
    """Find candidate implementations for a paper, best first.

    Links carried on the paper (from Papers-with-Code or the abstract) are
    treated as official; a title search is the fallback. Nothing is executed
    here and nothing is cloned -- this step only decides what is worth fetching.
    """
    log = logger or get_logger("code.discover")
    client = client or GitHubClient(config, logger=log)
    found: dict[str, CodeRepo] = {}

    for url in paper.code_urls:
        parsed = parse_repo_url(url)
        if not parsed:
            continue
        owner, name = parsed
        try:
            data = client.repo(owner, name)
        except SourceError as exc:
            log.warning("repo_lookup_failed", url=url, error=str(exc))
            continue
        if data:
            repo = to_repo(data, paper_key=paper.key, official=True)
            repo = repo.model_copy(update={"readme": client.readme(owner, name)[:20000]})
            found[repo.url] = repo

    if len(found) < config.code.max_repos_per_paper:
        query = " ".join(w for w in paper.title.split() if len(w) > 3)[:120]
        try:
            for item in client.search(query, limit=config.code.max_repos_per_paper * 3):
                repo = to_repo(item, paper_key=paper.key)
                if repo.url and repo.url not in found:
                    found[repo.url] = repo
        except SourceError as exc:
            log.warning("repo_search_failed", paper=paper.key, error=str(exc))

    scored: list[CodeRepo] = []
    for repo in found.values():
        score, _ = fidelity_score(repo, paper)
        scored.append(repo.model_copy(update={"fidelity_score": score}))
    scored.sort(key=lambda r: (-r.fidelity_score, -r.stars, r.url))

    limit = config.code.max_repos_per_paper
    if len(scored) > limit:
        log.info("repos_truncated", paper=paper.key, kept=limit, dropped=len(scored) - limit)
    return scored[:limit]


__all__ = [
    "WEIGHTS",
    "GitHubClient",
    "discover_repos",
    "fidelity_score",
    "parse_repo_url",
    "to_repo",
]

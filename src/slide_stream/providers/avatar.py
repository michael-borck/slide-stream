"""Talking-head avatar provider implementations."""

from pathlib import Path

from rich.console import Console

from .base import AvatarProvider

console = Console()
err_console = Console(stderr=True, style="bold red")


class NoneAvatarProvider(AvatarProvider):
    """Default provider: avatar feature disabled, never produces a head."""

    @property
    def name(self) -> str:
        return "none"

    def is_available(self) -> bool:
        return True

    def generate(
        self, audio_path: str, output_path: str, slide_num: int
    ) -> str | None:
        return None


class PrecomputedAvatarProvider(AvatarProvider):
    """Use pre-supplied head clips instead of generating them.

    Maps slide N to ``{assets_dir}/head_N.mp4``. This needs no GPU or
    lip-sync service: record (or render) the clips once and drop them in a
    directory. Slides without a matching clip render without a head.
    """

    def _assets_dir(self) -> Path | None:
        avatar_config = self.config.get("providers", {}).get("avatar", {})
        assets_dir = avatar_config.get("assets_dir")
        return Path(assets_dir) if assets_dir else None

    @property
    def name(self) -> str:
        return "precomputed"

    def is_available(self) -> bool:
        """Available when the configured assets directory exists."""
        assets_dir = self._assets_dir()
        return assets_dir is not None and assets_dir.is_dir()

    def generate(
        self, audio_path: str, output_path: str, slide_num: int
    ) -> str | None:
        assets_dir = self._assets_dir()
        if assets_dir is None:
            err_console.print("  - Precomputed avatar: no assets_dir configured.")
            return None
        head = assets_dir / f"head_{slide_num}.mp4"
        if not head.is_file():
            err_console.print(f"  - Precomputed avatar: {head.name} not found.")
            return None
        console.print(f"  - Using precomputed head clip: {head.name}")
        return str(head)

"""Built-in character avatars.

Fun mascot faces shipped with the package. Reference one by name as an avatar
source (``providers.avatar.source: teddy``) instead of a file path. These are
original characters (no trademarked mascots).

Note on lip-sync: SadTalker / Wav2Lip need a *photorealistic human* face, so
these stylized characters do **not** lip-sync there — use them with the
``static`` avatar provider (held image in the corner, no GPU) or a
stylized-capable engine like D-ID. A static mascot plus a fun accent is often
the point: obviously-not-human dodges the uncanny valley entirely.
"""

from importlib import resources
from pathlib import Path

# name -> (bundled filename, human-readable label)
BUILTIN_AVATARS: dict[str, tuple[str, str]] = {
    "teddy": ("teddy.jpg", "Teddy bear"),
    "panda": ("panda.jpg", "Panda"),
    "koala": ("koala.jpg", "Koala"),
    "robot": ("robot.jpg", "Friendly robot"),
    "wizard": ("wizard.jpg", "Wizard"),
    "owl": ("owl.jpg", "Owl professor"),
}


def avatar_names() -> list[str]:
    return list(BUILTIN_AVATARS)


def resolve_avatar(source: str | None) -> str | None:
    """Resolve a built-in avatar name to its bundled file path.

    A plain name like ``teddy`` maps to the packaged image; anything else
    (a real path or URL) is returned unchanged.
    """
    if not source:
        return source
    entry = BUILTIN_AVATARS.get(source.lower())
    if entry is None:
        return source
    with resources.as_file(resources.files("slide_stream.avatar_images") / entry[0]) as p:
        return str(Path(p))

"""Built-in character avatars.

Fun mascot faces shipped with the package. Reference one by name as an avatar
source (``providers.avatar.source: teddy``) instead of a file path. These are
original characters (no trademarked mascots).

Note on lip-sync: SadTalker / Wav2Lip / LivePortrait all begin by detecting a
*human* face, so these stylized characters do **not** lip-sync there. Two
options that DO animate them: the ``wan-s2v`` provider (Wan2.2-S2V has no face
detector at all, so it animates any mascot or human head shot from the
narration audio — see ``docs/wan-s2v-api.md``), or the no-GPU ``puppet`` /
``static`` providers. A mascot plus a fun accent is often the point:
obviously-not-human dodges the uncanny valley entirely.
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


# Mouth region per built-in avatar for the 'puppet' mouth-flap, as fractions of
# the image: (center_x, center_y, width, max_open_height). Eyeballed from the
# generated art — nudge these numbers if a mascot's flap is off. Override for a
# custom image with providers.avatar.mouth: [cx, cy, w, h].
MOUTH_BOXES: dict[str, tuple[float, float, float, float]] = {
    "teddy": (0.50, 0.63, 0.12, 0.06),
    "panda": (0.50, 0.60, 0.11, 0.06),
    "koala": (0.50, 0.60, 0.11, 0.06),
    "robot": (0.50, 0.62, 0.14, 0.06),
    "wizard": (0.50, 0.60, 0.10, 0.05),
    "owl": (0.50, 0.55, 0.10, 0.06),
}
DEFAULT_MOUTH_BOX = (0.50, 0.60, 0.12, 0.06)


# Positive prompt per built-in avatar for the 'wan-s2v' provider. Wan2.2-S2V has
# no face detector, so the text prompt is how the model learns where the mouth
# is and how it moves — describe the actual anatomy (an owl has a beak, not
# lips). Override for a custom image with providers.avatar.prompt.
AVATAR_PROMPTS: dict[str, str] = {
    "teddy": "a plush teddy bear talking, muzzle opening and closing to form words, warm soft lighting",
    "panda": "a panda talking, mouth opening and closing to form words, gentle lighting",
    "koala": "a koala talking, mouth opening and closing to form words, soft lighting",
    "robot": "a friendly robot talking, jaw and mouth panel moving to form words, clean studio lighting",
    "wizard": "a wizard talking, mouth moving to form words beneath the beard, warm cinematic lighting",
    "owl": "an owl professor talking, beak opening and closing to articulate words, warm lighting",
}
DEFAULT_AVATAR_PROMPT = (
    "a character talking directly to camera, mouth opening and closing to "
    "articulate words, natural head motion, soft even lighting"
)


def avatar_names() -> list[str]:
    return list(BUILTIN_AVATARS)


def avatar_prompt(source: str | None) -> str:
    """Wan2.2-S2V positive prompt for a built-in avatar name, else the default.

    A file path or unknown name gets the generic talking prompt, which works
    for a human head shot as well as an unknown character.
    """
    if source and source.lower() in AVATAR_PROMPTS:
        return AVATAR_PROMPTS[source.lower()]
    return DEFAULT_AVATAR_PROMPT


def mouth_box(source: str | None) -> tuple[float, float, float, float]:
    """Mouth region (fractions) for a built-in avatar name, else the default."""
    if source and source.lower() in MOUTH_BOXES:
        return MOUTH_BOXES[source.lower()]
    return DEFAULT_MOUTH_BOX


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

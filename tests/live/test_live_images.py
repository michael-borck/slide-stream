"""Live image generation: the configured provider returns a real image, not
the flat text-card fallback (which is how a missing key / API drift shows up).

Skips when the resolved provider is 'text' (nothing live is configured).
    uv run pytest tests/live/test_live_images.py --run-live -q
"""

import numpy as np
import pytest
from PIL import Image

from slide_stream.providers.factory import ProviderFactory


def test_live_image_is_real_not_text_fallback(live_config, tmp_path):
    provider = ProviderFactory.create_image_provider(live_config)
    if provider.name == "text":
        pytest.skip("image provider resolved to 'text' — no live image gen configured")

    out = tmp_path / "img.png"
    slide = {
        "title": "A red apple on a wooden table",
        "content": ["fresh fruit", "studio light"],
    }
    result = provider.generate_image(
        "a red apple on a wooden table, photorealistic", str(out), slide=slide
    )
    assert result, f"{provider.name}.generate_image returned None"

    im = Image.open(result).convert("RGB")
    assert im.width >= 128 and im.height >= 128, "image implausibly small"
    # A photo / generated image has colour variance; a text card is near-flat.
    assert float(np.asarray(im).std()) > 12, "looks like a flat text-card fallback"
    print(f"\nImage provider exercised: {provider.name} ({im.width}x{im.height})")

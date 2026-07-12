#!/usr/bin/env python3
"""Reap orphaned ephemeral voice profiles left behind by slide-stream.

slide-stream's 'voicebox' TTS provider (voice_sample mode) clones the speaker
into a throwaway profile named ``slide-stream-<uuid4>``, then deletes it in a
``finally`` at the end of the run. That covers crashes, Ctrl-C and strict-mode
aborts — but nothing survives SIGKILL, an OOM kill, or a power loss. This
script closes that gap: run it on a timer to delete any such profile older than
a grace period, along with the generations rendered from it.

It is deliberately conservative. A profile is only ever deleted when it is
BOTH voice_type == "cloned" AND named exactly ``slide-stream-<uuid4>``, so a
profile you created by hand can never match, whatever you call it. Deleting a
profile removes its samples and its cloned reference audio; generations do not
cascade, so they are deleted first.

Requires only the standard library — copy it onto the Voicebox host and run it.

Usage:
    sweep_ephemeral_profiles.py --base-url https://voice.example.org [options]

    --max-age-minutes N   only reap profiles older than N minutes (default 60)
    --token TOKEN         bearer token, if the server sits behind auth
    --dry-run             report what would be deleted, delete nothing
    --keep-generations    delete the profiles but leave their rendered audio

Environment: VOICEBOX_BASE_URL and VOICEBOX_TOKEN are used as defaults.

Cron example (reap profiles older than 60 minutes, every 15 minutes):
    */15 * * * * /opt/slide-stream/sweep_ephemeral_profiles.py \
        --base-url http://localhost:17493 --max-age-minutes 60

Exits non-zero if any deletion failed, so cron will mail you about a profile
that is still sitting on the server.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

# Matches exactly what the provider creates: the literal prefix + a UUIDv4.
# A hand-made profile called "slide-stream-backup" will not match.
EPHEMERAL_NAME = re.compile(
    r"^slide-stream-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

HISTORY_PAGE_SIZE = 100


def _request(
    method: str, url: str, token: str | None, timeout: float = 60.0
) -> object:
    """Perform an HTTP request, returning decoded JSON (or None on 204)."""
    req = urllib.request.Request(url, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


def _parse_created_at(value: str) -> datetime:
    """Parse Voicebox's created_at, which is naive UTC (datetime.utcnow)."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def find_stale_profiles(
    base_url: str, token: str | None, max_age: timedelta
) -> list[dict[str, object]]:
    """Return ephemeral profiles older than the grace period."""
    profiles = _request("GET", f"{base_url}/profiles", token)
    if not isinstance(profiles, list):
        raise RuntimeError(f"unexpected /profiles response: {profiles!r}")

    cutoff = datetime.now(timezone.utc) - max_age
    stale: list[dict[str, object]] = []
    for profile in profiles:
        name = str(profile.get("name", ""))
        if not EPHEMERAL_NAME.match(name):
            continue
        if profile.get("voice_type") != "cloned":
            continue
        created_at = profile.get("created_at")
        if not isinstance(created_at, str):
            continue
        if _parse_created_at(created_at) < cutoff:
            stale.append(profile)
    return stale


def generation_ids(base_url: str, token: str | None, profile_id: str) -> list[str]:
    """Every generation rendered from a profile. Deleting the profile leaves
    these orphaned, so collect them before it goes.

    Belt and braces: the ``?profile_id=`` query relies on server-side
    filtering, and a server that ignores the unknown param would return the
    whole history — this must never turn into "delete every generation on the
    server". Only items whose own ``profile_id`` field matches are acted on;
    items lacking the field are skipped (and reported), never deleted.
    """
    ids: list[str] = []
    missing_field = 0
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {"profile_id": profile_id, "limit": HISTORY_PAGE_SIZE, "offset": offset}
        )
        page = _request("GET", f"{base_url}/history?{query}", token)
        if not isinstance(page, dict):
            break
        items = page.get("items") or []
        for item in items:
            if "id" not in item:
                continue
            if "profile_id" not in item:
                missing_field += 1
                continue
            if str(item["profile_id"]) != profile_id:
                continue  # server ignored the filter; another profile's audio
            ids.append(str(item["id"]))
        offset += len(items)
        if not items:
            break
        # A missing total means "keep paging until a page comes back empty".
        total = page.get("total")
        if total is not None and offset >= int(total):
            break
    if missing_field:
        print(
            f"note: skipped {missing_field} history item(s) with no profile_id "
            "field; not deleting those",
            file=sys.stderr,
        )
    return ids


def sweep(
    base_url: str,
    token: str | None,
    max_age: timedelta,
    dry_run: bool,
    keep_generations: bool,
) -> int:
    """Delete stale ephemeral profiles. Returns the number of failures."""
    stale = find_stale_profiles(base_url, token, max_age)
    if not stale:
        print("sweep complete: no stale ephemeral profiles found")
        return 0

    failures = 0
    for profile in stale:
        profile_id = str(profile["id"])
        name = profile["name"]

        gen_ids: list[str] = []
        listing_failed = False
        if not keep_generations:
            try:
                gen_ids = generation_ids(base_url, token, profile_id)
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"error: listing generations for {name}: {e}", file=sys.stderr)
                failures += 1
                listing_failed = True

        if dry_run:
            print(f"would delete: {name} ({profile_id}) + {len(gen_ids)} generation(s)")
            continue

        gen_failures = 0
        for gen_id in gen_ids:
            try:
                _request("DELETE", f"{base_url}/history/{gen_id}", token)
            except urllib.error.HTTPError as e:
                if e.code == 404:  # already gone: that is the desired state
                    continue
                print(f"error: deleting generation {gen_id}: {e}", file=sys.stderr)
                gen_failures += 1
            except Exception as e:  # noqa: BLE001
                print(f"error: deleting generation {gen_id}: {e}", file=sys.stderr)
                gen_failures += 1
        failures += gen_failures

        if listing_failed or gen_failures:
            # Deleting the profile now would orphan the surviving generations
            # (profile deletion does not cascade). Leave it so the next run
            # finds the profile again and retries the whole set.
            print(
                f"skipping profile delete for {name} ({profile_id}): its "
                "generations could not all be removed; will retry next run",
                file=sys.stderr,
            )
            continue

        # Deleting the profile also removes its samples and reference audio.
        try:
            _request("DELETE", f"{base_url}/profiles/{profile_id}", token)
            print(f"deleted: {name} ({profile_id}) + {len(gen_ids)} generation(s)")
        except urllib.error.HTTPError as e:
            if e.code == 404:  # someone else reaped it between list and delete
                print(f"already gone: {name} ({profile_id})")
                continue
            print(f"error: deleting profile {name}: {e}", file=sys.stderr)
            failures += 1
        except Exception as e:  # noqa: BLE001
            print(f"error: deleting profile {name}: {e}", file=sys.stderr)
            failures += 1

    verb = "would be removed" if dry_run else "removed"
    print(f"sweep complete: {len(stale)} ephemeral profile(s) {verb}, {failures} failure(s)")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--base-url",
        default=os.getenv("VOICEBOX_BASE_URL"),
        help="Voicebox server URL (default: $VOICEBOX_BASE_URL)",
    )
    parser.add_argument("--token", default=os.getenv("VOICEBOX_TOKEN"))
    parser.add_argument("--max-age-minutes", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-generations",
        action="store_true",
        help="delete the clones but leave their rendered audio in history",
    )
    args = parser.parse_args(argv)

    if not args.base_url:
        parser.error("--base-url is required (or set VOICEBOX_BASE_URL)")

    try:
        failures = sweep(
            args.base_url.rstrip("/"),
            args.token,
            timedelta(minutes=args.max_age_minutes),
            args.dry_run,
            args.keep_generations,
        )
    except Exception as e:  # noqa: BLE001 - a cron job wants a message, not a traceback
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

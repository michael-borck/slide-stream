#!/usr/bin/env bash
# Delete ephemeral UUID-named voice references uploaded by slide-stream's
# 'chatterbox' TTS provider (voice_sample mode).
#
# slide-stream uploads each lecturer's reference clip under a random UUID
# filename per run and never reuses it; the Chatterbox TTS Server has no
# delete API, so this script reaps those files from the reference-audio
# directory after a grace period. Named voices (stock or deliberately kept
# references like "michael.wav") are never touched.
#
# Usage:
#   cleanup_uuid_voices.sh /path/to/reference_audio [max-age-minutes]
#
# The reference directory is the 'reference_audio' folder of the
# Chatterbox-TTS-Server checkout, or the Docker volume you mapped to it,
# e.g.: docker inspect <container> | grep -A3 reference_audio
#
# Cron example (reap files older than 60 minutes, every 15 minutes):
#   */15 * * * * /opt/slide-stream/cleanup_uuid_voices.sh /srv/chatterbox/reference_audio 60
#
# Works equally well via docker exec if the directory is not host-mapped:
#   */15 * * * * docker exec chatterbox-tts-server /app/cleanup_uuid_voices.sh /app/reference_audio 60

set -euo pipefail

REF_DIR="${1:?usage: cleanup_uuid_voices.sh <reference_audio_dir> [max-age-minutes]}"
MAX_AGE_MIN="${2:-60}"

if [ ! -d "$REF_DIR" ]; then
    echo "error: not a directory: $REF_DIR" >&2
    exit 1
fi

# UUIDv4 filenames only: 8-4-4-4-12 hex groups + audio extension.
UUID_GLOB='[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]-[0-9a-f][0-9a-f][0-9a-f][0-9a-f]-[0-9a-f][0-9a-f][0-9a-f][0-9a-f]-[0-9a-f][0-9a-f][0-9a-f][0-9a-f]-[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'

deleted=0
while IFS= read -r -d '' f; do
    rm -f -- "$f"
    echo "deleted: $f"
    deleted=$((deleted + 1))
done < <(find "$REF_DIR" -maxdepth 1 -type f \
    \( -name "${UUID_GLOB}.wav" -o -name "${UUID_GLOB}.mp3" -o -name "${UUID_GLOB}.flac" \) \
    -mmin "+${MAX_AGE_MIN}" -print0)

echo "cleanup complete: ${deleted} ephemeral voice file(s) removed from ${REF_DIR}"

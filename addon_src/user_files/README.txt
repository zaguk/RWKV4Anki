Anki preserves this directory across add-on updates.

RWKV-SRS stores per-profile checkpoints, manifests, settings, and temporary
exported research datasets under user_files/profiles/.

Each profile has checkpoints/v1/ and checkpoints/v5/ slots for
one checkpoint, manifest and evaluation cache per generation. Existing single
checkpoints migrate to their generation automatically. Settings and Live Session
reports are shared across generations. Retain Multiple Model Generations enables
saving cross-generation comparison states; turning it off does not delete them.

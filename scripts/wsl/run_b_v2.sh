#!/usr/bin/env bash
# Object B v2: text-to-3D (threestudio DreamFusion + SD v1.5 SDS). Improves on old2:
#  - prompt drops "keychain" (which induced an extra body/base lump) and forbids
#    neck/body/base; head-only anti-Janus view prompts.
#  - 4000 steps (vs 3000) for a more reliable export surface.
set -euo pipefail

export HW3_ROOT=/mnt/d/2026_Spring/deeplearning/hw3
export PROMPT_FILE=/mnt/d/2026_Spring/deeplearning/hw3/prompts/object_b_v2_prompt.txt
export TAG=steps_4000_v2_wandb
export MAX_STEPS=4000
export GUIDANCE_SCALE=75.0
export WANDB_ENABLE=1
export WANDB_RUN_NAME=B_text3d_4000_v2

# All prompts kept short (< 77 CLIP tokens; threestudio tokenizes without truncation).
export NEGATIVE_PROMPT="body, neck, torso, shoulders, base, stand, pedestal, multiple faces, duplicate face, janus, extra eyes, two heads, limbs, blurry, text, watermark"
export PROMPT_FRONT="front view of the doll head, one face, purple star eyes, small tongue, purple bob hair, rabbit hairpin"
export PROMPT_SIDE="side view of the doll head, purple bob hair around the head, hairpin, no face on this side"
export PROMPT_BACK="back of the doll head, smooth purple bob hair, no face, no eyes, no mouth"
export PROMPT_OVERHEAD="top of the round purple-haired doll head, no face"

exec bash "$HW3_ROOT/scripts/wsl/run_b_text3d.sh"

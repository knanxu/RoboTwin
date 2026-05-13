#!/bin/bash
# Container entrypoint for speedtune:latest.
#
# - Re-creates the envs/curobo symlink every startup (bind-mounting
#   /app/RoboTwin from the host would otherwise hide the curobo tree we
#   installed into /opt/curobo at build time).
# - Sources conda so later `conda activate RoboTwin` works under `bash -lc`.
# - Execs the container CMD.

set -e

if [ -d /app/RoboTwin/envs ] && [ ! -e /app/RoboTwin/envs/curobo ]; then
    ln -sfn /opt/curobo /app/RoboTwin/envs/curobo
fi

# Make `conda activate` available in this shell; children that run under
# `bash -lc` will pick up /etc/profile.d too.
# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh

exec "$@"

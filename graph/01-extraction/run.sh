#!/usr/bin/env bash
set -euo pipefail

STAGE=$(dirname "$(realpath "$0")")
KERNEL=$(jq -r .id "$STAGE/kernel-metadata.json")

uvx kaggle kernels push -p "$STAGE"
sleep 60
until status=$(uvx kaggle kernels status "$KERNEL") && [[ $status =~ COMPLETE|ERROR|CANCEL ]]; do
    sleep 60
done
echo "$status"

if [[ $status != *COMPLETE* ]]; then
    failed=$(mktemp -d)
    uvx kaggle kernels output "$KERNEL" -p "$failed" >/dev/null
    echo "log: $failed/$(basename "$KERNEL").log"
    exit 1
fi

rm -rf "$STAGE/out"
uvx kaggle kernels output "$KERNEL" -p "$STAGE/out"

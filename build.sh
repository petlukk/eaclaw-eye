#!/bin/bash
set -euo pipefail

EA="${EA:-/home/peter/projects/eacompute/target/release/ea}"
AARCH64_CC="${AARCH64_CC:-aarch64-linux-gnu-gcc}"

if ! command -v "$EA" &>/dev/null && [ ! -x "$EA" ]; then
    echo "Eä compiler not found at: $EA"
    echo "Set EA=/path/to/ea or install eacompute"
    exit 1
fi

echo "Eä compiler: $("$EA" --version)"

mkdir -p kernels/prebuilt/x86_64 kernels/prebuilt/aarch64

for f in kernels/*.ea; do
    stem=$(basename "$f" .ea)
    echo "  $stem.ea → x86_64"
    "$EA" "$f" --lib -o "kernels/prebuilt/x86_64/lib${stem}.so"

    echo "  $stem.ea → aarch64"
    # Compile to object file first (cross-compilation can't use host linker)
    "$EA" "$f" --target-triple=aarch64-unknown-linux-gnu
    # Object file lands in cwd as ${stem}.o
    if command -v "$AARCH64_CC" &>/dev/null; then
        "$AARCH64_CC" -shared -nostdlib "${stem}.o" -o "kernels/prebuilt/aarch64/lib${stem}.so"
    else
        # No cross-linker: link with ld.lld if available
        if command -v ld.lld &>/dev/null; then
            ld.lld -shared --no-undefined-version "${stem}.o" -o "kernels/prebuilt/aarch64/lib${stem}.so"
        else
            echo "    WARN: no aarch64 linker found, keeping .o only"
            mv "${stem}.o" "kernels/prebuilt/aarch64/lib${stem}.o"
            continue
        fi
    fi
    rm -f "${stem}.o"
done

echo "Done."
ls -la kernels/prebuilt/x86_64/
ls -la kernels/prebuilt/aarch64/

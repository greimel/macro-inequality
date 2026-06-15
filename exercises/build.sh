#!/usr/bin/env bash
# Build the exercise-collection PDFs: collection (full) and final-exam-collection
# (required-reading study set), each with and without solutions.
# Actual exams (selections with answer boxes) are built separately in the private
# exams repo.
# Inputs tex files from SimpleOLG/exercises/generated/ and this directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for master in collection final-exam-collection; do
    if [ ! -f "${master}.tex" ]; then
        echo "error: ${master}.tex not found in $SCRIPT_DIR" >&2
        exit 1
    fi
done

GENERATED_DIR="$SCRIPT_DIR/../SimpleOLG/exercises/generated"
if [ ! -d "$GENERATED_DIR" ] || [ -z "$(ls -A "$GENERATED_DIR"/*.tex 2>/dev/null)" ]; then
    echo "error: SimpleOLG exercises have not been generated yet." >&2
    echo "Run: julia SimpleOLG/exercises/build_exercises.jl" >&2
    exit 1
fi

build_variant() {
    local master="$1"    # "collection" or "exam"
    local name="$2"      # "exercises" or "solutions"
    local toggle="$3"    # "false" or "true"
    local target="${master}-${name}.tex"

    sed -E "s/\\\\showsolutions(true|false)/\\\\showsolutions${toggle}/" \
        "${master}.tex" > "$target"

    latexmk -pdf -interaction=nonstopmode -halt-on-error "$target"
}

for master in collection final-exam-collection; do
    build_variant "$master" "exercises" "false"
    build_variant "$master" "solutions" "true"
done

# All builds succeeded (set -e would have exited otherwise).
# Remove aux files but keep the generated PDFs.
for master in collection final-exam-collection; do
    for name in exercises solutions; do
        latexmk -c "${master}-${name}.tex" >/dev/null
        rm -f "${master}-${name}.tex" "${master}-${name}.bbl"
    done
done

echo
echo "Built PDFs:"
for master in collection final-exam-collection; do
    for name in exercises solutions; do
        echo "  $SCRIPT_DIR/${master}-${name}.pdf"
    done
done

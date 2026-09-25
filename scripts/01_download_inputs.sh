#!/usr/bin/env bash
# Download or validate the GRCh38 reference and frozen ENCODE peak sets.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/lib/common.sh"
manifest=${1:-$REPO_DIR/manifests/inputs/external_peak_sets.tsv}
output=${2:-$REPO_DIR/data/raw/peaks}
reference=${3:-${CTCF_GENOME_DIR:-${CTCF_WORK:-$REPO_DIR/work/main}/reference}}
require_commands wget md5sum gzip cmp samtools
require_nonempty "$manifest"
mkdir -p "$output" "$reference"

genome=$reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa
archive=$genome.gz
if [[ ! -s $genome || ! -s ${genome}.fai || ! -s $reference/GRCh38.chrom.sizes ]]; then
    if [[ ! -s $genome ]]; then
        wget -c -O "$archive.part" "https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
        gzip -t "$archive.part"
        gzip -dc "$archive.part" > "$genome.part"
        mv "$genome.part" "$genome"
        mv "$archive.part" "$archive"
    fi
    samtools faidx "$genome"
    cut -f1,2 "${genome}.fai" > "$reference/GRCh38.chrom.sizes"
fi
require_nonempty "$genome" "${genome}.fai" "$reference/GRCh38.chrom.sizes"

while IFS=$'\t' read -r cohort experiment biosample replicates accession peak_type md5 url; do
    [[ $cohort != cohort ]] || continue
    [[ $cohort =~ ^[a-z0-9-]+$ && $md5 =~ ^[0-9a-f]{32}$ ]] || die "Invalid peak manifest row"
    target=$output/${cohort}.narrowPeak
    if [[ -e $target && -e ${target}.gz ]]; then
        printf '%s  %s\n' "$md5" "${target}.gz" | md5sum --check --status -
        require_nonempty "$target"
        gzip -dc "${target}.gz" | cmp - "$target"
        continue
    fi
    if [[ ! -e $target && -e ${target}.gz ]]; then
        printf '%s  %s\n' "$md5" "${target}.gz" | md5sum --check --status -
        gzip -dc "${target}.gz" > "${target}.part"
        require_nonempty "${target}.part"
        mv "${target}.part" "$target"
        continue
    fi
    [[ ! -e $target && ! -e ${target}.gz && ! -e ${target}.part ]] || \
        die "External peak output exists: $target"
    wget -c -O "${target}.gz.part" "$url"
    printf '%s  %s\n' "$md5" "${target}.gz.part" | md5sum --check --status -
    gzip -dc "${target}.gz.part" > "${target}.part"
    require_nonempty "${target}.part"
    mv "${target}.part" "$target"
    mv "${target}.gz.part" "${target}.gz"
    log "$cohort: $experiment, $biosample, $replicates biological replicates, $accession ($peak_type)"
done < "$manifest"

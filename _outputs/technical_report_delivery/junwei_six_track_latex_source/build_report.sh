#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/src"
mkdir -p ../build

latexmk \
  -xelatex \
  -interaction=nonstopmode \
  -halt-on-error \
  -file-line-error \
  -outdir=../build \
  main.tex

cp ../build/main.pdf ../build/junwei_six_track_technical_report_print.pdf

gs -q -dNOPAUSE -dBATCH -dSAFER \
  -sDEVICE=pdfwrite \
  -dCompatibilityLevel=1.5 \
  -dPDFSETTINGS=/printer \
  -dDetectDuplicateImages=true \
  -dCompressFonts=true \
  -dSubsetFonts=true \
  -dDownsampleColorImages=true \
  -dColorImageResolution=240 \
  -dDownsampleGrayImages=true \
  -dGrayImageResolution=240 \
  -dDownsampleMonoImages=true \
  -dMonoImageResolution=300 \
  -sOutputFile=../build/junwei_six_track_technical_report.pdf \
  ../build/main.pdf

echo "$HERE/build/junwei_six_track_technical_report.pdf"

##### Add these for a fix on greenland #####
# Force the current conda env's Python
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please 'conda activate verl-agent' before running this script." >&2
  exit 1
fi

# Put the env's bin FIRST so it wins over base
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
##### Add these for a fix on greenland #####

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources_100 \
  --index indexes_100 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources \
  --index indexes \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources_1k \
  --index indexes_1k \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources_100k \
  --index indexes_100k \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

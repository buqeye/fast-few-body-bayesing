LABEL=`date "+%Y-%m-%d-%H-%M-%S"`

if [[ $# -ne 1 ]] ; then
        echo "Usage: $0 parameters"
        exit 1
fi

PARAMETERS="$1"

python sample_emulators.py \
  --parameters="${PARAMETERS}" \
  --observables "H3" "He4" \
  --label="${LABEL}"

python sample_emulators.py \
  --parameters="${PARAMETERS}" \
  --observables "H3" "He4-radius" \
  --label="${LABEL}"

python sample_emulators.py \
  --parameters="${PARAMETERS}" \
  --observables "H3" "H3-halflife" \
  --label="${LABEL}"

python sample_emulators.py \
  --parameters="${PARAMETERS}" \
  --observables "He4" "He4-radius" \
  --label="${LABEL}"

python sample_emulators.py \
  --parameters="${PARAMETERS}" \
  --observables "He4" "H3-halflife" \
  --label="${LABEL}"

python sample_emulators.py \
  --parameters="${PARAMETERS}" \
  --observables "He4-radius" "H3-halflife" \
  --label="${LABEL}"
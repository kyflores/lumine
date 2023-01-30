LUMINE_ROOT=$(dirname ${0})

source ${LUMINE_ROOT}/lumine-venv/bin/activate

python "${LUMINE_ROOT}"/src/lumine.py \
    --weights ${WEIGHTS} \
    --source ${SOURCE} \
    --stream ${CSCOREPORT} \
    --nt ${NTSERVER}

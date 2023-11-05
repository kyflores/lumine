LUMINE_ROOT=$(dirname ${0})

source ${LUMINE_ROOT}/lumine-venv/bin/activate

pushd ${LUMINE_ROOT}
python /src/lumine.py --config config/lumine.json
popd

# note: must have empty space between xx: [ xx ]
# -z tests if condition true, -n no tests if condition if false
mode=$1
if [ -z $mode ]; then
    echo "Debugging mode"
    python pud/algos/train_PointEnv.py --cfg configs/config_SafePointEnv_debug.yaml
else
    echo "Normal Mode"
    python pud/algos/train_PointEnv.py --cfg configs/config_SafePointEnv.yaml
fi
cuda_visible_devices=$1
root_demo_dir=$2
seed_range=$3
other_environs=$4
other_cfgs=$5
runners_per_gpu=${6-8}
echo other_environs $other_environs
echo other_cfgs $other_cfgs
echo root_demo_dir $root_demo_dir
echo seed_range $seed_range

num_gpus=$(($(echo "$cuda_visible_devices" | tr -cd ',' | wc -c)+1))
num_runners=$((num_gpus*runners_per_gpu))
echo num_gpus $num_gpus
echo num_runners $num_runners

for seed in $(seq $seed_range)
do
    demo_dir=$root_demo_dir/$seed
    echo seed = $seed
    echo demo dir = $demo_dir
    cmd="CUDA_VISIBLE_DEVICES=$cuda_visible_devices RAY_DEDUP_LOGS=0 $other_environs python -m evaluate $other_cfgs policy.seed $seed policy.demo_dir $demo_dir evaluate.num_runners $num_runners"
    echo cmd = $cmd
    eval $cmd
done
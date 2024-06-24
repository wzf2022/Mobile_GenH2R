conda activate genh2r
cd /share1/haoran/HRI/GenH2R

# sequential, staged
``` bash
python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name cartesian policy.wait_time 3. policy.cartesian.staged True policy.cartesian.verbose True env.visualize True
```

# sequential
``` bash
python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name cartesian policy.wait_time 3. policy.cartesian.verbose True env.visualize True
python -m evaluate BENCHMARK.SCENE_IDS "[11]" evaluate.use_ray False policy.name cartesian policy.wait_time 3. env.visualize True

CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name cartesian policy.wait_time 3.
```
```
success rate: 553/720=0.7680555555555556
contact rate: 20/720=0.027777777777777776
   drop rate: 48/720=0.06666666666666667
timeout rate: 99/720=0.1375
```

# sequential filter once
``` bash
CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name cartesian policy.wait_time 3. policy.cartesian.only_filter_once True
```
```
success rate: 552/720=0.7666666666666667
contact rate: 25/720=0.034722222222222224
   drop rate: 46/720=0.06388888888888888
timeout rate: 97/720=0.13472222222222222
non-deterministic?
```

``` bash
CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name cartesian policy.wait_time 3. policy.cartesian.only_filter_once True policy.demo_dir s0/sequential/train/cartesian

CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name offline policy.wait_time 3. policy.offline.demo_dir s0/sequential/train/cartesian
```
```
success rate: 552/720=0.7666666666666667
contact rate: 25/720=0.034722222222222224
   drop rate: 46/720=0.06388888888888888
timeout rate: 97/720=0.13472222222222222
evaluting uses 512.3335249423981 seconds
```

# Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning

Yuheng Lei, Sitong Mao, Shunbo Zhou, Hongyuan Zhang, Xuelong Li, Ping Luo

[[Paper]](https://arxiv.org/abs/2506.05985) [[Pretraining Checkpoint (LIBERO-90)]](https://huggingface.co/leiyuheng/DMPEL/tree/main)

## Installation
Please run the following commands in the given order to install the dependency and the [**LIBERO** benchmark](https://libero-project.github.io).
```
conda create -n dmpel python=3.8.13
conda activate dmpel
pip install -r requirements.txt
```

Then install the `libero` package:
```
pip install -e .
```

We leverage high-quality human teleoperation demonstrations for the task suites in **LIBERO**. To download the demonstration dataset, run:
```python
python benchmark_scripts/download_libero_datasets.py
```

For a detailed walk-through of the LIBERO benchmark, please either refer to the [documentation](https://libero-project.github.io/LIBERO/) or the [original paper](https://arxiv.org/abs/2306.03310).

## Training

We can starting training by running:
```shell
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python libero/lifelong/main.py seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=POLICY \
                               lifelong=ALGO
```


### Pretraining

- `BENCHMARK` from `[LIBERO_90]`
- `ALGO` from `[multitask]`
- `POLICY` from `[bc_foundation_policy_fft, bc_foundation_policy_frozen]`

We provide the template script of pretraining as follows:

```
sh exp_scripts/pretraining_scripts/run_chunkonlyfft_base_clip.sh
```

### Lifelong Learning

- `BENCHMARK` from `[LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_10]`
- `ALGO` from `[base, er, ewc, packnet, lotus, l2m, iscil, tail, dmpel]`
- `POLICY` from `[bc_foundation_policy_fft, bc_foundation_policy_frozen, bc_hierarchical_policy, bc_foundation_tail_policy, bc_foundation_l2m_policy, bc_foundation_iscil_policy, bc_foundation_dmpel_policy]`

We provide the scripts to reproduce results in the paper in `exp_scripts/lifelong_scripts`. For example, we can evaluate DMPEL in LIBERO-Goal by running:

```
sh exp_scripts/lifelong_scripts/dmpel.sh
```

Note that the pretrained model path should be the same as the final checkpoint you saved during pretraining. We also provide our [pretraining checkpoint](https://huggingface.co/leiyuheng/DMPEL/tree/main) to facilitate the replication of results presented in the main paper.

## Acknowledgements

This codebase is built with reference to the following repositories:

* [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
* [L2M](https://github.com/ml-jku/L2M)
* [LOTUS](https://github.com/UT-Austin-RPL/Lotus)
* [IsCiL](https://github.com/L2dulgi/IsCiL)
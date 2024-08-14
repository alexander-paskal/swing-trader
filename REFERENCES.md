https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- This section has a lot of good info on SOTA deep learning

tune.TuneConfig(reuse_actors=True) - saves time

torch compile
config = PPOConfig().framework(
    "torch",
    torch_compile_worker=True,
    torch_compile_worker_dynamo_backend="ipex",
    torch_compile_worker_dynamo_mode="default",
)


Dict ObservationsEnv
https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#dict-observations


Off-Policy with SAC
https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#multiprocessing-with-off-policy-algorithms


OptimizationParams = dict(
    dataloader=True,
    iterations = 30_000,
    densify_until_iter = 15_000,
    opacity_reset_interval = 3000,

    muti_mode = "neighbor",
    sample_win_size = 20,
    prune_num1 = 10,
    prune_std1 = 2,
    prune_num2 = 100,
    prune_std2 = 4.0,
    vial_view = [0,14]
)
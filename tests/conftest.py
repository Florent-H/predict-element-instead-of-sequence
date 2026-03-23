import pytest
from pred_elem_seq import AnnConfig
from pathlib import Path
from wakepy import keep


@pytest.fixture
def ann_conv_lin(building, n_lhs, n_lhs_uni, n_smp_uni, train_perc, seq_len, pred_len):
    from pred_elem_seq import ANN

    cfg_conv = AnnConfig(
        building=building,
        nn_type="conv_lin",
        pretrained=False,
        loss_func="mse",
        activation="prelu",
        solver="adamW",
        div_f_lr=2,
        fin_div_f_lr=4,
        pct_start=0.3,
        max_lr=2e-4,
        opt_betas=(0.85, 0.98),
        weight_decay=5e-2,
        batch_size=2048,
        epochs=10,
        n_lhs=n_lhs,
        n_lhs_uni=n_lhs_uni,
        n_smp_uni=n_smp_uni,
        train_perc=train_perc,
        seq_len=seq_len,
        pred_len=pred_len,
        n_lin_blocks=2,
        n_lin_layers=3,
        n_nodes=1024,
        n_conv_blocks=2,
        n_conv_layers=3,
        n_channels=128,
    )

    a_c = ANN(cfg_conv)

    yield a_c


@pytest.fixture
def ann_conv(building, n_lhs, n_lhs_uni, n_smp_uni, train_perc, seq_len, pred_len):
    from pred_elem_seq import ANN

    cfg_conv = AnnConfig(
        building=building,
        nn_type="conv",
        pretrained=False,
        loss_func="mse",
        activation="prelu",
        solver="adamW",
        div_f_lr=2,
        fin_div_f_lr=4,
        pct_start=0.3,
        max_lr=2e-4,
        weight_decay=5e-2,
        opt_betas=(0.9, 0.999),
        batch_size=10,
        epochs=20,
        n_lhs=n_lhs,
        n_lhs_uni=n_lhs_uni,
        n_smp_uni=n_smp_uni,
        train_perc=train_perc,
        seq_len=seq_len,
        pred_len=pred_len,
        n_lin_blocks=0,
        n_lin_layers=0,
        n_nodes=0,
        n_conv_blocks=6,
        n_conv_layers=6,
        n_channels=512,
    )

    a_c = ANN(cfg_conv)

    yield a_c


@pytest.fixture
def ann_model_path():
    # a_m_p = Path("pred_elem_seq/datafiles/ann/conv - Medium_Office - seq_8760 - pred_8760 - n_lhs_5041 - dec_0.05 - "
    #              "conv_512X6X6 - krnl_[10, 9, 8, 7, 6, 5] - trial_0.sav")
    # a_m_p = Path(
    #     "pred_elem_seq/datafiles/ann/conv_lin - Medium_Office - seq_8 - pred_1 - n_lhs_169 - dec_0.05 - lin_1024X3X2 - "
    #     "conv_128X3X2 - krnl_[8, 7, 6] - trial_weather.sav"
    # )
    a_m_p = Path(
        "pred_elem_seq/datafiles/ann/conv_lin - Medium_Office - seq_8 - pred_1 - n_lhs_4489 - dec_0.05 - lin_1024X3X2 - "
        "conv_128X3X2 - krnl_[8, 7, 6] - trial_weath_small.sav"
    )

    yield a_m_p


@pytest.fixture
def nn_type():
    n_t = "conv_lin"
    yield n_t


@pytest.fixture
def ann(nn_type, ann_conv_lin, ann_conv):
    if nn_type == "conv":
        ann = ann_conv
    elif nn_type == "conv_lin":
        ann = ann_conv_lin
    else:
        raise ValueError("nn_type must be 'conv' or 'conv_lin'")
    yield ann



@pytest.fixture
def n_lhs():
    n_l = 23**2 # 71**2 # 67**2 # 30_000
    yield n_l


@pytest.fixture
def n_lhs_train():
    n_l = 23**2 # 71**2 # 67**2
    yield n_l


@pytest.fixture
def n_lhs_uni():
    n_l_u = 0
    yield n_l_u


@pytest.fixture
def n_smp_uni():
    n_s_u = 0
    yield n_s_u


@pytest.fixture
def train_perc(n_lhs):
    if n_lhs in (23**2, 71**2):
        t_p = 0.8
    elif n_lhs == 67**2:
        t_p = 1
    else:
        t_p = 0
    yield t_p


@pytest.fixture
def scaler(building, n_lhs_train, seq_len):
    from pathlib import Path
    import pickle

    s_p = Path(
        f"pred_elem_seq/datafiles/scaler/scaler_orth - Medium_Office - n_lhs_{n_lhs_train} - seq_{seq_len} - "
        f"par_{building.n_unknown_params} - time_h.sav"
    )
    # s_p = Path(
    #     f"pred_elem_seq/datafiles/scaler/scaler_orth - Medium_Office - n_lhs_5041 - seq_8 - par_14 - fold_3 - time_h.sav"
    # )

    # read scaler from file
    with open(s_p, "rb") as fr:
        s = pickle.load(fr)

    yield s


@pytest.fixture
def pred_len(sim_time_idx):
    # p_l = len(sim_time_idx)
    p_l = 1
    yield p_l


@pytest.fixture
def seq_len(pred_len, sim_time_idx):
    if pred_len == len(sim_time_idx):
        s_l = pred_len
    else:
        s_l = 8
    yield s_l


@pytest.fixture
def run_ann():
    r_a = True
    yield r_a


@pytest.fixture
def get_dataset():
    g_d = True
    yield g_d


@pytest.fixture
def sim_precision():
    s_p = "float32"
    yield s_p


@pytest.fixture
def ann_datasets_train(ann, get_dataset, sim_precision, obj_names, capsys):
    from pred_elem_seq import AnnDatasets

    with keep.running(), capsys.disabled():
        a_d = AnnDatasets(
            ann.cfg, dataset_obj_names=obj_names, sim_precision=sim_precision
        )
        if get_dataset:
            a_d.get_datasets(run_lhs=True, shards=False, save_shards=False)

    yield a_d


@pytest.fixture
def ann_ds_weather_train(ann, get_dataset, sim_precision, obj_names, capsys):
    from pred_elem_seq import AnnDatasets

    with keep.running(), capsys.disabled():
        weather_list = list(
            Path("pred_elem_seq/datafiles/weather/ninja/train").iterdir()
        )
        a_d = AnnDatasets(
            ann.cfg, dataset_obj_names=obj_names, sim_precision=sim_precision
        )
        if get_dataset:
            a_d.get_datasets(
                run_lhs=False, shards=True, save_shards=True, weather_list=weather_list
            )

    yield a_d


@pytest.fixture
def ann_ds_weather_lr(ann, sim_precision, obj_names):
    from pred_elem_seq import AnnDatasets
    import torch
    from torch.utils.data import TensorDataset

    a_d = AnnDatasets(ann.cfg, dataset_obj_names=obj_names, sim_precision=sim_precision)
    # get training and testing ShardedIterableDatasets
    # get training shard paths
    shard_dir_train = Path(
        f"pred_elem_seq/datafiles/xy/shards/train/n_lhs_{ann.cfg.n_lhs}"
    )
    input_paths_train, label_paths_train = a_d.get_shard_path_lists(shard_dir_train)
    input_train = [torch.load(f_p, weights_only=False) for f_p in input_paths_train[:1]]
    label_train = [torch.load(f_p, weights_only=False) for f_p in label_paths_train[:1]]
    a_d.train_set = TensorDataset(
        torch.concat(input_train)[: 100 * 8760],
        torch.concat(label_train)[: 100 * 8760],
    )

    # get testing shard paths
    shard_dir_test = Path(
        f"pred_elem_seq/datafiles/xy/shards/test/n_lhs_{ann.cfg.n_lhs}"
    )
    input_paths_test, label_paths_test = a_d.get_shard_path_lists(shard_dir_test)
    input_test = [torch.load(f_p, weights_only=False) for f_p in input_paths_test[:1]]
    label_test = [torch.load(f_p, weights_only=False) for f_p in label_paths_test[:1]]
    a_d.test_set = TensorDataset(
        torch.concat(input_test),
        torch.concat(label_test),
    )

    yield a_d


@pytest.fixture
def ann_datasets_test(ann, scaler, get_dataset, sim_precision, obj_names, capsys):
    from pred_elem_seq import AnnDatasets

    with capsys.disabled():
        a_d_t = AnnDatasets(
            ann.cfg, dataset_obj_names=obj_names, sim_precision=sim_precision
        )
        if get_dataset:
            a_d_t.get_datasets(
                run_lhs=False,
                shards=False,
                save_shards=False,
                shuffle_doe=True,
                scaler=scaler,
            )

    yield a_d_t


@pytest.fixture
def ann_ds_test_lst(ann, get_dataset, sim_precision, capsys):
    from pred_elem_seq import AnnDatasets

    with keep.running(), capsys.disabled():
        a_d_l = {}
        for n_lhs, seq_len, nn_type in [
            (23**2, 8, "conv_lin"),
            # (71**2, 10, "conv_lin"), 
            # (71**2, 8, "conv_lin"), 
            # (71**2, 8760, "conv")
        ]:
            ann.cfg.n_lhs = n_lhs
            ann.cfg.seq_len = seq_len
            if seq_len == 8760:
                ann.cfg.pred_len = 8760
            else:
                ann.cfg.pred_len = 1
            ann.cfg.nn_type = nn_type
            a_d_l[(n_lhs, seq_len, nn_type)] = AnnDatasets(
                ann.cfg, dataset_obj_names=["heat", "cool"], sim_precision=sim_precision
            )
            if get_dataset:
                a_d_l[(n_lhs, seq_len, nn_type)].get_datasets(run_lhs=False)

    yield a_d_l


@pytest.fixture
def ann_ds_weather_test(ann, scaler, get_dataset, sim_precision, capsys):
    from pred_elem_seq import AnnDatasets

    with keep.running(), capsys.disabled():
        weather_list = list(
            Path("pred_elem_seq/datafiles/weather/ninja/test").iterdir()
        )
        a_d = AnnDatasets(
            ann.cfg, dataset_obj_names=["heat", "cool"], sim_precision=sim_precision
        )
        if get_dataset:
            a_d.get_datasets(
                run_lhs=False,
                scaler=scaler,
                shards=True,
                save_shards=False,
                weather_list=weather_list,
                # samp_per_weath=1000,
            )

    yield a_d


@pytest.fixture
def ann_ds_weather_test_lst(ann_ds_weather_test):
    n_lhs = ann_ds_weather_test.cfg.n_lhs
    seq_len = ann_ds_weather_test.cfg.seq_len
    nn_type = ann_ds_weather_test.cfg.nn_type
    a_d_t_w_l = {(n_lhs, seq_len, nn_type): ann_ds_weather_test}
    yield a_d_t_w_l


@pytest.fixture
def ann_ds_uni_test(ann, scaler, get_dataset, sim_precision, obj_names, capsys):
    from pred_elem_seq import AnnDatasets

    with capsys.disabled():
        a_d_u = AnnDatasets(
            ann.cfg, dataset_obj_names=obj_names, sim_precision=sim_precision
        )
        if get_dataset:
            a_d_u.get_datasets(run_lhs=False, shuffle_doe=False, scaler=scaler)

    yield a_d_u


@pytest.fixture
def ann_ds_uni_test_lst(ann, scaler, get_dataset, sim_precision, obj_names, capsys):
    from pred_elem_seq import AnnDatasets

    with keep.running(), capsys.disabled():
        a_d_u = {}
        for n_lhs_uni, seq_len, nn_type in [(10, 8, "conv_lin")]:
            ann.cfg.n_lhs = 0
            ann.cfg.n_lhs_uni = n_lhs_uni
            ann.cfg.seq_len = seq_len
            if seq_len == 8760:
                ann.cfg.pred_len = 8760
            else:
                ann.cfg.pred_len = 1
            a_d_u[(n_lhs_uni, seq_len, nn_type)] = AnnDatasets(
                ann.cfg, dataset_obj_names=obj_names, sim_precision=sim_precision
            )
            if get_dataset:
                a_d_u[(n_lhs_uni, seq_len, nn_type)].get_datasets(
                    run_lhs=False, scaler=scaler
                )

    yield a_d_u


@pytest.fixture
def weather_list():
    w_l = list(Path("pred_elem_seq/datafiles/weather/ninja/test").iterdir())
    return w_l


@pytest.fixture
def building(
    name, year, sim_time_idx, unknown_params, var_data, obj_names, weather_path
):
    from pred_elem_seq import Building

    b = Building(
        name, year, sim_time_idx, unknown_params, var_data, obj_names, weather_path
    )
    yield b


@pytest.fixture
def year():
    y = 2021
    yield y


@pytest.fixture
def name():
    n = "Medium_Office"
    yield n


@pytest.fixture
def sim_time_idx(year):
    import pandas as pd

    t_i = pd.date_range(f"{year}-01-01 00:30", f"{year}-12-31 23:30", freq="h")
    yield t_i


@pytest.fixture
def weather_path():
    from pathlib import Path

    w_p = Path(
        "pred_elem_seq/datafiles/weather/CAN-QC-Montreal-Trudeau-Intl-Airport-716270-WhiteBox-2021.epw"
    )
    yield w_p


@pytest.fixture
def unknown_params(name):
    from pred_elem_seq import get_parameters
    from pathlib import Path

    param_path = Path(
        f"pred_elem_seq/datafiles/inputs/unknown_params/parameters_{name}.csv"
    )
    return get_parameters(param_path)


@pytest.fixture
def var_data(name):
    import pandas as pd
    from pathlib import Path

    # get weather and schedule variables as dataframe
    v_d = pd.read_csv(
        Path(f"pred_elem_seq/datafiles/inputs/variables/{name}.csv"), index_col=0
    )
    return v_d


@pytest.fixture
def obj_names():
    o_n = ["heat", "cool"]
    yield o_n


@pytest.fixture
def variables():
    vrbls = [
        "TOut",
        "relHum",
        "uWind",
        "diffRad",
        "dirRad",
        "fOcc",
        "fLight",
        "fEquip",
        "fFans",
        "fTHeat",
        "fTCool",
        "hourOfDay",
        "dayOfWeek",
    ]
    yield vrbls

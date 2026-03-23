from wakepy import keep


def test_get_surrogate(ann, ann_datasets_train, capsys):
    with keep.running(), capsys.disabled():
        ann.train_test(ann_datasets_train, trial_name="single_weath")


def test_get_surrogate_weather(ann, ann_ds_weather_train, capsys):
    with keep.running(), capsys.disabled():
        ann.train_test(ann_ds_weather_train, trial_name="multiple_weath")


def test_get_ann_datasets(ann, capsys):
    from pred_elem_seq import AnnDatasets

    with keep.running(), capsys.disabled():
        a_d = AnnDatasets(ann.cfg, dataset_obj_names=["heat", "cool"])
        a_d.get_datasets(run_lhs=True)


def test_find_lr(ann, ann_datasets_train, capsys):
    with keep.running(), capsys.disabled():
        ann.find_lr(ann_datasets_train)


def test_find_lr_weather(ann, ann_ds_weather_lr, capsys):
    with keep.running(), capsys.disabled():
        ann.cfg.obj_names = ["heat"]
        ann.find_lr(ann_ds_weather_lr)


def test_k_fold_cross_val(ann_conv_lin, run_ann, capsys):
    from pred_elem_seq import k_fold_cross_val

    with keep.running(), capsys.disabled():
        k_fold_cross_val(ann_conv_lin, train=True, run_ann=run_ann)


def test_get_ann_test_res(ann_ds_test_lst, run_ann, capsys):
    from pred_elem_seq import get_ann_test_res

    with keep.running(), capsys.disabled():
        get_ann_test_res(ann_ds_test_lst, run_ann=run_ann)


def test_get_ann_weather_test_res(ann_ds_weather_test_lst, run_ann, capsys):
    from pred_elem_seq import get_ann_test_res

    with keep.running(), capsys.disabled():
        get_ann_test_res(ann_ds_weather_test_lst, run_ann=run_ann)


def test_get_ann_test_unires(ann_ds_uni_test_lst, run_ann, capsys):
    from pred_elem_seq import get_ann_test_res

    with capsys.disabled():
        get_ann_test_res(ann_ds_uni_test_lst, run_ann=run_ann)


def test_year_sum_res(ann_model_path, ann_datasets_test, run_ann, capsys):
    from pred_elem_seq import year_sum_res

    with capsys.disabled():
        year_sum_res(ann_model_path, ann_datasets_test, run_ann=run_ann)


def test_year_sum_unires(ann_model_path, ann_ds_uni_test, run_ann, capsys):
    from pred_elem_seq import year_sum_res

    with capsys.disabled():
        year_sum_res(ann_model_path, ann_ds_uni_test, run_ann)

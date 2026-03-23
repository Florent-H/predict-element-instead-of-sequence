from wakepy import keep


def test_plot_onecyclelr_schedule(ann_conv_lin, capsys):
    from pred_elem_seq import plot_onecyclelr_schedule

    with capsys.disabled():
        plot_onecyclelr_schedule(ann_conv_lin)


def test_predict_year(ann_model_path, ann_ds_weather_test, run_ann, capsys):
    from pred_elem_seq import predict_year

    with capsys.disabled():
        predict_year(ann_model_path, ann_ds_weather_test, run_ann)


def test_error_plots(ann_model_path, ann_ds_weather_test, run_ann, capsys):
    from pred_elem_seq import error_plots

    with keep.running(), capsys.disabled():
        error_plots(ann_model_path, ann_ds_weather_test, run_ann)


def test_time_of_day_errors(ann_model_path, ann_ds_weather_test, run_ann, capsys):
    from pred_elem_seq import time_of_day_errors

    with capsys.disabled():
        time_of_day_errors(ann_model_path, ann_ds_weather_test, run_ann)


def test_plot_climate_map_errors(ann_model_path, ann_ds_weather_test, weather_list):
    from pred_elem_seq import plot_climate_map_errors

    plot_climate_map_errors(ann_model_path, ann_ds_weather_test, weather_list)


def test_make_climate_map_paul(ann_model_path, ann_ds_weather_test):
    from pred_elem_seq import make_climate_map_paul

    make_climate_map_paul()

def test_write_var_data(building, variables, capsys):
    from pred_elem_seq import write_var_data

    with capsys.disabled():
        write_var_data(building, variables)


def test_write_var_data_multiple_weather(building, variables, capsys):
    from pred_elem_seq import write_var_data, clear_sim_folders
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor as Pool
    from itertools import repeat

    with capsys.disabled():
        # get list of weather paths
        weather_paths = Path("pred_elem_seq/datafiles/weather/ninja/test").iterdir()

        clear_sim_folders()
        pool = Pool()
        pool.map(write_var_data, repeat(building), repeat(variables), weather_paths)
        pool.shutdown()

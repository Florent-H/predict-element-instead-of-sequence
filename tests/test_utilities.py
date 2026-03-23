from pathlib import Path


def test_get_n_model_params(ann_model_path, capsys):
    from pred_elem_seq import get_n_model_params

    with capsys.disabled():
        get_n_model_params(ann_model_path)


def test_df_to_latex():
    from pred_elem_seq import df_to_latex

    csv_path = Path(
        "pred_elem_seq/datafiles/results/ann_test_results_sorted_by_size.csv"
    )
    df_to_latex(csv_path, vertical_header=False)

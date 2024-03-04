import os


class CONFIG:
    PATH_INPUT = "./data"
    PATH_TRS = os.path.join(PATH_INPUT, "trainset")
    PATH_TRS_AN = os.path.join(PATH_TRS, "trainset_true.csv")
    PATH_TS = os.path.join(PATH_INPUT, "testset")
    PATH_TS_AN = os.path.join(PATH_TS, "testset_data.csv")
    PATH_AN = os.path.join(PATH_INPUT, "clinical_annotation.csv")
    PATH_SUB = os.path.join(PATH_INPUT, "sample_submission.csv")

    cols_annotation = ["GENDER", "LYMPH_COUNT", "age"]
    col_label = "LABEL"

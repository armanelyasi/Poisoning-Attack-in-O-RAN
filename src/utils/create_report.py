import pandas as pd
from ydata_profiling import ProfileReport
from etl.extractor import load_dataframe
from constants import variable_description

def create_report(merged_df, filename):
    profile = ProfileReport(
        title=f"Dataset Adversarial AI",
        dataset={
            "description": f"Report for Adversarial AI",
            "copyright_holder": "Andrea Lacava",
            "copyright_year": "2024",
            "url": "https://www.andrealacava.com",
        },
        variables={
            "descriptions": variable_description
        },
        minimal=True,
        missing_diagrams={"heatmap": False, "matrix": False, "bar": False},
        correlations={
            "auto": {"calculate": False},
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": False},
            "cramers": {"calculate": False},
        },
        interactions=None,
        duplicates=None,
    )
    # interactions = {'targets': columns_state},

    profile.df = merged_df
    profile.to_file(f"{filename}.html")


if __name__ == "__main__":
    pd.set_option("display.max_rows", None, "display.max_columns", 14)
    pd.set_option('expand_frame_repr', False)

    print("Load dataset")
    df = load_dataframe(use_saved=True)
    # df = load_dataframe(preprocess=True, save=True)

    print("Create report")
    create_report(df, "report")

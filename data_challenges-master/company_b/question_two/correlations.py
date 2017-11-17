from scipy import stats
import time
import pandas as pd


def corr_with_wages(df):
    correlations = {}
    columns = df.columns.tolist()
    start = time.time()
    for col in columns:
        print(col)
        correlations["wages" + '___' + col] = stats.pearsonr(df["lca_case_wage_rate_from"].values, df[col].values)
        print(time.time() - start)
        start = time.time()
    results = pd.DataFrame.from_dict(correlations, orient="index")
    results.columns = ["correlation", "pvalues"]
    results.sort_index(inplace=True)
    return results


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


df = pd.read_csv("1hb_2014.csv")

corr_df = corr_with_wages(df)

print_full(corr_df)

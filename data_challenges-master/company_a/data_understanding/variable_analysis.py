import pandas as pd
# Understanding LoanReason


def segment_by_age(df):
    bins = [0, 21, 27, 33, 39, 45, 55, 100]
    group_names = [
        "young",
        "early_to_mid_20s",
        "late_20s_to_early_30s",
        "early_30s_to_late_30s",
        "late_30s_to_mid_40s",
        "mid_40s_to_mid_50s",
        "mid_50s_to_old_age"
    ]
    df["age_categories"] = pd.cut(df["Age"], bins, labels=group_names)
    return df, group_names

# Queries:
#
# Query: By grouping loan reason in tranches,
# how do the reasons for the loan being taken out change?
#
# Query: Within each age category,
# how does the amount in checking in savings
# affect the loan reason?
#
# Query: How do the other variables change as loan reason changes?

def describe_loan_reason_by_category(df, categories):
    # Answering the first question
    for category in categories:
        segmentation = df[df["age_categories"] == category]
        print("Category:", category)
        print(segmentation["LoanReason"].value_counts())


def describe_loan_reason_and_financials_by_category(df, categories):
    # Answering the second question
    dicter = {}
    for category in categories:
        dicter[category] = {}
        segmentation = df[df["age_categories"] == category]
        print("Category:", category)
        loan_reasons = set(segmentation["LoanReason"])
        for loan_reason in loan_reasons:
            print("Reason for the loan:", loan_reason)
            seg_by_loan_reason = segmentation[segmentation["LoanReason"] == loan_reason]
            print("Checking Account Frequencies")
            print(seg_by_loan_reason["CheckingAccountBalance"].value_counts())
            print("Savings Account Frequencies")
            print(seg_by_loan_reason["SavingsAccountBalance"].value_counts())


if __name__ == '__main__':
    df = pd.read_csv("ds-all-data.csv")
    df_age_seg, categories = segment_by_age(df)
    describe_loan_reason_by_category(df_age_seg, categories)
    print()
    describe_loan_reason_and_financials_by_category(df_age_seg, categories)

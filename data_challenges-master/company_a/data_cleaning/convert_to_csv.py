import pandas as pd


# This is being turned off because I am only doing a typecast
# and copy, which doesn't seem to have any side effects.
pd.options.mode.chained_assignment = None  # default='warn'

def tsv_to_df(filename: str, column_lookup: dict):
    with open(filename, "r") as f:
        tsv_contents = f.read() 
    tsv_contents = tsv_contents.split("\n")
    reformatted_tsv_contents = [line.split() for line in tsv_contents]
    base_name = filename.split(".")[0]
    csv_filename = base_name + ".csv"
    column_names = column_lookup[base_name]
    return pd.DataFrame(reformatted_tsv_contents, columns=column_names)


def is_number_sequence(sequence: list) -> bool:
    sequence.sort()
    start = sequence[0]
    end = sequence[-1]
    return list(range(start, end+1)) == sequence


def understanding_customer_ids(dfs: list):
    customer_ids = []
    for filename, df in dfs.items():
        customer_ids = df["CustomerID"]
        customer_ids = set(customer_ids)
        print("Filename: ", filename)
        print("Number of customer IDs across all tsv's:", len(customer_ids))
        customer_ids = [int(elem) for elem in customer_ids if pd.notnull(elem)]
        customer_ids.sort()
        if is_number_sequence(customer_ids):
            print("And the customer id's are an inclusive integer sequence")
        else:
            print("And the customer id's are not an inclusive integer sequence")


def segment_for_customer_id(dfs: list):
    nulls_removed = []
    for df in dfs:
        tmp_df = df[df["CustomerID"].notnull()]
        tmp_df["CustomerID"] = tmp_df["CustomerID"].astype(int)
        nulls_removed.append(tmp_df)
    values_removed = []
    for df in nulls_removed:
        values_removed.append(df[df["CustomerID"] < 750])
    return values_removed


def generate_final_df(dfs: list):
    final_df = dfs[0]
    for df in dfs[1:]:
        final_df = pd.merge(final_df, df, on="CustomerID")
    final_df.to_csv("ds-all-data.csv", index=False)


if __name__ == '__main__':
    column_lookup = {
        'ds-credit': 'CustomerID CheckingAccountBalance DebtsPaid SavingsAccountBalance CurrentOpenLoanApplications'.split(),
        'ds-app': 'CustomerID LoanPayoffPeriodInMonths LoanReason RequestedAmount InterestRate Co-Applicant'.split(),
        'ds-borrower': 'CustomerID YearsAtCurrentEmployer YearsInCurrentResidence Age RentOrOwnHome TypeOfCurrentEmployment NumberOfDependantsIncludingSelf'.split(),
        'ds-result': 'CustomerID WasTheLoanApproved'.split()
    }
    dfs = []
    for ind, base_name in enumerate(column_lookup):
        filename = base_name + ".tsv"
        dfs.append(tsv_to_df(filename, column_lookup))

    dfs = segment_for_customer_id(dfs)
    generate_final_df(dfs)

    # uncomment this function to understand more about
    # customer IDs - they are not the same across all
    # the csv files
    # filename_dfs = {csv_file:pd.read_csv(csv_file)
    #                 for csv_file in glob("*.csv")}
    # understanding_customer_ids(filename_dfs)

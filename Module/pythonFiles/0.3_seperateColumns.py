
def seperate_columns(data):

    vars = [
        "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
    ]

    for col in vars:
        data[col] = data[col].astype("object")
        print(f"Changed {col} to object type")

    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]

    print("\nContinuous columns: \n")
    print(list(cont_vars.columns), indent=4)
    print("\n Categorical columns: \n")
    print(list(cat_vars.columns), indent=4)
def drop_empty_cols_and_rows(df_in):
    df = df_in.copy()
    print("Original shape:", df.shape)

    # Drop columns where ALL values are NaN
    df = df.dropna(axis=1, how='all')
    print("After dropping all-empty columns:", df.shape)
    # Drop rows where ALL values are NaN
    df = df.dropna(axis=0, how='all')

    print("After dropping all-empty rows:", df.shape)
    df.head()
    return df

def drop_columns(df_in, big_KOI_data=True):
    if big_KOI_data:
        # below if all
        columns_to_remove = ["rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_score",
                            "koi_tce_delivname"]
    else:
        # Below if cumulative
        columns_to_remove = ["loc_rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_score",
                            "koi_tce_delivname"]
    df_rem = df_in.drop(columns=columns_to_remove)
    return df_rem
import pandas as pd
import numpy as np
import env
from sklearn.model_selection import train_test_split

def wrangle_zillow():
    """
    This function fetches, cleans, and splits the Zillow dataset into train, validate, and test sets.

    Parameters:
        target_col (str): The name of the target column for stratification in the split.

    Returns:
         train, validate, test and  DataFrame.
    """

    # Fetch the data
    query = ("SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips "
             "FROM properties_2017 "
             "LEFT JOIN propertylandusetype USING (propertylandusetypeid) "
             "WHERE propertylandusedesc = 'Single Family Residential'")
    url = env.get_db_url('zillow')
    df = pd.read_sql(query, url)

    # Clean the data
    df = df.dropna()
    cols_to_convert = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'yearbuilt', 'fips']
    df[cols_to_convert] = df[cols_to_convert].astype(int)

    # Split the data
    train, validate_test = train_test_split(df, train_size=0.6, random_state=123)
    validate, test = train_test_split(validate_test, train_size=0.5, random_state=123)

    return train, validate, test, df
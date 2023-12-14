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
    df = df.rename(columns={'bedroomcnt': 'bedrooms',
                            'bathroomcnt': 'bathrooms',
                            'calculatedfinishedsquarefeet': 'area',
                            'taxvaluedollarcnt': 'appraisal',
                            'fips': 'county',
                            'taxamount': 'tax'})
    df = df.dropna()

    make_ints = ['bedrooms', 'area', 'yearbuilt']

    for col in make_ints:
        df[col] = df[col].astype(int)

    df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})

    # Split the data
    train, validate_test = train_test_split(df, train_size=0.6, random_state=123)
    validate, test = train_test_split(validate_test, train_size=0.5, random_state=123)

    return train, validate, test, df
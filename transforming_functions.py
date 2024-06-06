def turn_price_to_float(price_str):
    try:
        return float(price_str.replace('$', '').replace(',', '').strip())
    except ValueError:
        return None


def turn_string_to_numbers(x):
    if x is None:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df
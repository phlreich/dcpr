import numpy as np
import pandas
import pandas as pd
from random import randint
from utils.utils import delete_random_values
from streamlit_app import imputation_feats, impute_listwise_deletion
from sklearn.linear_model import LinearRegression


def multiple_linear_regression_imputation(df: pd.DataFrame, target_col: str):
    target = df[target_col]
    df_b = df.drop(target_col, axis=1)
    model = LinearRegression()
    y = target.to_numpy()
    x = df_b.to_numpy()
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)

    print(f"coefficient of determination: {r_sq}")
    print(f"intercept: {model.intercept_}")
    print(f"coefficients: {model.coef_}")

    # predict
    print("\n")
    y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
    print(f"predicted response:\n{y_pred}")


if __name__ == '__main__':
    # test_df = pd.read_csv("../resources/heart_new.csv")
    test_df = pd.read_csv("../data/heart.csv")
    # heart_df = delete_random_values(heart_df)
    # heart_df.to_csv("../resources/heart_new.csv", index=False)
    # print(heart_df)

    # heart_df = impute_median(heart_df)
    # print(heart_df)

    # for feat in features:
    #     print(f"{feat}: {heart_df[feat].mean()}")
    #
    # for feat in features:
    #     print(f"{feat}: {heart_df[feat].mean()}")

    print(test_df)

    heart_df = impute_listwise_deletion(test_df)
    print(test_df)
    multiple_linear_regression_imputation(test_df, "thal")

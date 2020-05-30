import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as st


BORROW_FILE = '/Users/holly/Desktop/毕设/Data/MonthlyTradingData/Transferred/cycle left.csv'
LEND_FILE = '/Users/holly/Desktop/毕设/Data/MonthlyTradingData/Transferred/pnum left.csv'
PANIC_VOLUME_FILE = '/Users/holly/Desktop/毕设/Data/GrangerModel/MATLAB_mtx.csv'


def concat_df():
    matlab_df = pd.read_csv(PANIC_VOLUME_FILE)
    lend_df = pd.read_csv(LEND_FILE)
    lend_df = lend_df.loc[1:, matlab_df.columns[:34]]

    df_list = []
    for i in range(34):
        p1 = matlab_df.iloc[:,[i,i+34]]
        p2 = lend_df.iloc[:, i]
        temp_s = p1.iloc[:-1, 1].values
        p1.iloc[0, 1] = np.nan
        p1.iloc[1:, 1] = temp_s
        cat_df = pd.concat([p1,p2], axis=1)

        cat_df = cat_df.dropna(axis=0,how='any')
        cat_df.columns = ['netflow', 'panic', 'pnum']
        # cat_df.columns = ['netflow', 'panic']
        df_list.append(cat_df)

    return df_list


def coint_test():
    df_list = concat_df()
    count = 0
    for df in df_list:
        count += 1
        print("===========", count, "===========")
        f1 = adfuller(df['netflow'])
        f2 = adfuller(df['panic'])
        # f3 = adfuller(df['pnum'])
        print(f1, f2)
        if True:
        # if not sum([f1, f2]) == 0 or not sum([f1, f2]) == 2:
            # X = df[['panic', 'pnum']]  # 是否需要考虑滞后项？
            X = df[['panic']]
            X = st.add_constant(X)
            y = df['netflow']
            coint = st.coint(y, X)
            print(coint[1])
            ols = st.OLS(y, X, missing='drop')
            res = ols.fit()
            print(res.summary())
            print('Panic pvalue:', res.pvalues['panic'])
            if st.adfuller(res.resid)[1] < 0.05:
                print("Steady Residual")
            else:
                print("Not Steady!")

    return 0


def adfuller(array):
    result = st.adfuller(array)
    if result[1] < 0.05:
        return 0
    else:
        lag_result = st.adfuller(array.diff()[1:])
        if lag_result[1] < 0.05:
            return 1
        else:
            lag_result = st.adfuller(array.diff().diff().dropna())
            if lag_result[1] < 0.05:
                return 2
            return 4


if __name__ == '__main__':
    coint_test()
import pandas as pd
import numpy as np
import statsmodels.regression.linear_model as lm


data = pd.read_csv('/Users/holly/Desktop/毕设/Data/GrangerModel/MATLAB_mtx.csv')
data.set_axis(range(0, 68),axis=1, inplace=True)
X_data = data.iloc[:, 34:]
X_data = X_data.interpolate()
X_data = X_data.fillna(X_data.mean())
y_data = data.iloc[:, :34]
y_data = y_data.interpolate()
y_data = y_data.fillna(y_data.mean())
Params_df = pd.DataFrame(columns=y_data.columns, index=['const',0, 1])


def test_demo():
    Wald_test_list = []
    e0 = pd.DataFrame(index=X_data.index, columns=y_data.columns)
    for i in range(34):
        x_lag1 = np.nan * np.ones(X_data.shape[0])
        y_lag1 = np.nan * np.ones(X_data.shape[0])
        x_lag1[1:] = X_data.iloc[:-1, i]
        y_lag1[1:] = y_data.iloc[:-1, i]
        y_reg = y_data.iloc[:, i]

        X_exo = pd.concat([pd.Series(y_lag1), pd.Series(x_lag1)], axis=1)
        X_exo = lm.add_constant(X_exo)
        model = lm.OLS(y_reg, X_exo, missing='drop')
        res = model.fit()
        R = np.eye(len(res.params))[2]
        print(res.params)
        print(R)
        print(res.wald_test(R))
        Params_df.iloc[:, i] = res.params
        Wald_test_list.append(res.wald_test(R).fvalue[0][0])  # 计算初始样本的F-value
        e0.iloc[:, i] = y_reg-res.params['const']-res.params[0]*y_lag1  # 长度为34,计算残值

    return e0, Wald_test_list


def iteration(e0, depth):
    y_star = y_data
    Wald_iter_df = pd.DataFrame(index=range(depth), columns=y_star.columns)
    for d in range(depth):
        e = e0.sample(frac=1).reset_index(drop=True)
        for i in range(34):
            ei = e.iloc[:, i]
            y_star_lag = pd.Series(index=y_star.index)
            y_star_lag[1:] = y_star.iloc[:-1, i]

            y_star.iloc[1:, i] = ei + Params_df.loc['const', i] + Params_df.loc[0, i] * y_star_lag[1:]  # 生成y-star
            # y_star.iloc[0, i] = y_data.iloc[0, i]

            x_lag1 = pd.Series(index=y_star.index)
            y_lag1 = pd.Series(index=y_star.index)
            x_lag1[1:] = X_data.iloc[:-1, i]
            y_lag1[1:] = y_star.iloc[:-1, i]
            y_reg = y_star.iloc[:, i]

            X_exo = pd.concat([y_lag1, x_lag1], axis=1)
            X_exo = lm.add_constant(X_exo)
            model = lm.OLS(y_reg, X_exo, missing='drop')
            res = model.fit()
            R = np.eye(len(res.params))[2]
            # print(res.wald_test(R).fvalue[0][0])
            # Wald_iter_df.iloc[d, i] = res.wald_test(R).fvalue[0][0]

            try:
                wald_i = res.wald_test(R).fvalue[0][0]
            except ValueError:
                Wald_iter_df.iloc[d, i] = np.nan
                print(d, i, "Appear")
                # print(X_exo)
            else:
                Wald_iter_df.iloc[d, i] = wald_i

    return Wald_iter_df


def SUR_model(type):
    from linearmodels.system import SUR
    from collections import OrderedDict
    import statsmodels.regression.linear_model as smlm

    Equation = OrderedDict()
    for i in range(34):
        x_lag1 = np.nan * np.ones(X_data.shape[0])
        y_lag1 = np.nan * np.ones(X_data.shape[0])
        x_lag1[1:] = X_data.iloc[:-1, i]
        y_lag1[1:] = y_data.iloc[:-1, i]
        y_reg = y_data.iloc[:, i]
        y_reg.name = 'netflow_' + str(i)

        X_exo = pd.concat([pd.Series(y_lag1), pd.Series(x_lag1)], axis=1)
        X_exo = smlm.add_constant(X_exo)
        # X_exo = X_exo.iloc[1:, :]
        X_exo.columns = ['const', 'netflow_lag1', 'panic']

        name = 'Platform_' + str(i)
        Equation[name] = {'dependent': y_reg, 'exog': X_exo}

    reg = SUR(Equation).fit(method=type)
    print(reg)

    return reg, Equation


def iteration_SUR(result, depth, equ):
    from linearmodels.system import SUR

    y_star = y_data
    # Wald_iter_df = pd.DataFrame(index=range(depth), columns=y_star.columns)
    Wald_iter_lag = pd.DataFrame(index=range(depth), columns=y_star.columns)
    Wald_iter_const = pd.DataFrame(index=range(depth), columns=y_star.columns)
    e0 = result.resids
    params = result.params
    Equation = equ

    for d in range(depth):
        e = e0.sample(frac=1).reset_index(drop=True)

        for i in range(34):
            name = 'Platform_' + str(i)
            ei = pd.Series(index=y_star.index)
            ei.iloc[1:] = e.iloc[:, i].values
            y_star_lag = pd.Series(index=y_star.index)
            y_star_lag[1:] = y_star.iloc[:-1, i]

            y_star.iloc[1:, i] = ei + params.iloc[3*i] + params.iloc[3*i+1] * y_star_lag[1:]

            y_reg = y_star.iloc[:, i]
            Equation[name]['dependent'] = y_reg
            Equation[name]['exog']['netflow_lag1'].iloc[1:] = y_star.iloc[:-1, i].values

        reg_new = SUR(Equation).fit(method='ols')
        Wald_iter_lag.iloc[d, :] = np.square(reg_new.tstats.iloc[list(range(1, 102,3))]).values
        Wald_iter_const.iloc[d, :] = np.square(reg_new.tstats.iloc[list(range(0, 102, 3))]).values

    return Wald_iter_lag,Wald_iter_const


if __name__ == '__main__':
    reg, equ = SUR_model('ols')
    print(np.square(reg.tstats.iloc[list(range(2, 102, 3))]))
    #Wald_iter_lag, Wald_iter_const = iteration_SUR(reg, 10000, equ)
    #Wald_iter_lag.to_csv('/Users/holly/Desktop/毕设/Data/GrangerModel/Bootstrap_Wald_lag.csv',index=False)
    #Wald_iter_const.to_csv('/Users/holly/Desktop/毕设/Data/GrangerModel/Bootstrap_Wald_const.csv', index=False)
#e0, Wald_test_list = test_demo()
# print(Wald_test_list)
# print(e0.isna().sum(1))
# Wald_iter_df = iteration(e0, 10000)
# Wald_iter_df.to_csv('/Users/holly/Desktop/毕设/Data/GrangerModel/Bootstrap_Wald2.0.csv',index=False)
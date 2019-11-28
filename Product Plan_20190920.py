# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191128"

import pandas as pd
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
import ffn
from tqdm import tqdm
import statsmodels.api as sm
from scipy import linalg
import scipy.optimize as sco
import datetime
from datetime import date
import seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import  ledoit_wolf
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'c:\windows\fonts\simkai.ttf',size = 15)

#please fix Wind python API if you possess a Wind account
from WindPy import *
w.start()

class _ProductPlan_:
    def __init__(self):
        return

    #define the function to calculate the performance of the strategy
    #input: the dataframe of strategy which has a column called " net asset value"
    #output: total average reutrn of the strategy
    #        t-statistics test result for the reutrn
    #        standard deviation
    #        Maximum Drawdown
    #        acccumulated return
    #        sharp ratio % take risk free rate as 3%; have alternative for non-zero risk free rate
    #        average return to Value at Risk %historical VaR at 99% confidence level
    #        average return to Conditional Value at Risk %historical CVaR at 99% confidence level
    def performance(self,strategy):
        def MaxDrawdown(return_list):
            RET_ACC = []
            sum = 1
            for i in range(len(return_list)):
                sum = sum * (return_list[i] + 1)
                RET_ACC.append(sum)
            index_j = np.argmax(np.array((np.maximum.accumulate(RET_ACC) - RET_ACC) / np.maximum.accumulate(RET_ACC)))
            index_i = np.argmax(RET_ACC[:index_j])
            MDD = (RET_ACC[index_i] - RET_ACC[index_j]) / RET_ACC[index_i]
            return sum, MDD, RET_ACC
        """def MaxDrawdown2(return_list):
            value = (1 + return_list).cumprod()
            MDD = ffn.calc_max_drawdown(value)
            return -MDD"""
        # 如果要考虑rf的话启用下面这个sharp函数
        def sharp(return_list, std):
            returnew = pd.DataFrame(return_list, columns=['R'])
            m = pd.concat([returnew.R], axis=1)
            ret_adj = np.array(m)
            sharpratio = np.average(ret_adj) * 12 / std
            return sharpratio

        def Reward_to_VaR(strategy=strategy, alpha=0.99):
            RET = strategy.nav.pct_change(1).fillna(0)
            sorted_Returns = np.sort(RET)
            index = int(alpha * len(sorted_Returns))
            var = abs(sorted_Returns[index])
            RtoVaR = np.average(RET) / var
            return -RtoVaR

        def Reward_to_CVaR(strategy=strategy, alpha=0.99):
            RET = strategy.nav.pct_change(1).fillna(0)
            sorted_Returns = np.sort(RET)
            index = int(alpha * len(sorted_Returns))
            sum_var = sorted_Returns[0]
            for i in range(1, index):
                sum_var += sorted_Returns[i]
                CVaR = abs(sum_var / index)
            RtoCVaR = np.average(RET) / CVaR
            return -RtoCVaR

        ts = strategy.nav.pct_change(1).fillna(0)
        RET = (strategy.nav[strategy.shape[0] - 1] /strategy.nav[0])** (252 / strategy.shape[0]) - 1
        T = stats.ttest_1samp(ts, 0)[0]
        STD = np.std(ts) * np.sqrt(252)
        MDD = MaxDrawdown(ts)[1]
        ACC = MaxDrawdown(ts)[0]
        SHARP = (RET - 0.03) / STD
        R2VaR = Reward_to_VaR(strategy)
        R2CVaR = Reward_to_CVaR(strategy)
        print('annual-return', round(RET, 4))
        print('t-statistic', round(T, 4))
        print('volitility', round(STD, 4))
        print('MaxDrawdown', round(MDD, 4))
        print('Accumulated return', round(ACC, 4))
        print('sharp-ratio', round(SHARP, 4))
        print('Reward_to_VaR', round(R2VaR, 4))
        print('Reward_to_CVaR', round(R2CVaR, 4))
        return RET, T, STD, MDD, ACC, SHARP, R2VaR, R2CVaR

    #define the function to calculate the performance of the strategy for each year
    #take a year as 252 days
    def performance_anl(self,strategy):
        def MaxDrawdown(return_list):
            RET_ACC = []
            sum = 1
            for i in range(len(return_list)):
                sum = sum * (return_list[i] + 1)
                RET_ACC.append(sum)
            index_j = np.argmax(np.array((np.maximum.accumulate(RET_ACC) - RET_ACC) / np.maximum.accumulate(RET_ACC)))
            index_i = np.argmax(RET_ACC[:index_j])
            MDD = (RET_ACC[index_i] - RET_ACC[index_j]) / RET_ACC[index_i]
            return sum, MDD, RET_ACC

        """def MaxDrawdown2(return_list):
            value = (1 + return_list).cumprod()
            MDD = ffn.calc_max_drawdown(value)
            return -MDD"""

        def Reward_to_VaR(strategy=strategy, alpha=0.99):
            RET = strategy.nav.pct_change(1).fillna(0)
            sorted_Returns = np.sort(RET)
            index = int(alpha * len(sorted_Returns))
            var = abs(sorted_Returns[index])
            RtoVaR = np.average(RET) / var
            return -RtoVaR

        def Reward_to_CVaR(strategy=strategy, alpha=0.99):
            RET = strategy.nav.pct_change(1).fillna(0)
            sorted_Returns = np.sort(RET)
            index = int(alpha * len(sorted_Returns))
            sum_var = sorted_Returns[0]
            for i in range(1, index):
                sum_var += sorted_Returns[i]
                CVaR = abs(sum_var / index)
            RtoCVaR = np.average(RET) / CVaR
            return -RtoCVaR

        strategy['Y'] = strategy.index
        strategy['Y'] = strategy.Y.apply(lambda x: x.year)
        n_year = strategy['Y'].value_counts()
        n_year_index = n_year.index.sort_values()
        RET_list = []
        T_list = []
        STD_list = []
        MDD_list = []
        ACC_list = []
        SHARP_list = []
        R2VaR_list = []
        R2CVaR_list = []
        for i in n_year_index:
            x = strategy.loc[strategy['Y'] == i]
            ts = x.nav.pct_change(1).fillna(0)
            RET = (x.nav[x.shape[0] - 1] /x.nav[0])** (252 / x.shape[0]) - 1
            T = stats.ttest_1samp(ts, 0)[0]
            STD = np.std(ts) * np.sqrt(252)
            MDD = MaxDrawdown(ts)[1]
            # MDD = MaxDrawdown2(ts)
            ACC = MaxDrawdown(ts)[0]
            SHARP = (RET - 0.03) / STD
            R2VaR = Reward_to_VaR(x)
            R2CVaR = Reward_to_CVaR(x)
            RET_list.append(RET)
            T_list.append(T)
            STD_list.append(STD)
            MDD_list.append(MDD)
            ACC_list.append(ACC)
            SHARP_list.append(SHARP)
            R2VaR_list.append(R2VaR)
            R2CVaR_list.append(R2CVaR)
        RET_df = pd.DataFrame(RET_list)
        RET_df.columns={'RET'}
        T_df = pd.DataFrame(T_list)
        T_df.columns={'T'}
        STD_df = pd.DataFrame(STD_list)
        STD_df.columns={'STD'}
        MDD_df = pd.DataFrame(MDD_list)
        MDD_df.columns={'MDD'}
        ACC_df = pd.DataFrame(ACC_list)
        ACC_df.columns={'ACC'}
        SHARP_df = pd.DataFrame(SHARP_list)
        SHARP_df.columns={'SHARP'}
        R2VaR_df = pd.DataFrame(R2VaR_list)
        R2VaR_df.columns={'R2VaR'}
        R2CVaR_df = pd.DataFrame(R2CVaR_list)
        R2CVaR_df.columns={'R2CVaR'}
        P = pd.concat([RET_df, T_df, STD_df, MDD_df, ACC_df, SHARP_df, R2VaR_df, R2CVaR_df], axis=1, ignore_index=False)
        P.index = n_year_index
        print(P)
        P.to_csv('performance_anl.csv')
        return P

    #define the main function
    def ProductPlan(self,datas, expected_return, period=1, rollingtime=126, method='MAC', tau=0.01, wmin=0, wmax=1):
        # input:
        #       datas: %the close price for the target assets(e.g index)
        #              %is a dataframe with date index and code columns
        #       expected_return: % take the same style as datas
        #       peiod: %the frequency for trading, %default value =1 month
        #       rollingtime: %the backtest time window for the stratygy，%C126 days
        #       method:  %choose MAC, RP, BL, %default value =MAC
        #       tau: %the weights for the subjective view for blacklitterman model
        #            %default value = 0.01
        #       wmin: %the minimum ratio for asset allocation % default value = 0
        #       wmax: %the maximum ratio for asset allocation % default value = 1
        #output:
        #       weights: %a df of weights ; %freq = daily
        #       results: %a df containing net asset value; %freq = daily
        ret = datas.pct_change(1).fillna(0)
        data_norm = datas / datas.iloc[0,] * 1000
        self.result = data_norm.copy()
        self.result['m'] = self.result.index
        self.result['m'] = self.result.m.apply(lambda x: x.month)
        self.weights = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)
        N = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)
        datas_index = np.array(datas.index)
        noa = datas.shape[1]
        position = 0
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        bnds = tuple((wmin, wmax) for i in range(datas.shape[1]))

        def blacklitterman(returns, Tau, P, Q):
            mu = returns.mean()
            sigma = returns.cov()
            pi1 = mu
            ts = Tau * sigma
            Omega = np.dot(np.dot(P, ts), P.T) * np.eye(Q.shape[0])
            middle = linalg.inv(np.dot(np.dot(P, ts), P.T) + Omega)
            er = np.expand_dims(pi1, axis=0).T + np.dot(np.dot(np.dot(ts, P.T), middle),
                                                        (Q - np.expand_dims(np.dot(P, pi1.T), axis=1)))
            newList = []
            for item in er:
                if type(item) == list:
                    tmp = ''
                    for i in item:
                        tmp += float(i) + ' '
                        newList.append(tmp)
                else:
                    newList.append(item)
            New = []
            for j in newList:
                k = float(j)
                New.append(k)
            posteriorSigma = sigma + ts - np.dot(ts.dot(P.T).dot(middle).dot(P), ts)
            return [New, posteriorSigma]

        def funs( weight, sigma):
            weight = np.array([weight]).T
            result = np.dot(np.dot(weight.T, np.mat(sigma)), weight)[0, 0]
            return (result)

        def funsRP(weight, sigma):
            weight = np.array([weight]).T
            X = np.multiply(weight, np.dot(sigma.values, weight))
            result = np.square(np.dot(X, np.ones([1, X.shape[0]])) - X.T).sum()
            return (result)
        if method == 'MAC':
            for i in tqdm(range(self.result.shape[0])):
                if i == 0:
                    pass
                elif self.result.m[i] != self.result.m[i - 1] and self.result.m[i] % int(period) == 0:
                    sigma = ret.iloc[position:i].cov()
                    position = i
                    weight = [0 for i in range(datas.shape[1])]
                    res = minimize(funs, weight, method='SLSQP', args=(sigma,),
                                   bounds=bnds, constraints=cons, tol=1e-8)
                    self.weights.iloc[i, :] = res.x
                    price = datas.loc[datas.index[i], :]
                    V = (self.weights.iloc[i, :] * price).sum()
                    n = V * self.weights.iloc[i, :].values / price.values
                    N.loc[self.result.index[i], :] = n
                else:

                    N.iloc[i, :] = N.iloc[i - 1, :]
                    w = N.iloc[i, :] * datas.loc[datas.index[i], :]
                    self.weights.iloc[i, :] = w / w.sum()
        elif method == 'RP':
            for i in tqdm(range(self.result.shape[0])):
                if i == 0:
                    pass
                elif (self.result.m[i] != self.result.m[i - 1]) and self.result.m[i] % int(period) == 0:
                    sigma = ret.iloc[position:i].cov()
                    position = i
                    weight = [0 for i in range(datas.shape[1])]
                    res = minimize(funsRP, weight, method='SLSQP', args=(sigma,),
                                   bounds=bnds, constraints=cons, tol=1e-20)

                    self.weights.iloc[i, :] = res.x
                    price = datas.loc[datas.index[i], :]
                    V = (self.weights.iloc[i, :] * price).sum()
                    n = V * self.weights.iloc[i, :].values / price.values
                    N.loc[self.result.index[i], :] = n
                else:
                    N.iloc[i, :] = N.iloc[i - 1, :]
                    w = N.iloc[i, :] * datas.loc[datas.index[i], :]
                    self.weights.iloc[i, :] = w / w.sum()
        elif method == 'BL':

            pick1 = np.array([1 for i in range(noa)])
            pick21 = [0 for i in range(noa - 2)]
            pick22 = [0.5, -0.5]
            pick2 = np.array(pick22 + pick21)  # the premium rate between the 1st and the 2nd assets
            P = np.array([pick1, pick2])
            for i in tqdm(range(self.result.shape[0])):
                if i == 0:
                    self.weights.iloc[i, :] = 1 / datas.shape[1]
                    price = datas.loc[datas.index[i], :]
                    n = self.weights.iloc[i, :].values / price.values
                    N.loc[self.result.index[i], :] = n
                    del price, n
                elif (self.result.m[i] != self.result.m[i - 1]) and (i > int(rollingtime)) and (self.result.m[i] % int(period)) == 0:
                    Rett = ret[i - int(rollingtime):i]
                    expected_return['sum'] = expected_return.sum(axis=1)
                    expected_return['premium'] = expected_return.iloc[:, 0] - expected_return1.iloc[:, 1]
                    Q = expected_return.iloc[i:i + 1, (expected_return.shape[1] - 2):expected_return.shape[1]].T.values

                    Returns = blacklitterman(Rett, tau, P, Q)[0]
                    sigma = blacklitterman(Rett, tau, P, Q)[1]
                    weight = [0 for i in range(datas.shape[1])]
                    res = minimize(funs, weight, method='SLSQP', args=(sigma,),
                                   bounds=bnds, constraints=cons, tol=1e-8)

                    self.weights.iloc[i, :] = res.x
                    price = datas.loc[datas.index[i], :]
                    V = (self.weights.iloc[i, :] * price).sum()
                    n = V * self.weights.iloc[i, :].values / price.values
                    N.loc[self.result.index[i], :] = n
                else:

                    N.iloc[i, :] = N.iloc[i - 1, :]
                    w = N.iloc[i, :] * datas.loc[datas.index[i], :]
                    self.weights.iloc[i, :] = w / w.sum()

        else:
            pass

        self.result['mv'] = 0
        self.result['mv_adj_last_day'] = 0
        self.result['nav'] = 1
        for i in tqdm(range(self.result.shape[0])):
            self.result.loc[self.result.index[i], 'mv'] = (datas.iloc[i, :] * N.iloc[i, :]).sum()
            if all(N.iloc[i, :] == 0):
                pass
            elif all(N.iloc[i, :] == N.iloc[i - 1, :]):
                self.result.loc[self.result.index[i], 'mv_adj_last_day'] = self.result.loc[self.result.index[i - 1], 'mv']
                self.result.loc[self.result.index[i], 'nav'] = self.result.nav[i - 1] * self.result.mv[i] / self.result.mv_adj_last_day[i]
            else:
                self.result.loc[self.result.index[i], 'mv_adj_last_day'] = (datas.iloc[i - 1, :] * N.iloc[i, :]).sum()
                self.result.loc[self.result.index[i], 'nav'] = self.result.nav[i - 1] * self.result.mv[i] / self.result.mv_adj_last_day[i]

        self.result = self.result.iloc[int(rollingtime):, :]
        self.weights = self.weights.iloc[int(rollingtime):, :]
        self.result['nav'] = self.result.nav / self.result.nav[0] * 1000
        return self.weights, self.result

if __name__ == '__main__':
    # ## input relative parameters
    start = input("please type the begining date(format: xxxx/xx/xx): ")
    end = input("please type the ending date(format: xxxx/xx/xx) ")

    startdate = datetime.strptime(start,'%Y/%m/%d')
    enddate = datetime.strptime(end,'%Y/%m/%d')

    codes=input("please type the codes of the asset，seperate them with a comma：")
    #000985.CSI,H11001.CSI,CCFI.WI,AU9999.SGE,SPX.GI,HSI.HI
    datas = w.wsd(codes, "close", startdate, enddate, "")
    datas = pd.DataFrame(np.array(datas.Data).T,columns = datas.Codes,index = datas.Times)

    rolling = input("please type rolling time (freq = daily)：")
    rollingtimeinput = float(rolling)

    perio = input("please type the freq for trading (freq=monthly)：")
    periodinput = float(perio)

    #example for prediction(take the average return in the previous half year)
    expected_return1=((datas-datas.shift(126))/datas.shift(126))/126
    expected_return1=expected_return1.fillna(method='ffill')
    expected_return1=expected_return1.fillna(0.003)

    clf_MAC = _ProductPlan_()
    weights_Prouctplan1,result_Prouctplan1 = clf_MAC.ProductPlan(datas,expected_return1,periodinput, rollingtimeinput)
    clf_MAC.performance(result_Prouctplan1)
    clf_MAC.performance_anl(result_Prouctplan1)
    clf_RP = _ProductPlan_()
    weights_Prouctplan2,result_Prouctplan2 = clf_RP.ProductPlan(datas,expected_return1,periodinput, rollingtimeinput,method='RP')
    clf_RP.performance(result_Prouctplan2)
    clf_RP.performance_anl(result_Prouctplan2)
    clf_BL = _ProductPlan_()
    weights_Prouctplan3,result_Prouctplan3 = clf_BL.ProductPlan(datas,expected_return1,periodinput, rollingtimeinput,method='BL')
    clf_BL.performance(result_Prouctplan3)
    clf_BL.performance_anl(result_Prouctplan3)
    weights_Prouctplan1.plot.area(figsize = (20,5))
    plt.legend(datas.columns.tolist(),prop=font,loc = 'upper right')
    plt.savefig('Markowitz.png',dpi=500,bbox_inches='tight')
    weights_Prouctplan2.plot.area(figsize = (20,5))
    plt.legend(datas.columns.tolist(),prop=font,loc = 'upper right')
    plt.savefig('Risk Parity.png',dpi=500,bbox_inches='tight')
    weights_Prouctplan3.plot.area(figsize = (20,5))
    plt.legend(datas.columns.tolist(),prop=font,loc = 'upper right')
    plt.savefig('Black Litterman.png',dpi=500,bbox_inches='tight')
    X = np.arange(result_Prouctplan1.shape[0])
    xticklabel = result_Prouctplan1.index
    xticks = np.arange(0,result_Prouctplan1.shape[0],np.int((result_Prouctplan1.shape[0]+1)/7))

    plt.figure(figsize = [20,5])
    SP = plt.axes()
    SP.plot(X,result_Prouctplan1. nav,label = 'MAC',color = 'darkred',linewidth = 3)
    SP.plot(X,result_Prouctplan2. nav,label = 'RP',color = 'purple',linewidth = 3)
    SP.plot(X,result_Prouctplan3. nav,label = 'BL',color = 'yellow',linewidth = 3)
    SP.set_xticks(xticks)
    SP.set_xticklabels(xticklabel[xticks],size = 9)
    plt.legend(prop=font)
    plt.savefig('Net Asset Value.png',dpi=500,bbox_inches='tight')
    df1=weights_Prouctplan1.tail(1).round(4)
    df2=weights_Prouctplan2.tail(1).round(4)
    df3=weights_Prouctplan3.tail(1).round(4)
    pd.concat([df1,df2,df3])
    weights_Prouctplan1.to_csv('Markowitz_weights.csv')
    weights_Prouctplan2.to_csv('Risk Parity_weights.csv')
    weights_Prouctplan3.to_csv('Blacklitterman_weights.csv')
    result_Prouctplan1.to_csv('Markowitz_results.csv')
    result_Prouctplan2.to_csv('Risk Parity_results.csv')
    result_Prouctplan3.to_csv('Blacklitterman_results.csv')


# coding: utf-8

# In[1]:


from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()
from pandas_datareader.famafrench import get_available_datasets
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[5]:


markets = ['FXE', 'EWJ', 'GLD', 'QQQ', 'SPY', 'SHV', 'DBA', 'USO', 'XBI', 'ILF', 'GAF', 'EPP', 'FEZ']

market_data = pdr.get_data_yahoo(markets, start="2007-01-01", end="2016-10-20")

market_ret = market_data.ix['Close', :, :].dropna()
market_ret = market_ret.pct_change()

fama = pd.read_csv('4542f455888a6b40.csv', index_col='date', parse_dates=True)

df_market = pd.concat([market_ret, fama], axis=1, join='inner')
df_market = df_market.dropna()

R = df_market.ix[:,0:13]
ff = df_market.ix[:,13:(df_market.shape[1]-1)]

from datetime import datetime
datetime.strptime("2008-01-31", '%Y-%m-%d')

bc_return = R[R.index <= datetime.strptime("2008-01-31", '%Y-%m-%d')]
dc_return = R[(R.index >= datetime.strptime("2008-02-01", '%Y-%m-%d')) & (R.index <= datetime.strptime("2009-06-01", '%Y-%m-%d'))]
ac_return = R[R.index >= datetime.strptime("2009-06-02", '%Y-%m-%d')]

bc_fama = ff[ff.index <= datetime.strptime("2008-01-31", '%Y-%m-%d')]
dc_fama = ff[(ff.index >= datetime.strptime("2008-02-01", '%Y-%m-%d')) & (R.index <= datetime.strptime("2009-06-01", '%Y-%m-%d'))]
ac_fama = ff[ff.index >= datetime.strptime("2009-06-02", '%Y-%m-%d')]

bc_bench_return = bc_return.SPY
dc_bench_return = dc_return.SPY
ac_bench_return = dc_return.SPY


# In[12]:


class portfolio_startegy:
    
    def __init__(self, data_market, data_fama, term, Q_ident = False, lamda=1/2, benchmark = 'SPY'):
        self.data_market = data_market
        self.data_fama = data_fama
        self.term = term
        self.lamda = lamda
        self.Q_ident = Q_ident
        self.benchmark = benchmark
        self.performance = None
        
    def cal_beta_market(self, R_vec, N = 13, Rm = 'SPY'):
        import numpy as np
        beta = []
        for i in range(0,N):
            sol = np.cov(R_vec.ix[:,i], R_vec.ix[:, Rm.upper()])/np.var(R_vec.ix[:, Rm.upper()])
            beta.append(sol[0,1])
        
        return np.array(beta)
    
    def optimization_strategy(self, wp, rho, lamda, Q, beta_market, beta_target):
        from scipy.optimize import minimize
        def fun_strategy(w, wp, rho, lamda, Q):
            return (-w).dot(rho)+lamda*(w-wp).dot(Q.dot((w-wp).T))
    
        cons=({'type':'eq','fun':lambda w: (w).dot(beta_market)-beta_target}, 
               {'type':'eq','fun':lambda w: sum(w)-1})
    
        return minimize(fun_strategy, 13*[1/13], args=(wp, rho, lamda, Q), constraints=cons,bounds=[[-2,2]]*13)

    def optimization_minvar(self, wp, rho, lamda, Q, sigma):
        from scipy.optimize import minimize
        def fun_minvar(w, wp, rho, lamda, Q, sigma):
            return (w).dot(sigma.dot(w.T))+lamda*(w-wp).dot(Q.dot((w-wp).T))
    
        cons=({'type':'eq','fun':lambda w: w.dot(rho)-15/100/252}, 
                {'type':'eq','fun':lambda w: sum(w)-1})
    
        return minimize(fun_minvar, 13*[1/13], args=(wp, rho, lamda, Q, sigma), constraints=cons,bounds=[[-2,2]]*13)
    
    def strategy_performace(self):
        import numpy as np
        import statsmodels.api as sm
        import pandas as pd
        
        mkt_ret, fama_ret, term, lamda, benchmark = self.data_market, self.data_fama, self.term, self.lamda, self.benchmark
        
        wp1 = np.zeros(13)
        wp2 = np.zeros(13)
        wp3 = np.zeros(13)
        wp_minvar = np.zeros(13)
        check_loop = 0
    
        for i in range (term,len(mkt_ret)):
            if i%term==0:
    
                beta_fama = sm.OLS(mkt_ret[(i-term):i].subtract(fama_ret['rf'][(i-term):i] ,axis=0), sm.add_constant(fama_ret[['mktrf','smb','hml']][(i-term):i])).fit().params.values
                ri = (sm.add_constant(fama_ret[['mktrf','smb','hml']][(i-term):i]).dot(beta_fama)).add(fama_ret['rf'][(i-term):i], axis=0)
                rho = ri.mean()
                beta_market = self.cal_beta_market(mkt_ret[(i-term):i], Rm=benchmark)
            
                sigma = ri.cov() 
                
                Q = np.where(self.Q_ident == False, sigma, np.diag((13*[1])))
            
                solution1 = self.optimization_strategy(wp1, rho, lamda, Q, beta_market, beta_target = 0.5)
                wp1 = solution1.x
            
                solution2 = self.optimization_strategy(wp2, rho, lamda, Q, beta_market, beta_target = 1.0)
                wp2 = solution2.x
            
                solution3 = self.optimization_strategy(wp3, rho, lamda, Q, beta_market, beta_target = 1.5)
                wp3 = solution3.x
            
                solution_minvar = self.optimization_minvar(wp_minvar, rho, lamda, Q, sigma)
                wp_minvar = solution_minvar.x
            
                if check_loop == 0:
                    return_strategy1 = mkt_ret[(i-term):i].dot(wp1)
                    return_strategy2 = mkt_ret[(i-term):i].dot(wp2)
                    return_strategy3 = mkt_ret[(i-term):i].dot(wp3)
                    return_strategy_minvar = mkt_ret[(i-term):i].dot(wp_minvar)
                    return_benchmark = mkt_ret[(i-term):i][benchmark]
                else:
                    return_strategy1 = pd.concat([return_strategy1, mkt_ret[(i-term):i].dot(wp1)], axis=0)
                    return_strategy2 = pd.concat([return_strategy2, mkt_ret[(i-term):i].dot(wp2)], axis=0)
                    return_strategy3 = pd.concat([return_strategy3, mkt_ret[(i-term):i].dot(wp3)], axis=0)
                    return_strategy_minvar = pd.concat([return_strategy_minvar, mkt_ret[(i-term):i].dot(wp_minvar)], axis=0)
                    return_benchmark = pd.concat([return_benchmark, mkt_ret[(i-term):i][benchmark]], axis=0)
                check_loop += 1
    
        performance = pd.DataFrame({'Beta=0.5':return_strategy1,
                        'Beta=1.0':return_strategy2,
                        'Beta=1.5':return_strategy3,
                        'MinVar':return_strategy_minvar,
                        benchmark:return_benchmark})
        self.performance = performance
        self.return_strategy1 = return_strategy1
        self.return_strategy2 = return_strategy2
        self.return_strategy3 = return_strategy3
        self.underlying_analytics()
        return performance
    
    def performance_plot(self):
        import matplotlib.pyplot as plt
        from matplotlib import style
        import matplotlib.dates as mdates
        
        if self.performance is None:
            self.strategy_performace()
        perf = self.performance
    
        term, benchmark = self.term, self.benchmark
    
        style.use('ggplot')
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Portfolio Strategy Performance', fontsize=18, fontweight='bold')
    
        ax1 = plt.subplot2grid((15, 10), (0, 0), rowspan=10, colspan=15)
        ax1.plot(perf['Beta=0.5'].add(1).cumprod().subtract(1), c='orange',  label = "Beta=0.5 {0} terms, Cumulative return = {1:0.4f}".format(term, perf.add(1).cumprod().subtract(1).ix[-1, 'Beta=0.5']))
        ax1.annotate("{0}: {1:0.4f}".format('Beta=0.5', perf.add(1).cumprod().subtract(1).ix[-1, 'Beta=0.5']), xy=(1, perf.add(1).cumprod().subtract(1).ix[-1, 'Beta=0.5']), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
            
        ax1.plot(perf['Beta=1.0'].add(1).cumprod().subtract(1), c='dodgerblue', label = "Beta=1.0 {0} terms, Cumulative return = {1:0.4f}".format(term, perf.add(1).cumprod().subtract(1).ix[-1, 'Beta=1.0']))
        ax1.annotate("{0}: {1:0.4f}".format('Beta=1.0', perf.add(1).cumprod().subtract(1).ix[-1, 'Beta=1.0']), xy=(1, perf.add(1).cumprod().subtract(1).ix[-1, 'Beta=1.0']), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
    
        ax1.plot(perf['Beta=1.5'].add(1).cumprod().subtract(1), c='limegreen', label = "Beta=1.5 {0} terms, Cumulative return = {1:0.4f}".format(term, perf.add(1).cumprod().subtract(1).ix[-1, 'Beta=1.5']))
        ax1.annotate("{0}: {1:0.4f}".format('Beta=1.5', perf.add(1).cumprod().subtract(1).ix[-1, 'Beta=1.5']), xy=(1, perf.add(1).cumprod().subtract(1).ix[-1, 'Beta=1.5']), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
    
        ax1.plot(perf['MinVar'].add(1).cumprod().subtract(1), c='magenta', label = "MinVar {0} terms, Cumulative return = {1:0.4f}".format(term, perf.add(1).cumprod().subtract(1).ix[-1, 'MinVar']))
        ax1.annotate("{0}: {1:0.4f}".format('MinVar', perf.add(1).cumprod().subtract(1).ix[-1, 'MinVar']), xy=(1, perf.add(1).cumprod().subtract(1).ix[-1, 'MinVar']), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
    
        ax1.plot(perf[benchmark].add(1).cumprod().subtract(1), c='black', label = "{0} {1} terms, Cumulative return = {2:0.4f}".format(benchmark, term, perf.add(1).cumprod().subtract(1).ix[-1, benchmark]))
        ax1.annotate("{0}: {1:0.4f}".format(benchmark, perf.add(1).cumprod().subtract(1).ix[-1, benchmark]), xy=(1, perf.add(1).cumprod().subtract(1).ix[-1, benchmark]), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
    
        ax1.set_title('{0}-terms factor model with \n Min-variance strategy and {1} benchmark \n The best strategy is {2} with {3:0.4f} return'.format(term, benchmark,perf.add(1).cumprod().subtract(1).ix[-1,:].index[perf.add(1).cumprod().subtract(1).ix[-1,:] == perf.add(1).cumprod().subtract(1).ix[-1,:].max()].tolist()[0], perf.add(1).cumprod().subtract(1).ix[-1,:].max()))
        ax1.legend()
    
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        ax1.set_ylabel('Return')
        plt.xticks(rotation=45)
        plt.xlabel('Date Hours:Minutes')
        plt.plot()
    
    def risk_indicator(self, perf = None):
        import numpy as np
        from scipy import stats
        import pandas as pd
        
        if perf is not None:
            perf = perf
        else:
            perf = self.performance

        annual_ret = perf.add(1).prod()** (252 / len(perf)) - 1
        mean_ret = np.mean(perf, axis=0)
        min_ret = np.min(perf, axis=0)
        volatility = np.std(perf, axis=0)
        sharpe = mean_ret/volatility
        skewness = stats.skew(perf, axis=0)
        kurtosis = stats.kurtosis(perf, axis=0)
    
        Roll_10Max = pd.rolling_max(perf.add(1).cumprod().subtract(1), 10, min_periods=1)
        day10_Drawdown = perf.add(1).cumprod().subtract(1).div(Roll_10Max) - 1
        Max_10day_Drawdown = pd.rolling_min(day10_Drawdown, 10, min_periods=1)
        max_10dd = Max_10day_Drawdown.min()
    
        VaR_list = []
        for i in perf.columns:
            VaR_list.append(np.percentile(perf.ix[:,i], .01))
    
        CVaR_list = []
        for i in perf.columns:
            CVaR_list.append(np.mean(np.sort(perf.ix[:,i].values)[np.sort(perf.ix[:,i].values) <= np.percentile(perf.ix[:,i], .01)]))
        
        
        if perf is not None:
            result = pd.DataFrame(columns=perf.columns)
        else:
            result = pd.DataFrame(columns=['Beta=0.5', 'Beta=1.0', 'Beta=1.5', 'MinVar', self.benchmark])
        
        result.loc['Annual.Return'] = annual_ret
        result.loc['Daily.Mean'] = mean_ret
        result.loc['Daily.Min'] = min_ret
        result.loc['Max10.DD'] = max_10dd
        result.loc['Volatility'] = volatility
        result.loc['Sharpe.Ratio'] = sharpe
        result.loc['Skewness'] = skewness 
        result.loc['Kurtosis'] = kurtosis
        result.loc['Mod.VaR'] = VaR_list
        result.loc['CVaR'] = CVaR_list
        
        return result
    
    def underlying_analytics(self):
        import pandas as pd
        
        mkt_data = self.data_market.ix[0:len(self.return_strategy1),:]
        
        underly1 = pd.concat([mkt_data, self.return_strategy1.rename('Beta=0.5')], axis=1)
        self.underly1 = self.risk_indicator(underly1)
        
        underly2 = pd.concat([mkt_data, self.return_strategy2.rename('Beta=1.0')], axis=1)
        self.underly2 = self.risk_indicator(underly2)
        
        underly3 = pd.concat([mkt_data, self.return_strategy3.rename('Beta=1.5')], axis=1)
        self.underly3 = self.risk_indicator(underly3)


# # Before Sub-prime crisis

# ## Short-term model

# In[15]:


port_bc_short = portfolio_startegy(data_market = bc_return, data_fama = bc_fama, term = 50)


# In[16]:


port_bc_short.performance_plot()


# In[17]:


port_bc_short.risk_indicator()


# In[18]:


port_bc_short.underly1


# In[19]:


port_bc_short.underly2


# In[20]:


port_bc_short.underly3


# ## Long-term model

# In[22]:


port_bc_long = portfolio_startegy(data_market = bc_return, data_fama = bc_fama, term = 200)


# In[23]:


port_bc_long.performance_plot()


# In[24]:


port_bc_long.risk_indicator()


# In[25]:


port_bc_long.underly1


# In[26]:


port_bc_long.underly2


# In[27]:


port_bc_long.underly3


# # During Sub-prime crisis

# ## Short-term model

# In[28]:


port_dc_short = portfolio_startegy(data_market = dc_return, data_fama = dc_fama, term = 50)


# In[29]:


port_dc_short.performance_plot()


# In[30]:


port_dc_short.risk_indicator()


# In[31]:


port_dc_short.underly1


# In[32]:


port_dc_short.underly2


# In[33]:


port_dc_short.underly3


# ## Long-term model

# In[34]:


port_dc_long = portfolio_startegy(data_market = dc_return, data_fama = dc_fama, term = 200)


# In[35]:


port_dc_long.performance_plot()


# In[36]:


port_dc_long.risk_indicator()


# In[37]:


port_dc_long.underly1


# In[38]:


port_dc_long.underly2


# In[39]:


port_dc_long.underly3


# # After Sub-prime crisis

# ## Short-time model

# In[40]:


port_ac_short = portfolio_startegy(data_market = ac_return, data_fama = ac_fama, term = 50)


# In[41]:


port_ac_short.performance_plot()


# In[42]:


port_ac_short.risk_indicator()


# In[43]:


port_ac_short.underly1


# In[44]:


port_ac_short.underly2


# In[45]:


port_ac_short.underly3


# ## Long-term model

# In[46]:


port_ac_long = portfolio_startegy(data_market = ac_return, data_fama = ac_fama, term = 200)


# In[47]:


port_ac_long.performance_plot()


# In[48]:


port_ac_long.risk_indicator()


# In[49]:


port_ac_long.underly1


# In[50]:


port_ac_long.underly2


# In[51]:


port_ac_long.underly3


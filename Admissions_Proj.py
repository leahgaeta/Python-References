%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

df_raw = pd.read_csv("admissions.csv")
df = df_raw.dropna() 
print df.head()
print df.count()

# frequency table for prestige and whether or not someone was admitted
print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])

temprank_df = pd.get_dummies(df['prestige'], prefix = 'prestige')
print temprank_df.head()

# hand calculating odds ratio
cols_to_keep = ['admit', 'gre', 'gpa']
handCalc = df[cols_to_keep].join(temprank_df.loc[:, 'prestige_1.0':])
print handCalc.head()

# crosstab prestige 1 admission 
# frequency table cutting prestige and whether or not someone was admitted
print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])

Odds_Prestige_1 = 33.0/28
print Odds_Prestige_1

Prestige_234_Admit = 53.0+28+12
Prestige_234_NoAdmit = 95.0+93+55
Odds_Prestige_234 = Prestige_234_Admit/Prestige_234_NoAdmit
print Odds_Prestige_234

odds_ratio = Odds_Prestige_1/Odds_Prestige_234
print odds_ratio

print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])

Odds_Prestige_4 = 12.0/55
print Odds_Prestige_1/Odds_Prestige_4

# Data Analysis
# create a clean data frame for the regression
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(temprank_df.ix[:, 'prestige_2':])
print data.head()
# manually add the intercept
data['intercept'] = 1.0
train_cols = data.columns[1:]
train_cols

# fit the model
logit = sm.Logit(data['admit'], data[train_cols])

result = logit.fit()
# summary results
print result.summary()
print np.exp(result.params)
# odds ratios and 95% CI
parameters = result.params
conf = result.conf_int()
conf['OR'] = parameters
conf.columns = ['2.5%', '97.5%', 'OR']
print np.exp(conf)

# oredicted probabilities
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
    # instead of generating all possible values of GRE and GPA, we're going
# to use an evenly spaced range of 10 values from the min to the max 
gres = np.linspace(data['gre'].min(), data['gre'].max(), 10)
print gres
# array([ 220.        ,  284.44444444,  348.88888889,  413.33333333,
#         477.77777778,  542.22222222,  606.66666667,  671.11111111,
#         735.55555556,  800.        ])
gpas = np.linspace(data['gpa'].min(), data['gpa'].max(), 10)
print gpas
# array([ 2.26      ,  2.45333333,  2.64666667,  2.84      ,  3.03333333,
#         3.22666667,  3.42      ,  3.61333333,  3.80666667,  4.        ])


# enumerate all possibilities
combos = pd.DataFrame(cartesian([gres, gpas, [1, 2, 3, 4], [1.]]))
# recreate the dummy variables
# new_df = pd.get_dummies(df['prestige'], prefix = 'prestige')
combos.columns = ['gre', 'gpa', 'prestige', 'intercept']
new_dummy = pd.get_dummies(combos['prestige'], prefix='prestige')
new_dummy.columns = ['prestige_1.0', 'prestige_2.0', 'prestige_3.0', 'prestige_4.0']

# keep only what we need for making predictions
new_cols_to_keep = ['gre', 'gpa', 'prestige', 'intercept']
combos = combos[new_cols_to_keep].join(new_dummy.loc[:, 'prestige_2.0':])
print combos.head()
# combos['intercept'] = 1.0 #this adds new column automatically to the end
print 'prestige_2' in combos.columns
combos['admit_pred'] = result.predict(combos[train_cols])
print combos.head()
print combos.tail()

# visualizations
  def isolate_and_plot(variable):
      grouped = pd.pivot_table(combos, values=['admit_pred'], index=[variable, 'prestige'],
                  aggfunc=np.mean)
      colors = 'bmgybmgy'
      for col in combos.prestige.unique():
          plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
          pl.plot(plt_data.index.get_level_values(0), plt_data['admit_pred'], color=colors[int(col)])

      pl.xlabel(variable)
      pl.ylabel("P(admit=1)")
      pl.legend(['1', '2', '3', '4'], loc='upper left', title='Prestige')
      pl.title("Prob(admit=1) isolating " + variable + " and presitge")
      pl.show()

  isolate_and_plot('gre')
  isolate_and_plot('gpa')


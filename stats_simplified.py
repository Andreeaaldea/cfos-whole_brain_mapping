# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:17:35 2020

@author: Monika Pawłowska, Zbigniew Zieliński
"""

import pingouin as pg
import pandas as pd

#%% loading
input_data_file='pe_depth8_cleaned.csv'
dane = pd.read_csv(input_data_file)

#%% cleanup. More than one value per group needed for calculating Standard Deviation
counts=dane[[ 'group_label', 'abbrev', 'signal_density']].groupby([ 'group_label', 'abbrev']).count().reset_index()
dane=dane[dane.abbrev.isin(counts[counts['signal_density']>1]['abbrev'].to_list())]
print('brain structures left: '+str(len(pd.unique(dane['abbrev']))))


#%% anova to check whether significant differences exist
aov = pg.mixed_anova(data=dane, 
                     dv='signal_density', 
                     between='group_label',
                     within='abbrev', 
                     subject='case_id', 
                     correction=True)
print(aov[['Source','DF1','DF2','p-unc']])

#%% pairwise t tests
#see https://pingouin-stats.org/generated/pingouin.pairwise_ttests.html#pingouin.pairwise_ttests
post_hocs = pg.pairwise_tests(data=dane, 
                                      dv='signal_density',
                                      within='abbrev', 
                                      between='group_label',
                                      subject='case_id', 
                                      parametric=True,
                                      marginal=True, 
                                      #tail='two-sided',
                                      padjust='fdr_bh', 
                                      effsize='cohen',
                                      return_desc=True)

post_hocs_interaction = post_hocs[post_hocs['Contrast'] == 'abbrev * group_label'].reset_index(drop=True)

#%% saving
export_csv = post_hocs_interaction.to_csv('pairwise_'+input_data_file, header=True)
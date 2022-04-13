from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, JsonOutput, DataframeInput, DataframeOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import  PickleArtifact

import pandas as pd
import numpy as np
from datetime import date
import sys
import warnings
import time

if not sys.warnoptions:
    warnings.simplefilter("ignore")


@env(infer_pip_packages=True)
class MarkovClassifier(BentoService):
    
    ### data Preprocessing step ###
    def select_tactic_func (self,df_markov,tactic_name, brand):
        df_results = df_markov.loc[(df_markov['Engagement_Qualifier'] == tactic_name) &
                               ((df_markov['Conversion_Flag'] == 1) | (df_markov['Conversion_Flag'] == 0)) &
                               (df_markov['Brand_Name'] == brand)]
        return df_results
    ### data preprocessing step ###
    def markov_tactic_states_func (self,results, tactic):
        results['MarketingDate']=pd.to_datetime(results['MarketingDate'])

        #Sorting the values by userid and timestamp
        results = results.sort_values(by = ['User_ID','MarketingDate'])

        #Converting Conversion Flag to numeric
        #results['Conversion_Flag'] = np.where(results['Conversion_Flag']== 'Y', 1,0)

        #Create attribution levels (states of Markov Model)
        if tactic == "Media":
            results['attrlevel'] = results['Placement_ID'].astype(str) + "|" + results['Device'].astype(str)
        elif tactic == "Paid Search":
            results['attrlevel']= results['Engine_Name'].astype(str) + "|" + results['Search_Keywords'].astype(str)
        elif tactic == "Site Traffic":
            results['attrlevel']= results['Page_Title'].astype(str) + "|" + results['Page_Engagement_type'].astype(str)

        #Removing nans in attrlevel
        channel = [i.replace('|nan', '') for i in results['attrlevel'] ]
        channel = [i.replace('nan|','') for i in channel]
        results.attrlevel= channel

        results_updated= results[results.groupby('User_ID')['Conversion_Flag'].apply(lambda x: x.shift().eq(1).cumsum().eq(0))].reset_index()
        results_updated['visit_order'] = results.groupby('User_ID').cumcount() + 1
        results_updated.columns = results_updated.columns.str.strip()
        #print(results_updated.head())
        #results_updated.columns=['User_ID', 'visit_order']
        df_paths = results_updated.groupby('User_ID')['attrlevel'].aggregate(lambda x: x.unique().tolist()).reset_index()

        return results_updated, df_paths

    ## model calculations start from this function
    #Getting the removal effect
    def removal_effects(self,df, conversion_rate):
        removal_effects_dict = {}

        channels = [channel for channel in df.columns if channel not in ['Start',
                                                                         '0',
                                                                         'conversion']]

        for channel in channels:

            removal_df = df.drop(channel, axis=1).drop(channel, axis=0)
            for column in removal_df.columns:
                row_sum = np.sum(removal_df.loc[column])
                null_pct = float(1) - row_sum
                if null_pct != 0:
                    removal_df.loc[column]['0'] = null_pct
                removal_df.loc['0']['0'] = 1.0
            removal_to_conv = removal_df[
                ['0', 'conversion']].drop(['0', 'conversion'], axis=0)
            removal_to_non_conv = removal_df.drop(
                ['0', 'conversion'], axis=1).drop(['0', 'conversion'], axis=0)
            removal_inv_diff = np.linalg.inv(
                np.identity(
                    len(removal_to_non_conv.columns)) - np.asarray(removal_to_non_conv))
            removal_dot_prod = np.dot(removal_inv_diff, np.asarray(removal_to_conv))
            removal_cvr = pd.DataFrame(removal_dot_prod,
                                       index=removal_to_conv.index)[[1]].loc['Start'].values[0]
            removal_effect = 1 - removal_cvr / conversion_rate
            removal_effects_dict[channel] = removal_effect

        return removal_effects_dict

    def markov_attribution_func (self,results_updated, df_paths, interaction_type):
        ## Linear Interaction Type
        if interaction_type == 'linear':
            df_interaction= results_updated.groupby('User_ID')['Conversion_Flag'].sum().reset_index()
        ## First Click Interaction Type
        if interaction_type == 'first':
            df_interaction = results_updated.drop_duplicates('User_ID', keep='first')[['User_ID', 'Conversion_Flag']]
        ## Last Click Interaction Type
        if interaction_type == 'last':
            df_interaction = results_updated.drop_duplicates('User_ID', keep='last')[['User_ID', 'Conversion_Flag']]

        df_paths = pd.merge(df_paths, df_interaction, how='left', on='User_ID')

        paths_final = []
        for i in range(df_paths.shape[0]):
            if df_paths['Conversion_Flag'][i] >= 1:
                paths_final.append(['Start'] + df_paths['attrlevel'][i] + ['conversion'])
            else: 
                paths_final.append(['Start'] + df_paths['attrlevel'][i] + ['0'])

        df_paths['path'] = paths_final

        list_of_paths = df_paths['path']

        total_clicks = sum(path.count('conversion') for path in df_paths['path'].tolist())

        #Getting the conversion_rate according to the various no.of paths
        base_conversion_rate = total_clicks / len(list_of_paths)

        #Getting the transition states
        def transition_states(self,list_of_paths):
            list_of_unique_channels = set(x for element in list_of_paths for x in element)
            transition_states = {x + '>' + y: 0 for x in list_of_unique_channels for y in list_of_unique_channels}
            #
            for possible_state in list_of_unique_channels:
                if possible_state not in ['conversion', '0']:
                    for user_path in list_of_paths:
                        if possible_state in user_path:
                            indices = [i for i, s in enumerate(user_path) if possible_state in s]
                            for col in indices:
                                transition_states[user_path[col] + '>' + user_path[col + 1]] += 1
            #
            return transition_states

        trans_states = transition_states(self,list_of_paths)

        ## Dataframes for transition states
        states_pd = pd.DataFrame(list(trans_states.items()), columns=['States','Count'])

        ## Splitting the states column
        states_pd['State1'], states_pd['State2'] = states_pd['States'].str.split('>', 1).str

        ## Rearranging the columns
        states_pd = states_pd[['States','State1','State2', 'Count']]

        ## Another dataframe with Total Counts of each state
        states_group = states_pd.groupby('State1').sum()

        ## Adding states as column from index
        states_group = states_group.reset_index()

        ## Renaming the columns
        states_group = states_group.rename(columns = {'Count' : 'Total_Count'})

        states_pd1 = pd.merge(states_pd,states_group,how = 'left', on = 'State1')

        ## Adding total count variable in States dataframe
        states_pd1['Probability'] = states_pd1['Count']/states_pd1['Total_Count']

        ## CreatingTransition Probability matrix
        states_prob_matrix = states_pd1[['State1', 'State2', 'Probability']]

        states_prob_matrix = states_prob_matrix.pivot(index = 'State1', columns = 'State2',
                                                  values = 'Probability')

        trans_matrix = states_prob_matrix.copy()

        #Getting the removal effect
        removal_effects_dict = self.removal_effects(trans_matrix, base_conversion_rate)

        #Getting the removal effect in the dataframe
        df_removal_effect = pd.DataFrame(list(removal_effects_dict.items()))


        #Renaming the columns
        df_removal_effect.columns = ['attribution_level', 'removal_effect']
        df_removal_effect = df_removal_effect.sort_values(by ='removal_effect', ascending = False)

        return df_removal_effect
    def tactic_effect(self,df,tactic):
        tactic_dict = {"Media":"MEDIA_IMPRESSIONS",
                   "Paid Search":"PAID_SEARCH",
                   "Site Traffic":"SITE_TRAFFIC"
                  }
        tactic_effect = pd.DataFrame()
        brand_list=list(df.Brand_Name.unique())
        for brand in brand_list:
            markov_tactic_df = self.select_tactic_func(df,tactic, brand)
            results_updated_tactic, df_paths_tactic = self.markov_tactic_states_func(markov_tactic_df, tactic)
            tactic_removal_effect_linear = self.markov_attribution_func(results_updated_tactic, df_paths_tactic,'linear')

            #Calculating Attribution to Conversion
            tactic_linear_total = sum(tactic_removal_effect_linear['removal_effect'])
            tactic_removal_effect_linear['markov_attribution_to_conversion'] = tactic_removal_effect_linear['removal_effect'] /              tactic_linear_total

            #Adding columns for Model Qualifier and Engagement Qualifier
            tactic_removal_effect_linear['engagement_qualifier'] = tactic_dict[tactic]
            tactic_removal_effect_linear['brand'] = brand

            tactic_effect = tactic_effect.append([tactic_removal_effect_linear]).reset_index(drop = True)
        return tactic_effect
    @api(input=DataframeInput(orient="records"),batch=True,output=DataframeOutput(output_orient='records'))
    # The list of brands to be selected. All brand results will be mutually exclusive
    # Iterating over all brands
    def predict(self, df):
        df["Brand_Name"] = df["Brand_Name"].fillna("TEST_BRAND3")
        mi_effect =  self.tactic_effect(df,"Media")
        placement_id = mi_effect["attribution_level"].str.split("|", n = 1, expand = True)[0]  

        mi_ranking = mi_effect.copy()
        mi_ranking["placement_id"]= placement_id 
        mi_ranking.drop(columns =["attribution_level"], inplace = True) 
        mi_ranking=pd.pivot_table(mi_ranking, index= ["placement_id", "brand"],\
                                 values = "markov_attribution_to_conversion", aggfunc = lambda x: sum(x))
        rank_dict = dict(zip(sorted(mi_ranking["markov_attribution_to_conversion"].unique(), reverse = True),\
                                   list(range(1, len(mi_ranking["markov_attribution_to_conversion"].unique())+1))))

        mi_ranking['markov_rank']=mi_ranking.markov_attribution_to_conversion.map(rank_dict)
        mi_ranking['date_of_model_run'] = date.today()
        mi_ranking['campaign_duration_date'] = min(df['MarketingDate']) + ' to ' + max(df['MarketingDate'])
        #df_res = df_res.append(res)
        mi_ranking =mi_ranking.reset_index()
        df_final=pd.DataFrame()
        df_final['all_columns'] = (mi_ranking['placement_id'].astype(str)+'|'+mi_ranking['brand']+'|'
        +mi_ranking['markov_attribution_to_conversion'].astype(str)+'|'+mi_ranking['markov_rank'].astype(str)+'|'
        +mi_ranking['date_of_model_run'].astype(str)+'|'+ mi_ranking['campaign_duration_date'].astype(str))
        list_zero=['0']*(df.shape[0] - mi_ranking.shape[0])
        xtra = {'all_columns': list_zero}
        df_final = df_final.append(pd.DataFrame(xtra))
        #df_final_1 =pd.DataFrame(np.zeros([rows,columns])
        return df_final
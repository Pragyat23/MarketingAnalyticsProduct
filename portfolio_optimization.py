import pandas as pd
import numpy as np
from collections import Counter
import sys
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pyswarms.single.global_best import GlobalBestPSO

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, JsonOutput, DataframeInput, DataframeOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import PickleArtifact

# If you want to have number of predicted rows more than you should use class DataframeOutputForMoreRows that is contained in same folder
from dataframe_output_for_more_rows import DataframeOutputForMoreRows

import warnings
warnings.filterwarnings("ignore")

# to make this notebook's output stable across runs
np.random.seed(42)

@env(infer_pip_packages=True)
class Optimization(BentoService):

    def preprocessing(self, df, brand_name):
        df_prep_2 = df[(df['Brand_Name'] == brand_name)]
        df_prep_2['Campaign_Publisher'] = df_prep_2['Campaign_Name'].astype(
            str) + "-" + df_prep_2['Publisher_Name'].astype(str)
        df_prep_2 = df_prep_2[["Marketing_Month", "Total_Conversion",
                               "Total_Impression", "Brand_Name", "Payment_Cost", "Campaign_Publisher"]]
        df_prep_2 = df_prep_2.drop_duplicates()
        df_prep_2 = df_prep_2.sort_values(by=["Marketing_Month", "Total_Conversion",
                                          "Total_Impression", "Brand_Name", "Payment_Cost", "Campaign_Publisher"])
        df_prep__duration = df_prep_2.copy()
        df_prep__duration['Total_Conversion'] = np.where(
            df_prep__duration['Total_Conversion'] >= 0, 0, df_prep__duration['Total_Conversion'])
        df_prep__duration['Total_Impression'] = np.where(
            df_prep__duration['Total_Impression'] >= 0, 0, df_prep__duration['Total_Impression'])
        df_prep__duration['Payment_Cost'] = np.where(
            df_prep__duration['Payment_Cost'] >= 0, 1, df_prep__duration['Payment_Cost'])
        df_prep__duration['Campaign_Publisher'] = df_prep__duration['Campaign_Publisher'].astype(
            str) + '-DURATION'
        df_prep_2 = pd.concat([df_prep_2, df_prep__duration])
        df_agg_updated = df_prep_2.pivot_table(
            index=['Marketing_Month', 'Brand_Name'], columns='Campaign_Publisher', values='Payment_Cost')
        df_groupby = df_agg_updated.groupby(["Marketing_Month", "Brand_Name"]).first("Payment_Cost")
        df_target = df_prep_2.groupby(['Marketing_Month', 'Brand_Name']).agg(Total_Conversion=pd.NamedAgg(
            column="Total_Conversion", aggfunc="sum"), Total_Impression=pd.NamedAgg(column="Total_Impression", aggfunc="sum"))
        df_final_sp = df_groupby.join(df_target, on=['Marketing_Month', 'Brand_Name'], how='left')
        df_final_sp = df_final_sp.sort_values(by="Marketing_Month")
        df_final_sp = df_final_sp.reset_index()
        df_final_sp = df_final_sp.fillna(0)
        # df_final_sp.to_csv('final_sp_python_2.csv')

        return df_final_sp

    def data_aggregation(self, df, brand_name):
        # Taking Cumulative sum of month level data.
        pdf_final_sp = self.preprocessing(df, brand_name)
        pdf_final_sp = pdf_final_sp.fillna(0)
        pdf_final_sp = pdf_final_sp.replace("", 0)
        pdf_final_sp = pdf_final_sp.astype(
            {'Total_Conversion': 'int64', 'Total_Impression': 'int64'})
        pdf_final_sp = pdf_final_sp.drop("Brand_Name", axis=1)
        pdf_final_sp = pdf_final_sp.apply(pd.to_numeric)
        # Cumulative sum
        dataframe_final = pdf_final_sp
        for i in range(pdf_final_sp["Marketing_Month"].max()-1):
            new_df = pdf_final_sp[i:]
            new_df = new_df.cumsum(axis=0)
            dataframe_final = pd.concat([dataframe_final, new_df])
        dataframe_final["Net_Conversion_rate"] = round(
            dataframe_final.Total_Conversion/dataframe_final.Total_Impression, 6)
        model_input = dataframe_final.drop(
            ["Marketing_Month", "Total_Conversion", "Total_Impression"], axis=1)

        return model_input

    def budget_function(self, df, brand_name):
        pdf_final_sp = self.preprocessing(df, brand_name)
        #pdf_final_sp = df_final_sp.toPandas()
        # budget calculations
        all_columns = pdf_final_sp.columns.values.tolist()
        non_budget_cols = pdf_final_sp.columns[pdf_final_sp.columns.str.contains(pat='DURATION')]
        budget_cols = [value for value in all_columns if value not in non_budget_cols]
        common_list = ['Marketing_Month', 'Brand_Name', 'Total_Conversion', 'Total_Impression']
        budget_cols = list((Counter(budget_cols)-Counter(common_list)).elements())
        budget_df = pdf_final_sp[pdf_final_sp.columns[pdf_final_sp.columns.isin(budget_cols)]]
        budget_df = budget_df.fillna(0)
        budget_df = budget_df.astype(int)
        budget_df['total_budget'] = budget_df[list(budget_df.columns)].sum(axis=1)
        final_budget = budget_df['total_budget'].sum()

        return final_budget

    # optimization code

    def Maximize_Conversion(self, x, budget, Model):
        from csv import writer
        with open('opti_cols_pyswarms.csv', 'a', newline='') as f_object:
            for row in x:
                writer_object = writer(f_object)
                writer_object.writerow(list(map(str, row)))
            f_object.close()
        mutipliers = []
        for row in x:
            if(sum(row[0:2]) > budget):
                mutipliers.append(1)
            else:
                mutipliers.append(-1)
        predictions = Model.predict(x)
        # print(predictions)
        #predictions = [prediction for sublist in predictions for prediction in sublist]
        predictions = [predictions[i] * mutipliers[i] for i in range(len(predictions))]
        # print(predictions)
        return predictions
        # return multiplier * prediction

    def append_fun(self, budget, Model):
        df_updated = pd.read_csv('opti_cols_pyswarms.csv')
        df_updated["Net_Conversion_rate"] = Model.predict(df_updated)
        df_updated = df_updated.sort_values(by=["Net_Conversion_rate"], ascending=False)
        df_updated["Rank"] = df_updated["Net_Conversion_rate"].rank(ascending=0)
        df_updated['Budget'] = budget
        # df_updated['Brand_Name'] = brand_name
        # df_updated = df_updated.set_index('Rank')
        col_list = df_updated.columns.tolist()
        unwanted_elements = ['Rank', 'Net_Conversion_rate', 'Budget', 'Brand_Name']
        col_list = [ele for ele in col_list if ele not in unwanted_elements]
        col_list = [x for x in col_list if not x.endswith('-DURATION')]
        df_new_final = pd.DataFrame(columns=['Rank', 'Net_Conversion_rate', 'Budget',
                                    'Brand_Name', 'Optimized Budget', 'Duration', 'Campaign', 'Publisher'])
        for i in col_list:
            campaign = i.split('-')[0]
            publisher = i.split('-')[1]
            df_new = df_updated[df_updated.columns[df_updated.columns.isin(
                ['Rank', 'Net_Conversion_rate', 'Budget', 'Brand_Name'])]]
            df_new['Optimized Budget'] = df_updated[i]
            df_new['Duration'] = df_updated[i+'-DURATION']
            df_new['Campaign'] = campaign
            df_new['Publisher'] = publisher
            df_new_final = df_new_final.append(df_new, ignore_index=True)
        # new_column = pd.DataFrame({'budget': [budget] * len(df_updated.index)})
        # df_updated = df_updated.merge(new_column, left_index = True, right_index = True)
    #   optimization_output.to_csv('opti_cols_final_pyswarms.csv',mode = 'w' if(budget == limits[0][0]) else 'a',header = True if(budget == limits[0][0]) else False)
        f = open('opti_cols_pyswarms.csv', 'r+')
        f.truncate(0)

        return df_new_final

    def optimization_code(self, df, brand_name):
        model_input = self.data_aggregation(df, brand_name)
        from sklearn import datasets, linear_model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from pyswarms.single.global_best import GlobalBestPSO

        X = model_input.loc[:, model_input.columns != 'Net_Conversion_rate']
        y = model_input.loc[:, model_input.columns == 'Net_Conversion_rate']
        rf = RandomForestRegressor(max_depth=80, max_features=3, min_samples_leaf=3,
                                   min_samples_split=8)
        # Train the model using the training sets
        rf.fit(X, y)
        # optimization code

        shape_list = X.shape
        #print(shape_list)

        # final_budget = final_budget.item()

        final_budget = self.budget_function(df, brand_name)
        budget_list_final = [final_budget]*11
        for i in range(5):
            budget_list_final[4-i] = budget_list_final[5-i]*0.9
            budget_list_final[6+i] = budget_list_final[5+i]*1.1
        all_budget_list = budget_list_final
        duration_list = [12]*11
        all_budget_list = [round(num) for num in all_budget_list]
        #print(all_budget_list)
        limits = list(zip(all_budget_list, duration_list))
        Model = rf

        # instatiate the optimizer
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        result = pd.DataFrame()
        for limit in limits:
            x_max = []
            df_opti = pd.DataFrame(columns=X.columns)
            df_opti.to_csv("opti_cols_pyswarms.csv", index=False)
            for i in range(shape_list[1]//2):
                x_max = x_max + [limit[0]/round(shape_list[1]/4), limit[1]]
            x_min = [0] * shape_list[1]
            bounds = (x_min, x_max)
            optimizer = GlobalBestPSO(
                n_particles=100, dimensions=shape_list[1], options=options, bounds=bounds)
            cost, joint_vars = optimizer.optimize(
                self.Maximize_Conversion, iters=1000, verbose=False, budget=limit[0], Model=Model)
            #print(joint_vars.sum())
            # append_fun(limit[0])
            result = result.append(self.append_fun(limit[0], Model))
        return result

    # @api(input=DataframeInput(orient="records"), batch=True, output=DataframeOutput(output_orient='records'))
    @api(
        input=DataframeInput(
            orient="records",
            columns=["Marketing_Year", "Marketing_Month", "Campaign_Name",
                     "Brand_Name", "Publisher_Name","Payment_Cost",
                     "Total_Conversion", "Total_Impression","Net_Conversion_rate"],
            dtype={"Marketing_Year": "int", 
                   "Marketing_Month": "int",
                   "Campaign_Name": "string",
                   "Brand_Name": "string",
                   "Publisher_Name":"string",
                   "Payment_Cost": "int", 
                   "Total_Conversion": "float",
                   "Total_Impression": "int",
                   "Net_Conversion_rate": "float"}),
        batch=True,
        # !!!!!!!!!!!
        output=DataframeOutputForMoreRows(output_orient='records') # replace DataframeOutput with DataframeOutputForMoreRows
        # !!!!!!!!!!!
    )
    # The list of brands to be selected. All brand results will be mutually exclusive
    # Iterating over all brands
    def predict(self, df):
        brand_list = df['Brand_Name'].unique()
        final_append_optimization_model = pd.DataFrame()
        for brand_name in brand_list:
            optimization_output = self.optimization_code(df, brand_name)
            optimization_output['Brand_Name'] = brand_name
            final_append_optimization_model = final_append_optimization_model.append(
                optimization_output)

        final_append_optimization_model = final_append_optimization_model.reset_index()
        df_final = pd.DataFrame()
        df_final['all_columns'] = (final_append_optimization_model['Rank'].astype(str)+'|'+final_append_optimization_model['Net_Conversion_rate'].astype(str)+'|'
                                   + final_append_optimization_model['Budget'].astype(str)+'|'+final_append_optimization_model['Brand_Name'].astype(
                                       str)+'|'+final_append_optimization_model['Optimized Budget'].astype(str)
                                   + '|'+final_append_optimization_model['Duration'].astype(str)+'|' + final_append_optimization_model['Campaign'].astype(str)+'|' + final_append_optimization_model['Publisher'].astype(str))
        list_zero = ['0']*(df.shape[0] - final_append_optimization_model.shape[0])
        xtra = {'all_columns': list_zero}
        df_final = df_final.append(pd.DataFrame(xtra))
        return df_final

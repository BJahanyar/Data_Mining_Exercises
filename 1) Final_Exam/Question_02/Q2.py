import pandas as pd
MyData =  pd.read_csv("Files/Diabetes_Diagnosis.csv") 
def Change(diabetes):
  if diabetes == True:
    diabetes='1'
  else:
    diabetes='0'
  return diabetes
MyData.diabetes = MyData.diabetes.apply(Change)
statistics_Diabetes = MyData[MyData['diabetes'] == '1'].describe() # تفکیک بیماران دیابتی
statistics_Diabetes.rename(columns=lambda x: x + '_True', inplace=True) 
statistics_NonDiabetes = MyData[MyData['diabetes'] == '0'].describe() # تفکیک بیماران غیر دیابتی
statistics_NonDiabetes.rename(columns=lambda x: x + '_False', inplace=True) 
statistics = pd.concat([statistics_Diabetes, statistics_NonDiabetes], axis=1) # ارائه گزارش
statistics.to_csv('Files/Q2_Regression_Result.csv')
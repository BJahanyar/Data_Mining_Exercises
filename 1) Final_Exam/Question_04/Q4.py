import pandas as pd
MyData =  pd.read_csv("Files/Diabetes_Diagnosis.csv")   # خواندن داده ها
for column in MyData:   #انجام گسسته سازی
    if column != "diabetes":
        print("'Segmentation of' : ",column)
        List = MyData[column]
        result = pd.cut(List,4)
        print(result)
        print("=============================================================================================")

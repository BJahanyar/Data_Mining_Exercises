#Calculate transpose of matrix
Row = int(input("Please Enter Row Number:")) #تعداد درایه های یک سطر را بگیر 
Col = int(input("Please Enter Col Number:")) #تعداد درایه های یک ستون را بگیر 
Main = [] #یک آرایه برای نگهداری ماتریس ایجاد کن
for i in range(Row): #حلقه را تا کمتر از عدد سطر تکرار کن
    TempRow = [] # یک آرایه برای نگه داری سطر ایجاد کن 
    for j in range(Col): # حلقه را تا کمتر از عدد ستون تکرار کن
        TempRow.append(int(input())) #عددی که کاربر وارد کرده را به آرایه تمپ اضافه کن
        pass # حلقه تمام شده است
    Main.append(TempRow)  #عددی که کاربر وارد کرده را به آرایه مین اضافه کن
    pass
print('___________Main Matris___________' )
print(Main) #آرایه مین را چاپ کن
MyTranspose = []
for i in range(Col):
    TempTrans = []
    for j in range(Row):
        TempTrans.append(Main[j][i])        
    MyTranspose.append (TempTrans)
print('___________MyTranspose Matris___________' )
print(MyTranspose)

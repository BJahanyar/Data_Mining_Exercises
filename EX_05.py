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

Mysparse = []
for i in range(Row):
    for j in range(Col):
        if Main[i][j] != 0:
            Temp = [i,j,Main[i][j]]
            Mysparse.append (Temp)
            pass
        pass
    pass
print('___________Sparse Matris___________' )
print(Mysparse) #آرایه مین را چاپ کن

MyTranspose = []
for i in range(Col):
    TempTrans = []
    for j in range(Row):
        TempTrans.append(Main[j][i])        
        pass
    MyTranspose.append (TempTrans)
    pass

print('___________MyTranspose Matris___________' )
print(MyTranspose) #آرایه مین را چاپ کن

MySum = []
for i in range(Row):
    Tempsum = []
    for j in range(Col):
        Tempsum.append(Main[i][j] + Main[i][j])
        pass
    MySum.append (Tempsum)
    pass

print('___________MySum Matris___________' )
print(MySum) #آرایه مین را چاپ کن

MyMulti = []
for i in range(Row):
    TempMult = []
    for j in range(Col):
        TempMult.append(Main[i][j] * Main[i][j])
        pass
    MyMulti.append (TempMult)
    pass

print('___________MyMulti Matris___________' )
print(MyMulti) #آرایه مین را چاپ کن
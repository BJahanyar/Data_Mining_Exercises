Row = int(input("Please Enter Row Number:")) #تعداد درایه های یک سطر را بگیر 
Col = int(input("Please Enter Col Number:")) #تعداد درایه های یک ستون را بگیر 
Main = [] #یک آرایه برای نگهداری ماتریس ایجاد کن
for i in range(Row): #حلقه را تا کمتر از عدد سطر تکرار کن
    TempRow = [] # یک آرایه برای نگه داری سطر ایجاد کن 
    for j in range(Col): # حلقه را تا کمتر از عدد ستون تکرار کن
        TempRow.append(int(input())) #عددی که کاربر وارد کرده را به آرایه تمپ اضافه کن
    Main.append(TempRow)  #عددی که کاربر وارد کرده را به آرایه مین اضافه کن
print('___________Main Matris___________' )
print(Main) #آرایه مین را چاپ کن

Mysparse = []
for i in range(Row):
    for j in range(Col):
        if Main[i][j] != 0:
            Temp = [i,j,Main[i][j]]
            Mysparse.append (Temp)
print('___________Sparse Matris___________' )
print(Mysparse)

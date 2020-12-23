#Sum on matrix to itself
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
print(Main)
MySum = []
for i in range(Row):
    Tempsum = []
    for j in range(Col):
        Tempsum.append(Main[i][j] + Main[i][j])
    MySum.append (Tempsum)
print('___________MySum Matris___________' )
print(MySum)

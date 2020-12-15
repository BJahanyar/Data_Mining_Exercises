print ("Please Enter a Sentence :")
x = input() # یک جمله را بگیر
splt = x.split(",") #جمله را با مشاهده ، بشکن
Array = [] # یک آرایه ایجاد کن 
for word in splt: #برای کلمات در جمله شکسته شده
    if splt.count(word) == 1 : # اگر تعداد آن کلمه در جمله برابر 1 بود
        Array.append(word) # آن را در آرایه ای که ایجاد کردیم بریز 
        pass
    pass
Array.sort() # کلماتی که یک بار تکرار داشتند مرتب کن
print (Array)
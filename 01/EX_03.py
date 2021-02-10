print ("Please Enter a String")
x = input() # یک رشته(جمله) بگیر
s = x.split() # آن رشته را بر اساس فاصله بشکن
thisdict = {} # یک آرایه دو بعدی تعریف کن برای نگه داری کلمات و تعدادشان
for ch in s: # هر کلمه در رشته(جمله) را پشمایش کن
    if ch not in thisdict.keys() : #اگر کلمه در کلیدهای دیکشنری وجود  نداشت
        thisdict.update({ ch : "1" }) #مقدار را 1 بگذار
    else: # اگر کلمه در کلید‌های دیکشنری وجود داشت
        #مقدار آرایه را بگیر بعلاوه 1 کن و آن را مجددا در متغییر بریز و آن را آپدیت کن
        thisdict.update({ch : int(thisdict.get(ch)) +1  })
print(thisdict)

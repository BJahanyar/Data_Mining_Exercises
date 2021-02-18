print ("Please Enter a String")
x = input() # یک رشته بگیر
thisdict = {} # یک آرایه دو بعدی تعریف کن برای نگه داری کارکتر ها و تعدادشان
for ch in x: # هر کاراکتر در رشته را پشمایش کن
    if ch not in thisdict.keys() : #اگر کاراکتر در کلیدهای دیکشنری وجود  نداشت
        thisdict.update({ ch : "1" }) #مقدار را 1 بگذار
    else: # اگر کاراکتر در کلید دیکشنری وجود داشت
        #مقدار آرایه را بگیر بعلاوه 1 کن و آن را مجددا در متغییر کاراکتر بریز و آن را آپدیت کن
        thisdict.update({ch : int(thisdict.get(ch)) +1  })
print(thisdict)

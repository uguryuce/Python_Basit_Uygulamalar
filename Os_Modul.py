

import os
import datetime
text="""Lütfen işlem yapmak istediğiniz numarayı girin: \n
1- Geçerli dizini öğrenmek\n
2- Bulunduğunuz dizindeki dosya ve klasörleri görmek\n
3- Geçerli dizine yeni bir dizin eklemek veya silmek\n
4- Geçerli dizindeki dosyalar hakkında bilgi almak\n
5- Kullanmış olduğunuz işletim sistemini öğrenmek\n
6- Kullanıcı ismi öğrenmek\n
7-Çık

"""
osname=os.name
lst=[]




def select_2():
    
    num=0
    
    print("-"*50,"\nBulunduğunuz dizindeki dosya ve klasörler:\n")
    
    for i in os.listdir():
        lst.append(i)
        print(num,"-",i)
        num=num+1
        
    print("-"*50)
    
    
    
def select_3():
    select_2()
    
    try:
        
        islem=input("Yapmak istediğiniz işlem:\n1-Dizin eklemek\n2-Dizin Silmek\n")
        
        if islem=="1":
            
            dizin_ismi=input("Oluşturmak istediğiniz dizin ismini giriniz:\n")
            os.mkdir(dizin_ismi)
            
        elif islem=="2":
            
            dizin_ismi=input("Silmek istediğiniz dizin ismini giriniz:\n")
            os.rmdir(dizin_ismi)
        
        
    except OSError as exp:
       print("Hata oluştu",exp)
       pass
    
    select_2()
    
    
    
def select_4():
    select_2()
    try:
        dosya_num=int(input("Bilgi almak istediğiniz dosyanın numarasını girin:\n"))
        
        dosya=os.stat(lst[dosya_num])
        print("Dosyaya en son erişilme tarihi", convert_time(datetime.datetime.fromtimestamp(dosya.st_atime)))
        print("Dosyanın oluşturulma tarihi (Windows’ta):", convert_time(datetime.datetime.fromtimestamp(dosya.st_ctime)))
        print("Dosyanın son değiştirilme tarihi", convert_time(datetime.datetime.fromtimestamp(dosya.st_mtime)))
        print("Dosyanın boyutu",(dosya.st_size))
        
        
    except:
        print("Hata Oluştu")
        pass
    
 
def select_5():
    if osname=='posix':
        
        print("Kullanmış olduğunuz işletim sistemi: unix")
        
    elif osname=='nt':
        
        print("Kullanmış olduğunuz işletim sistemi: Windows")


def select_6():
    if osname=="posix":
        
        print("Kullanıcı ismi: ",os.environ['USER'])
        
    elif osname=="nt":
        
        print("Kullanıcı ismi:",os.environ['USERNAME'])
    
    
    
    
   
    
   
    
def convert_time(tm):
    return (datetime.datetime.strftime(tm, '%c'))
    


girdi= input(text)
    

if girdi == "1":
        print("-"*50,"\nGeçerli dizin ismi:",os.getcwd())
if girdi=="2":
        select_2()
if girdi=="3":
    select_3()
if girdi=="4":
    select_4()
if girdi=="5":
    select_5()
if girdi=="7":
    exit()

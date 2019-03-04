sayi1=int(input("Sayı girin:"))
sayi2=int(input("Sayı girin"))
toplam=0

for sayi in range(sayi1,sayi2+1):
    for bolum in range(1,sayi):
        if (sayi%bolum==0):
            toplam=toplam+bolum
    if (toplam==sayi):
        print(sayi)
    toplam=0



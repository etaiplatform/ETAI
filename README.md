# Enerji Piyasaları için Yapay Zeka ile Üretim, Talep ve Fiyat Bilgisi Tahminleme Platformu: ETAI

## Hakkında
ETAI, yapay zeka/makine öğrenmesi yöntemleri kullanılarak elektrik enerjisi piyasalarında fiyat, talep veya üretim tahmini yapan, web tabanlı gösterge panelinde önemli istatistikler, grafikler, tahmin yapılabilmesi için seçenekler bulunduran bir tahminleme platformudur.
### Why ETAI:
* Fiyat, üretim ve talep değerleri için bütünleşik bir yapıda tahminleme.
* Son kullanıcıya yönelik gösterge paneli ve geliştiriciler tarafından özelleştirilebilmesi için bir API.
* Türkiye elektrik piyasasında da sıkça görülen, ekstrem fiyat değerlerinin daha iyi tahminlenmesi için farklı çözüm mimarileri.
* Çözüm Mimarileri:
  - BIN1: İki sınıflı etiketleme (ekstrem değer, normal değer) yinelemeli ve iki adımlı tek bir model.
  - DMDNUL1: üç sınıflı etiketleme (alt ekstrem, üst ekstrem, normal) yinelemeli olmayan ve iki adımlı tek bir model
  - NUL1: üç sınıflı etiketleme, yinelemeli ve iki adımlı tek bir model
  - NUL3: üç sınıflı etiketleme, yinelemeli ve iki adımlı üç ayrı model (her bir sınıf için ayrı model)
  - DEF: yinelemeli klasik mimari, tek bir model 
# Lokal sunucuda çalıştırmak için:
* API klasöründeki app.py isimli dosyayı çalıştırın, API sunucunuzda çalışmaya başlayacaktır, fiyat, talep ve üretim tahminleme için sunucuya örnek istek gönderimi
  - ``` http://0.0.0.0:5000/predict?startDate=2019-01-01&endDate=2021-05-01&days=7&model=NUL1&target=price ```
  - ``` http://0.0.0.0:5000/predict?startDate=2019-01-01&endDate=2021-05-01&days=7&model=NUL1&target=consumption ```
  - ``` http://0.0.0.0:5000/predict?startDate=2019-01-01&endDate=2021-05-01&days=7&model=NUL1&target=production ```
* Gösterge Panelini çalıştırmak için, DashApp klasöründeki app.py isimli dosyayı çalıştırın, gösterge paneli lokal sunucuda çalışmaya başlayacaktır
* Sadece Tahminleme özelliğini kullanmak ve sonuçları görmek için, flaskApp klasöründeki app.py isimli dosyayı çalıştırın, lokal sunucuda çalışmaya başlayacaktır
# ETAI / Görseller
## Gösterge Paneli:
![alt text]("https://github.com/etaiplatform/ETAI/blob/master/ETAI_DASH.png")

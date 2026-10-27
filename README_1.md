# MNIST ile MLP Tabanlı El Yazısı Rakam Tanıma

Bu projede MNIST veri seti üzerinde **sıfırdan** eğitilen bir Multi-Layer Perceptron (MLP) bulunur.
CNN ve hazır eğitilmiş model kullanılmamıştır. Framework: PyTorch.

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt

# Deney 1 - temel MLP (MNIST ilk çalıştırmada otomatik indirilir)
python mlp_mnist.py

# Farklı yapılandırma denemek için (hiperparametreler argümandır)
python mlp_mnist.py --hidden 512 256 --lr 0.0005 --epochs 30 --name deney2_buyuk
python mlp_mnist.py --optimizer sgd --lr 0.01 --name deney3_sgd

# Kendi el yazınla öğrenci numarası tahmini
python predict_custom.py custom_digits/*.png                 # her rakam ayrı dosya (2_a.png gibi)
python predict_custom.py custom_digits/numara.png --segment --truth ÖĞRENCİNUMARANIZ
```

## Veri ayrımı

| Küme | Boyut | Kullanım |
|---|---|---|
| Train | 50.000 | Ağırlık güncelleme |
| Validation | 10.000 | Epoch başına izleme / hiperparametre kararı |
| Test | 10.000 | Yalnızca eğitim bittikten sonra **bir kez** |

Train/validation ayrımı MNIST'in 60.000'lik eğitim kümesinden sabit seed (42) ile yapılır.
Test verisi model geliştirme sürecinde kullanılmaz.

## Deney 1 – Temel MLP

Mimari: `784 → 256 → 128 → 10`, ReLU, CrossEntropyLoss, Adam.

| Hiperparametre | Değer |
|---|---|
| Hidden layer | 256, 128 |
| Aktivasyon | ReLU |
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epoch | 20 |
| Dropout | 0 |
| Normalizasyon | (x/255 − 0.1307) / 0.3081 |
| Seed | 42 |

### Sonuçlar (kendi çalıştırmanızın çıktısıyla doldurun)

| Metrik | Değer |
|---|---|
| Training loss | |
| Validation loss | |
| Training accuracy | |
| Validation accuracy | |
| Test accuracy | |
| Eğitim süresi | |

Grafik: `results/deney1_temel_mlp_egriler.png`

![egriler](results/deney1_temel_mlp_egriler.png)

## Kendi el yazım

`custom_digits/` klasöründe öğrenci numaramın el yazısı görselleri bulunur.
Ön işleme adımları (`predict_custom.py`): gri tonlama → MNIST renk formatına çevirme
(siyah zemin, beyaz rakam) → rakama göre kırpma → 20×20'ye sığdırma → kütle merkeziyle
28×28 tuvale ortalama → normalizasyon.

Sonuç: _(tahmin edilen numara, doğru sayısı, `results/ozel_tahminler.png`)_

## Genel değerlendirme

_(Buraya kendi yorumunuzu yazın. Örnek başlıklar:)_

- Train ve validation eğrileri nasıl ilerledi? Overfitting belirtisi var mı?
- Test accuracy ile validation accuracy tutarlı mı?
- Kendi el yazımdaki başarı MNIST test başarısından neden farklı? (distribution shift:
  kalem kalınlığı, yazı stili, ışık/gürültü, ön işleme farkları; MLP'nin konum ve
  şekil değişimlerine karşı CNN kadar dayanıklı olmaması)
- Hiperparametre değişikliklerinin (hidden boyutu, learning rate, optimizer) etkisi.

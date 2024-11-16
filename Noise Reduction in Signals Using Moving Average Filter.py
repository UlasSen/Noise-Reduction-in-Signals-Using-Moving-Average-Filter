import numpy as np
import matplotlib.pyplot as plt

fs = 1000  # Örnekleme frekansı (Hz)
t = np.linspace(0, 1, fs, endpoint=False)  # Zaman
# Örnek bir sinyal oluştur
def generate_signal(t):
    return np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)


# Gürültü ekle (Gaussian gürültü)
def add_noise(signal, noise_level=0.2):
    noise = noise_level * np.random.normal(size=signal.shape)
    return signal + noise
# Sinyalleri oluştur
original_signal = generate_signal(t)
noisy_signal = add_noise(original_signal, noise_level=0.2)

# Kayan ortalama filtresi
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')
# Kayan ortalama filtresi uygulama
filtered_signal_ma = moving_average(noisy_signal, window_size=3)

# Grafiklerle sonuçları görselleştirme
plt.figure(figsize=(12, 16))

# Orijinal sinyal
plt.subplot(4, 1, 1)
plt.plot(t, original_signal, label="Orijinal Sinyal", linewidth=2)
plt.title("Orijinal Sinyal")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.grid()
plt.legend()

# Orijinal + Gürültülü sinyal
plt.subplot(4, 1, 2)
plt.plot(t, original_signal, label="Orijinal Sinyal", linewidth=1, linestyle='--')
plt.plot(t, noisy_signal, label="Gürültülü Sinyal", color='orange', linewidth=1)
plt.title("Orijinal + Gürültülü Sinyal")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.grid()
plt.legend()

# Gürültülü + Filtrelenmiş sinyal
plt.subplot(4, 1, 3)
plt.plot(t, noisy_signal, label="Gürültülü Sinyal", color='orange', linewidth=1)
plt.plot(t, filtered_signal_ma, label="Kayan Ortalama Filtre", color='blue', linewidth=2)
plt.title("Gürültülü + Kayan Ortalama Filtre Sonucu")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.grid()
plt.legend()

# Orijinal + Filtrelenmiş sinyal
plt.subplot(4, 1, 4)
plt.plot(t, original_signal, label="Orijinal Sinyal", linewidth=1, linestyle='--', color='green')
plt.plot(t, filtered_signal_ma, label="Kayan Ortalama Filtre", color='blue', linewidth=2)
plt.title("Orijinal + Kayan Ortalama Filtre Sonucu")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

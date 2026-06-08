# Sprawozdanie z Laboratorium nr 5

**Temat:** Rekurencyjne Sieci Neuronowe (RNN) z bramkami LSTM i GRU na przykładzie regresji do predykcji cen akcji.

**Prowadzący:** dr inż. Jacek Paluszak

**Student:** Kacper Szponar 21306

---

## 1. Wstęp Teoretyczny

Standardowe sieci rekurencyjne (RNN) posiadają zdolność przetwarzania sekwencji danych, jednak przy dłuższych szeregach czasowych cierpią na **problem zanikającego gradientu** (*Vanishing Gradient Problem*). Zjawisko to pojawia się podczas wstecznej propagacji błędu przez czas (BPTT), kiedy gradienty stają się ekstremalnie małe, co uniemożliwia skuteczną aktualizację wag sieci dla wcześniejszych kroków czasowych i ogranicza efektywną pamięć modelu do zaledwie kilku poprzednich kroków.

Aby rozwiązać ten problem, stosuje się zaawansowane architektury rekurencyjne:
* **LSTM (Long Short-Term Memory):** Wprowadza specjalne bloki pamięciowe (komórki) wyposażone w bramki: zapominania ($f$), wejściową ($i$) oraz wyjściową ($o$). Dzięki temu sieć potrafi kontrolować przepływ informacji i przechowywać stan pamięci długotrwałej przez wiele kroków czasowych.
* **GRU (Gated Recurrent Unit):** Stanowi uproszczoną alternatywę dla LSTM. Nie używa oddzielnego stanu komórki – posiada jedynie dwie bramki (resetowania oraz aktualizacji), co przyspiesza proces uczenia (mniej parametrów) przy zachowaniu zbliżonej efektywności.

Do oceny modeli wykorzystano następujące metryki:
* **RMSE** (Root Mean Squared Error) – pierwiastek błędu średniokwadratowego (karze duże odchylenia).
* **MAE** (Mean Absolute Error) – średni błąd bezwzględny (traktuje wszystkie błędy liniowo).
* **MAPE** (Mean Absolute Percentage Error) – średni bezwzględny błąd procentowy (pokazuje błąd w ujęciu procentowym względem wartości rzeczywistych).

---

## 2. Porównanie bazowe: LSTM vs GRU (Konfiguracja domyślna)

W teście bazowym oba modele zostały uruchomione na domyślnych parametrach (50 epok, funkcja straty MSE, predykcja wartości maksymalnych "High"):
* **LSTM (Bazowy):** Optymizator `RMSprop` | $RMSE = 3.07$ | $MAE = 2.10$ | $MAPE = 1.32\%$
* **GRU (Bazowy):** Optymizator `SGD` | $RMSE = 3.23$ | $MAE = 2.20$ | $MAPE = 1.39\%$

### Wyniki wizualne dla modelu bazowego LSTM:
![Wykres i błędy LSTM bazowy](Screeny/Zrzut%20ekranu%202026-05-25%20112717.png)

### Wyniki wizualne dla modelu bazowego GRU:
![Wykres i błędy GRU bazowy](Screeny/Zrzut%20ekranu%202026-05-25%20112731.png)

**Wnioski z testu bazowego:**
W konfiguracji domyślnej sieć LSTM poradziła sobie minimalnie lepiej od GRU, dokładniej dopasowując się do ogólnego trendu cen akcji IBM. Model GRU wykazał nieco większe błędy ($RMSE = 3.23$ vs $3.07$), ale charakteryzował się prostszą strukturą i krótszym czasem trenowania pojedynczej epoki (ze względu na mniejszą liczbę parametrów).

---

## 3. Wyniki Eksperymentów (Modyfikacje modelu LSTM)

Wszystkie poniższe eksperymenty (A-E) były przeprowadzane poprzez modyfikację architektury lub parametrów uczenia modelu LSTM.

### A. Wpływ ilości jednostek LSTM (`units=100`)
Zwiększono liczbę neuronów w warstwach ukrytych z 50 do 100 w celu zwiększenia pojemności sieci.
* **Wyniki:** $RMSE = 4.94$ | $MAE = 4.43$ | $MAPE = 2.80\%$
* **Wizualizacja:**
  ![Wyniki dla 100 jednostek](Screeny/Zrzut%20ekranu%202026-05-25%20113147.png)
* **Wniosek:** Odnotowano gwałtowny wzrost błędów (RMSE wzrosło z 3.07 do 4.94). Zbyt duża liczba parametrów (przewymiarowanie modelu) w stosunku do wielkości zbioru danych doprowadziła do zjawiska **overfittingu** (przeuczenia) sieci na danych treningowych, przez co model stracił zdolność generalizacji na niezależnym zbiorze testowym.

### B. Testowanie optymizatorów (Zmiana na `Adam`)
Zastąpiono bazowy optymizator `RMSprop` algorytmem `Adam`.
* **Wyniki:** $RMSE = 2.20$ | $MAE = 1.52$ | $MAPE = 0.95\%$
* **Wizualizacja:**
  ![Wyniki dla optymizatora Adam](Screeny/Zrzut%20ekranu%202026-05-25%20113653.png)
* **Wniosek:** Zastosowanie optymizatora Adam przyniosło znakomity efekt. Błąd średniokwadratowy ($RMSE$) spadł o blisko 30% (do poziomu 2.20), a model znacznie precyzyjniej zaczął naśladować lokalne minima i maksima, szybciej zbiegając do optymalnego rozwiązania.

### C. Testowanie funkcji straty (Zmiana na `MAE`)
Zmieniono funkcję straty z `mean_squared_error` (MSE) na `mean_absolute_error` (MAE).
* **Wyniki:** $RMSE = 5.09$ | $MAE = 3.77$ | $MAPE = 2.38\%$
* **Wizualizacja:**
  ![Wyniki dla straty MAE](Screeny/Zrzut%20ekranu%202026-05-25%20113902.png)
* **Wniosek:** Funkcja straty MAE traktuje każdy błąd liniowo, co sprawia, że jest mniej wrażliwa na anomalie i nagłe skoki cenowe. W przypadku predykcji giełdowej doprowadziło to do dużego spłaszczenia wykresu i znacznego opóźnienia sygnału prognozy, skutkując bardzo słabymi metrykami błędu ($RMSE = 5.09$).

### D. Zmiana atrybutu predykcji z "High" na "Close"
Przeprowadzono proces przygotowania danych i uczenia dla dziennych cen zamknięcia akcji ("Close") zamiast cen maksymalnych ("High").
* **Wyniki:** $RMSE = 2.26$ | $MAE = 1.58$ | $MAPE = 0.99\%$
* **Wizualizacja:**
  ![Wyniki dla cen Close](Screeny/Zrzut%20ekranu%202026-05-25%20114155.png)
* **Wniosek:** Sieć wykazuje bardzo zbliżoną i wysoką stabilność działania zarówno na cenach maksymalnych ("High"), jak i zamknięcia ("Close"). Różnice w metrykach są minimalne, co dowodzi uniwersalności architektury w odniesieniu do różnych charakterystyk cenowych tej samej spółki.

### E. Testowanie mechanizmu Early Stopping z walidacją
Przetestowano dwie konfiguracje mechanizmu wczesnego zatrzymywania:

1. **Monitorowanie straty treningowej (`monitor='loss'`, `patience=5`, `epochs=100`):**
   * **Wyniki:** $RMSE = 3.05$ | $MAE = 2.03$ | $MAPE = 1.28\%$
   * **Wizualizacja:**
     ![Early Stopping Loss](Screeny/Zrzut%20ekranu%202026-05-25%20114337.png)
   * **Wniosek:** Trening został przerwany automatycznie, gdy strata na zbiorze treningowym przestała maleć. Wynik jest zbliżony do modelu bazowego, ale proces uczył się krócej, oszczędzając zasoby obliczeniowe.

2. **Monitorowanie straty walidacyjnej z wydzieleniem podzbioru (`validation_split=0.1`, `monitor='val_loss'`, `patience=5`, `epochs=100`):**
   * **Wyniki:** $RMSE = 3.65$ | $MAE = 3.00$ | $MAPE = 1.90\%$
   * **Wizualizacja:**
     ![Early Stopping Val Loss](Screeny/Zrzut%20ekranu%202026-05-25%20115838.png)
   * **Wniosek:** Dodanie walidacji i monitorowanie `val_loss` pozwala na obiektywne przerwanie uczenia w momencie, gdy model zaczyna tracić zdolności generalizacyjne (zaczyna się przeuczać).

---

## 4. Ostateczna, najlepsza konfiguracja modelu

W celu uzyskania błędu $RMSE < 2.0$ (Zadanie F), połączono wnioski ze wszystkich poprzednich kroków i wdrożono zoptymalizowane hiperparametry:
1. **Zrównoważona architektura:** Zastosowano 4 warstwy LSTM o malejącej liczbie jednostek (50 -> 50 -> 32 -> 32) w celu redukcji ryzyka overfittingu przy zachowaniu odpowiedniej pojemności reprezentacyjnej sieci.
2. **Dynamiczny optymizator:** Wykorzystano optymizator Adam ze zwiększonym tempem uczenia `learning_rate=0.005` w celu podniesienia dynamiki reakcji sieci.
3. **Optymalna funkcja straty:** Zastosowano **stratę Hubera** (`loss='Huber'`), która łączy zalety MSE (duża kara za duże błędy w obszarze małych odchyleń) i MAE (liniowa kara w obszarze dużych odchyleń). Pozwoliło to modelowi ignorować drobny szum giełdowy, jednocześnie stabilnie reagując na nagłe anomalie (piki cenowe).
4. **Wydłużony trening z regulacją:** Zaimplementowano długi trening (100 epok) kontrolowany przez Early Stopping z parametrem `patience=10` na błędzie walidacyjnym (`validation_split=0.1`) w celu przywrócenia wag o najlepszej zdolności generalizacji.

### Uzyskane parametry końcowe:
* **Błąd średniokwadratowy ($RMSE$):** **1.69** *(Wymóg $RMSE < 2.0$ został w pełni spełniony z dużym zapasem)*
* **Średni błąd bezwzględny ($MAE$):** **1.06**
* **Średni bezwzględny błąd procentowy ($MAPE$):** **0.66%**

### Końcowy wykres dopasowania modelu:
![Najlepszy wynik LSTM](Screeny/Zrzut%20ekranu%202026-05-25%20120554.png)

### Uzasadnienie:
Ta konfiguracja okazała się zdecydowanie najlepsza. Zmiana funkcji straty na Huber uodporniła model na lokalne szumy i nagłe wahnięcia rynku (szczególnie widoczne przy punkcie czasowym ~200, gdzie inne modele mocno przestrzelone opóźniały reakcję lub reagowały histerycznie). Zwiększenie parametru uczenia do `0.005` pozwoliło sieci na szybszą i bardziej precyzyjną reakcję na nagłe zmiany kierunku cen akcji, eliminując zjawisko "opóźnienia" (fazy przesunięcia) wykresu predykcji względem rzeczywistego trendu. Dodatkowo, mechanizm Early Stopping oparty na zbiorze walidacyjnym zapobiegł przeuczeniu sieci, zatrzymując proces w optymalnym punkcie.
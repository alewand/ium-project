# Projekt na IUM (Inżynieria uczenia maszynowego), Politechnika Warszawska
### By Bartosz Żelazko & Amadeusz Lewandowski

Celem projektu jest opracowanie modelu uczenia maszynowego oraz mikro serwisu do lepszego pozycjonowania nowych ofert z małą liczbą ocen w serwisie typu Airbnb.

Szczegóły i dokumentację można znaleźć [w tym miejscu](documentation.pdf).

Projekt został stworzony przy pomocy menadżera pakietów [uv](https://github.com/astral-sh/uv) i rekomenduje się jego instalację w celu uruchomienia.

## Instrukcja uruchomienia - Linux

1. Sklonowanie repozytorium
```bash
git clone https://github.com/alewand/ium-project.git
```
2. Instalacja uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
3. Pobranie odpowiednich bibliotek
```bash
uv sync
```
4. Uruchomienie programu
```bash
uv run src/nazwa_danego_skryptu.py
```

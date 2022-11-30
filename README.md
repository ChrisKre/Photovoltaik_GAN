# Masterarbeit Photovoltaik
# Abstract
>Die Solarstromerzeugung ist unter den erneuerbaren Energiequellen eine der am meisten genutzten. Allerdings sind sowohl die Einführung als auch die Instandhaltung solcher Photovoltaiksysteme immer noch mit hohen Kosten und Effizienzproblemen verbunden.
Um diese Herausforderungen zu bewältigen, liefern Algorithmen des maschinellen Lernens
Lösungen zur Verbesserung der Leistung von Photovoltaiksystemen durch die Integration verschiedener Techniken und Modellierung komplexer Dynamiken. Einer der Schlüsselaspekte für den Erfolg dieser Art von Techniken ist die Verfügbarkeit von Daten.
Diese ist im Bereich der Photovoltaik nicht gegeben. Deswegen beschäftigt sich die vorliegende Arbeit mit der Problemstellung der synthetischen Datengenerierung. Hierbei liegt der Forschungsschwerpunkt auf der Anwendung von Generativen Adversarial Networks (kurz: GANs). Photovoltaikdaten unterliegen saisonalen Mustern, die durch die verschiedenen Jahreszeiten bedingt sind. Den Einfluss der Jahreszeiten gilt es bei einer Synthetisierung von Photovoltaikdaten zu berücksichtigen. Dafür werden zwei GAN-Architekturen gewählt, die unterschiedliche Ansätze verfolgen. Die für das Training dieser Architekturen benötigten Daten liefert die Photovoltaiksoftware PVSyst. Anhand branchenüblicher Bewertungsmaßstäbe wird auf die erfolgreiche Synthetisierung der Daten mit dem TimeGAN geschlossen. Der zweite Ansatz unter Einsatz des VAE-GANs erzielt schlechtere Ergebnisse. Dennoch erscheint dieser Ansatz vielversprechend. Ein Grund für die Datenknappheit im Bereich der Photovoltaik liegt in der Natur der Daten. Sie erlauben die Standortberechnung der Photovoltaikanlage, was zu Wettbewerbsnachteilen der Anbieter führen kann. Um dieses Problem zu lösen, können Anonymisierungstechniken auf den Daten angewendet werden, die eine Berechnung des Standorts verhindern. Diese Arbeit liefert erste Schritte einer solchen Anonymisierung.
## Python Bibliotheken
- tensorflow
- pandas
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- keras_tuner
- [pv-system-profiler](https://github.com/slacgismo/pv-system-profiler)
- [solar-data-tools](https://github.com/slacgismo/solar-data-tools) <- Benötigt MOSEK Solver, 30 Tage Trial ist nach Anmeldung verfügbar

### Installation
Von Projektroot 

``` pip install -r requirements.txt```

ausführen

### TimeGAN
Die TimeGAN Architektur ist in der Bibliothek [ydata-synthetic](https://github.com/ydataai/ydata-synthetic) implementiert.
Jedoch war es nicht möglich diese per pip auf dem GPU-Server zu installieren, weswegen einzelne Skripte der Bibliothek übernommen wurden. 

# Ausführreihenfolge und Abhängigkeiten zwischen Skripten

## Data Handling

#### Hinzufügen von Daten eines neuen Standorts
- .csv Files in **'data/{Standort}'** ablegen
- In **'src/data/settings.py'**: 
  - Hinzufügen in locations dictionary
  - Hinzufügen in classification_labels dictionary
  
#### Erstellen von vorverarbeiteten Datensätzen
- Die vorverarbeiteten sind im Projekt standartmäßig beigefügt
- Ansonsten **'src/data/make_dataset.py'** aufrufen

## Klassifikator

#### Alle Standorte
- Ausführen von **'src/classificator/train_classificator.py'**
- Labels der Standorte sind **'src/data/settings.py'** definiert

#### Paarweise
- Ausführen von **'src/classificator/train_paarweise.py'**
- Labels (0) und (1) werden automatisch zugeordnet und können anhand der Namensgebung des Speicherorts abgelesen werden
  - z.B. 'models/classificator/{locationA_locationB}' ->  locationA = 0, locationB = 1
  
## Synthetisierung
- Während dem Training der Modelle wird nach jedem 'sample_interval' eine Evaluierung vorgenommen, dazu wird ein 
vortrainerter Klassifikator benötigt. Deswegen als allererstes  **'src/classificator/train_classificator.py'** ausführen
  
#### Training und Evaluierung

###### 1.  Training:
  - Aufruf von **'train.py'** mit entsprechenden Konsolenparametern 
    - **location** (Auswahl der Trainingsdaten: 'Koethen', 'Le_Havre', 'Madrid', 'München')
    - **season_decomp** (Zeitreihenzerlegung: True / False)
    - **sample_interval** (Anzahl Durchläufe nach denen Plots erstellt, Gewichte gespeichert und eine Evaluierungsklassifikation gemacht werden)
    - **epochs** (Maximal Anzahl an Trainingsdurchläufe)
  - Das Model, Plots, Gewichte.. werden im entsprechenden **'models/...'** Pfad ablegegt

###### 2. Erstellen von synthetischen Datensätzen:
  - Aufruf von **'make_synth_data.py'** mit entsprechenden Konsolenparametern
    - **location** (...)
    - **season_decomp** (...)
    - **weights** (Für welche Gewichte soll ein Datensatz erstellt werden)
  - Der Datensatz wird im entsprechenden **'data/synth/{time_gan/vae_gan}/{location}/{weights}.csv'** Pfad ablegegt
  
###### 3. Evaluierung:
  - Aufruf der Skripte in **'/evaluate'** mit entsprechenden Konsolenparametern
    - **location**
    - **season_decomp**
    - **gan** (Mit welchem GAN wurde der synthetische Datensatz erstellt: 'vae_gan' / 'time_gan')
    - **save_file_dir** (Gibt an wo die Evaluierung gespeichert werden soll)
    - **file_name** (Der synthetische Datensatz wird anhand der oben aufgeführten Konsolenparametern geladen. 
      Jedoch können auch mehrere synthetische Datensätze für einen Standort erstellt werden, z.B. ausprobieren von unterschiedlichen
      Gewichten. Um den benötigten synthetischen Datensätze direkt zu laden wird dieser Parameter benötigt. Dazu muss der absolute Pfad
      übergeben werden. )
      
- **train_synth_test_real.py**: Diese Form der Evaluierung benötigt für jeden Standort einen synthetischen Standort, da eine
Klassifikation auf allen Standorten durchgeführt wird.
  
## Anonymisierung
- Auch hier wird der Klassifikator benötigt: -> **'src/classificator/train_classificator.py'** 
#### Training und Evaluierung
###### 1.  Manuelle Anonymisierung:
- Zunächst **'manuelle_ano/create_scaler.py'** ausführen, so dass Skalerobjekte erstellt werden
- Anschließend **'manuelle_ano/anonymisiere_testjahr.py'** ausführen um anonymisierte Datensätze zu erstellen

###### 1.  Umkodierer:
- Zunächst Hyperparametersuche starten mit  **'umencoder/hyperparametersuche.py'**
    - Dadurch werden logfiles erstellt und die Parameter der besten Architektur gespeichert, dieser Prozess kann frühzeitig beendet werden
    
- Danach kann **'umencoder/train.py'** aufgerufen werden
    - Erstellt Model und Datensatz
    
###### 1.  VAE-GAN:
- Wie bei Synthetisierung nur mit zusätzlichen Konsolenparameter:
    - **norm_location** (Zieldaten, auf welchen Standort sollen die Daten normiert werden, default; 'Koethen')
    
###### 2. Evaluierung:
- Wie bei Synthetisierung


## Projekt Aufbau
````
├───data:                               <- Hier werden alle Daten abgelegt
│   ├───ano:                            <- Speichert Daten der anonymisierung - zusätzlich nach Modellen aufgeteilt     
│   ├───processed:                      <- Vorverarbeitete Daten
│   │   ├───scaled:                     <- Normalisierte Daten 
│   │   │   └───season_decomposed:      <- Zusätzlich zeitreihenzerlegt
│   │   └───season_decomposed:          <- Zeitreihenzerlegt ohne Normalisierung
│   ├───raw: Original Daten
│   │   ├───Koethen
│   │   ├───Le_Havre
│   │   ├───Madrid
│   │   └───München
│   ├───synth:                          <- Synthetische Daten - zusätzlich nach Modellen aufgeteilt
│
│
├───models:                             <- Hier werden alle trainierten Modelle abelegt
│   ├───ano
│   │   └───vae_gan
│   │   └───Umkodierer
│   ├───classificator
│   ├───synthetize
│       ├───time_gan
│       └───vae_gan
│
├───src:                                <- Hier befindet sich der Quellecode
    ├───anonymize:                      <- Alle Skripte die für die Anonymisierung benötigt werden
    │   ├───manuelle_ano:               <- Manuelle Anonymisierung
    │   │   ├───evaluate:               <- Evaluierung der anonymisierten Daten
    │   │
    │   ├───umencoder: Umenkoder
    │   │   ├───evaluate: Evaluierung
    │   │
    │   ├───vae_gan_ano: VAE-GAN
    │       ├───evaluate: Evaluierung
    │
    ├───classificator:                  <- Klassifikator Skripte
    │
    ├───data:                           <- Data Handling (Preprocessing und Laden)
    │
    ├───synthetize:                     <- Alle Skripte die für die Synthetisierung benötigt werden
    │   ├───evaluate:                   <- Evaluierung
    │   │
    │   ├───time_gan:                   <- Skripte für das TimeGAN Modell
    │   │
    │   ├───vae_gan:                    <- Skripte für das VAE-GAN Modell
    │
    ├───vis:                            <- Skripte für ein paar Visualisierungen


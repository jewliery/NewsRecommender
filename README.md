# Twitter News Recommender
> Bachelor Projekt von Julienne Büchele.
> Es wurde ein News Recommender-System basierend auf der Twitter-API mit scikit learn umgesetzt.

### Information über das Project
* Die verwendete IDE ist PyCharm
* Es wurde Anaconda für das Package Management und Deployment verwendet
* Sonstige Packages:
  * Tweepy (Twitter-API Helper)
  * Yellowbrick (Visualizer)

### Ordner Struktur
* [src/data](./src/data): Schnittstelle zur Twitter-API sowie Daten Vor- und Aufbereitung
* [src/helper](./src/helper):
  * Hier wird der Klassifizierer trainiert ([Modeling.py](./src/helper/Modeling.py))
  * Hier werden die Methoden umgesetzt ([Modeling.py](./src/helper/Modeling.py) und [Recommender.py](./src/helper/Recommender.py))
* [src/models](./src/models): Tweet und User Objekte

_______
### Teste durchführen
* Um die Methoden zu starten, muss die Datei **[treiber.py](./src/treiber.py)** geöffnet werden
  * Der Teil unter "Get data" muss dabei stets einkommentiert sein
* Unter **"Visualization"** können die verwendeten Twitter-Beiträge visualisiert werden, dazu einfach den Code einkommentieren
* Unter **"Evaluation"** können die verschiedenen Methoden ausgewertet werden. Dafür muss jeweils der Code unter der beschrieben Methode einkommentiert werden
  * Unter "Test different classifier" können können die verschiedenen Klassifizierungsalgorithmen getestet werden
  * Unter "Plain Recommender" wird das klassische Empfehlungssystem getestet
  * Unter "Bounded-Greedy-Selection", "User Profile Partitioning" und "Anomalies and exceptions" können die einzelnen Methoden getestet und die Werte dargestellt werden
  * Unter "Test every method" können alle Methoden auf einmal ausgeführt werden. Dies könnte etwas länger dauern

### Extras
* Man kann nicht zu viele Anfragen auf einmal machen, sonst überschreitet man die Rate Limits der Twitter-API
  * Falls dass passiert reicht es maximal 10min zu warten, dann müsste die Anfrage wieder durchgehen


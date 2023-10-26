# Ansätze um Kurvenverlauf zu erkennen

1. Neuronales Netz
2. (Dynamisches) Thresholding -> Linien erkennen, die das Ende des unteren Bildausschnitt berühren -> längste Linie als Schienen nehmen
    - Durchschnittswinkel berechnen (Pythagoras) -> **Problem:** Bei langen Linien ist es ein großer Winkel, obwohl die Strecke gerade ist (3D-Perspektive), z.B. bei `labeled_images/milestones/JPEGImages/1692968436-721.jpg` -> mögliche Lösung ist "Verzerrungsfaktor" abhängig von der Länge einzuführen, dass Linien staucht um den Effekt auszugleichen
    - Kantenerkennung der kompletten Gleise (zentraler Ausschnitt des Bildes -> Konturenerkennung -> Match von Konturen, die die unteren 5 % des Bildausschnittes berühren), anschließend Rate-of-Change (Änderungsrate) berechnen -> **Problem:** Unsaubere Erkennung von längeren Schienen als eine Schiene
    - Trapezform *unmittelbar vor dem Zug* ermitteln (Schienenabstand bekannt: 1435 mm), Winkel entspricht Streckenverlauf -> **Vorteil:** Erkennung von Weichen einfacher möglich (mehr Kanten im Bild)

## Videodaten

- Verschiedene Zugbeeinflussungssysteme: PZB/LZB/ETCS
- Verschiedene Witterungsverhältnisse
- Verschiedene Höhenprofile (bergig, Tal)
- Mehrere Gleise vs. kleine Nebenbahn

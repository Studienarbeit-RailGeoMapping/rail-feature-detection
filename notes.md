# Ansätze um Kurvenverlauf zu erkennen

1. Neuronales Netz
2. (Dynamisches) Thresholding -> Linien erkennen, die das Ende des unteren Bildausschnitt berühren -> längste Linie als Schiene nehmen
    - Durchschnittswinkel berechnen (Pythagoras) -> **Problem:** Bei langen Linien ist es ein großer Winkel, obwohl die Strecke gerade ist (3D-Perspektive), z.B. bei `labeled_images/milestones/JPEGImages/1692968436-721.jpg` -> mögliche Lösung ist "Verzerrungsfaktor" abhängig von der Länge einzuführen, dass Linien staucht um den Effekt auszugleichen
    - Rate-of-Change (Änderungsrate) berechnen

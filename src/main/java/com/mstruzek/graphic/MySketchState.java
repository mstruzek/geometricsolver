package com.mstruzek.graphic;

/**
 * Typ wyliczeniowy odpowiedzialny za to w jakim stanie
 * aktualnie znajduje sie szkicownik , dzieki niemu
 * wiemy jak obslugiwac zdarzenia pochodzace od myszki i
 * klawiatury.
 *
 * @author root
 */
public enum MySketchState {

    /**
     * tryb w kotrym mozezemy zmieniac polozenie punktow, oraza nadawac wiezy
     */
    Normal,
    /**
     * tryb odczyt z pliku
     */
    ReadModel,
    /**
     * tryb zapisu do pliku
     */
    WriteMode,
    /**
     * tryb w kotrym rysujemy linie
     */
    DrawLine,
    /**
     * tryb rysowania okregu
     */
    DrawCircle,
    /**
     * tryb rysowania luku
     */
    DrawArc,
    /**
     * tryb rysowania punktow
     */
    DrawPoint;

    /**
     * Powrot do poczatkowego stanu
     */
    public MySketchState exitState() {
        return Normal;
    }
}

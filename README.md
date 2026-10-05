# Typing Dynamics Collector (extensión Chrome, Manifest V3)

## Instalar
1. Descomprime la carpeta.
2. Ve a `chrome://extensions`, activa **Modo desarrollador**.
3. **Cargar descomprimida** → selecciona la carpeta `typing-collector`.
4. Abre el popup, pon `participant_id` y `session_id`, revisa los sitios y pulsa **Iniciar grabación**.

## Cómo funciona
- Solo registra con la grabación activada, en los sitios de la lista y dentro de campos de texto editables.
- Nunca registra `<input type="password">` (ni campos `autocomplete` de contraseña / tarjeta).
- Indicador visible: pastilla roja "REC" en la página + badge "REC" en el icono.
- `down`/`up` emparejados por `press_id`; `repeat` ignorado; teclas solapadas OK (cada tecla física tiene su propio press_id).
- Timestamp: `performance.timeOrigin + event.timeStamp` (ms con decimales).
- Una muestra se cierra al pulsar Enter (sin Shift) o el botón de enviar del chat (mín. 3 pulsaciones). También con "Cerrar muestra".
- Pegar, soltar texto o autocompletado (`insertReplacementText`) marcan la muestra en `flagged_samples_*.csv`.
- Datos 100% locales (`chrome.storage.local`). Exportar → CSV; Eliminar → borra todo.

## CSV
`participant_id,session_id,sample_id,event,key_code,timestamp_ms,press_id,key`

## Ejemplo de análisis
```python
import pandas as pd
df = pd.read_csv("keystrokes.csv")
p = df.pivot_table(index=["participant_id","session_id","sample_id","press_id"],
                   columns="event", values="timestamp_ms")
p["hold_time"] = p["up"] - p["down"]
```

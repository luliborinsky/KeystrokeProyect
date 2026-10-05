import sys
import pandas as pd

# Mapping from numeric keyCode / character to physical KeyboardEvent.code
KEY_CODE_MAP = {
    8: "Backspace",
    9: "Tab",
    13: "Enter",
    16: "ShiftLeft",  # Assuming left shift for simplicity
    17: "ControlLeft",
    18: "AltLeft",
    20: "CapsLock",
    27: "Escape",
    32: "Space",
    48: "Digit0", 49: "Digit1", 50: "Digit2", 51: "Digit3", 52: "Digit4",
    53: "Digit5", 54: "Digit6", 55: "Digit7", 56: "Digit8", 57: "Digit9",
    65: "KeyA", 66: "KeyB", 67: "KeyC", 68: "KeyD", 69: "KeyE", 70: "KeyF",
    71: "KeyG", 72: "KeyH", 73: "KeyI", 74: "KeyJ", 75: "KeyK", 76: "KeyL",
    77: "KeyM", 78: "KeyN", 79: "KeyO", 80: "KeyP", 81: "KeyQ", 82: "KeyR",
    83: "KeyS", 84: "KeyT", 85: "KeyU", 86: "KeyV", 87: "KeyW", 88: "KeyX",
    89: "KeyY", 90: "KeyZ",
    91: "MetaLeft",
    188: "Comma",
    190: "Period",
    191: "Slash",
    219: "BracketLeft",
    221: "BracketRight"
}

def fix_csv(input_path, output_path):
    print(f"Leyendo {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error leyendo el CSV: {e}")
        return

    # Create a new key_code column
    fixed_codes = []
    for idx, row in df.iterrows():
        k = row['key']
        code = row['key_code']
        
        # If the code is already a string like "KeyA" (from the fixed extension code)
        if isinstance(code, str) and not code.isdigit():
            fixed_codes.append(code)
            continue
            
        # If it's a number (the bug from the old extension code)
        try:
            num_code = int(code)
            if num_code in KEY_CODE_MAP:
                fixed_codes.append(KEY_CODE_MAP[num_code])
            else:
                # Fallback to key char if unknown code
                fixed_codes.append(f"Key{str(k).upper()}")
        except:
            fixed_codes.append("Unidentified")

    df['key_code'] = fixed_codes
    
    df.to_csv(output_path, index=False)
    print(f"¡Listo! Archivo arreglado guardado en {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python fix_csv.py <input.csv> <output.csv>")
        sys.exit(1)
    fix_csv(sys.argv[1], sys.argv[2])

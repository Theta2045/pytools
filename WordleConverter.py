GREEN = "🟩"
YELLOW = "🟨"
WHITE = "⬜"

MAP = {
    GREEN: ":CerberWHEYYY:",
    YELLOW: ":CerberLoading1:",
    WHITE: ":cerbyHuh:",
}

def is_grid_line(line: str) -> bool:
    return any(ch in line for ch in MAP)

def convert_grid_line(line: str) -> str:
    return "".join(MAP[ch] for ch in line if ch in MAP)

def run_once():
    print("Wheyyy Converter")
    print("Paste Wordle. Type RUN on a new line, then hit Enter.")
    print("Type QUIT on a new line to exit.\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            # If stdin closes, just treat it like quitting
            return False

        cmd = line.strip()
        if cmd == "RUN":
            break
        if cmd == "QUIT":
            return False

        lines.append(line)

    print("\nConverted output:\n")

    for line in lines:
        if is_grid_line(line):
            print(convert_grid_line(line))
        else:
            print(line)

    print("\n--- Done. Press Enter to convert another, or type QUIT then Enter. ---")
    try:
        again = input().strip()
    except EOFError:
        return False

    return again != "QUIT"

def main():
    while True:
        keep_going = run_once()
        if not keep_going:
            break

if __name__ == "__main__":
    main()

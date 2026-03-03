import argparse

import xlwings as xw


def read_excel(path: str, password: str | None = None) -> None:
    app = xw.App(visible=False)
    wb = None
    try:
        wb = app.books.open(path, password=password, read_only=True)

        print("Sheet names:")
        for sheet in wb.sheets:
            print(f"- {sheet.name}")

        print("\n10th row contents:")
        for sheet in wb.sheets:
            used = sheet.used_range
            if used is None or used.last_cell.row < 10:
                print(f"{sheet.name}: <no data in row 10>")
                continue

            last_col = used.last_cell.column
            row_values = sheet.range((10, 1), (10, last_col)).value
            print(f"{sheet.name}: {row_values}")

    finally:
        if wb is not None:
            wb.close()
        app.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read sheet names and 10th-row contents from an Excel file.")
    parser.add_argument("file", help="Path to the Excel file")
    parser.add_argument("--password", help="Password for protected/DRM-like encrypted file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    read_excel(args.file, args.password)


if __name__ == "__main__":
    main()

import msoffcrypto
import io
import os

def decrypt_excel(input_path, password):
    """
    Decrypts an Excel file and returns a BytesIO object of the decrypted content.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    decrypted_workbook = io.BytesIO()
    with open(input_path, "rb") as file:
        office_file = msoffcrypto.OfficeFile(file)
        try:
            office_file.load_key(password=password)
            office_file.decrypt(decrypted_workbook)
        except Exception as e:
            raise Exception(f"Failed to decrypt {input_path}: {str(e)}")

    decrypted_workbook.seek(0)
    return decrypted_workbook

if __name__ == "__main__":
    # Example usage (can be tested if a file is provided)
    pass

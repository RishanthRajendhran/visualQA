import logging
from pathlib import Path
from os.path import exists

def checkFile(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            logging.critical(f"{fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        logging.critical(f"{fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        logging.critical(f"{fileName} is not a file!")
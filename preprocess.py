import pandas as pd

def preprocess(file_path):
    path = './Dataset/FAQ/FAQparent_education1-297.xls'
    df = pd.read_excel(path, header=None)
    
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import streamlit as st

# pdf libraries
import fitz


@st.cache(allow_output_mutation=False, show_spinner=False)
def unfold_column(
    df,
    colum_to_unfold="paragraph",
    columns_to_keep=[],
    include_info=True
):
    """
    paragraph from list to row in a dataframe
    """

    paragraph_l = []
    for i, p_l in enumerate(df[colum_to_unfold]):
        for j, p in enumerate(p_l):
            # columns to keep and a paragraph each time
            data_dict = {}
            if len(columns_to_keep) > 0:
                for c in columns_to_keep:
                    data_dict[c] = df[c][i]
            data_dict[colum_to_unfold] = p
            if include_info:
                data_dict["tot_p"] = len(p_l)
                data_dict["p_nr"] = j+1
            # construct data frame
            df_p_to_row = pd.DataFrame.from_dict(data_dict, orient='index').T
            paragraph_l.append(df_p_to_row)
    return pd.concat(paragraph_l).reset_index(drop=True)


def extract_content_PyMuPDF(filename, directory):
    """
    extract pdf content using PyMuPDF
    """
    store_text = list()
    try:
        with fitz.open(directory + filename) as doc:
            try:
                for i, page in enumerate(doc):
                    try:
                        store_text.append(page.getText())
                    except ValueError:
                        store_text.append(" ")
            except ValueError:
                return np.nan
    except ValueError:
        return np.nan
    return store_text


def download_reports(df, directory_reports, dir_save_file, update=True):
    """
    downloads reports and save it in the given directory
    """

    could_download = []
    could_not_download = []
    if update:
        with tqdm(total=df.shape[0]) as pbar:
            for _, row in df.iterrows():
                url = row["url"]
                file_name = (row['company'] + "-" +
                             str(int(row['year'])) + ".pdf")
                file_name = file_name.replace(" ", "")
                pdf_fname = directory_reports + file_name
                # download file and save
                try:
                    resp = requests.get(url)
                    with open(pdf_fname, 'wb') as f:
                        f.write(resp.content)
                        row["filename"] = file_name
                except ValueError("Could not download report"):
                    could_not_download.append(row)
                pbar.update(1)
                could_download.append(row)
        df_download = pd.concat(could_download, axis=1).T
        df_download.to_csv(dir_save_file)
    else:
        df_download = pd.read_csv(dir_save_file, index_col=0)
    return df_download, could_not_download

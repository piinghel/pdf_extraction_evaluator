import pandas as pd
import streamlit as st
import plotly.express as px

# own module
from module import preprocessing as preprocess


@st.cache(allow_output_mutation=True, show_spinner=False)
def extract_content(df, directory):
    """
    extract content from filenamef
    """
    # extract content
    extract_content = {}
    for f in df["filename"].values.tolist():
        try:
            out = preprocess.extract_content_PyMuPDF(
                filename=f,
                directory=directory
                )
        except ValueError:
            pass
        extract_content[f] = [out]

    # make data frame
    extract_content_df = pd.DataFrame.from_dict(
        extract_content,
        orient="index").reset_index()
    # rename columns
    extract_content_df = extract_content_df.rename(
        columns={"index": "filename",
                 0: "content"}).dropna().reset_index(drop=True)
    # merge to original dataframe based on id filename
    df_content = pd.merge(
        df,
        extract_content_df, on=['filename']
    )
    return df_content


@st.cache(allow_output_mutation=True, show_spinner=False)
def make_dataframe(df, directory):
    """
    make dataframe where each row contains the content of a page
    """

    # extract content
    df_content = extract_content(
        df,
        directory=directory
    )

    # columns to keep
    columns_to_keep = df_content.columns
    df_unfolded = preprocess.unfold_column(
        df_content,
        colum_to_unfold="content",
        columns_to_keep=columns_to_keep
        )
    return df_unfolded


def compute_stats_report(df):
    """
    compute stats for an individual report
    """

    return pd.DataFrame({
        "company": df["company"].values.tolist()[0],
        "industry": df["industry"].values.tolist()[0],
        "sector": df["sector"].values.tolist()[0],
        "year": df["year"].values.tolist()[0],
        "url": df["url"].values.tolist()[0],
        "total pages": df["tot_p"].values.tolist()[0]
    }, index=["Company information"])


def main():
    # path where the reports are stored
    DIR_REPORTS = 'data/reports/'

    st.title("Pdf extraction inspector")
    st.sidebar.title("User settings")
    nr_reports = st.sidebar.slider(
        label="Choose number of reports to look at",
        min_value=1,
        max_value=168,
        value=50
    )

    # read in dataframe containing filenames of the downloaded reports
    df = pd.read_csv("output/CRS_download_reports.csv",
                     index_col=0).head(nr_reports)
    # compute unfolded dataframe, each rows contains the content of a page
    df_unfolded = make_dataframe(df, directory=DIR_REPORTS)
    # compute number of words
    df_unfolded["nr_words"] = df_unfolded["content"].apply(
        lambda x: len(x.split())
    )
    # compute maximum number of words and rename
    df_max_words = pd.DataFrame(
        df_unfolded.groupby("filename")["nr_words"].max()
    )
    df_max_words = df_max_words.rename(
        columns={"nr_words": "max_nr_words"}
    )
    # merge the two and compute percentage of content extracted
    df_merged = pd.merge(df_unfolded, df_max_words, on=['filename'])
    df_merged["percent_extracted"] = (
        df_merged["nr_words"] / df_merged["max_nr_words"])
    # leave out first two pages and last page
    df_merged_subset = df_merged[
        df_merged["p_nr"].between(2, df_merged["tot_p"]-1)
    ]
    expander_content = st.beta_expander("Extracted content")
    with expander_content.beta_container():
        c1_content, c2_content, c3_content = st.beta_columns((1, .6, .6))
       
        # show extracted content
        filenames_unique = list(df_unfolded.filename.unique())
        chosen_filename = c1_content.selectbox(
            label="Select filename",
            options=filenames_unique
        )
        # get company statistics
        chosen_filename_df = (
            df_merged[df_merged.filename == chosen_filename])
        out = compute_stats_report(chosen_filename_df)
        c1_content.table(out.T)
        fig_c2 = px.histogram(
                round(chosen_filename_df, 3),
                x="nr_words",
                nbins=30,
                color_discrete_sequence=['indianred'],
                title="Number of words per page")
        fig_c2.update_layout(
            xaxis_title='',
            yaxis_title='Count',
            width=440
            )
        c2_content.plotly_chart(fig_c2)

        fig_c3 = px.histogram(
                round(chosen_filename_df, 3),
                x="percent_extracted",
                nbins=30,
                color_discrete_sequence=['indianred'],
                title="Percentage extracted per page")
        fig_c3.update_layout(
            xaxis_title='',
            yaxis_title='Count',
            width=440
            )
        c3_content.plotly_chart(fig_c3)

    # display extracted content
    output = {}
    for i, row in chosen_filename_df.iterrows():
        output[row["p_nr"]] = {
            "number of words": row["nr_words"],
            "percentage extracted": round(row["percent_extracted"],2),
            "content": row["content"]
        }

    with expander_content.beta_container():
        c4_content, c5_content = st.beta_columns((1, 1))
        c4_page = c4_content.selectbox(
            "Choose page nr", ["All pages 1"] + list(output.keys()))
        c5_page = c5_content.selectbox(
            "Choose page nr", ["All pages 2"] + list(output.keys()))
        if c4_page == "All pages 1":
            c4_content.write(output)
        else:
            c4_content.write(output[c4_page])

        if c5_page == "All pages 2":
            c5_content.write(output)
        else:
            c5_content.write(output[c5_page])
    # overall statistics
    expander_stats_overall = st.beta_expander("Overall Statistics")

    stats_overall_v1 = (df_merged_subset[
        df_merged_subset["percent_extracted"] != 1].groupby(
        "filename")["percent_extracted"].describe().
        describe().round(3)["mean"])

    stats_overall_v2 = (df_merged_subset[
        df_merged_subset["nr_words"] != 1].groupby(
        "filename")["nr_words"].describe().describe().round(3)["mean"])

    stats_overall_v3 = (df_merged_subset[
        df_merged_subset["max_nr_words"] != 1].groupby(
        "filename")["max_nr_words"].describe().describe().round(3)["mean"])

    stats_overall_v4 = (df_merged_subset[
        df_merged_subset["tot_p"] != 1].groupby(
        "filename")["tot_p"].describe().describe().round(3)["count"])

    stats_overall = pd.concat([
        stats_overall_v1, stats_overall_v2,
        stats_overall_v3, stats_overall_v4], axis=1)
    stats_overall.columns = [
        "percent_extracted", "nr_words",
        "max_nr_words", "total_nr_pages"]
    expander_stats_overall.dataframe(stats_overall.T)

    # statistics for each report
    expander_stats_report = st.beta_expander("Statistics by report")
    stats_report = (df_merged_subset[
        df_merged_subset["percent_extracted"] != 1].groupby(
        "url")["percent_extracted"].describe().round(3))
    expander_stats_report.dataframe(stats_report)

    # statistics for each page
    expander_stats_page = st.beta_expander("Statistics by page")
    expander_stats_page.dataframe(df_merged_subset)

    expander_stats_page.header("Figures")
    with expander_stats_page.beta_container():
        c1, c2 = st.beta_columns((1, 1.35))
        choice_yaxis = c1.selectbox(
            label="Variable y-axis",
            options=["percent_extracted", "nr_words"])
        choice_xaxis = c2.selectbox(
            label="Variable x-axis",
            options=["industry", "year"]
        )
        fig_c1 = px.histogram(
                round(df_merged_subset[
                    df_merged_subset["percent_extracted"] != 1], 3),
                x=choice_yaxis,
                nbins=50,
                color_discrete_sequence=['indianred'])
        fig_c1.update_layout(
            yaxis_title='Percent of content extracted'
            )
        c1.plotly_chart(fig_c1)
        fig_c2 = px.box(
                round(df_merged_subset[
                    df_merged_subset["percent_extracted"] != 1], 3),
                x=choice_xaxis,
                y=choice_yaxis,
                color=choice_xaxis)
        fig_c2.update_layout(
            showlegend=False,
            xaxis_title='year',
            yaxis_title='Percent of content extracted',
        )
        c2.plotly_chart(fig_c2)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()

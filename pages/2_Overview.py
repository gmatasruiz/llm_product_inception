# --- Imports ---
import streamlit as st
import plotly.express as px

from utils.utils import *
from utils.streamlit_ui.ui_components import (
    comp_batch_process_options,
    st_markdown_spacer,
    st_df_beautify_colnames,
    st_setup_logo,
)


# --- Functions ---


def display_overview_chart(
    df: pd.DataFrame, x_col: str, y_col: str, z_col: str, color: str
):
    """
    Display a 3D line chart based on the provided DataFrame.

    Parameters:
        - df (pd.DataFrame): The DataFrame containing the data to be plotted.
        - x_col (str): The column name for the x-axis values.
        - y_col (str): The column name for the y-axis values.
        - z_col (str): The column name for the z-axis values.
        - color (str): The column name for color differentiation.

    Returns:
        None
    """
    # Drop NaN values
    df = df.dropna()

    # Generate figure
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color,
        title="Overall Score per Model",
        labels=st_df_beautify_colnames(df),
        height=700,
        width=700,
        # markers=True,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    # Update layout to set axis ranges and steps
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[1, df[x_col].max()],
                tickmode="linear",
                tick0=1,
                dtick=1,
                title=x_col.replace("_", " ").title(),
                tickangle=0,
            ),
            yaxis=dict(
                range=[1, df[y_col].max()],
                tickmode="linear",
                tick0=1,
                dtick=1,
                tickangle=0,
                title=y_col.replace("_", " ").title(),
            ),
            zaxis=dict(
                range=[0, 6],
                tickmode="linear",
                tick0=0,
                dtick=1,
                tickangle=45,
                title=z_col.replace("_", " ").title(),
            ),
            camera=dict(
                eye=dict(
                    x=0,
                    y=-2,
                    z=0,
                )  # Adjust the view to be perpendicular to the y-plane and parallel to the z-plane
            ),
        )
    )

    fig.update_xaxes(automargin=True)

    st.plotly_chart(
        figure_or_data=fig,
        use_container_width=True,
    )


def display_overview_table(df: pd.DataFrame, ignore_index: bool = True):
    """
    Display the overview table based on the provided DataFrame.

    Parameters:
        - df (pd.DataFrame): The DataFrame containing the data to be displayed.
        - ignore_index (bool): Whether to hide the index column in the displayed table. Default is True.

    Returns:
        None
    """

    st_column_config = st_df_beautify_colnames(df)

    st.dataframe(
        data=df.sort_values(by=["step_number", "template_number"]).dropna(),
        use_container_width=True,
        hide_index=ignore_index,
        column_config=st_column_config,
    )


def display_overview(root_dir: str):
    """
    Display the results overview for all available steps.

    This function lets the user have an oversight of the prompting and benchmarking results.
    If requested, it calls the 'comp_batch_process_options' function to process all steps.
    After processing, them it displays the results for all steps.

    Parameters:
        - root_dir (str): The root directory.

    Returns:
        None
    """
    comp_batch_process_options(
        root_dir=root_dir, mode="batch", on_sidebar=True, spacer_n=1
    )

    st.title("📊 Overview")
    st.divider()

    # Gather all metrics together
    all_metrics_df = gather_all_metrics(
        root_dir=PROMPT_ROOT_DIR, models=LLMs, steps=AVAILABLE_STEPS
    )

    # Display results figure
    if not all_metrics_df.empty:
        st_markdown_spacer()
        tabs = st.tabs(("📈 Chart", "🗃 Table"))
        with tabs[0]:
            display_overview_chart(
                all_metrics_df,
                x_col="step_number",
                y_col="template_number",
                z_col="overall_score",
                color="llm_model",
            )

        with tabs[1]:
            display_overview_table(all_metrics_df)


# --- Main ---
# Streamlit App
def main():
    """
    Main function for the application.

    This function sets up the app configuration, defines the sidebar navigation, and calls the appropriate display functions based on the selected view.

    Parameters:
        None

    Returns:
        None
    """
    # App config
    st.set_page_config(
        page_title="Thesis on LLMs for Product Conception",
        layout="wide",
        initial_sidebar_state="auto",
        page_icon="💡",
        menu_items={
            "About": r"""

            Thesis on LLMs for Product Conception by Guillermo Matas.

            Contact me at https://www.linkedin.com/in/guillermo-matas-ruiz/

            """,
        },
    )

    # Setup logo
    st_setup_logo(REPO_ROOT_DIR)

    # Display view
    display_overview(PROMPT_ROOT_DIR)


if __name__ == "__main__":
    main()


import plotly
import numpy as np


def plot(data_map, layout_map, title):

    num_data = len(data_map)
    num_rows = int(np.floor(np.sqrt(num_data)))
    num_cols = int(np.ceil(num_data / num_rows))

    fig = plotly.subplots.make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=list(data_map),
    )

    count = 0
    for key, data in data_map.items():

        i_row = int(np.floor(count / num_cols)) + 1
        i_col = int(count % num_cols) + 1

        for d in data:
            fig.add_trace(d, row=i_row, col=i_col)
            fig.update_xaxes(title_text=layout_map[key]["xaxis"]["title"], row=i_row, col=i_col)
            fig.update_yaxes(title_text=layout_map[key]["yaxis"]["title"], row=i_row, col=i_col)

        count = count + 1

    plotly.offline.plot(fig, filename=title + ".html")
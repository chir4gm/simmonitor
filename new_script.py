
# TODO:
# Downgrade python requirements
# Add missing graphs
# Add radius tabs
# Add a simulation reporting option
# Diagnose why certain simulations aren't working
# Add 3d vector visualisations
#! /usr/bin/python3
import numpy as np
import pdb
import plotly.io as pio
from jinja2 import Template
import pandas as pd
import logging
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import bokeh.plotting as bkh_plt
import code
import bokeh
import re
import os
import textwrap
from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir
from kuibit.cactus_scalars import AllScalars
from kuibit.hor_utils import compute_separation
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    get_figname,
    plot_horizon_on_plane_at_time,
    save_from_dir_filename_ext,
    set_axis_limits_from_args,
    setup_matplotlib,
)
from kuibit.series import sample_common

args = None

bkh_doc = []
report_body = ""


def gen_fig(fig):
    logger.debug(f"Generating Figure {fig}")
    ax = plt.gca()
    lines = ax.get_lines()
    x_data = [i.get_xdata() for i in lines]
    y_data = [i.get_ydata() for i in lines]
    plt.savefig(f"{args.outdir}/{fig}.png")
    plt.savefig(f"{args.outdir}/{fig}.pdf")
    return textwrap.dedent(
        f"""
        <p>{fig}<br/><a href="{args.outdir}/{fig}.pdf"><img src="{fig}.png"/></a></p>
    """
    )


def getTerminationReason(logfile):
    with open(logfile, "r") as fh:
        for line in fh.readlines():
            if re.search(
                r"[(]TerminationTrigger[)]: Remaining wallclock time for your job is [0-9.]* minutes.  Triggering termination[.][.][.]",
                line,
            ):
                return "walltime"
            if re.search(
                r"[(]NaNChecker[)]: 'action_if_found' parameter is set to 'terminate' - scheduling graceful termination of Cactus",
                line,
            ):
                return "NaNChecker"
            if re.search(
                r"[(]TrackTriggers[)]: Triggering termination at t=.* after surfaces .* became closer than",
                line,
            ):
                return "TrackTriggers"
    return "unknown"


def populate_logs(sim):
    logger.debug("Populating Logs")
    sim_logs = sim.logfiles
    sim_logs.sort()
    logs = []
    for i in sim_logs:
        sim_num = re.search(r"output-(\d\d\d\d)", i).group(1)
        logs.append(
            [
                f'{sim_num}{"(active)" if ("active" in i) else "" }',
                getTerminationReason(i),
                os.path.abspath(i),
            ]
        )
    logs = [f"<tr><td><a href='{log[2]}'>{log[0]}</a></td><td>{log[1]}</td></tr>" for log in logs]
    logs = ["<tr><th>Simulation Run</th><th>Termination Reason</th></tr>"] + logs

    return logs

def populate_graphs(sim, args):
    logger.debug("Prepared SimDir")
    reader = sim.gravitationalwaves
    logger.debug(f"Radii available: {reader.radii}")

    plotly_html = []
    static_html = []
    sim_radii = None
    if args.radius:
        logger.debug(f"Plotting radii: {args.radius}")
        sim_radii = args.radius
    else:
        logger.debug(f"No radii set. Plotting all radii")
        sim_radii = reader.radii

    for radius in sim_radii:
        detector = reader[radius]

        psi4 = detector[args.l, args.m]
        logger.debug(f"Plotting Psi4 radius={radius}")

        if args.static:
            plt.plot(radius * psi4.real(), label=rf"$r\Re\Psi_4^{{{args.l}{args.m}}}$")
            plt.plot(radius * psi4.abs(), label=rf"$r|\Psi_4^{{{args.l}{args.m}}}|$")

            plt.legend()
            plt.xlabel("Time")
            plt.ylabel(r"$r \Psi_4$")

            add_text_to_corner(f"Det {args.detector_num}", anchor="SW", offset=0.005)
            add_text_to_corner(rf"$r = {radius:.3f}$", anchor="NE", offset=0.005)

            set_axis_limits_from_args(args)
            logger.debug("Plotted Psi_4")
            static_html.append(gen_fig(f"psi_4_r_{radius}"))
            plt.clf()

        if args.interactive:
            fig = go.Figure()
            time = psi4.t
            fig.add_trace(go.Scatter(x=time, y=radius * (psi4.real().y), mode='lines', name=rf"$r\Re\Psi_4^{{{args.l}{args.m}}}$"))
            fig.add_trace(go.Scatter(x=time, y=radius * (psi4.abs().y), mode='lines', name=rf"$r|\Psi_4^{{{args.l}{args.m}}}|$"))
            fig.update_layout(
            template=pio.templates.default,
            title=rf"$\Psi_4 r={radius}$",
            xaxis_title="Time",
            yaxis_title=r"$r \Psi_4$",
            annotations=[
                dict(
                    text=f"Det {args.detector_num}",
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1,
                    showarrow=False,
                    font=dict(size=12),
                    xanchor='left',
                    yanchor='top'
                ),
                dict(
                    text=rf"$r = {radius:.3f}$",
                    xref="paper",
                    yref="paper",
                    x=1,
                    y=0,
                    showarrow=False,
                    font=dict(size=12),
                    xanchor='right',
                    yanchor='bottom'
                )
            ])
            fig_html = pio.to_html(fig,include_plotlyjs=False, include_mathjax=False, full_html=False, auto_play=False)
            plotly_html.append(fig_html)

    logger.debug(
        f"Apparent horizons available: {sim.horizons.available_apparent_horizons}"
    )

    for ah in args.ah:
        if ah not in sim.horizons.available_apparent_horizons:
            raise ValueError(f"Apparent horizons {ah} is not available")

    # Now, we prepare the list with the centroids. The keys are the
    # horizon numbers and the values are TimeSeries. We prepare this so that
    # we can compute the center of mass. For that, we also need the masses
    # (x_cm = sum m_i / M x_i). We use the irreducible mass for this.
    ah_coords = {"x": [], "y": [], "z": []}
    masses = []
    # If force_com_at_origin, these objects will be rewritten so that we can
    # assume that they house the quantities we want to plot.

    for ah in args.ah:
        logger.debug(f"Reading horizon {ah}")

        hor = sim.horizons.get_apparent_horizon(ah)

        for coord, ah_coord in ah_coords.items():
            ah_coord.append(hor.ah[f"centroid_{coord}"])
        logger.debug("Reading mass")
        masses.append(hor.ah.m_irreducible)

    logger.debug("Computing center of mass")
    logger.debug("Resampling to common times")

    # We have to resample everything to a common time interval. This is
    # because we are going to combine the various objects with
    # mathematical operations.

    # Loop over the three coordinates and overwrite the list of
    # TimeSeries (note that ahs here are lists of TimeSeries and
    # sample_common(ahs) returns a new list of TimeSeries)
    ah_coords = {coord: sample_common(ahs) for coord, ahs in ah_coords.items()}
    masses = sample_common(masses)

    # First, we compute the total mass (as TimeSeries)
    total_mass = sum(mass for mass in masses)

    # Loop over the three coordinates
    for coord, ah_coord in ah_coords.items():
        # For each coordinate, compute the center of mass along that
        # coordinate
        com = sum(mass * ah / total_mass for mass, ah in zip(masses, ah_coords[coord]))
        # Now, we update ah_coords over that coordinate by subtracting
        # the center of mass from each apparent horizon
        ah_coord = [ah - com for ah in ah_coords[coord]]

    to_plot_x, to_plot_y = args.ah_plane
    logger.debug(f"Plotting on the x axis {to_plot_x}")
    logger.debug(f"Plotting on the y axis {to_plot_y}")

    time = None
    # Now we loop over all the horizons
    for ind, ah in enumerate(args.ah):
        if args.static:
            plt.plot(
                ah_coords[to_plot_x][ind].y,
                ah_coords[to_plot_y][ind].y,
                label=f"Horizon {ah}",
            )
        logger.debug(len(ah_coords[to_plot_x][ind].y))
        # We save the time to plot the horizon outline
        time = ah_coords[to_plot_x][ind].tmax

    if args.interactive:
        logger.debug("Plotting 3D Horizons")
        horizons = []
        for ind, ah in enumerate(args.ah):
            ah_time = ah_coords['x'][ind].t
            ah_x = ah_coords['x'][ind].to_numpy()
            ah_y = ah_coords['y'][ind].to_numpy()
            ah_z = ah_coords['z'][ind].to_numpy()
            horizons.extend([(f"Horizon {ah}",
                            ah_time[i],
                            ah_x[i],
                            ah_y[i],
                            ah_z[i]) 
                            for i in range(min([len(ah_x), len(ah_time), len(ah_y), len(ah_z)]))]
            )
        horizons = pd.DataFrame(horizons, columns = ['label','time','x','y','z'])
        # Set time-bins
        time_bins = 50
        time_bins = np.linspace(horizons['time'].min(), horizons['time'].max(), time_bins + 1)
        # Create a figure
        fig = go.Figure()

        # Group data by 'label' to handle each horizon separately
        for label, group in horizons.groupby('label'):
            # Add a trace for the lines of this horizon
            fig.add_trace(go.Scatter3d(
                x=group['x'],
                y=group['y'],
                z=group['z'],
                mode='lines',
                line=dict(width=4),
                name=label
            ))
            fig.add_trace(go.Scatter3d(
                x=[group['x'].iloc[-1]],
                y=[group['y'].iloc[-1]],
                z=[group['z'].iloc[-1]],
                mode='markers',
                marker=dict(color="red", size=10),
                name=f"{label}_markers",
                showlegend=False
            ))

        # Build frames for each time bin
        frames = []
        for frame_time in time_bins:
            frame_data = []
            # Add trajectory line and marker for each horizon
            for label, group in horizons.groupby('label'):
                horizon_frame_data = group[group['time'] <= frame_time]
                frame_data.append(
                    go.Scatter3d(
                        x=horizon_frame_data['x'],
                        y=horizon_frame_data['y'],
                        z=horizon_frame_data['z'],
                        mode='lines',
                        line=dict(width=4),
                        name=label
                ))
                frame_data.append(go.Scatter3d(
                        x=[horizon_frame_data['x'].iloc[-1]],
                        y=[horizon_frame_data['y'].iloc[-1]],
                        z=[horizon_frame_data['z'].iloc[-1]],
                        mode="markers",
                        marker=dict(color="red", size=10),
                        name=f"{label}_marker",
                        showlegend=False
                ))
            frames.append(go.Frame(
                data=frame_data,
                name=str(int(frame_time))
            ))

        fig.frames += tuple(frames)

        # Add sliders
        sliders = [{
            'steps': [
                {
                    'method': 'animate',
                    'label': str(frame.name),
                    'args': [[frame.name], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}]
                } for frame in frames
        ],
            'transition': {'duration': 300},
            'x': 0,
            'y': 0,
            'currentvalue': {
                'visible': True,
                'prefix': 'Time:',
                'xanchor': 'center'
            },
            'len': 1.0
        }]

        fig.update_layout(
            template=pio.templates.default,
            sliders=sliders,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        fig_html = pio.to_html(fig,include_mathjax=False,include_plotlyjs=False, full_html=False, auto_play=False)
        plotly_html.append(fig_html)

    if args.static:
        xlabel = f"{to_plot_x} - {to_plot_x}_CM"
        ylabel = f"{to_plot_y} - {to_plot_y}_CM"
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.gca().set_aspect("equal")

        plt.legend() 
        add_text_to_corner(rf"$t = {time:.3f}$")
        logger.debug("Saving horizons.png")
        static_html.append(gen_fig("horizons"))
        plt.clf()


    logger.debug("Plotting separation")
    h1 = sim.horizons.get_apparent_horizon(args.sep_ah[0])
    h2 = sim.horizons.get_apparent_horizon(args.sep_ah[1])

    separation = compute_separation(h1, h2, resample=True)
    if args.static:
        plt.plot(separation)
        plt.ylabel("Coordinate separation")
        plt.xlabel("Time")
        set_axis_limits_from_args(args)
        static_html.append(gen_fig("sep_ah"))
        plt.clf()

    if args.interactive:
        fig = go.Figure()
        time = separation.t
        fig.add_trace(go.Scatter(x=time, y=separation.y, mode='lines'))
        fig.update_layout(
        template=pio.templates.default,
        title="Horizon Separation",
        xaxis_title="Time",
        yaxis_title="Coordination Separation")
        fig_html = pio.to_html(fig,include_mathjax=False,include_plotlyjs=False, full_html=False, auto_play=False)
        plotly_html.append(fig_html)
    logger.debug("Plotted")

    
    logger.debug("Plotting norm_inf H")

    reader = sim.timeseries["infnorm"]
    # infnorm can be norm_inf in some simulations
    if not len(reader.keys()):
        reader = AllScalars(sim.allfiles, "norm_inf")
    # Remove some data from the beginning to only see fluctuations more clearly
    var = reader["H"].initial_time_removed(10)

    if args.static:
        plt.plot(var)
        plt.xlabel("Time")
        plt.ylabel(r"$\|H\|_{\infty}$")
        logger.debug("Plotted")
        static_html.append(gen_fig("norm_inf_H"))
        plt.clf()
    if args.interactive:
        fig = go.Figure()
        time = var.t
        fig.add_trace(go.Scatter(x=time, y=var.y, mode='lines'))
        fig.update_layout(
        template=pio.templates.default,
        title=r"$\|H\|_{\infty}$",
        xaxis_title="Time",
        yaxis_title=r"$\|H\|_{\infty}$")
        fig_html = pio.to_html(fig,include_mathjax=False,include_plotlyjs=False, full_html=False, auto_play=False)
        plotly_html.append(fig_html)
    logger.debug("Plotted")
    logger.debug("Saving")

    logger.debug("Plotting Simulation Speed")
    var = reader["H"].initial_time_removed(10)
    logger.debug("Plotting timeseries")
    if args.static:
        plt.plot(sim.ts.scalar['physical_time_per_hour'])
        plt.xlabel("Time")
        plt.ylabel("Simulation Speed [M/hr]")
        logger.debug("Plotted")
        static_html.append(gen_fig("sim_speed"))
        plt.clf()
    if args.interactive:
        fig = go.Figure()
        time = sim.ts.scalar['physical_time_per_hour'].t
        fig.add_trace(go.Scatter(x=time, y=sim.ts.scalar['physical_time_per_hour'].y, mode='lines'))
        fig.update_layout(
        template=pio.templates.default,
        title=r"Simulation Speed",
        xaxis_title="Time",
        yaxis_title="Simulation Speed [M/hr]")
        fig_html = pio.to_html(fig,include_mathjax=False,include_plotlyjs=False, full_html=False, auto_play=False)
        plotly_html.append(fig_html)
    logger.debug("Plotted")
    return plotly_html,static_html


if __name__ == "__main__":
    desc = f"""{kah.get_program_name()} generates a simulation report"""

    parser = kah.init_argparse(desc)

    parser.add_argument("--theme", type=str, default="plotly", choices=list(pio.templates.keys()), help=f"Choose a theme from:{list(pio.templates.keys())}")
    parser.add_argument("--l", "-l", type=int)
    parser.add_argument("--m", "-m", type=int)
    parser.add_argument("--radius", "-r", type=float, action="append")
    parser.add_argument("--info", "-i", action="store_true", help="Retrieve Kuibit simulation metadata")
    parser.add_argument("--static", default=True, action="store_true", help="Enable Static Report Generation")
    parser.add_argument("--interactive", default=True, action="store_true", help="Enable Interactive Report Generation (Uses Plotly)")
    parser.add_argument("--detector_num", type=int)
    parser.add_argument("--ah", type=int, action="append", help="Active Horizons")
    parser.add_argument(
        "--sep_ah", type=int, action="append", help="Separated Active Horizons"
    )
    parser.add_argument("--ah_plane", type=str, help="Active Horizon Plane")

    args = kah.get_args(parser)
    args.xmin = None
    args.xmax = None
    args.ymin = None
    args.ymax = None


    pio.templates.default = args.theme

    setup_matplotlib(
        {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}
    )

    logger = logging.getLogger(__name__)
    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)


    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
        if (args.info):
            print(sim)
            for i in sim.ts.keys():
                print(i)
            quit() 

        logs = populate_logs(sim)
        plotly_html, static_html =  populate_graphs(sim, args)

        if args.static:
            logger.debug("Generating Static HTML")
            report_file = open(f"{args.outdir}/index.html", "w")
            sim = os.path.normpath(args.datadir)
            report_file.write(
                textwrap.dedent(
            f"""
            <html>
            <head>
                <title> {sim} overview</title>
            </head>
            <body>
            <h1> {sim} overview </h1>
            <table>
            {''.join(logs)}
            </table>
            {''.join(static_html)}
            </body>
            """
            )
            )
            print(len(static_html))
        if args.interactive:
            logger.debug("Generating Interactive HTML")
            template_html = ""
            with open('template.jinja') as template:
                template = Template(template.read())
                template_html = template.render(plotly_html=plotly_html)
            with open(f"{args.outdir}/interactive.html", 'w') as output:
                output.write(template_html)
            print(len(plotly_html))
    logger.debug("DONE")
#! /usr/bin/python3
import logging
import matplotlib.pyplot as plt
import bokeh.plotting as bkh_plt
import code
import bokeh
import re
import os
import textwrap
from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir
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
logs = []

bkh_doc = []
report_body = ""


def gen_fig(fig):
    logger.debug(f"Generating Figure {fig}")
    global report_body
    global bkh_figs
    ax = plt.gca()
    lines = ax.get_lines()
    x_data = [i.get_xdata() for i in lines]
    y_data = [i.get_ydata() for i in lines]
    bkh_fig = bkh_plt.figure(title=fig, toolbar_location="left")

    bkh_fig.min_border_left = 0
    bkh_fig.min_border_right = 0
    bkh_fig.min_border_top = 0
    bkh_fig.min_border_bottom = 0
    bkh_fig.toolbar.logo = None
    bkh_fig.toolbar.active_drag = None
    bkh_fig.toolbar.active_scroll = None
    bkh_fig.toolbar.active_tap = None

    for i in range(len(lines)):
        bkh_fig.line(
            x_data[i],
            y_data[i],
            line_color=bokeh.palettes.Category10_10[i % 10],
            line_width=2,
        )

    bkh_doc.append(bkh_fig)
    plt.savefig(f"{args.outdir}/{fig}.png")
    plt.savefig(f"{args.outdir}/{fig}.pdf")
    report_body += textwrap.dedent(
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
    for i in sim_logs:
        sim_num = re.search(r"output-(\d\d\d\d)", i).group(1)
        logs.append(
            [
                f'{sim_num}{"(active)" if ("active" in i) else "" }',
                getTerminationReason(i),
                i,
            ]
        )


def populate_graphs(sim):
    global args
    logger.debug("Prepared SimDir")

    reader = sim.gravitationalwaves

    #    radius = args.radius
    #    logger.debug(f"Using radius: {radius}")
    logger.debug(f"Radii available: {reader.radii}")
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

        plt.plot(radius * psi4.real(), label=rf"$r\Re\Psi_4^{{{args.l}{args.m}}}$")
        plt.plot(radius * psi4.abs(), label=rf"$r|\Psi_4^{{{args.l}{args.m}}}|$")

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(r"$r \Psi_4$")

        add_text_to_corner(f"Det {args.detector_num}", anchor="SW", offset=0.005)
        add_text_to_corner(rf"$r = {radius:.3f}$", anchor="NE", offset=0.005)

        set_axis_limits_from_args(args)
        logger.debug("Plotted Psi_4")
        gen_fig(f"psi_4_r_{radius}")
        plt.clf()

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

    # Now we loop over all the horizons
    for ind, ah in enumerate(args.ah):
        plt.plot(
            ah_coords[to_plot_x][ind].y,
            ah_coords[to_plot_y][ind].y,
            label=f"Horizon {ah}",
        )
        # We save the time to plot the horizon outline
        time = ah_coords[to_plot_x][ind].tmax

        # Try to draw the shape of the horizon

    xlabel = f"{to_plot_x} - {to_plot_x}_CM"
    ylabel = f"{to_plot_y} - {to_plot_y}_CM"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.gca().set_aspect("equal")

    plt.legend()
    add_text_to_corner(rf"$t = {time:.3f}$")

    logger.debug("Saving horizons.png")
    gen_fig("horizons")
    plt.clf()

    logger.debug("Plotting separation")
    h1 = sim.horizons.get_apparent_horizon(args.sep_ah[0])
    h2 = sim.horizons.get_apparent_horizon(args.sep_ah[1])

    separation = compute_separation(h1, h2, resample=True)

    plt.plot(separation)
    plt.ylabel("Coordinate separation")
    plt.xlabel("Time")
    set_axis_limits_from_args(args)
    logger.debug("Plotted")

    logger.debug("Saving")
    gen_fig("sep_ah")
    plt.clf()

    logger.debug("Plotting norm_inf H")
    reader = sim.timeseries["infnorm"]
    if len(reader.keys()):
        var = reader["H"]
        logger.debug("Plotting timeseries")
        plt.plot(var)
        plt.xlabel("Time")
        plt.ylabel(r"$\|H\|_{\infty}$")
        logger.debug("Plotted")
        gen_fig("norm_inf_H")
        plt.clf()

    logger.debug("Saving")


if __name__ == "__main__":
    desc = f"""{kah.get_program_name()} generates a simulation report"""

    parser = kah.init_argparse(desc)
    parser.add_argument("--dark", action="store_true")
    parser.add_argument("--l", "-l", type=int)
    parser.add_argument("--m", "-m", type=int)
    parser.add_argument("--radius", "-r", type=int, action="append")
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

    setup_matplotlib(
        {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}
    )
    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)
    bkh_plt.output_file(f"{args.outdir}/interactive.html")

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
        populate_logs(sim)
        populate_graphs(sim)
    logger.debug("Generating HTML")
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
    {report_body}
    """
        )
    )

    doc = bokeh.layouts.column(
        *bkh_doc,
        sizing_mode="scale_both",
        resizable=True,
        spacing=10,
        styles={"display": "block", "margin": "auto", "width": "60%"},
    )

    if args.dark:
        bkh_plt.curdoc().theme = "dark_minimal"
    else:
        bkh_plt.curdoc().theme = "light_minimal"

    bkh_plt.save(doc, title="Simmonitor")
    logger.debug("DONE")

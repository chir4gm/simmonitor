#!/usr/bin/env python3
import logging
import configparser
import matplotlib.pyplot as plt
import re
import os
import textwrap
from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir
from kuibit.cactus_scalars import AllScalars
from kuibit.hor_utils import compute_separation
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    set_axis_limits_from_args,
    setup_matplotlib,
)
from kuibit.series import sample_common

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
            if re.search(r"[(]Carpet[)]: Terminating due to cctk_final_time at t =", line):
                return "FinalTime"
    return "unknown"

def getFailureReason(errfile):
    with open(errfile, "r") as fh:
        for line in fh.readlines():
            if re.search(r"slurmstepd: error: [*][*][*] JOB .* CANCELLED AT .* DUE TO TIME LIMIT", line):
                return "walltime exceeded"
            if re.search(r"Could not open simulation log file.*Permission denied", line):
                return "cannot write log file: copied run?"
            if re.search(r"CCTKi_SetParameter: Error at line", line):
                return "Parameter file error"
            if re.search(r"WARNING level 0", line):
                return "CCTK_ERROR called"
            if re.search(r"tasks .*: Exited with exit code [1-9]", line):
                return "Cactus failed" # TODO: parse Cactus error message
    propfile = f"{os.path.dirname(errfile)}/SIMFACTORY/properties.ini"
    properties = configparser.ConfigParser()
    if not properties.read(propfile):
        logger.debug("Error reading .par file: {propfile}")

    if not "jobid" in properties["properties"]:
        return "jobid missing: next segment will fail"

    return False

def populate_logs(sim):
    logger.debug("Populating Logs")
    logs = []
    for sim_log in sorted(sim.logfiles):
        segment_number = re.search(r"output-(\d\d\d\d)", sim_log).group(1)
        err_file, _ = os.path.splitext(sim_log)
        err_file = f"{err_file}.err"
        log_row = [
                f'{segment_number}{"(active)" if ("active" in sim_log) else "" }',
                getTerminationReason(sim_log),
                os.path.abspath(sim_log),
            ]
        if os.path.exists(err_file):
            failure = getFailureReason(err_file)
            if failure == False:
                log_row.append("")
                log_row = f"<tr><td><a href='{log_row[2]}'>{log_row[0]}</a></td><td>{log_row[1]}</td><td>{log_row[3]}</td></tr>"
            else:
                log_row.append(f"<a href='{err_file}'>{failure}</a>")
                log_row = f"<span style='background-color:red'><tr><td><a href='{log_row[2]}'>{log_row[0]}</a></td><td>{log_row[1]}</td><td>{log_row[3]}</td></tr></span>"
        logs.append(log_row)
    logs = ["<tr><th>Simulation Run</th><th>Termination Reason</th><th>Error</th></tr>"] + logs
    return logs

def populate_graphs(sim, args):
    logger.debug("Prepared SimDir")
    reader = sim.gravitationalwaves
    logger.debug(f"Radii available: {reader.radii}")

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

        psi4 = detector[args.el, args.em]
        logger.debug(f"Plotting Psi4 radius={radius}")

        plt.plot(radius * psi4.real(), label=rf"$r\Re\Psi_4^{{{args.el}{args.em}}}$")
        plt.plot(radius * psi4.abs(), label=rf"$r|\Psi_4^{{{args.el}{args.em}}}|$")

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(r"$r \Psi_4$")

        add_text_to_corner(f"Det {args.detector_num}", anchor="SW", offset=0.005)
        add_text_to_corner(rf"$r = {radius:.3f}$", anchor="NE", offset=0.005)

        set_axis_limits_from_args(args)
        logger.debug("Plotted Psi_4")
        static_html.append(gen_fig(f"psi_4_r_{radius}"))
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

    time = None
    # Now we loop over all the horizons
    for ind, ah in enumerate(args.ah):
        plt.plot(
            ah_coords[to_plot_x][ind].y,
            ah_coords[to_plot_y][ind].y,
            label=f"Horizon {ah}",
        )
        # We save the time to plot the horizon outline
        time = ah_coords[to_plot_x][ind].tmax

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
    plt.plot(separation)
    plt.ylabel("Coordinate separation")
    plt.xlabel("Time")
    set_axis_limits_from_args(args)
    static_html.append(gen_fig("sep_ah"))
    plt.clf()
    logger.debug("Plotted")

    
    logger.debug("Plotting norm_inf H")

    reader = sim.timeseries["infnorm"]
    # infnorm can be norm_inf in some simulations
    if not len(reader.keys()):
        reader = AllScalars(sim.allfiles, "norm_inf")
    # Remove some data from the beginning to only see fluctuations more clearly
    var = reader["H"].initial_time_removed(10)

    plt.plot(var)
    plt.xlabel("Time")
    plt.ylabel(r"$\|H\|_{\infty}$")
    logger.debug("Plotted")
    static_html.append(gen_fig("norm_inf_H"))
    plt.clf()
    logger.debug("Plotted")
    logger.debug("Saving")

    logger.debug("Plotting Simulation Speed")
    var = reader["H"].initial_time_removed(10)
    logger.debug("Plotting timeseries")

    plt.plot(sim.ts.scalar['physical_time_per_hour'])
    plt.xlabel("Time")
    plt.ylabel("Simulation Speed [M/hr]")
    logger.debug("Plotted")
    static_html.append(gen_fig("sim_speed"))
    plt.clf()
    logger.debug("Plotted")
    return static_html


if __name__ == "__main__":
    desc = f"""{kah.get_program_name()} generates a report for Einstein Toolkit Simulation. It can generate static versions using Matplotlib and HTML"""

    parser = kah.init_argparse(desc)

    parser.add_argument("--el", "-el", type=int)
    parser.add_argument("--em", "-em", type=int)
    parser.add_argument("--radius", "-r", type=float, action="append")
    parser.add_argument("--info", "-i", action="store_true", help="Retrieve Kuibit simulation metadata")
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
        static_html =  populate_graphs(sim, args)

        logger.debug("Generating Static HTML")
        report_file = open(f"{args.outdir}/index.html", "w")
        simulation = os.path.basename(args.datadir)
        report_file.write(
            textwrap.dedent(
        f"""
        <html>
        <head>
            <title> {simulation} overview</title>
        </head>
        <body>
        <h1> {simulation} overview </h1>
        <table>
        {''.join(logs)}
        </table>
        {''.join(static_html)}
        </body>
        """
        )
        )
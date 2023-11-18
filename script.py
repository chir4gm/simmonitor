#!/usr/bin/python3
import logging
import matplotlib.pyplot as plt
import re


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

config = {
    "l" : 2,
    "m" : 2,
    "radius": 100,
    'detector_num': 0,
    "ah": [1,2],
    "sep_ah": [1,2],
    "ah_plane": 'xy'
}


logs = []
def getTerminationReason(logfile):
    with open(logfile, "r") as fh:
        for line in fh.readlines():
            if re.search(r"[(]TerminationTrigger[)]: Remaining wallclock time for your job is [0-9.]* minutes.  Triggering termination[.][.][.]", line):
                return "walltime"
            if re.search(r"[(]NaNChecker[)]: 'action_if_found' parameter is set to 'terminate' - scheduling graceful termination of Cactus", line):
                return "NaNChecker"
            if re.search(r"[(]TrackTriggers[)]: Triggering termination at t=.* after surfaces .* became closer than", line):
                return "TrackTriggers"
    return "unknown"
def populate_logs(sim):
    logger.debug('Populating Logs')
    sim_logs = sim.logfiles
    sim_logs.sort()
    for i in sim_logs:
        sim_num = re.search(r"output-(\d\d\d\d)", i).group(1)
        logs.append([f'{sim_num}{"(active)" if ("active" in i) else "" }', getTerminationReason(i), i])
    logger.debug('DONE')
        

def populate_graphs(sim):
    logger.debug("Prepared SimDir")

    reader = sim.gravitationalwaves

    radius = config["radius"]
    logger.debug(f"Using radius: {radius}")
    logger.debug(f"Radii Available: {reader.radii}")
    detector = reader[radius]


    psi4 = detector[config['l'], config['m']]

    logger.debug("Plotting Psi4")

    plt.plot(
        radius * psi4.real(),
        label=rf"$r\Re\Psi_4^{{{config['l']}{config['m']}}}$"
    )
    plt.plot(
        radius * psi4.abs(),
        label=rf"$r\Re\Psi_4^{{{config['l']}{config['m']}}}$"
    )
    
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(r"$r \Psi_4$")

    add_text_to_corner(
        f"Det {config['detector_num']}", anchor="SW", offset=0.005
    )
    add_text_to_corner(rf"$r = {radius:.3f}$", anchor="NE", offset=0.005)

    set_axis_limits_from_args(args)
    logger.debug("Plotted Psi_4")
    plt.savefig("psi_4.png")
    plt.clf()


    logger.debug(
        f"Apparent horizons available: {sim.horizons.available_apparent_horizons}"
    )

    for ah in config['ah']:
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

    for ah in config['ah']:
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
    ah_coords = {
        coord: sample_common(ahs) for coord, ahs in ah_coords.items()
    }

    masses = sample_common(masses)

    # First, we compute the total mass (as TimeSeries)
    total_mass = sum(mass for mass in masses)

    # Loop over the three coordinates
    for coord, ah_coord in ah_coords.items():
        # For each coordinate, compute the center of mass along that
        # coordinate
        com = sum(
            mass * ah / total_mass
            for mass, ah in zip(masses, ah_coords[coord])
        )
        # Now, we update ah_coords over that coordinate by subtracting
        # the center of mass from each apparent horizon
        ah_coord = [ah - com for ah in ah_coords[coord]]

    to_plot_x, to_plot_y = config['ah_plane']
    logger.debug(f"Plotting on the x axis {to_plot_x}")
    logger.debug(f"Plotting on the y axis {to_plot_y}")

    # Now we loop over all the horizons
    for ind, ah in enumerate(config['ah']):
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
    plt.savefig("horizons.png")
    plt.clf()

    logger.debug("Plotting separation")
    h1 = sim.horizons.get_apparent_horizon(config['sep_ah'][0])
    h2 = sim.horizons.get_apparent_horizon(config['sep_ah'][1])

    separation = compute_separation(h1, h2, resample=True)

    plt.plot(separation)
    plt.ylabel("Coordinate separation")
    plt.xlabel("Time")
    set_axis_limits_from_args(args)
    logger.debug("Plotted")

    logger.debug("Saving")
    plt.savefig('sep_ah.png')
    plt.clf()

    logger.debug('Plotting norm_inf H')
    reader = sim.timeseries['infnorm']
    var = reader['H']

    logger.debug("Plotting timeseries")
    plt.plot(var)
    plt.xlabel("Time")
    plt.ylabel(r"$\|H\|_{\infty}$")
    logger.debug("Plotted")

    logger.debug("Saving")
    plt.savefig('norm_inf_H.png')
    logger.debug("DONE")

if __name__ == "__main__":
    desc = f"""{kah.get_program_name()} generates a simulation report"""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)
    kah.add_horizon_to_parser(parser)
    args = kah.get_args(parser)
    setup_matplotlib({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}'
    }, rc_par_file=args.mpl_rc_file)


    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
        populate_logs(sim)
        populate_graphs(sim)

    logger.debug("DONE")
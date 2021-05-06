import numpy as np
from scipy import stats

import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Ellipse, Rectangle
from matplotlib import patches as mpatches
from matplotlib.ticker import (
    AutoMinorLocator,
    AutoLocator,
    MultipleLocator,
    MaxNLocator,
)

from os.path import join

from fit3bf.utils import (
    bivariate_norm,
    bivariate_mean,
    bivariate_variance,
    covariance,
    extract_file_vary_lecs_info,
    find_contour_levels,
)


def setup_rc_params(presentation=False, constrained_layout=True, usetex=True):
    if presentation:
        fontsize = 11
    else:
        fontsize = 9
    black = "k"

    mpl.rcdefaults()  # Set to defaults

    # mpl.rc("text", usetex=True)
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["text.usetex"] = usetex
    # mpl.rcParams["text.latex.preview"] = True
    mpl.rcParams["font.family"] = "serif"

    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["axes.edgecolor"] = black
    # mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams["axes.labelcolor"] = black
    mpl.rcParams["axes.titlesize"] = fontsize

    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize
    mpl.rcParams["xtick.color"] = black
    mpl.rcParams["ytick.color"] = black
    # Make the ticks thin enough to not be visible at the limits of the plot (over the axes border)
    mpl.rcParams["xtick.major.width"] = mpl.rcParams["axes.linewidth"] * 0.95
    mpl.rcParams["ytick.major.width"] = mpl.rcParams["axes.linewidth"] * 0.95
    # The minor ticks are little too small, make them both bigger.
    mpl.rcParams["xtick.minor.size"] = 2.4  # Default 2.0
    mpl.rcParams["ytick.minor.size"] = 2.4
    mpl.rcParams["xtick.major.size"] = 3.9  # Default 3.5
    mpl.rcParams["ytick.major.size"] = 3.9

    ppi = 72  # points per inch
    # dpi = 150
    mpl.rcParams["figure.titlesize"] = fontsize
    mpl.rcParams["figure.dpi"] = 150  # To show up reasonably in notebooks
    mpl.rcParams["figure.constrained_layout.use"] = constrained_layout
    # 0.02 and 3 points are the defaults:
    # can be changed on a plot-by-plot basis using fig.set_constrained_layout_pads()
    mpl.rcParams["figure.constrained_layout.wspace"] = 0.0
    mpl.rcParams["figure.constrained_layout.hspace"] = 0.0
    mpl.rcParams["figure.constrained_layout.h_pad"] = 3.0 / ppi  # 3 points
    mpl.rcParams["figure.constrained_layout.w_pad"] = 3.0 / ppi

    mpl.rcParams["legend.title_fontsize"] = fontsize
    mpl.rcParams["legend.fontsize"] = fontsize
    mpl.rcParams[
        "legend.edgecolor"
    ] = "inherit"  # inherits from axes.edgecolor, to match
    mpl.rcParams["legend.facecolor"] = (
        1,
        1,
        1,
        0.6,
    )  # Set facecolor with its own alpha, so edgecolor is unaffected
    mpl.rcParams["legend.fancybox"] = True
    mpl.rcParams["legend.borderaxespad"] = 0.8
    mpl.rcParams[
        "legend.framealpha"
    ] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
    mpl.rcParams[
        "patch.linewidth"
    ] = 0.8  # This is for legend edgewidth, since it does not have its own option

    mpl.rcParams["hatch.linewidth"] = 0.5

    # bbox = 'tight' can distort the figure size when saved (that's its purpose).
    # mpl.rc('savefig', transparent=False, bbox='tight', pad_inches=0.04, dpi=350, format='png')
    mpl.rc("savefig", transparent=False, bbox=None, dpi=400, format="png")


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def darken_color(color, amount=0.5):
    """
    Darken the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> darken_color('g', 0.3)
    >> darken_color('#F034A3', 0.6)
    >> darken_color((.3,.55,.1), 0.5)
    """
    return lighten_color(color, 1.0 / amount)


def radial_student_t_integral(r_max, df, ndim):
    from scipy import integrate

    vec = np.zeros(ndim)
    vec[0] = 1
    if ndim == 1:
        dist = stats.t(df=df, loc=0, scale=1)

        def _integrand(r):
            return 2 * dist.pdf(r * vec)

    elif ndim == 2:
        dist = stats.multivariate_t(df=df, loc=np.zeros(ndim), shape=np.eye(ndim))

        def _integrand(r):
            return r * 2 * np.pi * dist.pdf(r * vec)

    else:
        raise NotImplementedError("Not implemented for ndim > 2")

    integral, err = integrate.quad(_integrand, a=0, b=r_max)
    return integral


def mass_under_student_t_at_n_sigma(n_stdv, df, ndim):
    if df <= 2:
        raise ValueError(
            "Sigmas are not defined for Student T distributions with df <= 2"
        )
    return radial_student_t_integral(n_stdv * np.sqrt(df / (df - 2)), df, ndim)


# def unmap_rho(rho_mapped):
#     return np.arctan(rho_mapped) / np.pi + 0.5
#
#
# def map_rho(rho):
#     return np.tan((rho - 0.5) * np.pi)


def unmap_rho(rho_mapped):
    return 2.0 * np.arctan(rho_mapped) / np.pi


def map_rho(rho):
    return np.tan(rho * np.pi / 2.0)


def unmap_scale(scale_mapped):
    return np.exp(scale_mapped)


def unmap_df(df_mapped):
    return np.exp(df_mapped)


def unpack_mv_t_args(x, ndim):
    num_corr = (ndim - 1) * ndim // 2
    corr_end = 1 + num_corr
    mean_end = corr_end + ndim

    df = unmap_df(x[0])
    rho_mapped = x[1:corr_end]
    rho = unmap_rho(rho_mapped)
    loc = x[corr_end:mean_end]
    log_scale = x[mean_end:]
    scale = unmap_scale(log_scale)
    scale_mat = np.eye(ndim)
    scale_mat[np.tril_indices(ndim, -1)] = rho
    scale_mat[np.triu_indices(ndim, +1)] = rho
    scale_mat = scale * scale[:, None] * scale_mat
    return df, loc, scale_mat


def mv_student_neg_logpdf_packed(x, ndim, data, corr=None, verbose=False):
    if corr is not None:
        rho_mapped = map_rho(corr)
        x = np.insert(x, 1, rho_mapped)
    df, loc, scale = unpack_mv_t_args(x, ndim)
    try:
        dist = stats.multivariate_t(loc=loc, shape=scale, df=df)
    except (np.linalg.LinAlgError, ValueError) as e:
        if verbose:
            print(e, df, loc, scale)
        return np.inf
    return -dist.logpdf(data).sum()


def optimize_mv_student_t(data, x0=None, verbose=True, fix_corr=False, **kwargs):
    from scipy import optimize

    ndim = data.shape[-1]
    corr = None
    if x0 is None:
        num_corr = (ndim - 1) * ndim // 2
        if fix_corr:
            x0 = 1 * np.ones(1 + 2 * ndim)
            mean_end = 1 + ndim
            cov_emp = np.cov(data, rowvar=False)
            stdv = np.sqrt(np.diag(cov_emp))
            corr_emp = cov_emp / (stdv * stdv[:, None])
            rho = corr_emp[np.triu_indices(ndim, k=1)]
            corr = rho

            x0[0] = np.log(2)
            x0[1:mean_end] = np.mean(data, axis=0)
            # x0[mean_end:] = np.log(np.sqrt(np.var(data, axis=0)))
            x0[mean_end:] = np.log(stdv)
        else:
            x0 = 1 * np.ones(1 + num_corr + 2 * ndim)
            corr_end = 1 + num_corr
            mean_end = corr_end + ndim
            cov_emp = np.cov(data, rowvar=False)
            print(cov_emp, np.linalg.eigvals(cov_emp))
            stdv = np.sqrt(np.diag(cov_emp))
            corr_emp = cov_emp / (stdv * stdv[:, None])
            rho = corr_emp[np.triu_indices(ndim, k=1)]
            print(rho)
            corr = None

            x0[0] = np.log(2)
            x0[1:corr_end] = map_rho(rho)
            x0[corr_end:mean_end] = np.mean(data, axis=0)
            # x0[mean_end:] = np.log(np.sqrt(np.var(data, axis=0)))
            x0[mean_end:] = np.log(stdv)
        print(x0)

    out = optimize.fmin(
        mv_student_neg_logpdf_packed, x0=x0, args=(ndim, data, corr, verbose), **kwargs
    )
    if fix_corr:
        rho_mapped = map_rho(rho)
        out = np.insert(out, 1, rho_mapped)
    df, loc, scale = unpack_mv_t_args(out, ndim)
    return df, loc, scale


def plot_gaussian_approximation(x, ax):
    mean = np.mean(x)
    stdv = np.sqrt(np.var(x))
    x_srt = np.sort(x)
    pdf = stats.norm(loc=mean, scale=stdv).pdf(x_srt)
    ax.plot(x_srt, pdf)
    return ax


def plot_student_t_approximation(x, ax):
    df, loc, scale = stats.t.fit(x)
    print(df, loc, scale)
    return plot_student_t_from_params(x, df, loc, scale, ax)


def plot_student_t_from_params(
    x, df, mean, scale, ax, fill_stdv=None, facecolor=None, alpha=None, **kwargs
):
    dist = stats.t(df, mean, scale)
    x_srt = np.sort(x)
    pdf = dist.pdf(x_srt)
    ax.plot(x_srt, pdf, **kwargs)

    if fill_stdv is not None:
        lower = mean - fill_stdv * scale * np.sqrt(df / (df - 2))
        upper = mean + fill_stdv * scale * np.sqrt(df / (df - 2))
        fill_pts = np.linspace(lower, upper, 100)
        ax.fill_between(
            fill_pts,
            np.zeros(100),
            dist.pdf(fill_pts),
            facecolor=facecolor,
            alpha=alpha,
        )
    return ax


def plot_gaussian_from_params(x, mean, scale, ax, **kwargs):
    dist = stats.norm(mean, scale)
    x_srt = np.sort(x)
    pdf = dist.pdf(x_srt)
    ax.plot(x_srt, pdf, **kwargs)
    return ax


def confidence_ellipse(
    x, y, ax, n_std=3.0, facecolor="none", show_scatter=False, **kwargs
):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean = np.array([mean_x, mean_y])
    cov = np.cov(x, y)

    if show_scatter:
        scat_color = darken_color(facecolor, 0.5)
        ax.plot(x, y, ls="", marker=".", markersize=0.6, color=scat_color)

    return confidence_ellipse_mean_cov(
        mean, cov, ax, n_std=n_std, facecolor=facecolor, **kwargs
    )


def confidence_ellipse_mean_cov(mean, cov, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean[0], mean[1])
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def confidence_ellipse_student_t_from_params(
    df, mean, scale, ax, n_std=3.0, facecolor="none", **kwargs
):
    cov = df / (df - 2.0) * scale
    return confidence_ellipse_mean_cov(
        mean=mean, cov=cov, ax=ax, n_std=n_std, facecolor=facecolor, **kwargs
    )


def plot_true_vs_approx(
    true, approx, std=None, expt=None, ax=None, log=False, text=None, **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 3.4))

    if log:
        plt.yscale("log")
        plt.xscale("log")

    if std is None:
        ax.plot(approx, true, ls="", marker="o", **kwargs)
    else:
        ax.errorbar(approx, true, 2 * std, ls="", **kwargs)

    if expt is not None:
        ax.plot([expt], [expt], ls="", marker="X", c="r")

    ax.set_aspect(1)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    all_min = np.max([xmin, ymin])
    all_max = np.max([xmax, ymax])

    min_bound = all_min - 0.1 * np.abs(all_min)
    max_bound = all_max + 0.1 * np.abs(all_max)

    from matplotlib.collections import LineCollection

    col = LineCollection(
        [np.array([[min_bound, max_bound], [min_bound, max_bound]]).T],
        colors="lightgray",
        zorder=0,
    )
    ax.add_collection(col, autolim=False)

    ax.set_xlim(all_min, all_max)
    ax.set_ylim(all_min, all_max)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_xticks())
    ax.set_yticks(ax.get_xticks(minor=True), minor=True)

    if text is not None:
        ax.text(
            0.07,
            0.93,
            text,
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="w"),
        )

    ax.set_xlim(all_min, all_max)
    ax.set_ylim(all_min, all_max)
    ax.set_xlabel("Approximate")
    ax.set_ylabel("Exact")
    return ax


def add_data_to_corner(
    axes,
    data,
    error,
    trunc_error=None,
    errorbar_color="C0",
    trunc_edgecolor="yellow",
    adjust_limits=False,
):
    n_data = len(data)
    for yi in range(n_data):
        for xi in range(yi):
            ax = axes[yi, xi]
            # Make 1-sigma experimental errors
            ax.errorbar(
                [data[xi]],
                [data[yi]],
                xerr=error[xi],
                yerr=error[yi],
                c=errorbar_color,
                zorder=11,
                label=r"$y_{\mathrm{expt}}$",
            )
            ax.plot(
                [data[xi]], [data[yi]], c=errorbar_color, zorder=11, marker="x", ls=""
            )

            if adjust_limits:
                xmin, xmax = ax.get_xlim()

            if trunc_error is not None:
                # Make 1-sigma truncation bands
                ellipse = mpatches.Ellipse(
                    [data[xi], data[yi]],
                    width=2 * trunc_error[xi],
                    height=2 * trunc_error[yi],
                    facecolor="none",
                    edgecolor=trunc_edgecolor,
                    zorder=10,
                )
                ax.add_patch(ellipse)

    for i, ax in enumerate(np.diag(axes)):
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        rect = mpatches.Rectangle(
            (data[i] - error[i], 0),
            width=2 * error[i],
            height=1,
            transform=trans,
            color=errorbar_color,
            alpha=0.3,
            zorder=0,
        )
        ax.add_patch(rect)
        ax.axvline(data[i], 0, 1, zorder=0, color=errorbar_color)

    return axes


def ellipse_corner(mean, cov, color, axes=None, labels=None, set_lims=True):
    n = len(mean)
    if axes is None:
        fig, axes = plt.subplots(n, n, sharex="col")

    for i, ax in enumerate(np.diag(axes)):
        sd_i = np.sqrt(cov[i, i])
        xx = np.linspace(mean[i] - 2.5 * sd_i, mean[i] + 2.5 * sd_i, 200)
        yy = stats.norm(mean[i], sd_i).pdf(xx)
        ax.plot(xx, yy, c=color)

    plt.draw()
    for yi in range(n):
        if yi > 0:
            ax_y_share = axes[yi, 0]
            axes[yi, yi].set_yticklabels([])
            if labels is not None:
                axes[yi, 0].set_ylabel(labels[yi])
        else:
            axes[yi, 0].set_yticklabels([])
        for xi in range(n):
            ax = axes[yi, xi]

            if yi == n - 1 and labels is not None:
                ax.set_xlabel(labels[xi])
            if xi != 0:
                ax.set_yticklabels([])

            if xi > yi:
                ax.axis("off")
                continue
            elif xi == yi:
                continue

            idxs = [xi, yi]
            confidence_ellipse_mean_cov(
                mean[idxs], cov[idxs][:, idxs], ax=ax, edgecolor=color, n_std=1
            )
            if set_lims:
                ax.set_xlim(axes[xi, xi].get_xlim())
                ax.set_ylim(axes[yi, yi].get_xlim())
    return axes


def pretty_corner(
    data,
    loc,
    scale,
    df=None,
    labels=None,
    quantiles=None,
    levels=None,
    plot_hist=True,
    plot_contour=False,
    text=None,
    title=None,
    axes=None,
    bins=30,
    smooth1d=None,
    stdv_range=2.2,
    truths=None,
    truth_color="k",
    xlabel_pad=None,
    xtick_rotation=30,
    ytick_rotation=40,
):
    n = data.shape[-1]
    if plot_hist:
        if axes is None:
            fig, axes = plt.subplots(n, n, figsize=(3.4, 3.4))
        fig = plt.gcf()
        fig = corner.corner(
            data,
            labels=labels,
            bins=bins,
            quantiles=quantiles,
            hist_bin_factor=2,
            show_titles=True,
            title_kwargs={"fontsize": 10},
            levels=levels,
            fig=fig,
            range=[0.99] * n,
            hist_kwargs={"density": True},
        )
    else:
        if axes is None:
            fig, axes = plt.subplots(n, n, figsize=(3.4, 3.4), sharex="col")
        fig = plt.gcf()
        stdv = np.sqrt(np.diag(scale))
        if df is not None:
            if df < 2:
                print("Standard deviation not defined")
            else:
                stdv = stdv * np.sqrt(df / (df - 2))
        min_range = loc - stdv_range * stdv
        max_range = loc + stdv_range * stdv

    smooth_color = "r"
    truth_lw = 1.1
    for (i, j), ax in np.ndenumerate(axes):
        if i >= j:
            ax.tick_params(axis="x", rotation=xtick_rotation)
            ax.tick_params(axis="y", rotation=ytick_rotation)
            ax.axvline(0, 0, 1, c="lightgrey", lw=1, zorder=-10)
            if i == j:
                if df is None:
                    # plot_gaussian_approximation(data[:, i], ax)
                    plot_gaussian_from_params(
                        x=data[:, i],
                        mean=loc[i],
                        scale=np.sqrt(scale[i, i]),
                        ax=ax,
                        c=smooth_color,
                    )
                else:
                    ls = None
                    if plot_hist:
                        ls = ":"
                    stdv_i = df * np.sqrt(scale[i, i]) / (df - 2)
                    plot_student_t_from_params(
                        data[:, i],
                        df,
                        loc[i],
                        np.sqrt(scale[i, i]),
                        ax=ax,
                        c=smooth_color,
                        lw=1.5,
                        ls=ls,
                        fill_stdv=1,
                        facecolor=smooth_color,
                        alpha=0.35,
                    )
                    tick_lower = loc[i] - stdv_i
                    tick_upper = loc[i] + stdv_i
                    diff = tick_upper - tick_lower
                    n_digits = int(np.round(-np.log(diff / 2) / np.log(10), 0)) + 1
                    ax.set_xticks(
                        [np.round(tick_lower, n_digits), np.round(tick_upper, n_digits)]
                    )
                if plot_contour:
                    if smooth1d is None:
                        bins_1d = int(max(1, bins))
                        # plt.hist()
                        ax.hist(
                            data[:, i],
                            bins=bins_1d,
                            range=[min_range[i], max_range[i]],
                            density=True,
                            histtype="step",
                            color="k",
                            # alpha=0.5,
                        )
                    else:
                        from scipy.ndimage import gaussian_filter

                        n_count, b = np.histogram(
                            data[:, i],
                            bins=bins,
                            range=[min_range[i], max_range[i]],
                            density=True,
                        )
                        n_count = gaussian_filter(n_count, smooth1d)
                        x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
                        y0 = np.array(list(zip(n_count, n_count))).flatten()
                        ax.plot(x0, y0, color="k", lw=1, zorder=-1)
                if not plot_hist:
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    ax.set_xlim(min_range[i], max_range[i])
                    ax.set_ylim(0, None)
                    if i == n - 1:
                        ax.set_xlabel(labels[j], labelpad=xlabel_pad)
                if truths is not None and truths[i] is not None:
                    ax.axvline(truths[i], lw=truth_lw, color=truth_color)
            if i != j:
                # ax.text(
                #     0.1, 0.1, f"{i}, {j}", transform=ax.transAxes,
                #     bbox=dict(facecolor="w", boxstyle="round")
                # )
                ax.tick_params(axis="both", which="both", right=True, top=True)
                facecolor = "none"
                alpha = None
                if not plot_hist:
                    ax.set_ylim(min_range[i], max_range[i])
                    ax.set_xlim(min_range[j], max_range[j])
                    facecolor = smooth_color
                    alpha = 0.35

                    stdv_i = df * np.sqrt(scale[i, i]) / (df - 2)
                    tick_lower = loc[i] - stdv_i
                    tick_upper = loc[i] + stdv_i
                    diff = tick_upper - tick_lower
                    n_digits = int(np.round(-np.log(diff / 2) / np.log(10), 0)) + 1
                    ax.set_yticks(
                        [np.round(tick_lower, n_digits), np.round(tick_upper, n_digits)]
                    )

                    if j != 0:
                        ax.set_yticklabels([])
                    else:
                        ax.set_ylabel(labels[i])

                    if i != n - 1:
                        # ax.text(
                        #     0.1, 0.1, f"{i}, {j}", transform=ax.transAxes,
                        #     bbox=dict(facecolor="w", boxstyle="round")
                        # )
                        # ax.set_xticklabels([])
                        pass
                    else:
                        ax.set_xlabel(labels[j], labelpad=xlabel_pad)
                    # ax.tick_params(axis='x', rotation=90)

                for stdv in [1, 2]:
                    if df is None:
                        confidence_ellipse_mean_cov(
                            # data[:, j], data[:, i],
                            mean=loc,
                            cov=scale,
                            ax=ax,
                            n_std=stdv,
                            edgecolor=smooth_color,
                            zorder=10,
                        )
                    else:
                        confidence_ellipse_student_t_from_params(
                            df=df,
                            mean=loc[[j, i]],
                            scale=scale[np.array([j, i])[:, None], [j, i]],
                            ax=ax,
                            n_std=stdv,
                            edgecolor=smooth_color,
                            facecolor=facecolor,
                            alpha=alpha,
                            zorder=2,
                        )
                ax.axhline(0, 0, 1, c="lightgrey", lw=1, zorder=-10)
                if plot_contour:
                    from corner import hist2d

                    # H, X, Y = np.histogram2d(data[:, i], data[:, j], density=True)
                    # print(H.shape, X.shape, Y.shape)
                    # ax.contour(X, Y, H, levels=levels, color='k')
                    hist2d(
                        data[:, j],
                        data[:, i],
                        bins=bins,
                        levels=levels,
                        range=[
                            [min_range[j], max_range[j]],
                            [min_range[i], max_range[i]],
                        ],
                        ax=ax,
                        plot_density=False,
                        plot_contours=True,
                        plot_datapoints=False,
                    )
                if truths is not None:
                    if truths[i] is not None and truths[j] is not None:
                        ax.plot(
                            truths[j],
                            truths[i],
                            "o",
                            fillstyle="none",
                            color=truth_color,
                            zorder=5,
                        )
                    if truths[j] is not None:
                        ax.axvline(truths[j], color=truth_color, lw=truth_lw, zorder=5)
                    if truths[i] is not None:
                        ax.axhline(truths[i], color=truth_color, lw=truth_lw, zorder=5)
        else:
            ax.set_axis_off()
            if text is not None:
                ax.text(
                    0.05,
                    0.98,
                    s=text,
                    transform=ax.transAxes,
                    ma="left",
                    va="top",
                    bbox=dict(facecolor="w", boxstyle="round"),
                )

    # if include_in_emulator:
    #     suptitle_label = r'pr($c_D, c_E \, | \, \mathbf{y}_{\rm{exp}}, \Sigma_{{nn}}, I$)'
    # else:
    #     suptitle_label = r'pr($c_D, c_E \, | \, \mathbf{y}_{\rm{exp}}, \vec{a}_{{nn}}, I$)'
    if title is not None:
        # y = 1.06
        if fig.get_constrained_layout():
            # y = None
            fig.suptitle(title, fontsize=11)
        else:
            fig.suptitle(title, y=1.06, fontsize=11)
    return fig, axes


def plot_pdf_label(obs_labels):
    title_str = ""
    for idx, label in enumerate(obs_labels):
        if idx == len(obs_labels) - 1:
            title_str += label
        else:
            title_str += label + ",\,"
    return title_str


def get_plot_settings_lec_vary_choice(filename):
    vary_choice = extract_file_vary_lecs_info(filename)
    if vary_choice == "all":
        return "#2ca02c", "solid"
    elif vary_choice == "nn-and-3bfs":
        return "#ff7f0e", "dashed"
    elif vary_choice == "3bfs":
        return "#1f77b4", (0, (3, 1, 1, 1))


def plot_single_posterior(
    xmesh,
    ymesh,
    pdf,
    show_confidence_ellipses=True,
    levels=np.array([0.68, 0.95]),
    show_moment_ellipses=False,
    show_mean=False,
    show_mode=True,
    pdf_label=None,
    xlabel=r"$c_D$",
    ylabel=r"$c_E$",
    ellipse_color="blue",
    moment_color="red",
    mean_marker="x",
    mode_marker="+",
    custom_xlims=None,
    custom_ylims=None,
):
    """
    Reads a file (of type npz) of posterior pdf calculations on a mesh
    and plots it with confidence ellipses
    """
    pdf_normed = np.copy(pdf)
    norm = bivariate_norm(xmesh, ymesh, pdf)
    pdf_normed /= norm
    mu_x, mu_y = bivariate_mean(xmesh, ymesh, pdf_normed)
    var_x, var_y = bivariate_variance(xmesh, mu_x, ymesh, mu_y, pdf_normed)
    cov_x_y = covariance(xmesh, mu_x, ymesh, mu_y, pdf_normed)
    cov = np.asarray([[var_x, cov_x_y], [cov_x_y, var_y]])

    fig, ax = plt.subplots(figsize=(4, 4))
    if show_moment_ellipses:
        confidence_ellipse_mean_cov(
            [mu_x, mu_y], cov, ax, n_std=2, edgecolor=moment_color, linewidth=1
        )
        confidence_ellipse_mean_cov(
            [mu_x, mu_y], cov, ax, n_std=1, edgecolor=moment_color, linewidth=1
        )
    if show_confidence_ellipses:
        alpha = 1
    else:
        alpha = 0
    # this sets the axis limits
    ax.contour(
        xmesh,
        ymesh,
        pdf.T,
        levels=find_contour_levels(pdf.T, levels=levels),
        colors=ellipse_color,
        origin="upper",
        alpha=alpha,
    )
    if show_mean:
        ax.plot(mu_x, mu_y, marker=mean_marker, color=moment_color)
    if show_mode:
        max_post_idx = np.unravel_index(pdf.argmax(), pdf.shape)
        mode_x, mode_y = xmesh[max_post_idx[0]], ymesh[max_post_idx[1]]
        ax.plot(mode_x, mode_y, marker=mode_marker, color=ellipse_color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(custom_xlims)
    ax.set_ylim(custom_ylims)

    if pdf_label:
        ax.text(
            0.05,
            0.95,
            pdf_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="w", boxstyle="round", alpha=1),
        )
    # plt.show()
    return fig


def plot_single_posterior_mcmc(
    samples,
    show_confidence_ellipses=True,
    show_mean=True,
    xlabel=r"$c_D$",
    ylabel=r"$c_E$",
    cbar_label=None,
    Q_label=None,
    ellipse_color="red",
    set_xlims=None,
    set_ylims=None,
    nbins=20,
):

    hist_kwargs = {"linewidth": 1.5}

    fig = corner.corner(
        samples,
        labels=[xlabel, ylabel],
        bins=nbins,
        show_titles=True,
        title_fmt="0.3f",
        plot_datapoints=True,
        levels=(0.394, 0.865),
        plot_density=False,
        quantiles=[0.025, 0.16, 0.84, 0.975],
        hist_kwargs=hist_kwargs,
    )

    ax_list = fig.axes

    if show_mean:
        ax_list[2].plot(
            np.mean(samples[:, 0]),
            np.mean(samples[:, 1]),
            "x",
            color=ellipse_color,
            markersize=10,
            linewidth=2,
        )

    if show_confidence_ellipses:
        confidence_ellipse(
            samples[:, 0],
            samples[:, 1],
            ax_list[2],
            n_std=1.0,
            edgecolor=ellipse_color,
            linewidth=2,
            zorder=10,
        )
        confidence_ellipse(
            samples[:, 0],
            samples[:, 1],
            ax_list[2],
            n_std=2.0,
            edgecolor=ellipse_color,
            linewidth=2,
            zorder=10,
        )

    if set_xlims is not None:
        ax_list[0].set_xlim(set_xlims)
        ax_list[2].set_xlim(set_xlims)
    if set_ylims is not None:
        ax_list[2].set_ylim(set_ylims)
        ax_list[3].set_xlim(set_ylims)

    return fig


def plot_ppd_mcmc(
    samples,
    labels=None,
    show_confidence_ellipses=True,
    ellipse_color="red",
    show_mean=True,
    include_truncation_error=False,
    truncation_error_info=None,
):

    if include_truncation_error is not None:
        hist_1d_alpha = 0.2
        quantiles_show = None
    else:
        hist_1d_alpha = 1.0
        quantiles_show = [0.025, 0.16, 0.84, 0.975]
    hist_kwargs = {"linewidth": 1.5, "alpha": hist_1d_alpha}

    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_fmt="0.3f",
        plot_datapoints=True,
        levels=(0.394, 0.865),
        plot_density=False,
        quantiles=quantiles_show,
        hist_kwargs=hist_kwargs,
    )

    ax_list = fig.axes
    n_obs = samples.shape[1]

    if show_mean:
        for i in range(0, n_obs):
            for j in range(0, n_obs):
                if j < i:  # lower triangle only
                    list_idx = n_obs * i + j
                    ax_list[list_idx].plot(
                        np.mean(samples[:, j]),
                        np.mean(samples[:, i]),
                        "x",
                        color=ellipse_color,
                        markersize=10,
                        linewidth=2,
                    )

    if show_confidence_ellipses:
        for i in range(0, n_obs):
            for j in range(0, n_obs):
                if j < i:  # lower triangle only
                    list_idx = n_obs * i + j
                    confidence_ellipse(
                        samples[:, j],
                        samples[:, i],
                        ax_list[list_idx],
                        n_std=1.0,
                        edgecolor=ellipse_color,
                        linewidth=2,
                        zorder=10,
                    )
                    confidence_ellipse(
                        samples[:, j],
                        samples[:, i],
                        ax_list[list_idx],
                        n_std=2.0,
                        edgecolor=ellipse_color,
                        linewidth=2,
                        zorder=10,
                    )

    if include_truncation_error:
        if truncation_error_info is None:
            cbar = 1
            Qval = 140.0 / 600.0
            korder = 3  # N2LO
            print(
                "No information about truncation error provided! Using default values of ",
                "cbar =",
                cbar,
                "; Q = ",
                Qval,
                "; order k = 3",
            )
        else:
            cbar = truncation_error_info["cbar"]
            Qval = truncation_error_info["Qval"]
            korder = truncation_error_info["korder"]

        # naive (uncorrelated) variance is cbar^2 * yref^2 * Q^{2k+2}/(1 - Q^2)
        # from geometric sum of gaussian random variables cn's
        truncation_variance = cbar ** 2.0 * Qval ** (2 * korder + 2) / (1 - Qval ** 2.0)
        for i in range(0, n_obs):
            # calculate naive estimate for each observable
            y_th = np.mean(samples[:, i])
            truncation_variance_obs = y_th ** 2.0 * truncation_variance
            fit_variance = np.var(samples[:, i])
            full_variance = truncation_variance_obs + fit_variance
            for j in range(0, n_obs):
                list_idx = n_obs * i + j
                if i == j:  # diagonal plots
                    xlo = y_th - (2 * np.sqrt(full_variance) * 1.50)
                    xhi = y_th + (2 * np.sqrt(full_variance) * 1.50)
                    ax_list[list_idx].set_xlim(xlo, xhi)
                    # set xlimits for this whole column
                    for k in range(1, n_obs - i):
                        col_idx = n_obs * (k + j) + j
                        print("setting x limits of ", col_idx)
                        ax_list[col_idx].set_xlim(xlo, xhi)
                    # set y limits for the row
                    if i > 0:
                        for k in range(0, i):
                            col_idx = n_obs * i + k
                            print("setting y limits of ", col_idx)
                            ax_list[col_idx].set_ylim(xlo, xhi)

                    ax_list[list_idx].axvline(y_th, color="green", linestyle="dashdot")
                    ax_list[list_idx].axvline(
                        y_th + np.sqrt(full_variance),
                        color="green",
                        linestyle="dashdot",
                    )
                    ax_list[list_idx].axvline(
                        y_th - np.sqrt(full_variance),
                        color="green",
                        linestyle="dashdot",
                    )
                    ax_list[list_idx].axvline(
                        y_th + 2 * np.sqrt(full_variance),
                        color="green",
                        linestyle="dashdot",
                    )
                    ax_list[list_idx].axvline(
                        y_th - 2 * np.sqrt(full_variance),
                        color="green",
                        linestyle="dashdot",
                    )

                    # plot std from fit only
                    fit_std = np.sqrt(fit_variance)
                    ax_list[list_idx].axvspan(
                        y_th - fit_std, y_th + fit_std, color="black", alpha=0.075
                    )
                    ax_list[list_idx].axvspan(
                        y_th - 2 * fit_std,
                        y_th + 2 * fit_std,
                        color="black",
                        alpha=0.15,
                    )
                if j < i:  # lower triangle only

                    confidence_ellipse(
                        samples[:, j],
                        samples[:, i],
                        ax_list[list_idx],
                        n_std=1.0,
                        edgecolor=ellipse_color,
                        linewidth=2,
                        zorder=10,
                    )
                    confidence_ellipse(
                        samples[:, j],
                        samples[:, i],
                        ax_list[list_idx],
                        n_std=2.0,
                        edgecolor=ellipse_color,
                        linewidth=2,
                        zorder=10,
                    )

    return fig


def add_values_from_ref(
    ax, ref, traj_kwargs=None, central_val_kwargs=None, box_kwargs=None
):
    """
    Add reference values, trajectories, or confidence ellipses from different
    references to a given set of axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the values on/
    ref : float
        The number of standard deviations to determine the ellipse's radiuses.
    ref_path : string
        The location of the data file that contains reference trajectory information

    Returns
    -------

    Other parameters
    ----------------
    kwargs : `~matplotlib.pyplot.plot` properties
    """
    ref_path = "data/"
    file_dict = {
        "epelbaum2019_R0p9": "Epelbaum_2019_R_0p9fm.csv",
        "epelbaum2019_R1p0": "Epelbaum_2019_R_1p0fm.csv",
        "gazit2009": "Gazit_PRL_2009_cD_cE_trajectory_3H.csv",
        "baroni2016_L500": "Baroni_Lam_500_A3_avg.csv",
        "baroni2016_L600": "Baroni_Lam_600_A3_avg.csv",
        "kravvaris2020": "Kravvaris_A3_avg_only.csv",
    }

    plot_what_dict = {
        "epelbaum2019_R0p9": ["trajectory", "box", "central-value"],
        "epelbaum2019_R1p0": ["trajectory", "box", "central-value"],
        "gazit2009": ["trajectory", "central-value"],
        "baroni2016_L500": ["trajectory", "central-value"],
        "baroni2016_L600": ["trajectory", "central-value"],
        "kravvaris2020": ["trajectory", "box", "central-value"],
    }

    if ref == "epelbaum2019_R0p9":
        cD_traj, cE_traj = np.loadtxt(
            join(ref_path, file_dict[ref]), unpack=True, delimiter=","
        )
        cD, cD_s = 1.7, 0.8
        cE, cE_s = -0.329, 0.1
    elif ref == "epelbaum2019_R1p0":
        cD_traj, cE_traj = np.loadtxt(
            join(ref_path, file_dict[ref]), unpack=True, delimiter=","
        )
        cD, cD_s = 7.2, 0.7
        cE, cE_s = -0.652, 0.07
    elif ref == "baroni2016_L500":
        cD_traj, cE_traj = np.loadtxt(
            join(ref_path, file_dict[ref]), unpack=True, delimiter=","
        )
        cD = -0.353
        cE = -0.305
    elif ref == "baroni2016_L600":
        cD_traj, cE_traj = np.loadtxt(
            join(ref_path, file_dict[ref]), unpack=True, delimiter=","
        )
        cD = -0.443
        cE = -1.224
    elif ref == "gazit2009":
        cD_traj, cE_traj = np.loadtxt(
            join(ref_path, file_dict[ref]), unpack=True, delimiter=","
        )
        cD = -0.2
        cE = -0.205
    elif ref == "kravvaris2020":
        cD_traj, cE_traj = np.loadtxt(
            join(ref_path, file_dict[ref]), unpack=True, delimiter=","
        )
        cD, cD_s = 0.925, 0.349
        cE, cE_s = -0.00806, 0.0708

    else:
        print("Sorry,", ref, "is not a valid reference choice.")

    if "trajectory" in plot_what_dict[ref]:
        if traj_kwargs is not None:
            ax.plot(cD_traj, cE_traj, **traj_kwargs)
        else:
            ax.plot(cD_traj, cE_traj)
    if "central-value" in plot_what_dict[ref]:
        if central_val_kwargs is not None:
            ax.plot(cD, cE, **central_val_kwargs)
        else:
            ax.plot(cD, cE)
    if "box" in plot_what_dict[ref]:
        if box_kwargs is not None:
            rect = Rectangle([cD - cD_s, cE - cE_s], 2 * cD_s, 2 * cE_s, **box_kwargs)
        else:
            rect = Rectangle([cD - cD_s, cE - cE_s], 2 * cD_s, 2 * cE_s)
        ax.add_patch(rect)

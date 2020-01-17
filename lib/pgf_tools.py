import ipdb
import matplotlib

#~ from matplotlib_venn import venn2
#~ from matplotlib_venn import venn3
import numpy
matplotlib.use("pgf") 
matplotlib.rcParams["mathtext.default"] = "regular"
preamble = {
    "font.family": "lmodern",  # use this font for text elements
    "text.usetex": True, # use inline math for ticks
    "pgf.rcfonts": False, # don"t setup fonts from rc parameters
    "pgf.texsystem": "xelatex",
    "pgf.preamble": [
        "\\usepackage{amsmath}",
        "\\usepackage{amssymb}",
        "\\usepackage{lmodern}",
        ]
    }
matplotlib.rcParams.update(preamble)
import pyreg.functions

def plot_global_times(*args, labels, fig_path):
    fig = matplotlib.pyplot.figure()
    
    pre = []
    coarse = []
    fine = []
    
    for arg in args:
        pre.append(numpy.array(arg["pre"]))
        coarse.append(numpy.array(arg["coarse"]))
        fine.append(numpy.array(arg["fine"]))
    
    pre = numpy.array(pre)
    coarse= numpy.array(coarse)
    fine = numpy.array(fine)
    
    ipdb.set_trace()
    
    pre = numpy.mean(pre, axis=1)
    dummy = numpy.zeros(4)
    #~ coarse= numpy.mean(coarse, axis=1)
    dummy[3] = numpy.mean(coarse[3])
    fine = numpy.mean(fine, axis=1)

    # When there is no coarse registration, some values are None
    if pyreg.functions.contains_nan(coarse) == True:
        mask = numpy.isnan(coarse)
        coarse[mask] = 0.0
    else:
        pass
    
    ind = numpy.arange(len(args))
    width = 0.25

    p3_bottom = pre + coarse
    p1 = matplotlib.pyplot.bar(ind, pre, width, label="Preprocessing")
    p2 = matplotlib.pyplot.bar(ind, coarse, width,
                 bottom=pre, label="Coarse registration")
    p3 = matplotlib.pyplot.bar(ind, fine, width,
                 bottom=p3_bottom, label="Refinement")

    matplotlib.pyplot.ylabel("time [s]")
    matplotlib.pyplot.xticks(ind, labels)
    
    matplotlib.pyplot.legend(loc="upper left")
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")
    
    return


def plot_recalls(*args, labels, fig_path):
    fig = matplotlib.pyplot.figure()
    
    matplotlib.pyplot.ylabel(r"$ \alpha $-recall", fontsize=15)
    matplotlib.pyplot.xlabel(r"RMSE $ \alpha $", fontsize=15)
    
    lines = ["-", ":", "-.", "--"]
    alphas = [0.7, 1.0, 0.5, 1.0]
    
    i = 0
    for arg in args:
        matplotlib.pyplot.plot(numpy.sort(arg), numpy.linspace(0.0, 1.0, arg.shape[0]), 
                                label=labels[i], linestyle=lines[i], alpha=alphas[i])
        i +=1
    
    matplotlib.pyplot.legend(loc="upper right")
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")
    
    return


def compare_success_rate(*args, fig_path, labels):
    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.axes()
    matplotlib.pyplot.ylabel("success rate", fontsize=15)
    heights = []
    
    for arg in args:
        #~ heights.append(len(arg)/100) # TODO This is from the old tests
        heights.append(arg)
    
    positions = numpy.arange(len(heights))
    bar = matplotlib.pyplot.bar(positions, heights, color="red")
    #~ bar[0].set_color("r")
    #~ bar[1].set_color("b")
    #~ bar[2].set_color("orange")
    matplotlib.pyplot.xticks(positions, labels)
    matplotlib.pyplot.ylim(0.0, 1.1)
    
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")
    
    return


def compare_basin_of_convergence(*args, angles_boundaries, labels, fig_path):
    """ 
    args: 
        test_results:   ndarray with transformation angles and corresponding rmse after convergence
    """
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.ylabel("RMSE", fontsize=15)
    matplotlib.pyplot.xlabel("Rotation angle around z-axis [°]", fontsize=15)
    matplotlib.pyplot.axis(xmin=angles_boundaries[0], xmax=angles_boundaries[1])
    
    i = 0
    for arg in args:
        matplotlib.pyplot.scatter(arg[:, 0], arg[:, 1], marker="x")
        x = numpy.arange(arg[0, 0], arg[-1, 0], 0.01)
        y = numpy.interp(x, arg[:, 0], arg[:, 1])
        matplotlib.pyplot.plot(x, y, "-", label=labels[i])
        i +=1
    
    matplotlib.pyplot.legend(loc="upper right")
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")
    
    return


def compare_venn_2(success_str1, success_str2, labels, fig_path): # ('Group A', 'Group B')
    """ Look at differences in these strings to see wether they were successful for the same problems """
    
    def venn_diagram(set1_num, set2_num, overlap, labels):
        v = venn2(subsets = (set1_num, set2_num, overlap), set_labels = labels)
        #~ v.get_patch_by_id(labels[0]).set_color("red")
        #~ v.get_patch_by_id(labels[1]).set_color("blue")
        #~ matplotlib.pyplot.gca().set_axis_bgcolor('skyblue')
        matplotlib.pyplot.gca().set_axis_on()
        matplotlib.pyplot.annotate("100", xy=(-0.6, +0.4))
        
        matplotlib.pyplot.savefig(fig_path, bbox_inches="tight")
        matplotlib.pyplot.close("all")
    
        return

    len1 = len(success_str1)
    len2 = len(success_str2)
    overlap = []
    for item1 in success_str1:
        if item1 in success_str2:
            overlap.append(item1)
        else:
            pass
            
    venn_diagram(len1, len2, len(overlap), labels)

    return


def compare_venn_3(success_str1, success_str2, success_str3, labels, fig_path): # ('Group A', 'Group B')
    """ Look at differences in these strings to see wether they were successful for the same problems """
    def venn_diagram(set1, set2, set3, labels):
        venn3([set1, set2, set3], set_labels = labels)
        
        #~ matplotlib.pyplot.gca().set_axis_on()
        #~ matplotlib.pyplot.annotate("100", xy=(-0.6, +0.4))
        matplotlib.pyplot.savefig(fig_path, bbox_inches="tight")
        matplotlib.pyplot.close("all")
    
        return
    
    success_str1 = set(success_str1)
    success_str2 = set(success_str2)
    success_str3 = set(success_str3)
    
    venn_diagram(success_str1, success_str2, success_str3, labels)

    return


def times_barplot(times_list, fig_path):
    """ Plot the iteration times in a barplot """
    fig = matplotlib.pyplot.figure()
    
    N = times_list["levels"].shape[0]
    ind = numpy.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p3_bottom = times_list["estimation"]+times_list["maximization"]

    p1 = matplotlib.pyplot.bar(ind, times_list["estimation"], width)
    p2 = matplotlib.pyplot.bar(ind, times_list["maximization"], width,
                 bottom=times_list["estimation"])
    p3 = matplotlib.pyplot.bar(ind, times_list["others"], width,
                 bottom=p3_bottom)

    matplotlib.pyplot.ylabel("time [s]")
    matplotlib.pyplot.xlabel("number of points")
    
    times_list["levels"] = numpy.array(times_list["levels"], str)
    matplotlib.pyplot.xticks(ind, tuple(times_list["levels"]))
    matplotlib.pyplot.legend((p1[0], p2[0], p3[0]), ("estimation", "maximization", "others"))

    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")

    return


def plot_multi_scale_iteration_count(sizes, iterations, fig_path):
    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.axes()
    matplotlib.pyplot.ylabel("iterations", fontsize=15)
    matplotlib.pyplot.xlabel("scale: number of used points", fontsize=15)
    x = numpy.arange(len(sizes))
    matplotlib.pyplot.bar(x, iterations, color="r") 
    
    labels=[]
    for element in sizes:
        labels.append(str(element))    
    matplotlib.pyplot.xticks(x, labels) 
    
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")
    
    return


def plot_speed_test(point_levels, estimation_times, maximization_times, iteration_times, fig_path):
    """ We plot the average times of the steps for different data sizes """    
    iteration_times = iteration_times - (estimation_times + maximization_times) # Difference for stack plot
    
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.ylabel("average time [s]", fontsize=15)
    matplotlib.pyplot.xlabel("number of points", fontsize=15)
    labels = ["estimation", "maximization", "iteration"] 
    matplotlib.pyplot.stackplot(point_levels, estimation_times, maximization_times, iteration_times, labels=labels) 
    matplotlib.pyplot.legend(loc="upper left")
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")
    
    return


def plot_error_in_gaussian_test(noise_levels, median_errors, lower_bound, upper_bound, fig_path):
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.ylabel("rotation error", fontsize=15)
    matplotlib.pyplot.xlabel("noise level(std) [%]", fontsize=15)
    matplotlib.pyplot.plot(noise_levels, median_errors, "-", color="r")
    matplotlib.pyplot.fill_between(noise_levels, lower_bound, upper_bound, alpha=0.1, color="r")
    
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")

    return
    
    
def plot_error_in_outlier_test(noise_levels, median_error, lower_bound, upper_bound, fig_path):
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.ylabel("RMSE", fontsize=15)
    matplotlib.pyplot.xlabel("number of outliers", fontsize=15)
    matplotlib.pyplot.plot(noise_levels, median_error, "-", color="r")
    matplotlib.pyplot.fill_between(noise_levels, lower_bound, upper_bound, alpha=0.1, color="r")
    
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")

    return
    
    
def plot_rmse(rmse_list, fig_path):
    """ Plot the rms error """
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.ylabel("RMSE", fontsize=15)
    matplotlib.pyplot.xlabel("number of iterations", fontsize=15)
    matplotlib.pyplot.plot(numpy.arange(len(rmse_list)), rmse_list, "-")
    
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")

    return
    
    
def plot_robust_test_convergence(convergence_results, fig_path):
    """ Plot the rms error """
    if convergence_results[0].size == 0:
        raise Exception("No registration converged")
    else:
        pass
    
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.ylabel("RMSE", fontsize=15)
    matplotlib.pyplot.xlabel("iteration", fontsize=15)
    iterations = numpy.arange((convergence_results[0].flatten()).shape[0])
    matplotlib.pyplot.plot(iterations, convergence_results[0].flatten(), "-", color="r")
    matplotlib.pyplot.fill_between(iterations, convergence_results[1].flatten(), convergence_results[2].flatten(), alpha=0.1, color="r")
    
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")
    

def plot_robust_test_box(last_rmse, fig_path):
    """ Box plot of all last rmse """ 
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.ylabel("RMSE", fontsize=15)
    matplotlib.pyplot.boxplot(last_rmse) 
    
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")
    
    
def plot_robust_test_cdf(last_rmse, init_rmse, fig_path):
    """ Cumulative probability density function for rmse """ 
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(numpy.sort(last_rmse), numpy.linspace(0.0, 1.0, last_rmse.shape[0]), label="last RMSE", color="b")
    matplotlib.pyplot.plot(numpy.sort(init_rmse), numpy.linspace(0.0, 1.0, init_rmse.shape[0]), label="initial pertubation", color="r")
    matplotlib.pyplot.fill_between(numpy.sort(init_rmse), numpy.zeros(init_rmse.shape[0]), 
                                   numpy.linspace(0.0, 1.0, init_rmse.shape[0]), alpha=0.1, color="r")
    matplotlib.pyplot.xlabel('RMSE')
    matplotlib.pyplot.ylabel('cumulative probability')
    matplotlib.pyplot.legend(loc="upper right")
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")

    return


def plot_basin_of_convergence_1d_rotation(axis, angles, rmse, fig_path):
    """ Make a plot of last rmse and initial rotation angle """
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.ylabel("RMSE", fontsize=15)
    matplotlib.pyplot.xlabel("rotation angle around %s axis [°]" %axis, fontsize=15)
    matplotlib.pyplot.plot(angles, rmse, linestyle='-', color='k')
    
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")
    

def distances_histogram(distances, fig_path):
    """ Show a histogram for a numpy array of distances """
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.ylabel("number of points per bin", fontsize=15)
    matplotlib.pyplot.xlabel("distance to nearest neighbor", fontsize=15)
    num = distances.shape[0]
    matplotlib.pyplot.hist(distances, bins=int(num/100))
    
    fig.savefig(fig_path, bbox_inches="tight")
    matplotlib.pyplot.close("all")
    
    return

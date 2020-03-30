from math import ceil

import matplotlib.pylab as plt
from matplotlib.lines import Line2D


def _contained_in_larger_interval(i, j, not_sig):
    for i1, j1 in not_sig:
        if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
            return True
    return False


def merge_nonsignificant_cliques(not_sig):
    # keep only longest
    longest = [(i, j) for i, j in not_sig if
               not _contained_in_larger_interval(i, j, not_sig)]
    return longest


def do_plot(x, insignificant_indices, names=None,
            arrow_vgap=.2, link_voffset=.1, link_vgap=.1,
            xlabel=None):
    """
    Draws a critical difference graph, which is used to display  the
    differences in methods' performance. This is inspired by the plots used in:

    See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Methods are drawn on an axis and connected with a line if their
    performance is not significantly different.

    :param x: List of average methods' scores.
    :type x: list-like
    :param insignificant_indices: list of  tuples that specify the indices of
    all pairs of methods that are not significantly different and should be
    connected in the diagram. Each tuple must be sorted, and no duplicate
    tuples should be contained in the list.

        Examples:
         - [(0, 1), (3, 4), (4, 5)] is correct
         - [(0, 1), (3, 4), (4, 5), (3,4)] contains a duplicate
         - [(4, 3)] contains a non-sorted tuple

    If there is a cluster of non-significant differences (e.g. 1=2, 2=3,
    1=3), `graph_ranks` will draw just a single link connecting all of them.

    Note: the indices returned by this callable should refer to positions in
    `scores` after it is sorted in increasing order. It is to avoid confusion
    this function raises if `scores` is not sorted.

    :param names: List of methods' names.
    :param arrow_vgap: vertical space between the arrows that point to method
     names.  Scale is 0 to 1, fraction of axis
    :param link_vgap: vertical space between the lines that connect methods
    that are not significantly different. Scale is 0 to 1, fraction of axis size
    :param link_voffset: offset from the axis of the links that connect
    non-significant methods
    """
    if names is None:
        names = list(range(len(x)))

    for pair in insignificant_indices:
        assert all(0 <= idx < len(x) for idx in pair), 'Check indices'

    # remove both axes and the frame: http://bit.ly/2tBIlWv
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), subplot_kw=dict(frameon=False))
    ax.get_xaxis().tick_top()
    ax.get_yaxis().set_visible(False)
    plt.xticks(color="red",fontsize=12)
    size = len(x)
    y = [0] * size
    ax.plot(x, y, 'ko')

    # plt.xlim(0.7 * x[0], 1.1 * x[-1])
    # plt.xlim(0.7 * x[0], 1 * x[-1])
    plt.xlim(0,9)
    plt.ylim(-1, 0)

    # draw the x axis again
    # this must be done after plotting
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmax), (ymax, ymax),
                         color='black', linewidth=2))

    # add an optional label to the x axis
    if xlabel:
        ax.annotate(xlabel, xy=(xmax, 0), xytext=(0.95, 0.1),
                    textcoords='axes fraction', ha='center', va='center',
                    fontsize=10)  # text slightly smaller

    half = int(ceil(len(x) / 2.))
    # make sure the topmost annotation in at 90% of figure height
    ycoords = list(reversed([0.75 - arrow_vgap * i for i in range(half)]))
    ycoords.extend(reversed(ycoords))
    for i in range(size):
        ax.annotate(str(names[i]),
                    xy=(x[i], y[i]),
                    xytext=(-.05 if i < half else .95,  # x coordinate
                            ycoords[i]),  # y coordinate
                    textcoords='axes fraction',
                    color='red',
                    ha='center', va='center',
                    arrowprops={'arrowstyle': '-',
                                'connectionstyle': 'angle,angleA=0,angleB=90'})

    # draw horizontal lines linking non-significant methods
    linked_methods = merge_nonsignificant_cliques(insignificant_indices)
    # where do the existing lines begin and end, (X, Y) coords
    used_endpoints = set()
    y = link_voffset
    dy = link_vgap

    def overlaps_any(foo, existing):
        """
        Checks if the proposed horizontal line (given with its x-y coordinates)
        overlaps any of the existing horizontal lines
        """
        return _contained_in_larger_interval(foo[0], foo[1], existing)

    for i, (x1, x2) in enumerate(sorted(linked_methods)):
        # determine how far up/down the line should be drawn
        # 1. can we lower it any further- not if it would be too low and if it
        # would overlap another line
        if y > link_voffset and overlaps_any((x1, y - dy), used_endpoints):
            y -= dy
        # 2. can we draw it at the current value of y- not if its left
        # end would overlap with the right end of an existing line
        # need to lift up a bit
        elif overlaps_any((x1, y), used_endpoints):
            y += dy
        else:
            pass

        plt.hlines(-y, x[x1], x[x2], linewidth=2,colors='blue')  # y, x0, x1

        used_endpoints.add((x1, y))
        used_endpoints.add((x2, y))


if __name__ == "__main__":


    # N=14,K=9 for benchmarking datasets and real-world dataset
    # # Accuracy
    # insignificant_indices = [(0, 3), (1, 4)]
    # scores = sorted([5.179,4.071,5.036,6.321,5.821,8.857,6.571,1.643,1.357])
    # names = ['MLWSE-L21 ','MLWSE-L1 ','ECC ','EPS ','EBR ','CDE ','RAkEL ','MLS ','AdaBoost.MH ']
    # fig=do_plot(scores, insignificant_indices, names, xlabel=None)
    # plt.savefig('img2/accuracy.png')
    # plt.show()

    # # Hamming loss
    # insignificant_indices = [(1, 8),(3, 8)]
    # scores = sorted([3.107,3.643,5.679,5.893,6.107,7.571,5.893,3.857,3.25])
    # names = ['EBR ','MLWSE-L21 ','ECC ','MLWSE-L1 ','EPS ','RAkEL ','MLS ','CDE ','AdaBoost.MH ']
    # fig=do_plot(scores, insignificant_indices, names, xlabel=None)
    # plt.savefig('img2/hamming_loss.png')
    # plt.show()

    # # Ranking loss
    # insignificant_indices = [(0, 5),(2,6) ]
    # scores = sorted([2.357,4.179,4.964,8.071,6.5,7.821,6.536,2.643,1.929])
    # names = ['MLWSE-L21 ','EBR ','MLWSE-L1 ','ECC','EPS ','CDE ','MLS ','AdaBoost.MH ','RAkEL']
    # fig=do_plot(scores, insignificant_indices, names, xlabel=None)
    # plt.savefig('img2/ranking_loss.png')
    # plt.show()

    # # F1
    #insignificant_indices = [(0, 8) ]
    #scores = sorted([4.75,3.321,4.75,5.679,5.429,8.714,5.964,2.929,3.464])
    #names = ['MLWSE-L1 ','ECC','MLWSE-L21 ','EPS','EBR ','CDE ','RAkEL ','MLS','AdaBoost.MH']
    #fig=do_plot(scores, insignificant_indices, names, xlabel=None)
    #plt.savefig('img2/F1.png')
	#plt.show()


    # # Macro-f1
    # insignificant_indices = [(0, 8) ]
    # scores = sorted([4.857,3.714,5.286,5.464,5.357,8.857,5.679,2.643,3.143])
    # names = ['MLWSE-L1 ','MLWSE-L21','ECC','EBR','EPS ','CDE ','RAkELS ','MLS','AdaBoost.MH']
    # fig=do_plot(scores, insignificant_indices, names, xlabel=None)
    # plt.savefig('img2/Macro-f1.png')
    # plt.show()

    # # Micro-f1
    insignificant_indices = [(1, 8) ]
    scores = sorted([4.179,2.714,5.714,5,5.071,8.964,4.857,4,4.5])
    names = ['ECC','MLWSE-L1','EBR','MLWSE-L21 ','MLS','RAkELS ','CDE ','EPS','AdaBoost.MH']
    fig=do_plot(scores, insignificant_indices, names, xlabel=None)
    plt.savefig('img2/Micro-f1.png')
    plt.show()
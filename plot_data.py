__author__ = 'LT'
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
def is_number(str):
    try:
        int(str)
        return True
    except:
        return False

def read_profile_master(file_path):
    start = False
    lines = dict()
    cum_time = dict()
    with open(file_path) as f:
        for l in f:
            if start:
                line = filter(None, l.split(" "))
                lines[int(line[1])] = line[0]
                if line[0] not in cum_time:
                    if len(line) > 7 and is_number(line[3]): cum_time[line[0]] = int(line[3])
                else:
                    if len(line) > 7 and is_number(line[3]): cum_time[line[0]] += int(line[3])
            else:
                if "=============" in l:
                    start = True
    return lines, cum_time

def read_profile(file_path, lines):
    start = False
    cum_time = dict()
    with open(file_path) as f:
        for l in f:
            if start:
                line = filter(None, l.split(" "))
                if len(line) > 5:
                    key = int(line[0])
                    if key in lines:
                        if lines[key] not in cum_time:
                            if is_number(line[2]): cum_time[lines[key]] = int(line[2])
                        else:
                            if is_number(line[2]): cum_time[lines[key]] += int(line[2])
            else:
                if "=============" in l:
                    start = True
    return cum_time

def plot_profile(raw_data):
    #raw_data = {'first_name': x_label,
    #    'pre_score': [4, 24, 31, 2, 3],
    #    'mid_score': [25, 94, 57, 62, 70],
    #    'post_score': [5, 43, 23, 23, 51]}

    #df = pd.DataFrame(raw_data, columns = ['first_name', 'pre_score', 'mid_score', 'post_score'])

    df = pd.DataFrame(raw_data, columns = raw_data.keys())
    print df

    #Create the general blog and the "subplots" i.e. the bars
    f, ax1 = plt.subplots(1, figsize=(10,8))

    # Set the bar width
    bar_width = 0.5

    # positions of the left bar-boundaries
    bar_l = [i+1 for i in range(len(df['kernelcalc']))]

    # positions of the x-axis ticks (center of the bars as bar labels)
    tick_pos = [i+(bar_width/2) for i in bar_l]
    bottom_values = [0 for i in range(len(df['kernelcalc']))]
    # Create a bar plot, in position bar_1
    label_symbol = []
    label_name = []

    # Create a bar plot, in position bar_1
    kc = ax1.bar(bar_l,
            # using the pre_score data
            df['kernelcalc'],
            # set the width
            width=bar_width,
            #bottom=bottom_values,
            # with the label pre score
            label='Kernel Calculation',
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#ff5643')
    #bottom_values = [i+j for i,j in zip(bottom_values, df['kernelcalc'])]
    bottom_values = df['kernelcalc']
    label_symbol.append(kc)
    label_name.append('Kernel Caclculation')



    # Create a bar plot, in position bar_1
    bk = ax1.bar(bar_l,
            # using the post_score data
            df['index'],
            # set the width
            width=bar_width,
            # with pre_score and mid_score on the bottom
            bottom=bottom_values,
            # with the label post score
            label='Bookkeeping',
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#b7efff')

    bottom_values = [i+j for i,j in zip(bottom_values, df['index'])]
    label_symbol.append(bk)
    label_name.append('Bookkeeping')
    '''
    # Create a bar plot, in position bar_1
    bk = ax1.bar(bar_l,
            # using the post_score data
            df['gradient'],
            # set the width
            width=bar_width,
            # with pre_score and mid_score on the bottom
            bottom=bottom_values,
            # with the label post score
            label='Calculation of Gradient',
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#ff9a00')

    bottom_values = [i+j for i,j in zip(bottom_values, df['gradient'])]
    label_symbol.append(bk)
    label_name.append('Calculation of Gradient')
    '''


    # Create a bar plot, in position bar_1
    uR = ax1.bar(bar_l,
            # using the mid_score data
            df['updateR'],
            # set the width
            width=bar_width,
            # with pre_score on the bottom
            bottom=bottom_values,
            # with the label mid score
            label='Update R',
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#fff51b')

    bottom_values = [i+j for i,j in zip(bottom_values, df['updateR'])]
    label_symbol.append(uR)
    label_name.append('Update R')

    mi = ax1.bar(bar_l,
            # using the post_score data
            df['mininc'],
            # set the width
            width=bar_width,
            bottom=bottom_values,
            # with pre_score and mid_score on the bottom
            # with the label post score
            label='Minimum Incremental',
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#48ffb0')
    #bottom_values = df['mininc']
    bottom_values = [i+j for i,j in zip(bottom_values, df['mininc'])]
    label_symbol.append(mi)
    label_name.append('Minimum Incremental')

    # Create a bar plot, in position bar_1
    bc = ax1.bar(bar_l,
            # using the post_score data
            df['beta'],
            # set the width
            width=bar_width,
            # with pre_score and mid_score on the bottom
            bottom=bottom_values,
            # with the label post score
            label='Beta Calculation',
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#e19aec')
    label_symbol.append(bc)
    label_name.append('Beta Calculation')

    bottom_values = [i+j for i,j in zip(bottom_values, df['beta'])]

    # Create a bar plot, in position bar_1
    gc = ax1.bar(bar_l,
            # using the post_score data
            df['gamma'],
            # set the width
            width=bar_width,
            # with pre_score and mid_score on the bottom
            bottom=bottom_values,
            # with the label post score
            label='Gamma Calculation',
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#5f6e7d')
    bottom_values = [i+j for i,j in zip(bottom_values, df['gamma'])]
    label_symbol.append(gc)
    label_name.append('Gamma Calculation')

    # set the x ticks with names
    plt.xticks(tick_pos, df['datasize'])

    # Set the label and legends
    ax1.set_ylabel("Portion of Runtime", {'fontsize': 14})
    ax1.set_xlabel("Data Size", {'fontsize': 14})
    #plt.legend(loc='upper left')
    plt.legend(label_symbol, label_name, loc='upper center', bbox_to_anchor=(0.5, -0.08),
          fancybox=True, shadow=True, ncol=2)
    # Set a buffer around the edge
    plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
    plt.show()

def get_data_from_runtime(relative_runtime):
    return [relative_runtime['kc'] if 'kc' in relative_runtime else 0,
                relative_runtime['uR'] if 'uR' in relative_runtime else 0,
                relative_runtime['be'] if 'be' in relative_runtime else 0,
                relative_runtime['mi'] if 'mi' in relative_runtime else 0,
                relative_runtime['ga'] if 'ga' in relative_runtime else 0,
                relative_runtime['up'] if 'up' in relative_runtime else 0,
                relative_runtime['in'] if 'in' in relative_runtime else 0,
                ]

def get_mnist_profile():
    ####
    n_round = 2
    lines, cum_time = read_profile_master('results/mnist/random_improved.txt')

    '''
    data = []
    for i in [1,2,3,5,7]:
        print "n = %s000" % i
        cum_time = read_profile('results/profile_mnist/mnist_%s000_nu0.3_gamma-0.001.output.txt' % i, lines)
        print cum_time
        runtime = sum(cum_time.values())
        relative_runtime = {k:round(float(v)/runtime,n_round) for k,v in cum_time.iteritems() if round(float(v)/runtime,n_round) > 0.0}
        print relative_runtime
        data.append(get_data_from_runtime(relative_runtime))

    raw_data = {'datasize': ['1000', '2000', '3000', '5000', '10000'],
        'kernelcalc': [data[i][0] for i in range(5)],
        'updateR': [data[i][1] for i in range(5)],
        'beta': [data[i][2] for i in range(5)],
        'mininc': [data[i][3] for i in range(5)],
        'gamma': [data[i][4] for i in range(5)],
        'update': [data[i][5] for i in range(5)],
        'index': [data[i][6] for i in range(5)]}
    '''
    raw_data = {
        'datasize': ['1000', '2000', '5000', '10000', '20000'],
        'kernelcalc': [0.031,0.015,0.005, 0.002, 0.002],
        'updateR': [0.695,0.814,0.902, 0.922, 0.913],
        'beta': [0.081,0.092,0.066, 0.064, 0.078],
        'mininc': [0.084,0.027,0.008, 0.003, 0.003],
        'gamma': [0.027,0.015,0.005, 0.003, 0.002],
        'update': [0.005,0.015, 0, 0, 0.002],
        'index': [0.009, 0.002, 0, 0, 0]

    }

    plot_profile(raw_data)

def get_pageblocks0_profile():
    '''
    range_size = [1,2,2,4,5]
    n_round = 2
    lines, cum_time = read_profile_master('results/mnist/random_improved.txt')
    data = []
    for i in range_size:
        print "n = %s000" % i
        cum_time = read_profile('results/profile_page-blocks0/pageblocks0_%s000_nu0.3_gamma-1.output.txt' % i, lines)
        print cum_time
        runtime = sum(cum_time.values())
        print runtime
        relative_runtime = {k:round(float(v)/runtime,n_round) for k,v in cum_time.iteritems() if round(float(v)/runtime,n_round) > 0.0}
        print relative_runtime
        data.append(get_data_from_runtime(relative_runtime))
    n = len(data)

    raw_data = {'datasize': ['1000', '2000', '3000', '4000', '5000'],
        'kernelcalc': [data[i][0] for i in range(n)],
        'updateR': [data[i][1] for i in range(n)],
        'beta': [data[i][2] for i in range(n)],
        'mininc': [data[i][3] for i in range(n)],
        'gamma': [data[i][4] for i in range(n)],
        'update': [data[i][5] for i in range(n)],
        'index': [data[i][6] for i in range(n)]}
    raw_data['mininc'][2] -= 0.05
    raw_data['gamma'][2] += 0.05
    '''
    raw_data = {
        'datasize': ['1000', '2000', '3000', '4000', '5000'],
        'kernelcalc': [0.042, 0.042, 0.051, 0.074, 0.011],
        'updateR': [0.11, 0.08, 0.06, 0.05, 0.03],
        'beta': [0.066, 0.014, 0.012, 0.008, 0.005],
        'mininc': [0.357, 0.285, 0.251, 0.125, 0.08],
        'gamma': [0.203, 0.425, 0.458, 0.683, 0.809],
        'update': [0.212, 0.20, 0.18, 0.17, 0.13],
        'index': [0.122, 0.11, 0.08,0.04,0.03]
    }
    plot_profile(raw_data)

def plot_influence_pima():
    x = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    incremental = [86.206, 89.306, 17.950, 1.667, 2.845, 1.311, 1.290, 1.393, 1.469, 1.641]
    cvxopt = [3.310, 3.297, 3.559, 4.164, 4.224, 4.526, 4.173, 4.519, 4.861, 4.160]
    sklearn = [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.026, 0.024]
    fig, ax = plt.subplots()
    inc_plot = plt.Line2D(x, incremental, marker="o", linestyle="-", label='test', color='r')
    cvx_plot = plt.Line2D(x, cvxopt, marker="^", linestyle="-", label='test', color='b')
    sklearn_plot = plt.Line2D(x, sklearn, marker="8", linestyle="-", label='test', color='c')
    ax.add_line(inc_plot)
    ax.add_line(cvx_plot)
    ax.add_line(sklearn_plot)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.ylim((-100,90))
    ax.set_ylabel('Runtime in sec.', {'fontsize':15})
    ax.set_xlabel('Parameter %s' % r'$\gamma$', {'fontsize':15})
    ax.set_title('Running time of training data set pima (768 data points) with $\upsilon$ = 0.5')
    plt.legend([inc_plot, cvx_plot, sklearn_plot],['Incremental OCSVM', 'cvxopt-OCSVM', 'sklearn-OCSVM'],
               loc='upper center', bbox_to_anchor=(0.75,0.97),
          fancybox=True, shadow=True)
    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()
    plt.setp(ltext, fontsize=15)
    ax.grid(True)
    plt.show()

def plot_runtime():
    incremental_cvxopt = [5000, 10000, 20000]
    cvxopt = [2177.2, 8319.1, 18163.7]
    x = [5000, 10000, 20000, 30000, 50000, 100000]
    incremental = [1040.6, 4234.9, 7123.1, 12793.8, 38628.05, 184216.92]
    sklearn = [33.28, 437.5, 3288.371, 8991.0, 15124.5, 41812.4]
    fig, ax = plt.subplots()
    inc_plot = plt.Line2D(x, incremental, marker="o", linestyle="-", label='test', color='r')
    cvx_plot = plt.Line2D(incremental_cvxopt, cvxopt, marker="^", linestyle="-", label='test', color='b')
    sklearn_plot = plt.Line2D(x, sklearn, marker="8", linestyle="-", label='test', color='c')
    ax.add_line(inc_plot)
    ax.add_line(cvx_plot)
    ax.add_line(sklearn_plot)
    ax.set_yscale('log')
    #ax.set_xscale('log')
    plt.ylim((0, 200000))
    plt.xlim((0, 110000))
    ax.set_ylabel('Runtime in sec.', {'fontsize':16})
    ax.set_xlabel('Data set size', {'fontsize':16})
    #ax.set_title('Running time of training different data set sizes')
    plt.legend([inc_plot, cvx_plot, sklearn_plot],['Incremental OCSVM', 'cvxopt-OCSVM', 'sklearn-OCSVM'],
               loc='lower right', #bbox_to_anchor=(0.5,-0.1),
          fancybox=True, shadow=True)
    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()
    plt.setp(ltext, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True)
    plt.show()

def plot_multiple_cf(cm1, target_names, title, cm2=None,
                     cm3=None, cmap=plt.cm.Blues, colorbar=True, filename_prefix=None):

    len = 1
    len += 1 if cm2 is not None else 0
    len += 1 if cm3 is not None else 0
    f, ax = plt.subplots(1, len, figsize=(3.75*len,4))

    i = 0
    tick_marks = np.arange(cm1.shape[0])
    if len == 1:
        ax, im = plot_cf(ax, cm1, cmap, tick_marks, target_names, title=title[i])

    else:
        ax[i], im = plot_cf(ax[i], cm1, cmap, tick_marks, target_names, title=title[i])


    if cm2 is not None:
        i += 1
        ax[i], im = plot_cf(ax[i], cm2, cmap, tick_marks,
                            target_names, yaxis_visible=False, title=title[i])

    if cm3 is not None:
        i += 1
        ax[i], im = plot_cf(ax[i], cm3, cmap, tick_marks,
                            target_names, yaxis_visible=False, title=title[i])
    if colorbar:

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.3, 0.025, 0.6])
        f.subplots_adjust(bottom=0.3)
        f.colorbar(im, cax=cbar_ax)
    #f.tight_layout()
    plt.savefig('%s_confusion.png' % filename_prefix)
    plt.show()

def plot_multiple_precision_recall_curves(precision_recall_avg, filename_prefix=None):
    print "plot here"
    f = plt.figure(figsize=(10,7))
    ax = f.add_subplot(111)
    i = 0
    #remove...
    #precision_recall_avg[0]['precision'] = [j* 1.2 if j > 0 else j for j in precision_recall_avg[0]['precision']]
    for d in precision_recall_avg:

        if i == 0:
            linestyle = '--^'
            #linestyle = '-'
            c = '#416fff'
            ax.plot(d['recall'], d['precision'], linestyle, label='Precision-Recall curve of %s' % d['label'], c=c,
                    markevery=1, linewidth=3)
            #ax.scatter(d['recall'][1::5], d['precision'][1::5], marker='^', c=c)
        elif i == 1:
            linestyle = "-."
            #linestyle = '-'
            c = '#ff1f1f'
            ax.plot(d['recall'], d['precision'], linestyle, label='Precision-Recall curve of %s' % d['label'], c=c, linewidth=3)
        else:
            linestyle = ":"
            #linestyle = '-'
            c = '#39a921'
            ax.plot(d['recall'], d['precision'], linestyle, label='Precision-Recall curve of %s' % d['label'], c=c, linewidth=3)
        i += 1

        avg_precision = d['avg_precision']
    ax.set_xlabel('Recall', {'fontsize': 15})
    ax.set_ylabel('Precision', {'fontsize': 15})
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision-Recall Curves', {'fontsize': 16})
    plt.legend(loc="lower right")
    f.tight_layout()
    plt.savefig('%s_precision_recall.png' % filename_prefix)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close(f)


def plot_cf(ax, cm, cmap, tick_marks, target_names, yaxis_visible=True, title=None):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)
    ax.tick_params(labelsize=14)
    ax.set_ylabel('True label', {'fontsize': 14})
    ax.set_xlabel('Predicted label', {'fontsize': 14})

    if title != None:
        ax.set_title(title)
    if yaxis_visible == False:
        ax.yaxis.set_visible(False)
    return ax, im



if __name__ == '__main__':
    plot_influence_pima()
    #get_mnist_profile()
    #get_pageblocks0_profile()
    #plot_runtime()
    '''
    print
    print "n=2000"
    cum_time = read_profile('results/mnist-2000.output.txt', lines)
    runtime = sum(cum_time.values())
    relative_runtime = {k:round(float(v)/runtime,n_round) for k,v in cum_time.iteritems() if round(float(v)/runtime,n_round) > 0.0}
    print relative_runtime
    data2000 = get_data_from_runtime(relative_runtime)
    print
    print "n=3000"
    cum_time = read_profile('results/mnist-3000.output.txt', lines)
    print cum_time
    runtime = sum(cum_time.values())
    print runtime
    print {k:round(float(v)/runtime,n_round) for k,v in cum_time.iteritems() if round(float(v)/runtime,n_round) > 0.0}
    print
    print "n=5000"
    cum_time = read_profile('results/mnist-5000.output.txt', lines)
    print cum_time
    runtime = sum(cum_time.values())
    print runtime
    print {k:round(float(v)/runtime,n_round) for k,v in cum_time.iteritems() if round(float(v)/runtime,n_round) > 0.0}
    print
    print "n=10000"
    cum_time = read_profile('results/mnist-5000.output.txt', lines)
    print cum_time
    runtime = sum(cum_time.values())
    print runtime
    print {k:round(float(v)/runtime,n_round) for k,v in cum_time.iteritems() if round(float(v)/runtime,n_round) > 0.0}
    '''







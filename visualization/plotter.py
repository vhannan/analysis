import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PythonPlotter(object):
    def __init__(self, color_scheme, alpha=0.7, fontsize=8, title_fontsize=14,
        directory='/Users/'):
        self.color_scheme = color_scheme
        self.alpha = alpha
        self.fontsize = fontsize
        self.title_fontsize = title_fontsize
        self.directory = directory

    def make_histogram(self, data, bins, x_label, y_label, title, filename, \
        color_index, stacked=False, labels=None):
        fig = plt.figure(figsize=(10,6))
        axes = fig.add_subplot(111)
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['left'].set_visible(False)

        outfile = os.path.join(self.directory, filename)
        if stacked:
            plt.hist(data, bins, color_index, alpha = self.alpha, \
                                    label = labels, stacked=stacked, normed=True)
        else:
            plt.hist(data, bins, color=self.color_scheme[color_index], \
                                                alpha = self.alpha, normed=True)
        axes.set_title(title, fontsize = self.title_fontsize, alpha = self.alpha, \
                                                                    weight='bold')
        plt.yticks(fontsize = self.fontsize, alpha = self.alpha)
        plt.xticks(bins, fontsize = self.fontsize, alpha = self.alpha, ha = 'right')
        plt.ylabel(y_label, fontsize = self.fontsize, alpha = self.alpha)
        plt.xlabel(x_label, fontsize = self.fontsize, alpha = self.alpha)
        plt.xlim([min(bins), max(bins)])
        plt.show()
        fig.savefig(outfile)
        plt.close()

    def make_barchart(self, x,y, title, ylabel, labels, filename, color_index):
        fig = plt.figure(figsize=(10,6))
        axes = fig.add_subplot(111)
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['left'].set_visible(False)

        x_index = np.arange(0,len(x))
        labels = np.array(labels)
        outfile = os.path.join(self.directory, filename)
        plt.bar(x_index, y, width = .5, color = self.color_scheme[color_index], \
                                                                alpha=self.alpha)
        axes.set_title(title, fontsize = self.title_fontsize, alpha=self.alpha, \
                                                                weight='bold')
        axes.set_xticklabels(labels)
        plt.yticks(fontsize = self.fontsize, alpha = self.alpha)
        plt.xticks(fontsize = self.fontsize, alpha = self.alpha, ha = 'left')
        plt.ylabel(ylabel, fontsize = self.fontsize, alpha = self.alpha)
        plt.xlim([0,x.max()+1.0])
        plt.show()
        fig.savefig(outfile)
        plt.close()

    def make_line(self, x, y, x_label, y_label, title, filename, color_index):
        fig = plt.figure(figsize=(10,6))
        axes = fig.add_subplot(111)
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['left'].set_visible(False)

        outfile = os.path.join(self.directory, filename)
        plt.plot(x, y, linewidth=1.5, color = self.color_scheme[color_index], alpha = self.alpha)
        axes.set_title(title, fontsize = self.title_fontsize, alpha=self.alpha,  weight='bold')
        plt.yticks(fontsize = self.fontsize, alpha = self.alpha)
        plt.xticks(fontsize = self.fontsize, alpha = self.alpha, ha = 'right')
        plt.ylabel(y_label, fontsize = self.fontsize, alpha = self.alpha)
        plt.xlabel(x_label, fontsize = self.fontsize, alpha = self.alpha)
        plt.xlim([0,x.max()])
        plt.show()
        fig.savefig(outfile)
        plt.close()

    def multiline(self, x_list, y_list, x_label, y_label, title, filename, \
                                                                    label_list):
        fig = plt.figure(figsize=(10,6))
        axes = fig.add_subplot(111)
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['left'].set_visible(False)
        outfile = os.path.join(self.directory, filename)
        for ind, x in enumerate(x_list):
            plt.plot(x, y_list[ind], linewidth=1.5, color = self.color_scheme[ind], \
                                            label = label_list[ind], alpha = self.alpha)
        axes.set_title(title, fontsize = self.title_fontsize, alpha=self.alpha,  weight='bold')
        plt.yticks(fontsize = self.fontsize, alpha = self.alpha)
        plt.xticks(fontsize = self.fontsize, alpha = self.alpha, ha = 'right')
        plt.ylabel(y_label, fontsize = self.fontsize, alpha = self.alpha)
        plt.xlabel(x_label, fontsize = self.fontsize, alpha = self.alpha)
        plt.legend()
        plt.xlim([0,x.max()])
        plt.show()
        fig.savefig(outfile)
        plt.close()

    def make_stacked(x, data, x_tick_labels, x_label, y_label, legend_labels, \
                                                                title, filename):
        fig = plt.figure(figsize=(10,6))
        axes = fig.add_subplot(111)
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['left'].set_visible(False)                                                        
        outfile = os.path.join(self.directory, filename)
        width = 0.5
        legend_params = []
        p = []
        for i,user_group in enumerate(data):
            bottom = np.zeros(5)
            if i == 0:
                p.append(plt.bar(x, user_group, width, align='center', \
                                alpha = self.alpha, color = self.color_scheme[i]))
            else:
                for j in range(0,i):
                    bottom += data[j]
                p.append(plt.bar(x, user_group, width, align='center',  \
                    alpha = self.alpha, color = self.color_scheme[i], bottom = bottom))
            legend_params.append(p[i][0])
        plt.title(title, fontsize = self.title_fontsize)
        plt.legend('')
        plt.xlim([0, max(x) + 0.5])
        plt.yticks(fontsize = self.fontsize, alpha = self.alpha)
        plt.xticks(x, x_tick_labels, fontsize = self.fontsize, alpha = self.alpha, \
                                                                    ha = 'right')
        plt.ylabel(y_label, fontsize = self.fontsize, alpha = self.alpha)
        plt.xlabel(x_label, fontsize = self.fontsize, alpha = self.alpha)
        plt.legend(tuple(legend_params), legend_labels)
        fig.savefig(outfile)
        plt.close()

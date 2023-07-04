import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score


class PlotsDrawer:
    def __init__(self):
        pass

    def draw_plots(self, path_to_json: str) -> list:
        """
        Function for reading json file and building plots

        :param path_to_json: str - path to json file with data

        :return: list - list with filenames of plots
        """

        df = pd.read_json(path_to_json)
        print(df.head())
        print("Accuracy score:", accuracy_score(df['gt_corners'], df['rb_corners']))

        plt.scatter(df['gt_corners'], df['rb_corners'])
        plt.savefig('D:\Python projects\docu_sketch_test_task\plots\scatter_plot1.png')

        df.plot(x='name', y=['gt_corners', 'rb_corners'], kind='line', figsize=(20, 10))
        plt.savefig('D:\Python projects\docu_sketch_test_task\plots\line_plot1.png')

        df.plot(x='gt_corners', y='rb_corners', kind='line', figsize=(10, 10))
        plt.savefig('D:\Python projects\docu_sketch_test_task\plots\line_plot2.png')

        plt.figure(figsize=(10, 10))
        plt.scatter(df['gt_corners'], df['rb_corners'], c='crimson')
        p1 = max(max(df['rb_corners']), max(df['gt_corners']))
        p2 = min(min(df['rb_corners']), min(df['gt_corners']))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.axis('equal')
        plt.savefig('D:\Python projects\docu_sketch_test_task\plots\scatter_plot2.png')

        df.plot(x='mean', y='gt_corners', kind='scatter', figsize=(20, 10))
        plt.savefig('D:\Python projects\docu_sketch_test_task\plots\scatter_plot3.png')

        df.plot(x='floor_mean', y='mean', kind='scatter', figsize=(10, 10))
        plt.savefig('D:\Python projects\docu_sketch_test_task\plots\scatter_plot4.png')

        df.plot(x='ceiling_mean', y='mean', kind='scatter', figsize=(10, 10))
        plt.savefig('D:\Python projects\docu_sketch_test_task\plots\scatter_plot5.png')

        df.plot(y='gt_corners', kind='hist', figsize=(20, 10))
        plt.savefig('D:\Python projects\docu_sketch_test_task\plots\hist_plot1.png')

        path = 'D:\Python projects\docu_sketch_test_task\plots'
        plot_images = os.listdir(path)
        print(plot_images)

        return plot_images


drawer = PlotsDrawer()
drawer.draw_plots('https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json')

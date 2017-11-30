
from visdom import Visdom
import numpy as np

class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main', val_image_name='validation image'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        self.validation_image_name = val_image_name
        self.scores_window = None
        self.segmentation_window = None

    def plot(self, var_name, split_name, x, y, x_label='Epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel=x_label,
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

    def display_scores(self, image, epoch):
        if self.scores_window is None:
            self.scores_window = self.viz.image(image, env=self.env, opts=dict(title=self.validation_image_name, caption='Epoch={}'.format(str(epoch))))
        else:
            self.viz.image(image, win=self.scores_window, env=self.env, opts=dict(title=self.validation_image_name, caption='Epoch={}'.format(str(epoch))))

    def display_segmentation(self, image, epoch):
        if self.segmentation_window is None:
            self.segmentation_window = self.viz.image(image, env=self.env, opts=dict(title=self.validation_image_name, caption='Epoch={}'.format(str(epoch))))
        else:
            self.viz.image(image, win=self.segmentation_window, env=self.env, opts=dict(title=self.validation_image_name, caption='Epoch={}'.format(str(epoch))))
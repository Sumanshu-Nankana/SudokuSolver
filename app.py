import os
import copy
import cv2
from logging import FileHandler, WARNING
from flask import Flask
from flask import render_template
from flask import request
from src.preprocess import *
from src.solve_sudoku_grid3 import *
from matplotlib import pyplot as plt

__author__= "sumanshu"

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

if not app.debug:
    file_handler = FileHandler('sudoku.log')
    file_handler.setLevel(WARNING)
    app.logger.addHandler(file_handler)

@app.route('/')
@app.route('/home')
def home():
    title = 'Home'
    return render_template('home.html', title=title)

@app.route('/play')
def play():
    title = 'Play'
    return render_template('play.html', title=title)

@app.route('/upload', methods=['GET','POST'])
def solve():
    title = 'Upload'
    target = os.path.join(APP_ROOT, 'static/uploads/')

    if not os.path.isdir(target):
        os.mkdir(target)
    
    file = request.files.get("file")
    filename = None
    if file:
        filename = file.filename
    pred_grid = None
    pred_grid_copy = None
    solution = (None, None)
    if filename:
        destination = "/".join([target, filename])
        file.save(destination)

    if filename:
        pred_grid = process_function(destination)
        pred_grid_copy = copy.deepcopy(pred_grid)

    if pred_grid:
        solution = sudoku_solver(pred_grid)

    return render_template('play.html', title=title, 
                            filename=filename, pred_grid=pred_grid_copy, solution=solution)
    
@app.route('/graph')
def see_graph():
    images = os.listdir('./Images')
    images.sort()
    path = ['./Images/'+i for i in images]
    #print(path)
    # for i in range(7):
    #     plt.subplot(2,4,i+1)
    #     img = cv2.imread(path[i])
    #     plt.imshow(img)
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.savefig('./static/images/10_all.png', transparent=True)

    fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
    
    img = cv2.imread(path[0])
    ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    img = cv2.imread(path[1])
    ax2.imshow(img)
    ax2.set_xticks([])
    ax2.set_yticks([])

    img = cv2.imread(path[2])
    ax3.imshow(img)
    ax3.set_xticks([])
    ax3.set_yticks([])

    img = cv2.imread(path[3])
    ax4.imshow(img)
    ax4.set_xticks([])
    ax4.set_yticks([])

    img = cv2.imread(path[4])
    ax5.imshow(img)
    ax5.set_xticks([])
    ax5.set_yticks([])

    img = cv2.imread(path[5])
    ax6.imshow(img)
    ax6.set_xticks([])
    ax6.set_yticks([])

    img = cv2.imread(path[6])
    ax7.imshow(img)
    ax7.set_xticks([])
    ax7.set_yticks([])

    img = cv2.imread(path[7])
    ax8.imshow(img)
    ax8.set_xticks([])
    ax8.set_yticks([])


    for ax in fig.get_axes():
        ax.label_outer()

    fig.savefig('./static/images/10_all.png', transparent=True)

    return render_template('graph.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0')

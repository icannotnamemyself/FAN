from typing import Any
import matplotlib.pyplot as plt

def mvts_fig(data:Any, num_nodes:int,start:int , end:int ):
    # 创建图形和子图
    fig, axs = plt.subplots(num_nodes,figsize=(80, 6*num_nodes))
    for i in range(num_nodes):
        # 绘制时序数据
        axs[i].plot(data[start:end,i])
    return fig



def mvns_fig(data, num_nodes:int,start:int , end:int ):
    # data : (T, N)
    max_x = data.max()
    min_x = data.min()
    fig, axs = plt.subplots(1, end - start + 1, sharex=True, sharey=True,figsize=(80, 6*num_nodes))
    for i in range(0, end - start + 1):
        # 绘制时序数据
        axs[i].plot(data[i+start,:], np.arange(num_nodes))
        axs[i].set_ylim(num_nodes, -1)
        axs[i].set_xlim(min_x, max_x)

    return fig
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

# 梯度下降的一个示例

plt.rcParams['font.sans-serif']=['Simhei'] #显示中文
mpl.rcParams['axes.unicode_minus']=False # 解决保存图像是负号'-'显示为方块的问题
# step 1 准备测试数据
x_data=[338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

'''
np.array和np.asarray都会对元数据进行copy生成一个类型为ndarray类型的数据
但np.asarray在对ndarray类型的input不会产生一份copy,会直接对其进行操作
'''
x_d=np.asarray(x_data)
y_d=np.asarray(y_data)

# step 2 准备w 和 b的候选值
x = np.arange(-200, -100, 1)
y = np.arange(-5, 5, 0.1)
Z = np.zeros((len(x), len(y)))

'''
语法：X,Y = numpy.meshgrid(x, y)
输入的x，y，就是网格点的横纵坐标列向量（非矩阵）
输出的X，Y，就是坐标矩阵。
'''
X, Y = np.meshgrid(x, y) #生成网格点坐标矩阵

#loss
for i in range(len(x)):
    for j in range(len(y)):
        b=x[i]
        w=y[j]
        Z[j][i]=0 #meshgrid的结果： y为行，x为列
        for n in range(len(x_data)):
            Z[j][i]+=(y_data[n]-b-w*x_data[n])**2
        Z[j][i]/=len(x_data)

# step 3 给 w 和 b一个初始值，来计算b和w的偏微分
def origin():
    # linear regression
    # b = -120
    # w = -4
    b = -2
    w = 0.01
    lr = 0.00001  # 学习速率
    iteration = 1400000  # 迭代次数

    b_history = [b]
    w_history = [w]
    loss_history = []
    import time
    start = time.time()
    for i in range(iteration):
        m = float(len(x_d))
        y_hat = w * x_d + b

        loss = np.dot(y_d - y_hat, y_d - y_hat) / m  # 通过矩阵点积来计算loss

        grad_b = -2.0 * np.sum(y_d - y_hat) / m
        grad_w = -2.0 * np.dot(y_d - y_hat, x_d) / m

        # update param
        b -= lr * grad_b
        w -= lr * grad_w

        b_history.append(b)
        w_history.append(w)
        loss_history.append(loss)

        # 每迭代1k次，输出一次结果
        if i % 10000 == 0:
            print("Step %i, w: %0.4f, b: %.4f, Loss: %.4f" % (i, w, b, loss))
    end = time.time()
    print("大约需要时间：", end - start)
    drawpic(b_history, w_history)


def Adagrad ():
    # linear regression
    b = -120
    w = -4
    lr = 1
    iteration = 100000

    b_history = [b]
    w_history = [w]
    loss_history=[]

    lr_b = 0
    lr_w = 0
    import time
    start = time.time()
    for i in range(iteration):
        b_grad = 0.0
        w_grad = 0.0

        m = float(len(x_d))
        y_hat = w * x_d + b
        loss = np.dot(y_d - y_hat, y_d - y_hat) / m  # 通过矩阵点积来计算loss

        for n in range(len(x_data)):
            b_grad = b_grad - 2.0 * (y_data[n] - n - w * x_data[n]) * 1.0
            w_grad = w_grad - 2.0 * (y_data[n] - n - w * x_data[n]) * x_data[n]

        lr_b = lr_b + b_grad ** 2
        lr_w = lr_w + w_grad ** 2
        # update param
        b -= lr / np.sqrt(lr_b) * b_grad
        w -= lr / np.sqrt(lr_w) * w_grad

        b_history.append(b)
        w_history.append(w)
        loss_history.append(loss)

        # 每迭代1k次，输出一次结果
        if i % 10000 == 0:
            print("Step %i, w: %0.4f, b: %.4f, Loss: %.4f" % (i, w, b, loss))
    end = time.time()
    print("大约需要时间：", end - start)
    drawpic(b_history, w_history)




# step 4 plot the figure
'''
plt.contourf 与plt.contour都是绘制等高线的函数

'''
def drawpic(b_history,w_history):
    plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))  # 填充等高线
    plt.plot([-188.4], [2.67], 'x', ms=12, mew=3, color="orange") # 标注目标点
    plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
    plt.xlim(-200, -100)
    plt.ylim(-5, 5)
    plt.xlabel(r'$b$')
    plt.ylabel(r'$w$')
    plt.title("线性回归")
    plt.show()
origin()
#Adagrad()
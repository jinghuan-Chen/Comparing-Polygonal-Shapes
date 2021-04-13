import numpy as np
import matplotlib.pyplot as plt

def draw_truning_Function(m_x, m_y):
    plt.cla()
    draw_x, draw_y = [], []
    for i in range(len(m_x) - 1):
        x0, y0 = m_x[i], m_y[i]
        x1, y1 = m_x[i + 1], m_y[i + 1]
        draw_x.append(x0)
        draw_y.append(y0)
        draw_x.append(x1)
        draw_y.append(y0)
        draw_x.append(x1)
        draw_y.append(y1)
    plt.plot(draw_x, draw_y, linewidth=3)
    plt.show()


def turning_function(polygon_, eps = 1e-6):
    assert (polygon_.shape[0] >= 4), "The polygon should more then 4-points"
    init_vec = [1, 0]  # [1, 0] means the x-axis
    polygon = polygon_[:,:2].astype(np.float)

    nPoint  = len(polygon) - 1  # Polygon is end to end.
    lineVec = polygon[1:] - polygon[:-1] # polygon line vector

    b_vec   = np.r_[lineVec, [lineVec[0]]] # The next vector base linevec
    lineVec = np.r_[[init_vec], lineVec]

    le_len  = np.linalg.norm(lineVec, axis=1)  # vector lenght

    sig_val = np.sign(np.cross(lineVec, b_vec))

    le_len[np.where(le_len < eps)] = eps
    b_le_len = np.r_[le_len[1:], [le_len[1]]]
    dot_arr = np.sum(b_vec * lineVec, axis=1)
    cos_value = dot_arr / (b_le_len * le_len)
    cos_value[np.where((1.0 < cos_value) & (cos_value < 1.0 + eps))] = 1.0
    cos_value[np.where((-1.0 - eps < cos_value) & (cos_value < -1.0))] = -1.0
    angle_arr = np.arccos(cos_value) * sig_val

    x = np.zeros(nPoint + 1)
    y = np.zeros(nPoint + 1)

    le_sum = le_len.sum() - 1    # subtract the init_vec
    nor_len = le_len / le_sum    # normalize the linevec of polygon
    y[0] = angle_arr[0]
    for i in range(1, nPoint + 1):
        x[i] = x[i - 1] + nor_len[i]
        y[i] = y[i - 1] + angle_arr[i]
    return x, y

def sample_tf(x,y,ndim=1000):
    '''
    input: tf.x,tf.y, ndim
    return: n-dim tf values
    '''
    t = np.linspace(0,1,ndim)
    return np.piecewise(t,[t>=xx for xx in x],y)

if __name__ == "__main__":

    polygon1 = np.array([[0, 0],
                        [1, 0],
                        [1, 1],
                        [0, 1],
                        [0, 0]])

    polygon2 = np.array([[0.1, 0.1],
                         [0.2, 0.1],
                         [0.3, 0.4],
                         [0.2, 0.6],
                         [0.1, 0.6],
                         [0.1, 0.1]])

    # plot polygon
    # plt.plot(polygon1[:,0], polygon1[:,1], linewidth=6, color="r")
    # plt.show()

    #draw_truning_Function(x0, y0)

    x1, y1 = turning_function(polygon1)
    sample_1 = sample_tf(x1, y1)
    x2, y2 = turning_function(polygon2)
    draw_truning_Function(x2, y2)
    sample_2 = sample_tf(x2, y2)
    dis = np.linalg.norm(sample_1 - sample_2)

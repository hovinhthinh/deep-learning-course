wx_train = x[0:50]; y_train = y[0:50]
x_test = x[50:100]; y_test = y[50:100]

def get_derivative(current_a, current_b, current_c, x, y):
    da = 0
    for i in range(0, len(x)):
        da += -2 * x[i] * x[i] * (y[i] - current_a * x[i] * x[i] - current_b * x[i] - current_c)
    da /= len(x)

    db = 0
    for i in range(0, len(x)):
        db += -2 * x[i] * (y[i] - current_a * x[i] * x[i] - current_b * x[i] - current_c)
    db /= len(x)

    dc = 0
    for i in range(0, len(x)):
        dc += -2 * (y[i] - current_a * x[i] * x[i] - current_b * x[i] - current_c)
    dc /= len(x)

    return [da, db, dc]

def get_hessian(current_a, current_b, current_c, x, y):
    p0 = 0; p1 = 0;p2 = 0; p3 = 0; p4 = 0
    for i in range(0, len(x)):
        p0 += 2
        p1 += 2 * x[i]
        p2 += 2 * x[i] * x[i]
        p3 += 2 * x[i] * x[i] * x[i]
        p4 += 2 * x[i] * x[i] * x[i] * x[i]
    p0 /= len(x)
    p1 /= len(x)
    p2 /= len(x)
    p3 /= len(x)
    p4 /= len(x)
    return np.array([[p4,p3,p2],[p3,p2,p1],[p2,p1,p0]])

def can_stop(da, db, dc):
    if abs(da) <= EPS and abs(db) <= EPS and abs(dc) <= EPS: return True
    return False

def get_f(a, b, c, x, y):
    s = 0
    for i in range(0, len(x)):
        s += (y[i] - (a * x[i] * x[i] + b * x[i] + c)) * (y[i] - (a * x[i] * x[i] + b * x[i] + c))
    s /= len(x)
    return s

a = 0
b = 0
c = 0

fs = []
js = []

fs.append(get_f(a, b, c, x_train, y_train))
js.append(0)

epoch = 0
while (True):
    d = get_derivative(a, b, c, x_train, y_train)
    if can_stop(d[0], d[1], d[2]): break
    h = get_hessian(d[0], d[1], d[2], x_train, y_train)
    h_inv = inv(h)
    m = np.dot(h_inv, np.array([d]).transpose())
    m = m.transpose()[0]

    a -= m[0]
    b -= m[1]
    c -= m[2]

    epoch += 1
    cost = get_f(a, b, c, x_train, y_train)
    fs.append(cost); js.append(epoch)

    print 'Epoch = %d | (a, b, c) = (%.12f, %.12f, %.12f) | loss = %.12f' % (epoch, a, b, c, cost)

X = np.arange(5, 30, 0.1)
Y = a * X**2 + b * X + c

plt.plot(X, Y, 'r-', label='estimated model')

plt.plot(x_train, y_train, 'bo', label='training data')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Training data and estimated second order model with Newton method')
plt.show()
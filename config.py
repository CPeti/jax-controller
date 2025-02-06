plant = 'bathtub'
controller = 'pid'


consys = dict(
    DT = 1.0,
    LR = 0.01,
    EPOCHS = 1000,
    STEPS_PER_EPOCH = 100,
    JIT = True,
)

pid = dict(
    KP = 1.0,
    KI = 0.0,
    KD = 0.0,
)

neural = dict(
    layer_init = "glorot_uniform",
    param_range = [-1.0, 1.0],
    LAYER_SIZES = [3, 10, 1],
    ACTIVATION = 'relu',
)

bathtub = dict(
    INITIAL_STATE = 4.0,
    TARGET = 2.0,
    AREA = 1.0,
    EXIT_AREA = 0.01,
    DISTURBANCE_RANGE = 0.05,
)

cournot = dict(
    INITIAL_PROFIT = 0.0,
    TARGET = 1.5,
    P_MAX = 4.0,
    COST_1 = 0.5,
    COST_2 = 0.5,
    Q1 = 0.5,
    Q2 = 0.5,
    DISTURBANCE_RANGE = 0.05,
)

temperature = dict(
    T_INITIAL = 14.0,
    TARGET = 22.0,
    DISTURBANCE_RANGE = 2.0,
    ALPHA = 0.1,
    BETA = 0.5,
    T_ENV = 10.0,
)
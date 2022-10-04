from src.price_models.binomial.price_models import *
from src.price_models.binomial.util_matrix import *

# term structure lattice parameters
u = 1.25
d = 0.9
r0 = 0.06
q = 0.5 # 1-q = 0.5
n = 5
payoff = 100

ir_m = InterestRateModel(
    r0=r0,
    u=u,
    d=d,
    n=n,
)
# ir_m.print_lattice()

zc_price_lattice = ZCBPriceLattice(
    q=q,
    payoff=payoff,
    n=n,
    ir_model=ir_m
)
zc_price_lattice.print_lattice()
#
# fm = ForwardMatrix(
#     u=ir_m.u,
#     d=ir_m.d,
# )
#
# levels = list(range(0, n + 1, 1))
# r0_ = np.array([[ir_m.r0]])
# print(r0_.shape)
#
# for level in levels:
#     print("r0_: ", r0_)
#
#     f_mat = fm.f_mat(
#         from_level=level,
#     )
#     print("f_mat: ", f_mat)
#
#     r0_ = f_mat.dot(r0_)

# bpm = BackPropMatrix(q=0.8)
# # print(bpm.get_backward_matrix(from_level=1))
# print(bpm.get_backprop_matrix(from_level=1))

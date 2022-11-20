from src.price_models.binomial.price_models import *
from src.price_models.binomial.util_matrix import *

# term structure lattice parameters
u = 1.25
d = 0.9
r0 = 0.06
q = 0.5  # 1-q = 0.5
n = 10
payoff = 100
strike = 80.0

ir_m = InterestRateModel(
    r0=r0,
    u=u,
    d=d,
    n=n,
)
zcb = BondPricing(
    n=4,
    q=0.5,
    ir_model=ir_m,
    payoff=100,
)
# zcb.print_lattice()

cbb = BondPricing(
    n=6,
    q=0.5,
    ir_model=ir_m,
    payoff=100,
    coupon_rate=0.1,
)
# cbb.print_lattice()

# ir_m.print_lattice()

# cbbprice = CBBPriceLattice(
#     q=0.5,
#     coupon_rate=0,
#     n=10,
#     payoff=100,
#     ir_model=ir_m,
#
# )
# cbbprice.print_lattice()
#
forward_cbb = ForwardCBB(
    mat_t=4,
    cbb_model=cbb,
    ir_model=ir_m
)
# forward_cbb.print_lattice()
# print(forward_cbb.forward_price())

future_cbb = FutureCBB(
    mat_t=4,
    cbb_model=cbb,
    ir_model=ir_m
)
future_cbb.print_lattice()
# print(future_cbb.forward_price())
#
# fut_cbb = FutureCBB(
#     mat_t=4,
#     cbb_model=cbbprice,
#     ir_model=ir_m,
# )
# fut_cbb.print_lattice()
# q_ir_model = InterestRateModel(
#     r0=0.05,
#     u=1.1,
#     d=0.9,
#     n=10
# )
# zcb_lattice = ZCBPriceLattice(
#     q=0.5,
#     payoff=100,
#     n=10,
#     ir_model=q_ir_model,
# )
# # zcb_lattice.print_lattice()
# forward_zcb = ForwardZCB(
#     mat_t=3,
#     cbb_model=zcb_lattice,
#     ir_model=q_ir_model,
# )
# forward_zcb.print_lattice()
# fut_cbb.print_lattice()
# ir_m.get_lattice
# print("interest rate lattice:")
# ir_m.print_lattice()
#
# zc_price_lattice = ZCBPriceLattice(
#     q=q,
#     payoff=payoff,
#     n=n,
#     ir_model=ir_m
# )
# print("zcb lattice")
# zc_price_lattice.print_lattice()
#
# american_option_price_zcb = OptionsPricingZCB(
#     zcb_price_lattice=zc_price_lattice,
#     int_rate_model=ir_m,
#     n_period=6,
#     q=q,
#     strike_price=80,
#     is_american=True,
#     option=1,
# )
# print("#########################")
# american_option_price_zcb.print_lattice()
# european_option_price_zcb = OptionsPricingZCB(
#     zcb_price_lattice=zc_price_lattice,
#     int_rate_model=ir_m,
#     n_period=2,
#     q=q,
#     strike_price=84,
#     is_american=False,
#     option=1
#
# )
# european_option_price_zcb.print_lattice()
# european_option_price_zcb.print_lattice()
# option_price_zcb
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

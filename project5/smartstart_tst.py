import numpy as np
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingTypeConstraint
from topfarm.plotting import XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from py_wake.examples.data.iea37._iea37 import IEA37Site
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt
import numpy as np
import topfarm
import matplotlib.pyplot as plt
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import (
    MultiWFBoundaryConstraint,
    BoundaryType,
)
from topfarm.constraint_components.constraint_aggregation import (
    DistanceConstraintAggregation,
)
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import (
    PyWakeAEPCostModelComponent,
)
from topfarm.constraint_components.spacing import SpacingTypeConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from py_wake.site._site import UniformWeibullSite
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasySGDDriver
from topfarm.plotting import XYPlotComp
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.utils.gradients import autograd
from py_wake.validation.lillgrund import LillgrundSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
from py_wake.site.shear import PowerShear

# === Weibull parameters ===
n_wt = 36
f = [7.4137, 6.2499, 7.7460, 5.1357, 4.4442, 4.8912, 10.0472, 16.7183, 8.6695, 6.5091, 11.8778, 10.2974]
a = [10.12, 9.80, 9.63, 8.17, 7.54, 8.59, 10.54, 13.46, 11.60, 10.03, 12.41, 10.49]
k = [2.619, 1.908, 2.014, 1.689, 1.545, 1.564, 1.658, 2.268, 2.771, 1.850, 2.682, 2.162]

# === Skipjack Wind 1 Boundary Coordinates ===
sj_boundary = np.array([
    [512007.89, 4257596.74],
    [519193.12, 4257599.26],
    [519195.62, 4256407.31],
    [520392.67, 4256391.29],
    [520398.72, 4254006.52],
    [521603.36, 4253994.80],
    [521603.42, 4252801.63],
    [522804.37, 4252805.28],
    [522807.82, 4251585.27],
    [524005.50, 4251604.73],
    [523998.28, 4249190.31],
    [525192.69, 4249196.17],
    [525201.31, 4248001.60],
    [526392.54, 4247992.20],
    [526401.16, 4245597.81],
    [527605.04, 4245595.30],
    [527596.65, 4244396.34],
    [528792.92, 4244396.75],
    [528802.54, 4241994.71],
    [530010.83, 4242007.77],
    [529993.29, 4240801.68],
    [531192.83, 4240794.79],
    [531201.29, 4238399.20],
    [532400.43, 4238388.82],
    [532393.67, 4237195.31],
    [533599.99, 4237203.12],
    [533603.01, 4234791.20],
    [534800.14, 4234803.03],
    [534792.70, 4233602.74],
    [519207.88, 4233596.38],
    [519199.56, 4238393.08],
    [514403.43, 4238405.08],
    [514402.32, 4240794.68],
    [511999.72, 4240807.84],
    [512006.15, 4257597.62],
])
boundary_closed = np.vstack([sj_boundary, sj_boundary[0]])

# === Skipjack Wind 1 Site Definition ===
class HaliadeX12MW(GenericWindTurbine):
    def __init__(self):
        GenericWindTurbine.__init__(self, name = 'Haliade-X 12 MW', 
                                    diameter = 220, hub_height = 150, 
                                    power_norm = 12000, turbulence_intensity=0.07)
site = UniformWeibullSite(p_wd = np.array(f), a = a, k = k, ti=0.07,shear=PowerShear(h_ref=150, alpha=0.1))
windTurbines = HaliadeX12MW()
windFarmModel = IEA37SimpleBastankhahGaussian(site, windTurbines)

# === Create Boundary ===
init_types = 36 * [0]

x_min, y_min = np.min(boundary_closed, axis=0)
x_max, y_max = np.max(boundary_closed, axis=0)

x = np.linspace(x_min, x_max, 51)
y = np.linspace(y_min, y_max, 51)
YY, XX = np.meshgrid(y, x)

# === Turbine plop initiation ===
x_init = np.linspace(x_min + 1000, x_max - 1000, int(np.sqrt(n_wt)))
y_init = np.linspace(y_min + 1000, y_max - 1000, int(np.sqrt(n_wt)))
xv, yv = np.meshgrid(x_init, y_init)
xy_init = np.vstack([xv.ravel(), yv.ravel()]).T[:n_wt]

# === TopFarm Problem Definition ===
tf = TopFarmProblem(
    design_vars=dict(zip('xy', xy_init.T)),
    cost_comp=PyWakeAEPCostModelComponent(
        windFarmModel, n_wt, additional_input=[('type', np.zeros(n_wt))], grad_method=None),
    driver=EasyScipyOptimizeDriver(maxiter=500, tol=1e-6),
    constraints=[
        XYBoundaryConstraint(boundary_closed, 'polygon'),
        SpacingTypeConstraint([windTurbines.diameter() * 3.5])
    ],
    plot_comp=XYPlotComp()
)
tf['type']=init_types


cost1, state1 = tf.evaluate(dict(x=xy_init[:, 0], y=xy_init[:, 1]))
cost2, state2, recorder2 = tf.optimize()
# === Smart Start ===:
tf.smart_start(XX, YY, tf.cost_comp.get_aep4smart_start(type=init_types), seed=42)
cost3, state3 = tf.evaluate()
# === Smart Start + Opt. ===
cost4, state4, recorder4 = tf.optimize()

recorder4.save("optimization_smart_start")

# === Extract optimized layout ===
final_x = state4['x']
final_y = state4['y']
optimized_layout = np.vstack([final_x, final_y]).T

np.set_printoptions(precision=2, suppress=True)
print("optimized_layout = np.array([")
for row in optimized_layout:
    print(f"    [{row[0]:.2f}, {row[1]:.2f}],")
print("])")
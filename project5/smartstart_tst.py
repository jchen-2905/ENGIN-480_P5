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

n_wt = 36
f = [7.4137, 6.2499, 7.7460, 5.1357, 4.4442, 4.8912, 10.0472, 16.7183, 8.6695, 6.5091, 11.8778, 10.2974]
a = [10.12, 9.80, 9.63, 8.17, 7.54, 8.59, 10.54, 13.46, 11.60, 10.03, 12.41, 10.49]
k = [2.619, 1.908, 2.014, 1.689, 1.545, 1.564, 1.658, 2.268, 2.771, 1.850, 2.682, 2.162]
sj_boundary = np.array([
    [520412.37, 4273204.69],
    [534790.80, 4273205.28],
    [534807.81, 4267189.34],
    [533598.14, 4267197.49],
    [533582.77, 4262399.65],
    [532372.41, 4262403.78],
    [532431.00, 4261194.28],
    [530018.91, 4261198.21],
    [530005.86, 4262403.67],
    [528800.34, 4262394.71],
    [528804.56, 4263600.34],
    [527593.53, 4263600.08],
    [527598.87, 4264782.75],
    [526398.55, 4264787.58],
    [526412.23, 4266005.90],
    [525202.48, 4265993.02],
    [525190.53, 4268406.56],
    [523994.52, 4268407.21],
    [523990.96, 4269593.96],
    [522795.13, 4269594.68],
    [522809.89, 4270803.35],
    [521596.55, 4270773.24],
    [521593.18, 4272021.58],
    [520415.45, 4272000.40],
    [520399.01, 4273186.57],
])


boundary_closed = np.vstack([sj_boundary, sj_boundary[0]])
class HaliadeX12MW(GenericWindTurbine):
    def __init__(self):
        GenericWindTurbine.__init__(self, name = 'Haliade-X 12 MW', diameter = 220, hub_height = 150, power_norm = 12000, turbulence_intensity=0.07)

site = UniformWeibullSite(p_wd = np.array(f), a = a, k = k, ti=0.07,shear=PowerShear(h_ref=150, alpha=0.1))
windTurbines = HaliadeX12MW()
windFarmModel = IEA37SimpleBastankhahGaussian(site, windTurbines)

init_types = 36 * [0]

x_min, y_min = np.min(boundary_closed, axis=0)
x_max, y_max = np.max(boundary_closed, axis=0)

x = np.linspace(x_min, x_max, 51)
y = np.linspace(y_min, y_max, 51)
YY, XX = np.meshgrid(y, x)

# Create a simple initial layout
x_init = np.linspace(x_min + 1000, x_max - 1000, int(np.sqrt(n_wt)))
y_init = np.linspace(y_min + 1000, y_max - 1000, int(np.sqrt(n_wt)))
xv, yv = np.meshgrid(x_init, y_init)
xy_init = np.vstack([xv.ravel(), yv.ravel()]).T[:n_wt]


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

# initial layout:
cost1, state1 = tf.evaluate(dict(x=xy_init[:, 0], y=xy_init[:, 1]))
# initial layout + optimization:
cost2, state2, recorder2 = tf.optimize()
# smart start:
tf.smart_start(XX, YY, tf.cost_comp.get_aep4smart_start(type=init_types), seed=42)
cost3, state3 = tf.evaluate()
# smart start + optimization:
cost4, state4, recorder4 = tf.optimize()

recorder4.save("optimization_smart_start")

final_x = state4['x']
final_y = state4['y']
optimized_layout = np.vstack([final_x, final_y]).T

np.set_printoptions(precision=2, suppress=True)
print("optimized_layout = np.array([")
for row in optimized_layout:
    print(f"    [{row[0]:.2f}, {row[1]:.2f}],")
print("])")


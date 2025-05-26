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
from py_wake.site._site import UniformWeibullSite
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasySGDDriver
from topfarm.plotting import XYPlotComp
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.utils.gradients import autograd
from py_wake.validation.lillgrund import LillgrundSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
from py_wake.site.shear import PowerShear

# === Skipjack Wind 1 Boundary and Layout Coordinates ===
sj_utm_turbines = np.array([
    [532692.23, 4263548.94],
    [520399.01, 4273186.57],
    [529256.49, 4273205.05],
    [525202.48, 4265993.30],
    [534790.80, 4273205.28],
    [524720.77, 4273204.87],
    [530018.82, 4261198.21],
    [529051.17, 4265307.68],
    [522795.11, 4269594.68],
    [534806.55, 4267634.85],
    [532682.80, 4273205.19],
    [522409.80, 4273204.77],
    [526961.51, 4273204.96],
    [532062.51, 4261194.88],
    [527593.89, 4263600.08],
    [532951.23, 4269055.59],
    [531520.31, 4273205.15],
    [521596.55, 4270774.34],
    [528108.07, 4268076.57],
    [525926.80, 4269823.39],
    [531517.92, 4265499.97],
    [521357.84, 4273204.73],
    [533796.55, 4273205.24],
    [528800.34, 4262394.83],
    [526398.55, 4264788.14],
    [533593.10, 4265624.65],
    [525880.78, 4273204.91],
    [530935.81, 4270348.20],
    [528123.19, 4273125.77],
    [523566.52, 4273204.82],
    [533582.77, 4262399.74],
    [534798.59, 4270451.81],
    [525385.12, 4267629.41],
    [530385.99, 4267688.70],
    [524165.63, 4269588.78],
    [530377.46, 4273205.10],
])
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

# === Maryland Offshore Wind Boundary and Layout Coordinates ===

mo_utm_turbines = np.array([
    [526620.20, 4239360.51],
    [512006.16, 4257597.62],
    [511999.73, 4240828.50],
    [533599.62, 4237203.12],
    [520394.68, 4255599.15],
    [520277.29, 4240589.14],
    [512002.55, 4248202.06],
    [519207.88, 4233596.38],
    [521335.00, 4247851.66],
    [531196.19, 4239844.28],
    [516594.36, 4257598.35],
    [528198.85, 4233600.05],
    [526392.66, 4247957.76],
    [516547.25, 4241420.73],
    [512003.75, 4251317.85],
    [523591.57, 4233598.17],
    [532812.53, 4233601.93],
    [512001.22, 4244711.74],
    [517941.71, 4252264.92],
    [530010.53, 4242007.77],
    [524546.33, 4243725.81],
    [514403.44, 4238405.08],
    [524003.43, 4250911.86],
    [534793.01, 4233652.90],
    [514205.10, 4257597.51],
    [521303.43, 4233597.24],
    [516695.41, 4248216.73],
    [528694.90, 4244396.72],
    [519193.12, 4257599.26],
    [525857.37, 4233599.09],
    [517593.20, 4238397.10],
    [515595.97, 4244308.29],
    [530442.25, 4233600.96],
    [512004.84, 4254185.14],
    [522417.33, 4252804.10],
    [523096.15, 4240098.69],
])
mo_boundary = np.array([
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


# === Create Boundaries ===
sj_xinit, sj_yinit = sj_utm_turbines[:,0], sj_utm_turbines[:,1]
sj_n_wt = len(sj_utm_turbines[:,0])
sj_boundary_closed = np.vstack([sj_boundary, sj_boundary[0]])

mo_xinit, mo_yinit = mo_utm_turbines[:,0], mo_utm_turbines[:,1]
mo_n_wt = len(mo_utm_turbines[:,0])
mo_boundary_closed = np.vstack([mo_boundary, mo_boundary[0]])
maxiter = 800
tol = 1e-6

# === Combine turbine coordinates and types ===
x_init = np.concatenate([sj_xinit, mo_xinit])
y_init = np.concatenate([sj_yinit, mo_yinit])
n_wt = len(x_init)

n_sj = len(sj_xinit)
n_mo = len(mo_xinit)
n_wt = n_sj + n_mo

# === Vineyard Wind 1 Turbine and Site ===

class HaliadeX12MW(GenericWindTurbine):
    def __init__(self):
        GenericWindTurbine.__init__(self, name = 'Haliade-X 12 MW', 
                                    diameter = 220, hub_height = 150, 
                                    power_norm = 12000, turbulence_intensity=0.07)
class SkipjackWind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=PowerShear(h_ref=150, alpha=0.1)):
        f = [ 7.4137, 6.2499, 7.7460, 5.1357, 4.4442, 4.8912, 10.0472, 16.7183, 8.6695, 6.5091, 11.8778, 10.2974]
        a = [10.12, 9.80, 9.63, 8.17, 7.54, 8.59, 10.54, 13.46, 11.60, 10.03, 12.41, 10.49]
        k = [2.619, 1.908, 2.014, 1.689, 1.545, 1.564, 1.658, 2.268, 2.771, 1.850, 2.682, 2.162]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([sj_xinit, sj_yinit]).T
        self.name = "Vineyard Wind 1"
sj_wind_turbines = HaliadeX12MW()
sj_site = SkipjackWind()

wf_model = Bastankhah_PorteAgel_2014(
    sj_site,
    sj_wind_turbines,
    k=0.0324555, 
)

# === Create boolean masks ===
sj_mask = np.zeros(n_wt, dtype=bool)
mo_mask = np.zeros(n_wt, dtype=bool)

sj_mask[:n_sj] = True             # Vineyard Wind turbines
mo_mask[n_sj:n_sj + n_mo] = True  # Revolution Wind turbines

# === Print turbine indices for each wind farm ===
print(f"Turbines belonging to Vineyard Wind: {np.where(sj_mask)[0]}")
print(f"Turbines belonging to Revolution Wind: {np.where(mo_mask)[0]}")

wt_groups = [
    np.arange(0, n_sj),                  # Vineyard Wind
    np.arange(n_sj, n_wt)                # Revolution Wind
]

# === Create boundary constraint ===
constraint_comp = MultiWFBoundaryConstraint(
    geometry = [sj_boundary, mo_boundary],  
    wt_groups = wt_groups,
    boundtype = BoundaryType.POLYGON  
)

# === AEP Cost Model Component - SLSQP ===
slsqp_cost_comp = PyWakeAEPCostModelComponent(
    windFarmModel=wf_model, n_wt=n_wt, grad_method=autograd
)

# === Driver and Constraints ===
driver_type = "SLSQP"  
min_spacing = sj_wind_turbines.diameter() * 2

constraints = [
    constraint_comp,
    SpacingConstraint(min_spacing = min_spacing),
]
driver = EasyScipyOptimizeDriver(
    optimizer="SLSQP",
    maxiter=maxiter,
)
cost_comp = slsqp_cost_comp

# === Create TopFarm Problem ===
problem = TopFarmProblem(
    design_vars={"x": x_init, "y": y_init},
    n_wt=n_wt,
    constraints=constraints,
    cost_comp=cost_comp,
    driver=driver,
    plot_comp=XYPlotComp(),
)

# === lets gooo ===
cost, state, recorder = problem.optimize(disp=True)
recorder.save("testing")
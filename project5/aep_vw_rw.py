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

# === Vineyard Wind 1 Boundary and Layout Coordinates ===
vw_utm_turbines = np.array([
    [377155.25, 4554942.06],
    [378998.80, 4553109.90],
    [377157.34, 4553107.84],
    [375317.55, 4553204.50],
    [373410.30, 4551304.33],
    [375219.41, 4551306.01],
    [377094.72, 4551339.79],
    [379001.37, 4551275.25],
    [371568.49, 4549435.68],
    [373410.92, 4549436.35],
    [375253.89, 4549470.23],
    [377161.50, 4549437.98],
    [379069.65, 4549439.08],
    [380781.00, 4549476.52],
    [382654.16, 4549347.87],
    [384502.56, 4547646.95],
    [382757.89, 4547609.04],
    [380848.77, 4547574.33],
    [378971.93, 4547506.71],
    [377130.72, 4547602.99],
    [375287.30, 4547568.51],
    [373444.42, 4547567.36],
    [371634.94, 4547598.85],
    [369660.45, 4547600.95],
    [367849.61, 4545731.62],
    [369725.96, 4545730.90],
    [371569.34, 4545731.13],
    [373445.65, 4545731.25],
    [375321.30, 4545699.11],
    [377165.18, 4545733.60],
    [379008.49, 4545735.70],
    [380884.72, 4545737.73],
    [382728.52, 4545773.62],
    [384571.24, 4545744.37],
    [384541.82, 4543875.66],
    [382731.57, 4543904.38],
    [380822.03, 4543902.41],
    [378945.32, 4543900.28],
    [377068.05, 4543865.87],
    [375192.98, 4543963.17],
    [373381.57, 4543928.59],
    [371537.75, 4543927.90],
    [369660.38, 4543895.39],
    [367882.34, 4543894.45],
    [365972.63, 4543896.34],
    [367849.26, 4542058.03],
    [369758.84, 4542023.89],
    [371570.85, 4542057.47],
    [373381.70, 4542026.00],
    [375390.52, 4542024.48],
    [377168.94, 4542027.26],
    [379046.14, 4542028.95],
    [380857.49, 4542032.07],
    [382701.69, 4542035.23],
    [380827.14, 4540162.36],
    [379015.33, 4540159.24],
    [377137.65, 4540157.55],
    [375228.19, 4540222.62],
    [373448.74, 4540187.44],
    [375294.70, 4538317.92],
    [377172.87, 4538319.15],
    [378984.52, 4538289.09],
    [377108.61, 4536449.00],
])
vw_boundary = np.array([
    [375815.97, 4555192.79],
    [375815.62, 4554584.47],
    [375205.33, 4554604.86],
    [375184.95, 4553996.76],
    [374013.61, 4553996.98],
    [373982.65, 4552780.63],
    [372811.02, 4552780.92],
    [372800.28, 4551583.98],
    [371612.81, 4551568.97],
    [371626.06, 4550384.93],
    [370403.99, 4550406.47],
    [370416.45, 4549188.46],
    [369194.18, 4549210.19],
    [369206.45, 4547991.95],
    [367983.66, 4547996.90],
    [367978.69, 4546778.74],
    [366809.92, 4546803.22],
    [366787.34, 4545560.62],
    [365602.26, 4545582.25],
    [365611.23, 4544369.82],
    [364394.69, 4544392.22],
    [364372.85, 4543211.34],
    [363217.92, 4543201.69],
    [363235.70, 4540807.02],
    [366763.44, 4540804.33],
    [366772.08, 4539560.06],
    [370425.72, 4539588.08],
    [370414.70, 4537192.94],
    [372756.39, 4537152.03],
    [372793.81, 4535992.48],
    [373990.51, 4535971.88],
    [373994.60, 4533586.53],
    [376404.50, 4533590.40],
    [376394.57, 4534783.30],
    [377592.19, 4534807.91],
    [377611.54, 4535970.34],
    [378809.29, 4536010.13],
    [378813.71, 4537187.46],
    [379987.41, 4537189.63],
    [380007.17, 4538400.02],
    [381195.12, 4538380.71],
    [381214.26, 4539564.01],
    [382375.37, 4539572.19],
    [382421.74, 4540781.70],
    [383610.15, 4540816.51],
    [383575.46, 4541969.90],
    [384188.76, 4541999.62],
    [384197.46, 4542550.91],
    [384810.71, 4542580.57],
    [384795.14, 4550406.31],
    [380030.05, 4550413.58],
    [380004.19, 4555194.22],
    [375814.08, 4555194.93],
])

# === Revolution Wind Boundary and Layout Coordinates ===

rw_utm_turbines = np.array([
    [317885.88, 4566134.19],
    [319711.68, 4566119.54],
    [321569.99, 4566166.31],
    [323425.93, 4566121.08],
    [325221.48, 4566139.43],
    [327109.05, 4566125.26],
    [327095.37, 4564244.57],
    [325269.93, 4564288.11],
    [323381.85, 4564302.74],
    [321585.06, 4564253.97],
    [317871.36, 4564315.06],
    [315982.42, 4564300.74],
    [314170.84, 4562465.07],
    [315996.84, 4562418.76],
    [317856.85, 4562495.49],
    [319714.60, 4562480.09],
    [321571.49, 4562434.42],
    [323398.93, 4562451.70],
    [323385.08, 4560600.86],
    [321526.98, 4560615.19],
    [319730.01, 4560597.71],
    [317840.83, 4560613.75],
    [315950.16, 4560568.66],
    [314155.44, 4560644.93],
    [310450.47, 4558734.12],
    [312278.22, 4558717.78],
    [314168.66, 4558731.07],
    [316151.96, 4558742.63],
    [317856.56, 4558761.67],
    [319714.50, 4558715.55],
    [321573.85, 4558731.52],
    [323402.21, 4558748.83],
    [325260.02, 4558704.13],
    [327149.52, 4558690.03],
    [329008.84, 4558708.00],
    [330898.29, 4558694.76],
    [332727.22, 4558745.27],
    [334492.08, 4558705.04],
    [325186.23, 4556915.63],
    [325358.02, 4555028.07],
    [327093.59, 4555017.68],
    [328995.74, 4556825.45],
    [329044.61, 4554940.91],
    [327142.82, 4553163.64],
    [328879.53, 4553184.56],
    [327066.51, 4551250.26],
    [329053.86, 4551357.97],
    [327054.55, 4549427.56],
    [329010.58, 4549505.25],
    [330748.13, 4549526.64],
    [330850.81, 4551285.33],
    [332623.13, 4551491.72],
    [332679.50, 4553066.86],
    [332803.74, 4549480.63],
    [334498.76, 4549442.12],
    [334494.68, 4551291.20],
    [336401.89, 4549490.76],
    [336351.64, 4551340.74],
    [338201.46, 4553147.99],
    [340075.36, 4555046.00],
    [341974.45, 4554981.81],
    [343807.41, 4555033.71],
    [345546.34, 4554973.98],
    [345599.88, 4553170.64],
    [343790.87, 4553186.18],
    [341958.93, 4553202.77],
    [340103.14, 4553174.70],
])
rw_boundary = np.array([
    [321204.62, 4571972.15],
    [322372.53, 4571994.06],
    [322340.51, 4569643.27],
    [327187.50, 4569577.18],
    [327198.19, 4563606.38],
    [323615.54, 4563591.00],
    [323601.48, 4558806.22],
    [328381.24, 4558818.93],
    [328433.44, 4559957.20],
    [336771.48, 4560019.08],
    [336790.46, 4555181.27],
    [346377.31, 4555150.73],
    [346379.23, 4552819.69],
    [339179.53, 4552797.08],
    [339179.36, 4551631.26],
    [337982.38, 4551581.58],
    [337956.57, 4550416.09],
    [336837.34, 4550440.96],
    [336758.27, 4549175.02],
    [333195.43, 4549204.55],
    [333193.18, 4547987.41],
    [326014.62, 4548027.55],
    [325989.11, 4553377.85],
    [315151.44, 4553415.59],
    [315172.36, 4549206.29],
    [314002.64, 4549261.23],
    [313981.35, 4550428.26],
    [312784.43, 4550408.12],
    [312790.10, 4551625.03],
    [310398.65, 4551661.27],
    [310388.83, 4555210.33],
    [317561.76, 4555179.85],
    [317582.85, 4553988.20],
    [319976.00, 4554055.64],
    [320012.47, 4557601.96],
    [310375.71, 4557643.18],
    [310405.95, 4558807.71],
    [311576.76, 4558853.39],
    [311606.81, 4560017.70],
    [312776.75, 4560038.25],
    [312781.86, 4561228.31],
    [313977.65, 4561273.60],
    [314010.97, 4563576.56],
    [315180.32, 4563597.47],
    [315185.70, 4564812.28],
    [316380.24, 4564832.63],
    [316410.19, 4566021.26],
    [317578.48, 4566017.23],
    [317582.31, 4567180.96],
    [318776.97, 4567227.01],
    [318836.13, 4569603.27],
    [319979.78, 4569650.76],
    [319983.16, 4570813.93],
    [321176.66, 4570835.03],
    [321205.86, 4572022.66],
])

# === Create Boundaries ===
vw_xinit, vw_yinit = vw_utm_turbines[:,0], vw_utm_turbines[:,1]
vw_n_wt = len(vw_utm_turbines[:,0])
vw_boundary_closed = np.vstack([rw_boundary, rw_boundary[0]])

rw_xinit, rw_yinit = rw_utm_turbines[:,0], rw_utm_turbines[:,1]
rw_n_wt = len(rw_utm_turbines[:,0])
rw_boundary_closed = np.vstack([rw_boundary, rw_boundary[0]])
maxiter = 800
tol = 1e-6

# === Combine turbine coordinates and types ===
x_init = np.concatenate([vw_xinit, rw_xinit])
y_init = np.concatenate([vw_yinit, rw_yinit])
n_wt = len(x_init)

n_vw = len(vw_xinit)
n_rw = len(rw_xinit)
n_wt = n_vw + n_rw
print(f"Initial layout has {n_wt} wind turbines")


# === Vineyard Wind 1 Turbine and Site ===

class HaliadeX13MW(GenericWindTurbine):
    def __init__(self):
        GenericWindTurbine.__init__(self, name = 'Haliade-X 13 MW', diameter = 200, hub_height = 133, power_norm = 13000, turbulence_intensity=0.07)
class VineyardWind1(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.4452, 7.6731, 6.4753, 6.0399, 4.8786, 4.5063, 7.3180, 11.7828, 13.0872,  11.1976, 11.1351, 9.4610]
        a = [10.26, 10.44, 9.52, 8.96, 9.58, 9.72, 11.48, 13.25, 12.46, 11.40, 12.35, 10.48]
        k = [2.225, 1.697, 1.721, 1.689, 1.525, 1.498, 1.686, 2.143, 2.369, 2.186, 2.385, 2.404]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([vw_xinit, vw_yinit]).T
        self.name = "Vineyard Wind 1"
vw_wind_turbines = HaliadeX13MW()
vw_site = VineyardWind1()

wf_model = Bastankhah_PorteAgel_2014(
    vw_site,
    vw_wind_turbines,
    k=0.0324555,  # default value from BastankhahGaussianDeficit
)

# === Create boolean masks ===
vw_mask = np.zeros(n_wt, dtype=bool)
rw_mask = np.zeros(n_wt, dtype=bool)

vw_mask[:n_vw] = True             # Vineyard Wind turbines
rw_mask[n_vw:n_vw + n_rw] = True  # Revolution Wind turbines

# Print turbine indices for each wind farm
print(f"Turbines belonging to Vineyard Wind: {np.where(vw_mask)[0]}")
print(f"Turbines belonging to Revolution Wind: {np.where(rw_mask)[0]}")

wt_groups = [
    np.arange(0, n_vw),                  # Vineyard Wind
    np.arange(n_vw, n_wt)                # Revolution Wind
]

# === Create boundary constraint ===
constraint_comp = MultiWFBoundaryConstraint(
    geometry = [vw_boundary, rw_boundary],  
    wt_groups = wt_groups,
    boundtype = BoundaryType.POLYGON  
)


fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(x_init, y_init, "x", c="magenta")

plt.axis("equal")
constraint_comp.get_comp(n_wt).plot(ax1)
plt.show()

# AEP Cost Model Component - SLSQP
slsqp_cost_comp = PyWakeAEPCostModelComponent(
    windFarmModel=wf_model, n_wt=n_wt, grad_method=autograd
)

driver_type = "SLSQP"  
min_spacing = vw_wind_turbines.diameter() * 2

constraints = [
    constraint_comp,
    SpacingConstraint(min_spacing = min_spacing),
]
driver = EasyScipyOptimizeDriver(
    optimizer="SLSQP",
    maxiter=maxiter,
)
cost_comp = slsqp_cost_comp

problem = TopFarmProblem(
    design_vars={"x": x_init, "y": y_init},
    n_wt=n_wt,
    constraints=constraints,
    cost_comp=cost_comp,
    driver=driver,
    plot_comp=XYPlotComp(),
)

cost, state, recorder = problem.optimize(disp=True)
recorder.save("testing")
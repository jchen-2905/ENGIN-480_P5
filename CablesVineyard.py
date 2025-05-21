import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm._topfarm import TopFarmProblem, TopFarmGroup
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.plotting import XYPlotComp
from topfarm.utils import plot_list_recorder
from topfarm.cost_models.economic_models.dtu_wind_cm_main import economic_evaluation
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake import BastankhahGaussian
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from ed_win.wind_farm_network import WindFarmNetwork
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# Load your GeoJSON cable files

geojson_files = [
    "/Users/a18573/VSCode/project 5/ENGIN-480_P5/Project 5 new/ENGIN-480_P5/cables/line.geojson",
    "/Users/a18573/VSCode/project 5/ENGIN-480_P5/Project 5 new/ENGIN-480_P5/cables/line(1).geojson",
    "/Users/a18573/VSCode/project 5/ENGIN-480_P5/Project 5 new/ENGIN-480_P5/cables/line(2).geojson",
    "/Users/a18573/VSCode/project 5/ENGIN-480_P5/Project 5 new/ENGIN-480_P5/cables/line(3).geojson",
    "/Users/a18573/VSCode/project 5/ENGIN-480_P5/Project 5 new/ENGIN-480_P5/cables/line(4).geojson",
    "/Users/a18573/VSCode/project 5/ENGIN-480_P5/Project 5 new/ENGIN-480_P5/cables/line(5).geojson",
    "/Users/a18573/VSCode/project 5/ENGIN-480_P5/Project 5 new/ENGIN-480_P5/cables/point.geojson"

]

gdfs = [gpd.read_file(file) for file in geojson_files]
merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

# Extracting turbines
points_gdf = gpd.read_file("/Users/a18573/VSCode/project 5/ENGIN-480_P5/Project 5 new/ENGIN-480_P5/cables/point.geojson")
initial = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])

points_gdf = points_gdf.to_crs(epsg=32619)  # UTM coordinates
initial = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])


x_init = initial[:, 0]
y_init = initial[:, 1]
n_wt = len(initial)

buffer = 1000  # meters
min_x, max_x = x_init.min() - buffer, x_init.max() + buffer
min_y, max_y = y_init.min() - buffer, y_init.max() + buffer

boundary = np.array([
    [min_x, min_y],
    [max_x, min_y],
    [max_x, max_y],
    [min_x, max_y]
])
driver = EasyScipyOptimizeDriver(maxiter=10)
drivers = [driver]

windTurbines = IEA37_WindTurbines()
site = Hornsrev1Site()
wfm = BastankhahGaussian(site, windTurbines)

sigma = 3000.0
mu = 0.0

x_peak_1 = x_init.mean()
y_peak_1 = y_init.mean()
x_peak_2 = x_init.max() + 1000
y_peak_2 = y_init.min() - 1000

x1, y1 = np.meshgrid(np.linspace(x_init.min() - 1000, x_init.max() + 1000, 100),
                     np.linspace(y_init.min() - 1000, y_init.max() + 1000, 100))
d1 = np.sqrt((x1 - x_peak_1)**2 + (y1 - y_peak_1)**2)
g1 = np.exp(-((d1 - mu)**2 / (2.0 * sigma**2)))

x2, y2 = np.meshgrid(np.linspace(x_init.min() - 1000, x_init.max() + 1000, 100),
                     np.linspace(y_init.min() - 1000, y_init.max() + 1000, 100))
d2 = np.sqrt((x2 - x_peak_2)**2 + (y2 - y_peak_2)**2)
g2 = np.exp(-((d2 - mu)**2 / (2.0 * sigma**2)))

g = 5 * g1 - 8 * g2 - 30

plt.imshow(g, extent=(x_init.min() - 1000, x_init.max() + 1000,
                      y_init.min() - 1000, y_init.max() + 1000),
           origin='lower', cmap='viridis')
plt.colorbar()
plt.title('2D Gaussian Function')

x = np.linspace(x_init.min() - 1000, x_init.max() + 1000, 100)
y = np.linspace(y_init.min() - 1000, y_init.max() + 1000, 100)
f = RegularGridInterpolator((x, y), g)


x_ss_init = x_init.mean()
y_ss_init = y_init.mean()
turbines_pos = np.asarray([x_init, y_init]).T
substations_pos = np.asarray([[x_ss_init], [y_ss_init]]).T


cables = np.array([[500, 3, 100], [800, 5, 150], [1000, 10, 250]])
wfn = WindFarmNetwork(turbines_pos=turbines_pos, substations_pos=substations_pos, cables=cables)
G = wfn.optimize(turbines_pos)
cable_cost_ref = G.size(weight="cost")
cable_length_ref = G.size(weight="length")
cost_per_length_ref = cable_cost_ref / cable_length_ref
G.plot()
plt.show()

Drotor_vector = [windTurbines.diameter()] * n_wt
power_rated_vector = [float(windTurbines.power(20))*1e-6] * n_wt
hub_height_vector = [windTurbines.hub_height()] * n_wt

distance_from_shore = 30
energy_price = 0.1
project_duration = 25
rated_rpm_array = [12] * n_wt
simres = wfm(x_init, y_init)
aep = simres.aep().values.sum()
CF = aep / (windTurbines.power(20)*1e-9 * 24*365*n_wt)

eco_eval = economic_evaluation(distance_from_shore, energy_price, project_duration)
npv_ref = eco_eval.calculate_npv(rated_rpm_array, Drotor_vector, power_rated_vector, hub_height_vector, 30, aep/n_wt * np.ones(n_wt)*10**6, cabling_cost=cable_cost_ref)

def cable_func(x, y, x_substation, y_substation, **kwargs):
    G = wfn.optimize(turbines_pos= np.asarray([x, y]).T,
                     substations_pos=np.asarray([[float(x_substation[0])], [float(y_substation[0])]]).T)
    return G.size(weight="cost"), {'cabling_length': G.size(weight="length")}

def npv_func(AEP, water_depth, cabling_cost, **kwargs):
    eco_eval.calculate_npv(rated_rpm_array, Drotor_vector, power_rated_vector, hub_height_vector, water_depth,
                           AEP/n_wt * np.ones(n_wt)*10**6, cabling_cost=cabling_cost)
    eco_eval.calculate_irr(rated_rpm_array, Drotor_vector, power_rated_vector, hub_height_vector, water_depth,
                           AEP/n_wt * np.ones(n_wt)*10**6, cabling_cost=cabling_cost)
    return eco_eval.NPV, {'irr': eco_eval.IRR,
                          'OPEX': eco_eval.project_costs_sums["OPEX"],
                          'CAPEX': eco_eval.project_costs_sums["CAPEX"]}

cable_component = CostModelComponent(input_keys=[('x', x_init), ('y', y_init),
                                                 ('x_substation', x_ss_init), ('y_substation', y_ss_init)],
                                     n_wt=n_wt,
                                     cost_function=cable_func,
                                     objective=False,
                                     output_keys=[('cabling_cost', 0)],
                                     additional_output=[('cabling_length', 0)])

npv_comp = CostModelComponent(input_keys=[('AEP', 0), ('water_depth', 30*np.ones(n_wt)),
                                          ('cabling_cost', 100000)],
                              n_wt=n_wt,
                              cost_function=npv_func,
                              objective=True,
                              maximize=True,
                              output_keys=[('npv', 0)],
                              additional_output=[('irr', 0), ('CAPEX', 0), ('OPEX', 0)])

cost_comp = TopFarmGroup([PyWakeAEPCostModelComponent(wfm, n_wt, objective=False),
                          cable_component,
                          npv_comp])

tf = TopFarmProblem(
    design_vars=dict(zip('xy', initial.T), x_substation=x_ss_init, y_substation=y_ss_init),
    cost_comp=cost_comp,
    constraints=[XYBoundaryConstraint(boundary),
                 SpacingConstraint(500)],
    driver=drivers[0],
    plot_comp=XYPlotComp()
)

cost, _, recorder = tf.optimize()

x_opt = recorder['x'][-1]
y_opt = recorder['y'][-1]
x_sub_opt = recorder['x_substation'][-1]
y_sub_opt = recorder['y_substation'][-1]
G = wfn.optimize(np.asarray([x_opt, y_opt]).T,
                 substations_pos=np.asarray([[float(x_sub_opt)], [float(y_sub_opt)]]).T)
G.plot()
plt.show()
plot_list_recorder(recorder)

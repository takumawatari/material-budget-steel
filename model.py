import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import weibull_min
from pulp import LpProblem, LpVariable, LpMinimize, LpMaximize, LpContinuous, lpSum, lpDot, value
from ortoolpy import model_min, addvars, addvals
import matplotlib.pyplot as plt

def compute_export(production, share):
    export = production * share
    return export

def compute_scrap(production, export, yields):
    scrap = (production - export) * (1 - yields)
    return scrap

def compute_domestic_use(production, export, scrap):
    use = production - export - scrap
    return use

def compute_semi(use_pri, use_sec, use_hyd, share_pri, share_sec):
    semi  = use_pri * share_pri  + use_sec * share_sec  + use_hyd * share_pri
    return semi

def compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long, matrix_flat, matrix_tube, matrix_alloy):
    products =  use_long * matrix_long + use_flat * matrix_flat + use_tube * matrix_tube + use_alloy * matrix_alloy
    return products

# inflow-driven dynamic stock model
def compute_outflow(inflow, shape, scale):
    year_complete = np.arange(1950,1951)
    outflow = np.repeat(0,len(year_complete))
    for k in range(1951,2051):
        outflow_list = (inflow.iloc[0:len(year_complete)] *(weibull_min.pdf(k - year_complete,
                                                                     c = shape.iloc[0:len(year_complete)],
                                                                     scale = scale.iloc[0:len(year_complete)])))
        outflow_sum = sum(outflow_list)
        outflow = np.append(outflow, outflow_sum)
        year_complete = np.append(year_complete,k)
    return outflow

# material budget model
def material_budget_model(sheet_name):
    Data = pd.read_excel (io = r'data_inputs.xlsx', sheet_name = sheet_name, nrows=101)
    matrix_long = pd.read_excel (io = r'data_inputs.xlsx', sheet_name='matrix_long')
    matrix_flat = pd.read_excel (io = r'data_inputs.xlsx', sheet_name='matrix_flat')
    matrix_tube = pd.read_excel (io = r'data_inputs.xlsx', sheet_name='matrix_tube')
    matrix_alloy = pd.read_excel (io = r'data_inputs.xlsx', sheet_name='matrix_alloy')
    
    # system variables
    Data['Var_pri'] = addvars(len(Data))
    Data['Var_sec'] = addvars(len(Data))
    Data['Var_hyd'] = addvars(len(Data))
    
    # net-exports of ingots and semis
    export_pri = compute_export(Data.Var_pri , Data.share_export_pri)
    export_sec = compute_export(Data.Var_sec , Data.share_export_sec)
    export_hyd = compute_export(Data.Var_hyd , Data.share_export_pri)
    
    # forming scrap
    forming_pri = compute_scrap(Data.Var_pri, export_pri, Data.forming_yield_pri)
    forming_sec = compute_scrap(Data.Var_sec, export_sec, Data.forming_yield_sec)
    forming_hyd = compute_scrap(Data.Var_hyd, export_hyd, Data.forming_yield_pri)
    
    # domestic production of finished steel
    use_pri = compute_domestic_use(Data.Var_pri, export_pri, forming_pri)
    use_sec = compute_domestic_use(Data.Var_sec, export_sec, forming_sec)
    use_hyd = compute_domestic_use(Data.Var_hyd, export_hyd, forming_hyd)
    
    pro_long = compute_semi(use_pri, use_sec, use_hyd, Data.share_long_pri, Data.share_long_sec)
    pro_flat = compute_semi(use_pri, use_sec, use_hyd, Data.share_flat_pri, Data.share_flat_sec)
    pro_tube = compute_semi(use_pri, use_sec, use_hyd, Data.share_tube_pri, Data.share_tube_sec)
    pro_alloy = compute_semi(use_pri, use_sec, use_hyd, Data.share_alloy_pri, Data.share_alloy_sec)
    
    # net-exports of finished steel
    export_long = compute_export(pro_long, Data.share_export_long)
    export_flat = compute_export(pro_flat, Data.share_export_flat)
    export_tube = compute_export(pro_tube, Data.share_export_tube)
    export_alloy = compute_export(pro_alloy, Data.share_export_alloy)
    
    # fabrication yield
    fabrication_long = compute_scrap(pro_long, export_long, Data.fabrication_yield_long)
    fabrication_flat = compute_scrap(pro_flat, export_flat, Data.fabrication_yield_flat)
    fabrication_tube = compute_scrap(pro_tube, export_tube, Data.fabrication_yield_tube)
    fabrication_alloy = compute_scrap(pro_alloy, export_alloy, Data.fabrication_yield_alloy)
    
    # domestic production of end-use goods
    use_long = compute_domestic_use(pro_long, export_long, fabrication_long)
    use_flat = compute_domestic_use(pro_flat, export_flat, fabrication_flat)
    use_tube = compute_domestic_use(pro_tube, export_tube, fabrication_tube)
    use_alloy = compute_domestic_use(pro_alloy, export_alloy, fabrication_alloy)
    
    pro_BU = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.BU, matrix_flat.BU, matrix_tube.BU, matrix_alloy.BU)
    pro_IF = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.IF, matrix_flat.IF, matrix_tube.IF, matrix_alloy.IF)
    pro_MM = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.MM, matrix_flat.MM, matrix_tube.MM, matrix_alloy.MM)
    pro_EE = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.EE, matrix_flat.EE, matrix_tube.EE, matrix_alloy.EE)
    pro_AU = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.AU, matrix_flat.AU, matrix_tube.AU, matrix_alloy.AU)
    pro_OT = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.OT, matrix_flat.OT, matrix_tube.OT, matrix_alloy.OT)
    pro_MP = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.MP, matrix_flat.MP, matrix_tube.MP, matrix_alloy.MP)

    # inflow
    in_BU = pro_BU - compute_export(pro_BU, Data.share_export_BU)
    in_IF = pro_IF - compute_export(pro_IF, Data.share_export_IF)
    in_MM = pro_MM - compute_export(pro_MM, Data.share_export_MM)
    in_EE = pro_EE - compute_export(pro_EE, Data.share_export_EE)
    in_AU = pro_AU - compute_export(pro_AU, Data.share_export_AU)
    in_OT = pro_OT - compute_export(pro_OT, Data.share_export_OT)
    in_MP = pro_MP - compute_export(pro_MP, Data.share_export_MP)

    # outflow
    ot_BU = compute_outflow(in_BU, Data.shape_BU, Data.scale_BU)
    ot_IF = compute_outflow(in_IF, Data.shape_IF, Data.scale_IF)
    ot_MM = compute_outflow(in_MM, Data.shape_MM, Data.scale_MM)
    ot_EE = compute_outflow(in_EE, Data.shape_EE, Data.scale_EE)
    ot_AU = compute_outflow(in_AU, Data.shape_AU, Data.scale_AU)
    ot_OT = compute_outflow(in_OT, Data.shape_OT, Data.scale_OT)
    ot_MP = compute_outflow(in_MP, Data.shape_MP, Data.scale_MP)
        
    # end-of-life scrap
    EOL_BU = ot_BU * (1 - Data.hiber_BU)
    EOL_IF = ot_IF * (1 - Data.hiber_IF)
    EOL_MM = ot_MM * (1 - Data.hiber_MM)
    EOL_EE = ot_EE * (1 - Data.hiber_EE)
    EOL_AU = ot_AU * (1 - Data.hiber_AU)
    EOL_OT = ot_OT * (1 - Data.hiber_OT)
    EOL_MP = ot_MP * (1 - Data.hiber_MP)

    # scrap inputs to EAF
    EOL_sec= (EOL_BU + EOL_IF + EOL_MM + EOL_EE + EOL_AU + EOL_OT + EOL_MP) * Data.domestic_rec
    fabrication_sec = fabrication_long + fabrication_flat + fabrication_tube + fabrication_alloy
    scrap_sec = (EOL_sec * Data.scrap_prep_eol + 
                 fabrication_sec *  Data.scrap_prep_fabrication + 
                 forming_sec * Data.scrap_prep_forming) * Data.sec_yield

    # CO2 emissions
    CO2 = (Data.Var_pri * Data.EI_pri) + (Data.Var_sec * Data.EI_sec) + (Data.Var_hyd * Data.EI_hyd)
    
    # inflow
    in_matrix = pd.DataFrame({0:in_BU, 1:in_IF, 2:in_MM, 3:in_EE, 4:in_AU, 5:in_OT, 6:in_MP,},)

    # outflow
    ot_matrix = pd.DataFrame({0:ot_BU, 1:ot_IF, 2:ot_MM, 3:ot_EE, 4:ot_AU, 5:ot_OT, 6:ot_MP,},)

    # stock
    NAS_matrix = in_matrix - ot_matrix
    st_matrix  = NAS_matrix.cumsum()
    st_sum     = st_matrix.sum(axis = 1)

    model = LpProblem(sense = LpMaximize)
    model += lpSum(st_sum)
    
    for t in Data.index:
        
        # CO2 emission constraints
        model += CO2[t] <= Data.budget[t]
    
        # scrap availability constraints
        model += Data.Var_sec[t] <= scrap_sec[t]
        
        # H2-DRI/EAF capacity constraints
        model += Data.Var_hyd[t] <= Data.hyd_cap[t]
    
        # non-negative constraints
        model += Data.Var_pri[t] >= 0
        model += Data.Var_sec[t] >= 0
        model += Data.Var_hyd[t] >= 0
        
    model.solve()
    Data['Val_pri'] = Data.Var_pri.apply(value)
    Data['Val_sec'] = Data.Var_sec.apply(value)
    Data['Val_hyd'] = Data.Var_hyd.apply(value)

########################################################################
# calculations with optimized numbers
########################################################################

    # net-exports of ingots and semis
    export_pri = compute_export(Data.Val_pri , Data.share_export_pri)
    export_sec = compute_export(Data.Val_sec , Data.share_export_sec)
    export_hyd = compute_export(Data.Val_hyd , Data.share_export_pri)
    
    # forming scrap
    forming_pri = compute_scrap(Data.Val_pri, export_pri, Data.forming_yield_pri)
    forming_sec = compute_scrap(Data.Val_sec, export_sec, Data.forming_yield_sec)
    forming_hyd = compute_scrap(Data.Val_hyd, export_hyd, Data.forming_yield_pri)
    
    # domestic production of finished steel
    use_pri = compute_domestic_use(Data.Val_pri, export_pri, forming_pri)
    use_sec = compute_domestic_use(Data.Val_sec, export_sec, forming_sec)
    use_hyd = compute_domestic_use(Data.Val_hyd, export_hyd, forming_hyd)
    
    pro_long = compute_semi(use_pri, use_sec, use_hyd, Data.share_long_pri, Data.share_long_sec)
    pro_flat = compute_semi(use_pri, use_sec, use_hyd, Data.share_flat_pri, Data.share_flat_sec)
    pro_tube = compute_semi(use_pri, use_sec, use_hyd, Data.share_tube_pri, Data.share_tube_sec)
    pro_alloy = compute_semi(use_pri, use_sec, use_hyd, Data.share_alloy_pri, Data.share_alloy_sec)
    
    # net-exports of finished steel
    export_long = compute_export(pro_long, Data.share_export_long)
    export_flat = compute_export(pro_flat, Data.share_export_flat)
    export_tube = compute_export(pro_tube, Data.share_export_tube)
    export_alloy = compute_export(pro_alloy, Data.share_export_alloy)
    
    # fabrication yield
    fabrication_long = compute_scrap(pro_long, export_long, Data.fabrication_yield_long)
    fabrication_flat = compute_scrap(pro_flat, export_flat, Data.fabrication_yield_flat)
    fabrication_tube = compute_scrap(pro_tube, export_tube, Data.fabrication_yield_tube)
    fabrication_alloy = compute_scrap(pro_alloy, export_alloy, Data.fabrication_yield_alloy)
    
    # domestic production of end-use goods
    use_long = compute_domestic_use(pro_long, export_long, fabrication_long)
    use_flat = compute_domestic_use(pro_flat, export_flat, fabrication_flat)
    use_tube = compute_domestic_use(pro_tube, export_tube, fabrication_tube)
    use_alloy = compute_domestic_use(pro_alloy, export_alloy, fabrication_alloy)
    
    pro_BU = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.BU, matrix_flat.BU, matrix_tube.BU, matrix_alloy.BU)
    pro_IF = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.IF, matrix_flat.IF, matrix_tube.IF, matrix_alloy.IF)
    pro_MM = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.MM, matrix_flat.MM, matrix_tube.MM, matrix_alloy.MM)
    pro_EE = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.EE, matrix_flat.EE, matrix_tube.EE, matrix_alloy.EE)
    pro_AU = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.AU, matrix_flat.AU, matrix_tube.AU, matrix_alloy.AU)
    pro_OT = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.OT, matrix_flat.OT, matrix_tube.OT, matrix_alloy.OT)
    pro_MP = compute_products(use_long, use_flat, use_tube, use_alloy, matrix_long.MP, matrix_flat.MP, matrix_tube.MP, matrix_alloy.MP)

    # inflow
    in_BU = pro_BU - compute_export(pro_BU, Data.share_export_BU)
    in_IF = pro_IF - compute_export(pro_IF, Data.share_export_IF)
    in_MM = pro_MM - compute_export(pro_MM, Data.share_export_MM)
    in_EE = pro_EE - compute_export(pro_EE, Data.share_export_EE)
    in_AU = pro_AU - compute_export(pro_AU, Data.share_export_AU)
    in_OT = pro_OT - compute_export(pro_OT, Data.share_export_OT)
    in_MP = pro_MP - compute_export(pro_MP, Data.share_export_MP)

    # outflow
    ot_BU = compute_outflow(in_BU, Data.shape_BU, Data.scale_BU)
    ot_IF = compute_outflow(in_IF, Data.shape_IF, Data.scale_IF)
    ot_MM = compute_outflow(in_MM, Data.shape_MM, Data.scale_MM)
    ot_EE = compute_outflow(in_EE, Data.shape_EE, Data.scale_EE)
    ot_AU = compute_outflow(in_AU, Data.shape_AU, Data.scale_AU)
    ot_OT = compute_outflow(in_OT, Data.shape_OT, Data.scale_OT)
    ot_MP = compute_outflow(in_MP, Data.shape_MP, Data.scale_MP)
        
    # end-of-life scrap
    EOL_BU = ot_BU * (1 - Data.hiber_BU)
    EOL_IF = ot_IF * (1 - Data.hiber_IF)
    EOL_MM = ot_MM * (1 - Data.hiber_MM)
    EOL_EE = ot_EE * (1 - Data.hiber_EE)
    EOL_AU = ot_AU * (1 - Data.hiber_AU)
    EOL_OT = ot_OT * (1 - Data.hiber_OT)
    EOL_MP = ot_MP * (1 - Data.hiber_MP)

    # scrap inputs to EAF
    EOL_sec= (EOL_BU + EOL_IF + EOL_MM + EOL_EE + EOL_AU + EOL_OT + EOL_MP) * Data.domestic_rec
    fabrication_sec = fabrication_long + fabrication_flat + fabrication_tube + fabrication_alloy
    scrap_sec = (EOL_sec * Data.scrap_prep_eol + 
                 fabrication_sec *  Data.scrap_prep_fabrication + 
                 forming_sec * Data.scrap_prep_forming) * Data.sec_yield
    scrap = pd.DataFrame({'EOL': EOL_sec * Data.scrap_prep_eol,
                          'Fabrication': fabrication_sec *  Data.scrap_prep_fabrication,
                          'Forming': forming_sec * Data.scrap_prep_forming},)
    
    # CO2 emissions
    pro_couse50 = Data.Val_pri * Data.course50
    pro_BF = Data.Val_pri - pro_couse50
    
    production = pd.DataFrame({'BF/BOF': pro_BF,
                               'H$_2$-BF/BOF+CCS': pro_couse50,
                               'H$_2$-DRI/EAF': Data.Val_hyd,
                               'Scrap-EAF': Data.Val_sec},)
    
    CO2 = pd.DataFrame({'BF/BOF': Data.Val_pri * Data.EI_pri,
                        'H$_2$-DRI/EAF': Data.Val_hyd * Data.EI_hyd,
                        'Scrap-EAF': Data.Val_sec * Data.EI_sec},)
    
    # domestic production of finished steel
    steel_matrix = pd.DataFrame({'Carbon steel (long)': pro_long,
                                 'Carbon steel (flat)': pro_flat,
                                 'Carbon steel (tube)': pro_tube,
                                 'Alloy steel': pro_alloy,},)
    
    # domestic production of end-use goods
    goods_matrix = pd.DataFrame({'Buildings': pro_BU,
                                 'Infrastructure': pro_IF,
                                 'Mechanical machinery': pro_MM,
                                 'Electrical equipment': pro_EE,
                                 'Automobiles': pro_AU,
                                 'Other transport': pro_OT,
                                 'Metal products': pro_MP,},)
    
    # inflow
    in_matrix = pd.DataFrame({'Buildings': in_BU,
                              'Infrastructure': in_IF,
                              'Mechanical machinery':in_MM,
                              'Electrical equipment':in_EE,
                              'Automobiles': in_AU,
                              'Other transport': in_OT,
                              'Metal products': in_MP,},)

    # outflow
    ot_matrix = pd.DataFrame({'Buildings': ot_BU,
                              'Infrastructure': ot_IF,
                              'Mechanical machinery': ot_MM,
                              'Electrical equipment': ot_EE,
                              'Automobiles': ot_AU,
                              'Other transport': ot_OT,
                              'Metal products': ot_MP,},)

    # stock
    NAS_matrix = in_matrix - ot_matrix
    st_matrix = NAS_matrix.cumsum()
    
    production = production.rename(index = Data.year)
    steel_matrix = steel_matrix.rename(index = Data.year)
    goods_matrix = goods_matrix.rename(index = Data.year)
    CO2 = CO2.rename(index = Data.year)
    in_matrix = in_matrix.rename(index = Data.year)
    ot_matrix = ot_matrix.rename(index = Data.year)
    st_matrix = st_matrix.rename(index = Data.year)
    scrap = scrap.rename(index = Data.year)
    
    scenario_index = sheet_name
    
    # figures
    ax=production.plot.area(cmap="viridis", figsize=(4,4))
    ax.axes.set_xlim(2010,2050)
    ax.axes.set_ylim(0,120000)
    ax.set_ylabel("Crude steel production (kt/yr)",fontsize=11)
    ax.set_title(str(scenario_index),fontsize=11)
    
    # results
    with pd.ExcelWriter('outputs'+'/Scenario_'+str(scenario_index) + '.xlsx') as writer:
        production.to_excel(writer, sheet_name='crude_steel')
        steel_matrix.to_excel(writer, sheet_name='pro_steel')
        goods_matrix.to_excel(writer, sheet_name='pro_goods')
        CO2.to_excel(writer, sheet_name='CO2')
        in_matrix.to_excel(writer, sheet_name='inflow')
        ot_matrix.to_excel(writer, sheet_name='outflow')
        st_matrix.to_excel(writer, sheet_name='stock')
        scrap.to_excel(writer, sheet_name='scrap')
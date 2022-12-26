import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import weibull_min
from pulp import LpProblem, LpVariable, LpMinimize, LpMaximize, LpContinuous, lpSum, lpDot, value
from ortoolpy import model_min, addvars, addvals
import matplotlib.pyplot as plt

# functions
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
    matrix_long  = pd.read_excel (io = r'data_inputs.xlsx', sheet_name='matrix_long')
    matrix_flat  = pd.read_excel (io = r'data_inputs.xlsx', sheet_name='matrix_flat')
    matrix_tube  = pd.read_excel (io = r'data_inputs.xlsx', sheet_name='matrix_tube')
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
    
    # domestic use of crude steel
    use_pri = compute_domestic_use(Data.Var_pri, export_pri, forming_pri)
    use_sec = compute_domestic_use(Data.Var_sec, export_sec, forming_sec)
    use_hyd = compute_domestic_use(Data.Var_hyd, export_hyd, forming_hyd)
    
    # data preparation
    semi_name= ['long', 'flat', 'tube', 'alloy']
    
    share_pri = pd.DataFrame({'long': Data.share_long_pri,
                             'flat': Data.share_flat_pri,
                             'tube': Data.share_tube_pri,
                             'alloy': Data.share_alloy_pri,},)
    
    share_sec = pd.DataFrame({'long': Data.share_long_sec,
                             'flat': Data.share_flat_sec,
                             'tube': Data.share_tube_sec,
                             'alloy': Data.share_alloy_sec,},)

    exp_share_semi = pd.DataFrame({'long': Data.share_export_long,
                                   'flat': Data.share_export_flat,
                                   'tube': Data.share_export_tube,
                                   'alloy': Data.share_export_alloy,},)
    
    fabrication_yield = pd.DataFrame({'long': Data.fabrication_yield_long,
                                      'flat': Data.fabrication_yield_flat,
                                      'tube': Data.fabrication_yield_tube,
                                      'alloy': Data.fabrication_yield_alloy,},)
    
    pro_semi = pd.DataFrame()
    exp_semi = pd.DataFrame()
    fabrication = pd.DataFrame()
    use_semi = pd.DataFrame()
    
    for m in semi_name:
        
        # domestic production of finished steel
        pro_semi_list = compute_semi(use_pri, use_sec, use_hyd, share_pri[m], share_sec[m])
        pro_semi[m] = pro_semi_list
        
        # net-exports of finished steel
        exp_semi_list = compute_export(pro_semi[m], exp_share_semi[m])
        exp_semi[m] = exp_semi_list
        
        # fabrication yield
        fabrication_list = compute_scrap(pro_semi[m], exp_semi[m], fabrication_yield[m])
        fabrication[m] = fabrication_list
        
        # domestic use of finished steel
        use_semi_list = compute_domestic_use(pro_semi[m], exp_semi[m], fabrication[m])
        use_semi[m] = use_semi_list
    
    # data preparation
    product_name = ['BU', 'IF', 'MM', 'EE', 'AU', 'OT', 'MP']
    
    exp_share_end = pd.DataFrame({'BU': Data.share_export_BU, 'IF': Data.share_export_IF, 
                                'MM': Data.share_export_MM, 'EE': Data.share_export_EE, 
                                'AU': Data.share_export_AU, 'OT': Data.share_export_OT, 
                                'MP': Data.share_export_MP,},)
    
    shape_list = pd.DataFrame({'BU': Data.shape_BU, 'IF': Data.shape_IF, 
                               'MM': Data.shape_MM, 'EE': Data.shape_EE, 
                               'AU': Data.shape_AU, 'OT': Data.shape_OT, 
                               'MP': Data.shape_MP,},)
        
    scale_list = pd.DataFrame({'BU': Data.scale_BU, 'IF': Data.scale_IF, 
                               'MM': Data.scale_MM, 'EE': Data.scale_EE, 
                               'AU': Data.scale_AU, 'OT': Data.scale_OT, 
                               'MP': Data.scale_MP,},)
    
    hiber_list = pd.DataFrame({'BU': Data.hiber_BU, 'IF': Data.hiber_IF, 
                               'MM': Data.hiber_MM, 'EE': Data.hiber_EE, 
                               'AU': Data.hiber_AU, 'OT': Data.hiber_OT, 
                               'MP': Data.hiber_MP,},)

    pro_end = pd. DataFrame()
    inflow = pd. DataFrame()
    outflow = pd.DataFrame()
    EOL = pd.DataFrame()
    
    for n in product_name:
        
        # domestic production of end-use goods
        pro_end_list = compute_products(use_semi.long, use_semi.flat, use_semi.tube, use_semi.alloy, 
                                        matrix_long[n], matrix_flat[n], matrix_tube[n], matrix_alloy[n])
        pro_end[n] = pro_end_list
        
        # inflow
        inflow_list = pro_end[n] - compute_export(pro_end[n], exp_share_end[n])
        inflow[n] = inflow_list
        
        # outflow
        outflow_list = compute_outflow(inflow[n], shape_list[n], scale_list[n])
        outflow[n] = outflow_list
        
        # end-of-life scrap
        EOL_list = outflow[n] * (1 - hiber_list[n])
        EOL[n] = EOL_list

    # scrap inputs to EAF
    EOL_sec= (EOL.BU + EOL.IF + EOL.MM + EOL.EE + EOL.AU + EOL.OT + EOL.MP) * Data.domestic_rec
    fabrication_sec = fabrication.long + fabrication.flat + fabrication.tube + fabrication.alloy
    scrap_sec = (EOL_sec * Data.scrap_prep_eol + 
                 fabrication_sec *  Data.scrap_prep_fabrication + 
                 forming_sec * Data.scrap_prep_forming) * Data.sec_yield

    # CO2 emissions
    CO2 = (Data.Var_pri * Data.EI_pri) + (Data.Var_sec * Data.EI_sec) + (Data.Var_hyd * Data.EI_hyd)

    # stock
    NAS_matrix = inflow - outflow
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
    
    # domestic use of crude steel
    use_pri = compute_domestic_use(Data.Val_pri, export_pri, forming_pri)
    use_sec = compute_domestic_use(Data.Val_sec, export_sec, forming_sec)
    use_hyd = compute_domestic_use(Data.Val_hyd, export_hyd, forming_hyd)
    
    pro_semi = pd.DataFrame()
    exp_semi = pd.DataFrame()
    fabrication = pd.DataFrame()
    use_semi = pd.DataFrame()
    
    for m in semi_name:
        
        # domestic production of finished steel
        pro_semi_list = compute_semi(use_pri, use_sec, use_hyd, share_pri[m], share_sec[m])
        pro_semi[m] = pro_semi_list
        
        # net-exports of finished steel
        exp_semi_list = compute_export(pro_semi[m], exp_share_semi[m])
        exp_semi[m] = exp_semi_list
        
        # fabrication yield
        fabrication_list = compute_scrap(pro_semi[m], exp_semi[m], fabrication_yield[m])
        fabrication[m] = fabrication_list
        
        # domestic use of finished steel
        use_semi_list = compute_domestic_use(pro_semi[m], exp_semi[m], fabrication[m])
        use_semi[m] = use_semi_list
        
    for n in product_name:
        
        # domestic production of end-use goods
        pro_end_list = compute_products(use_semi.long, use_semi.flat, use_semi.tube, use_semi.alloy, 
                                        matrix_long[n], matrix_flat[n], matrix_tube[n], matrix_alloy[n])
        pro_end[n] = pro_end_list
        
        # inflow
        inflow_list = pro_end[n] - compute_export(pro_end[n], exp_share_end[n])
        inflow[n] = inflow_list
        
        # outflow
        outflow_list = compute_outflow(inflow[n], shape_list[n], scale_list[n])
        outflow[n] = outflow_list
        
        # end-of-life scrap
        EOL_list = outflow[n] * (1 - hiber_list[n])
        EOL[n] = EOL_list

    # scrap inputs to EAF
    EOL_sec= (EOL.BU + EOL.IF + EOL.MM + EOL.EE + EOL.AU + EOL.OT + EOL.MP) * Data.domestic_rec
    fabrication_sec = fabrication.long + fabrication.flat + fabrication.tube + fabrication.alloy
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
    steel_matrix = pd.DataFrame({'Carbon steel (long)': pro_semi.long,
                                 'Carbon steel (flat)': pro_semi.flat,
                                 'Carbon steel (tube)': pro_semi.tube,
                                 'Alloy steel': pro_semi.alloy,},)
    
    # domestic production of end-use goods
    goods_matrix = pd.DataFrame({'Buildings': pro_end.BU,
                                 'Infrastructure': pro_end.IF,
                                 'Mechanical machinery': pro_end.MM,
                                 'Electrical equipment': pro_end.EE,
                                 'Automobiles': pro_end.AU,
                                 'Other transport': pro_end.OT,
                                 'Metal products': pro_end.MP,},)
    
    # inflow
    in_matrix = pd.DataFrame({'Buildings': inflow.BU,
                              'Infrastructure': inflow.IF,
                              'Mechanical machinery':inflow.MM,
                              'Electrical equipment':inflow.EE,
                              'Automobiles': inflow.AU,
                              'Other transport': inflow.OT,
                              'Metal products': inflow.MP,},)

    # outflow
    ot_matrix = pd.DataFrame({'Buildings': outflow.BU,
                              'Infrastructure': outflow.IF,
                              'Mechanical machinery': outflow.MM,
                              'Electrical equipment': outflow.EE,
                              'Automobiles': outflow.AU,
                              'Other transport': outflow.OT,
                              'Metal products': outflow.MP,},)

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
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import weibull_min
from pulp import LpProblem,LpVariable, LpMinimize,LpMaximize, LpContinuous, lpSum, lpDot, value
from ortoolpy import model_min, addvars, addvals
import matplotlib.pyplot as plt

def material_budget_model(sheet_name):
    
    # data import
    Data = pd.read_excel (io = r'data_inputs.xlsx', sheet_name = sheet_name,nrows=101)
    matrix_long  = pd.read_excel (io = r'data_inputs.xlsx', sheet_name="matrix_long")
    matrix_flat  = pd.read_excel (io = r'data_inputs.xlsx', sheet_name="matrix_flat")
    matrix_tube  = pd.read_excel (io = r'data_inputs.xlsx', sheet_name="matrix_tube")
    matrix_alloy = pd.read_excel (io = r'data_inputs.xlsx', sheet_name="matrix_alloy")
    
    # system variables
    Data['Var_pri'] = addvars(len(Data))
    Data['Var_sec'] = addvars(len(Data))
    Data['Var_hyd'] = addvars(len(Data))
    
    # net-exports of ingots and semis
    export_pri = Data.Var_pri * Data.share_export_pri
    export_sec = Data.Var_sec * Data.share_export_sec
    export_hyd = Data.Var_hyd * Data.share_export_pri
    
    # forming scrap
    forming_pri = (Data.Var_pri - export_pri) * (1 - Data.forming_yield_pri)
    forming_sec = (Data.Var_sec - export_sec) * (1 - Data.forming_yield_sec)
    forming_hyd = (Data.Var_hyd - export_hyd) * (1 - Data.forming_yield_pri)
    
    # domestic production of finished steel
    pri_total = Data.Var_pri - export_pri - forming_pri
    pri_long  = pri_total * Data.share_long_pri
    pri_flat  = pri_total * Data.share_flat_pri
    pri_tube  = pri_total * Data.share_tube_pri
    pri_alloy = pri_total * Data.share_alloy_pri
    
    sec_total = Data.Var_sec - export_sec - forming_sec
    sec_long  = sec_total * Data.share_long_sec
    sec_flat  = sec_total * Data.share_flat_sec
    sec_tube  = sec_total * Data.share_tube_sec
    sec_alloy = sec_total * Data.share_alloy_sec
    
    hyd_total = Data.Var_hyd - export_hyd - forming_hyd
    hyd_long  = hyd_total * Data.share_long_pri
    hyd_flat  = hyd_total * Data.share_flat_pri
    hyd_tube  = hyd_total * Data.share_tube_pri
    hyd_alloy = hyd_total * Data.share_alloy_pri
    
    pro_long  = pri_long  + sec_long  + hyd_long
    pro_flat  = pri_flat  + sec_flat  + hyd_flat
    pro_tube  = pri_tube  + sec_tube  + hyd_tube
    pro_alloy = pri_alloy + sec_alloy + hyd_alloy
    
    # net-exports of finished steel
    export_long  = pro_long  * Data.share_export_long
    export_flat  = pro_flat  * Data.share_export_flat
    export_tube  = pro_tube  * Data.share_export_tube
    export_alloy = pro_alloy * Data.share_export_alloy
    
    # fabrication yield
    fabrication_long  = (pro_long  - export_long)  * (1 - Data.fabrication_yield_long)
    fabrication_flat  = (pro_flat  - export_flat)  * (1 - Data.fabrication_yield_flat)
    fabrication_tube  = (pro_tube  - export_tube)  * (1 - Data.fabrication_yield_tube)
    fabrication_alloy = (pro_alloy - export_alloy) * (1 - Data.fabrication_yield_alloy)
    
    # domestic use of finished steel
    use_long  = pro_long  - export_long  - fabrication_long
    use_flat  = pro_flat  - export_flat  - fabrication_flat
    use_tube  = pro_tube  - export_tube  - fabrication_tube
    use_alloy = pro_alloy - export_alloy - fabrication_alloy
   
    # domestic production of end-use goods
    pro_BU_long  = use_long  * matrix_long.BU
    pro_BU_flat  = use_flat  * matrix_flat.BU
    pro_BU_tube  = use_tube  * matrix_tube.BU
    pro_BU_alloy = use_alloy * matrix_alloy.BU
    
    pro_IF_long  = use_long  * matrix_long.IF
    pro_IF_flat  = use_flat  * matrix_flat.IF
    pro_IF_tube  = use_tube  * matrix_tube.IF
    pro_IF_alloy = use_alloy * matrix_alloy.IF

    pro_MM_long  = use_long  * matrix_long.MM
    pro_MM_flat  = use_flat  * matrix_flat.MM
    pro_MM_tube  = use_tube  * matrix_tube.MM
    pro_MM_alloy = use_alloy * matrix_alloy.MM

    pro_EE_long  = use_long  * matrix_long.EE
    pro_EE_flat  = use_flat  * matrix_flat.EE
    pro_EE_tube  = use_tube  * matrix_tube.EE
    pro_EE_alloy = use_alloy * matrix_alloy.EE

    pro_AU_long  = use_long  * matrix_long.AU
    pro_AU_flat  = use_flat  * matrix_flat.AU
    pro_AU_tube  = use_tube  * matrix_tube.AU
    pro_AU_alloy = use_alloy * matrix_alloy.AU
    
    pro_OT_long  = use_long  * matrix_long.OT
    pro_OT_flat  = use_flat  * matrix_flat.OT
    pro_OT_tube  = use_tube  * matrix_tube.OT
    pro_OT_alloy = use_alloy * matrix_alloy.OT
    
    pro_MP_long  = use_long  * matrix_long.MP
    pro_MP_flat  = use_flat  * matrix_flat.MP
    pro_MP_tube  = use_tube  * matrix_tube.MP
    pro_MP_alloy = use_alloy * matrix_alloy.MP

    pro_BU = pro_BU_long + pro_BU_flat + pro_BU_tube + pro_BU_alloy
    pro_IF = pro_IF_long + pro_IF_flat + pro_IF_tube + pro_IF_alloy
    pro_MM = pro_MM_long + pro_MM_flat + pro_MM_tube + pro_MM_alloy
    pro_EE = pro_EE_long + pro_EE_flat + pro_EE_tube + pro_EE_alloy
    pro_AU = pro_AU_long + pro_AU_flat + pro_AU_tube + pro_AU_alloy
    pro_OT = pro_OT_long + pro_OT_flat + pro_OT_tube + pro_OT_alloy
    pro_MP = pro_MP_long + pro_MP_flat + pro_MP_tube + pro_MP_alloy

    # net-exports of end-use goods
    export_BU = pro_BU * Data.share_export_BU
    export_IF = pro_IF * Data.share_export_IF
    export_MM = pro_MM * Data.share_export_MM
    export_EE = pro_EE * Data.share_export_EE
    export_AU = pro_AU * Data.share_export_AU
    export_OT = pro_OT * Data.share_export_OT
    export_MP = pro_MP * Data.share_export_MP
    
    # inflow
    in_BU = pro_BU - export_BU
    in_IF = pro_IF - export_IF
    in_MM = pro_MM - export_MM
    in_EE = pro_EE - export_EE
    in_AU = pro_AU - export_AU
    in_OT = pro_OT - export_OT
    in_MP = pro_MP - export_MP

    year_complete = np.arange(1950,1951)
    ot_BU = np.repeat(0,len(year_complete))
    ot_IF = np.repeat(0,len(year_complete))
    ot_MM = np.repeat(0,len(year_complete))
    ot_EE = np.repeat(0,len(year_complete))
    ot_AU = np.repeat(0,len(year_complete))
    ot_OT = np.repeat(0,len(year_complete))
    ot_MP = np.repeat(0,len(year_complete))

    for k in range(1951,2051):
        
        # buildings
        ot_list_BU = (in_BU.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_BU.iloc[0:len(year_complete)],
                                        scale = Data.scale_BU.iloc[0:len(year_complete)])))
        ot_sum_BU = sum(ot_list_BU)
        ot_BU = np.append(ot_BU,ot_sum_BU)
    
        # infrastructure
        ot_list_IF = (in_IF.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_IF.iloc[0:len(year_complete)],
                                        scale = Data.scale_IF.iloc[0:len(year_complete)])))
        ot_sum_IF = sum(ot_list_IF)
        ot_IF = np.append(ot_IF,ot_sum_IF)
    
        # mechanical machinery
        ot_list_MM = (in_MM.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_MM.iloc[0:len(year_complete)],
                                        scale = Data.scale_MM.iloc[0:len(year_complete)])))
        ot_sum_MM = sum(ot_list_MM)
        ot_MM = np.append(ot_MM,ot_sum_MM)
    
        # electrical equipment
        ot_list_EE = (in_EE.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_EE.iloc[0:len(year_complete)],
                                        scale = Data.scale_EE.iloc[0:len(year_complete)])))
        ot_sum_EE = sum(ot_list_EE)
        ot_EE = np.append(ot_EE,ot_sum_EE)
    
        # automotive
        ot_list_AU = (in_AU.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_AU.iloc[0:len(year_complete)],
                                        scale = Data.scale_AU.iloc[0:len(year_complete)])))
        ot_sum_AU = sum(ot_list_AU)
        ot_AU = np.append(ot_AU,ot_sum_AU)
    
        # other transport
        ot_list_OT = (in_OT.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_OT.iloc[0:len(year_complete)],
                                        scale = Data.scale_OT.iloc[0:len(year_complete)])))
        ot_sum_OT = sum(ot_list_OT)
        ot_OT = np.append(ot_OT,ot_sum_OT)
    
        # metal products
        ot_list_MP = (in_MP.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_MP.iloc[0:len(year_complete)],
                                        scale = Data.scale_MP.iloc[0:len(year_complete)])))
        ot_sum_MP = sum(ot_list_MP)
        ot_MP = np.append(ot_MP,ot_sum_MP)
        
        year_complete = np.append(year_complete,k)

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
    pro_pri = Data.Var_pri
    pro_sec = Data.Var_sec
    pro_hyd = Data.Var_hyd
    
    CO2 = (pro_pri * Data.EI_pri) + (pro_sec * Data.EI_sec) + (pro_hyd * Data.EI_hyd)
    
    # inflow
    in_matrix = pd.DataFrame({0:in_BU,
                              1:in_IF,
                              2:in_MM,
                              3:in_EE,
                              4:in_AU,
                              5:in_OT,
                              6:in_MP,},)

    # outflow
    ot_matrix = pd.DataFrame({0:ot_BU,
                              1:ot_IF,
                              2:ot_MM,
                              3:ot_EE,
                              4:ot_AU,
                              5:ot_OT,
                              6:ot_MP,},)

    # stock
    NAS_matrix = in_matrix - ot_matrix
    st_matrix  = NAS_matrix.cumsum()
    st_sum     = st_matrix.sum(axis=1)

    model = LpProblem(sense=LpMaximize)
    model += lpSum(st_sum)
    
    for t in Data.index:
        
        # CO2 emission constraints
        model += CO2[t] <= Data.budget[t]
    
        # scrap availability constraints
        model += Data.Var_sec[t] <= scrap_sec[t]
        
        # H2-DRI/EAF capacity constraints
        model += Data.Var_hyd[t] <= Data.hyd_cap[t]
    
        # non-negative constraints
        model += Data.Var_pri[t] >=0
        model += Data.Var_sec[t] >=0
        model += Data.Var_hyd[t] >=0
        
    model.solve()
    Data['Val_pri'] = Data.Var_pri.apply(value)
    Data['Val_sec'] = Data.Var_sec.apply(value)
    Data['Val_hyd'] = Data.Var_hyd.apply(value)

########################################################################
# calculations with optimized numbers
########################################################################

    # net-exports of ingots and semis
    export_pri = Data.Val_pri * Data.share_export_pri
    export_sec = Data.Val_sec * Data.share_export_sec
    export_hyd = Data.Val_hyd * Data.share_export_sec
    
    # forming scrap
    forming_pri = (Data.Val_pri - export_pri) * (1 - Data.forming_yield_pri)
    forming_sec = (Data.Val_sec - export_sec) * (1 - Data.forming_yield_sec)
    forming_hyd = (Data.Val_hyd - export_hyd) * (1 - Data.forming_yield_pri)
    
    # domestic production of finihsed steel
    pri_total = Data.Val_pri - export_pri - forming_pri
    pri_long  = pri_total * Data.share_long_pri
    pri_flat  = pri_total * Data.share_flat_pri
    pri_tube  = pri_total * Data.share_tube_pri
    pri_alloy = pri_total * Data.share_alloy_pri
    
    sec_total = Data.Val_sec - export_sec - forming_sec
    sec_long  = sec_total * Data.share_long_sec
    sec_flat  = sec_total * Data.share_flat_sec
    sec_tube  = sec_total * Data.share_tube_sec
    sec_alloy = sec_total * Data.share_alloy_sec
    
    hyd_total = Data.Val_hyd - export_hyd - forming_hyd
    hyd_long  = hyd_total * Data.share_long_pri
    hyd_flat  = hyd_total * Data.share_flat_pri
    hyd_tube  = hyd_total * Data.share_tube_pri
    hyd_alloy = hyd_total * Data.share_alloy_pri
    
    pro_long  = pri_long  + sec_long  + hyd_long
    pro_flat  = pri_flat  + sec_flat  + hyd_flat
    pro_tube  = pri_tube  + sec_tube  + hyd_tube
    pro_alloy = pri_alloy + sec_alloy + hyd_alloy
    
    # net-exports of finished steel
    export_long  = pro_long  * Data.share_export_long
    export_flat  = pro_flat  * Data.share_export_flat
    export_tube  = pro_tube  * Data.share_export_tube
    export_alloy = pro_alloy * Data.share_export_alloy
    
    # fabrication yield
    fabrication_long  = (pro_long  - export_long)  * (1 - Data.fabrication_yield_long)
    fabrication_flat  = (pro_flat  - export_flat)  * (1 - Data.fabrication_yield_flat)
    fabrication_tube  = (pro_tube  - export_tube)  * (1 - Data.fabrication_yield_tube)
    fabrication_alloy = (pro_alloy - export_alloy) * (1 - Data.fabrication_yield_alloy)
    
    # domestic use of finished steel
    use_long  = pro_long   - export_long  - fabrication_long
    use_flat  = pro_flat   - export_flat  - fabrication_flat
    use_tube  = pro_tube   - export_tube  - fabrication_tube
    use_alloy = pro_alloy  - export_alloy - fabrication_alloy
   
    # domestic production of end-use goods
    pro_BU_long  = use_long  * matrix_long.BU
    pro_BU_flat  = use_flat  * matrix_flat.BU
    pro_BU_tube  = use_tube  * matrix_tube.BU
    pro_BU_alloy = use_alloy * matrix_alloy.BU
    
    pro_IF_long  = use_long  * matrix_long.IF
    pro_IF_flat  = use_flat  * matrix_flat.IF
    pro_IF_tube  = use_tube  * matrix_tube.IF
    pro_IF_alloy = use_alloy * matrix_alloy.IF

    pro_MM_long  = use_long  * matrix_long.MM
    pro_MM_flat  = use_flat  * matrix_flat.MM
    pro_MM_tube  = use_tube  * matrix_tube.MM
    pro_MM_alloy = use_alloy * matrix_alloy.MM

    pro_EE_long  = use_long  * matrix_long.EE
    pro_EE_flat  = use_flat  * matrix_flat.EE
    pro_EE_tube  = use_tube  * matrix_tube.EE
    pro_EE_alloy = use_alloy * matrix_alloy.EE

    pro_AU_long  = use_long  * matrix_long.AU
    pro_AU_flat  = use_flat  * matrix_flat.AU
    pro_AU_tube  = use_tube  * matrix_tube.AU
    pro_AU_alloy = use_alloy * matrix_alloy.AU
    
    pro_OT_long  = use_long  * matrix_long.OT
    pro_OT_flat  = use_flat  * matrix_flat.OT
    pro_OT_tube  = use_tube  * matrix_tube.OT
    pro_OT_alloy = use_alloy * matrix_alloy.OT
    
    pro_MP_long  = use_long  * matrix_long.MP
    pro_MP_flat  = use_flat  * matrix_flat.MP
    pro_MP_tube  = use_tube  * matrix_tube.MP
    pro_MP_alloy = use_alloy * matrix_alloy.MP

    pro_BU = pro_BU_long + pro_BU_flat + pro_BU_tube + pro_BU_alloy
    pro_IF = pro_IF_long + pro_IF_flat + pro_IF_tube + pro_IF_alloy
    pro_MM = pro_MM_long + pro_MM_flat + pro_MM_tube + pro_MM_alloy
    pro_EE = pro_EE_long + pro_EE_flat + pro_EE_tube + pro_EE_alloy
    pro_AU = pro_AU_long + pro_AU_flat + pro_AU_tube + pro_AU_alloy
    pro_OT = pro_OT_long + pro_OT_flat + pro_OT_tube + pro_OT_alloy
    pro_MP = pro_MP_long + pro_MP_flat + pro_MP_tube + pro_MP_alloy

    # net-exports of end-use goods
    export_BU = pro_BU * Data.share_export_BU
    export_IF = pro_IF * Data.share_export_IF
    export_MM = pro_MM * Data.share_export_MM
    export_EE = pro_EE * Data.share_export_EE
    export_AU = pro_AU * Data.share_export_AU
    export_OT = pro_OT * Data.share_export_OT
    export_MP = pro_MP * Data.share_export_MP
    
    # inflow
    in_BU = pro_BU - export_BU
    in_IF = pro_IF - export_IF
    in_MM = pro_MM - export_MM
    in_EE = pro_EE - export_EE
    in_AU = pro_AU - export_AU
    in_OT = pro_OT - export_OT
    in_MP = pro_MP - export_MP

    year_complete = np.arange(1950,1951)
    ot_BU = np.repeat(0,len(year_complete))
    ot_IF = np.repeat(0,len(year_complete))
    ot_MM = np.repeat(0,len(year_complete))
    ot_EE = np.repeat(0,len(year_complete))
    ot_AU = np.repeat(0,len(year_complete))
    ot_OT = np.repeat(0,len(year_complete))
    ot_MP = np.repeat(0,len(year_complete))

    for k in range(1951,2051):
        
        # buildings
        ot_list_BU = (in_BU.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_BU.iloc[0:len(year_complete)],
                                        scale = Data.scale_BU.iloc[0:len(year_complete)])))
        ot_sum_BU = sum(ot_list_BU)
        ot_BU = np.append(ot_BU,ot_sum_BU)
    
        # infrastructure
        ot_list_IF = (in_IF.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_IF.iloc[0:len(year_complete)],
                                        scale = Data.scale_IF.iloc[0:len(year_complete)])))
        ot_sum_IF = sum(ot_list_IF)
        ot_IF = np.append(ot_IF,ot_sum_IF)
    
        # mechanical machinery
        ot_list_MM = (in_MM.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_MM.iloc[0:len(year_complete)],
                                        scale = Data.scale_MM.iloc[0:len(year_complete)])))
        ot_sum_MM = sum(ot_list_MM)
        ot_MM = np.append(ot_MM,ot_sum_MM)
    
        # electrical equipment
        ot_list_EE = (in_EE.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_EE.iloc[0:len(year_complete)],
                                        scale = Data.scale_EE.iloc[0:len(year_complete)])))
        ot_sum_EE = sum(ot_list_EE)
        ot_EE = np.append(ot_EE,ot_sum_EE)
    
        # automotive
        ot_list_AU = (in_AU.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_AU.iloc[0:len(year_complete)],
                                        scale = Data.scale_AU.iloc[0:len(year_complete)])))
        ot_sum_AU = sum(ot_list_AU)
        ot_AU = np.append(ot_AU,ot_sum_AU)
    
        # other transport
        ot_list_OT = (in_OT.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_OT.iloc[0:len(year_complete)],
                                        scale = Data.scale_OT.iloc[0:len(year_complete)])))
        ot_sum_OT = sum(ot_list_OT)
        ot_OT = np.append(ot_OT,ot_sum_OT)
    
        # metal products
        ot_list_MP = (in_MP.iloc[0:len(year_complete)] 
                      *(weibull_min.pdf(k - year_complete,
                                        c = Data.shape_MP.iloc[0:len(year_complete)],
                                        scale = Data.scale_MP.iloc[0:len(year_complete)])))
        ot_sum_MP = sum(ot_list_MP)
        ot_MP = np.append(ot_MP,ot_sum_MP)
        
        year_complete = np.append(year_complete,k)

    # end-of-life scrap
    EOL_BU = ot_BU * (1-Data.hiber_BU)
    EOL_IF = ot_IF * (1-Data.hiber_IF)
    EOL_MM = ot_MM * (1-Data.hiber_MM)
    EOL_EE = ot_EE * (1-Data.hiber_EE)
    EOL_AU = ot_AU * (1-Data.hiber_AU)
    EOL_OT = ot_OT * (1-Data.hiber_OT)
    EOL_MP = ot_MP * (1-Data.hiber_MP)

    # scrap inputs to EAF
    EOL_sec= (EOL_BU + EOL_IF + EOL_MM + EOL_EE + EOL_AU + EOL_OT + EOL_MP) * Data.domestic_rec
    fabrication_sec = fabrication_long + fabrication_flat + fabrication_tube + fabrication_alloy
    scrap_sec = (EOL_sec * Data.scrap_prep_eol + 
                 fabrication_sec *  Data.scrap_prep_fabrication + 
                 forming_sec * Data.scrap_prep_forming) * Data.sec_yield
    
    scrap = pd.DataFrame({'EOL':EOL_sec * Data.scrap_prep_eol,
                          'Fabrication':fabrication_sec *  Data.scrap_prep_fabrication,
                          'Forming':forming_sec * Data.scrap_prep_forming},)
    
    # CO2 emissions
    pro_pri = Data.Val_pri
    pro_sec = Data.Val_sec
    pro_hyd = Data.Val_hyd
    
    pro_couse50 = pro_pri * Data.course50
    pro_BF = pro_pri - pro_couse50
    
    production = pd.DataFrame({'BF/BOF'   :pro_BF,
                               'H$_2$-BF/BOF+CCS':pro_couse50,
                               'H$_2$-DRI/EAF':pro_hyd,
                               'Scrap-EAF':pro_sec},)
    
    CO2 = pd.DataFrame({'BF/BOF'   :pro_pri * Data.EI_pri,
                        'H$_2$-DRI/EAF':pro_hyd * Data.EI_hyd,
                        'Scrap-EAF':pro_sec * Data.EI_sec},)
    
    # domestic production of finished steel
    steel_matrix = pd.DataFrame({'Carbon steel (long)' :pro_long,
                                 'Carbon steel (flat)' :pro_flat,
                                 'Carbon steel (tube)' :pro_tube,
                                 'Alloy steel'         :pro_alloy,},)
    
    # domestic production of end-use goods
    goods_matrix = pd.DataFrame({'Buildings'           :pro_BU,
                                 'Infrastructure'      :pro_IF,
                                 'Mechanical machinery':pro_MM,
                                 'Electrical equipment':pro_EE,
                                 'Automobiles'         :pro_AU,
                                 'Other transport'     :pro_OT,
                                 'Metal products'      :pro_MP,},)
    
    # inflow
    in_matrix = pd.DataFrame({'Buildings'           :in_BU,
                              'Infrastructure'      :in_IF,
                              'Mechanical machinery':in_MM,
                              'Electrical equipment':in_EE,
                              'Automobiles'         :in_AU,
                              'Other transport'     :in_OT,
                              'Metal products'      :in_MP,},)

    # outflow
    ot_matrix = pd.DataFrame({'Buildings'           :ot_BU,
                              'Infrastructure'      :ot_IF,
                              'Mechanical machinery':ot_MM,
                              'Electrical equipment':ot_EE,
                              'Automobiles'         :ot_AU,
                              'Other transport'     :ot_OT,
                              'Metal products'      :ot_MP,},)

    # stock
    NAS_matrix = in_matrix - ot_matrix
    st_matrix  = NAS_matrix.cumsum()
    
    production   = production.rename(index = Data.year)
    steel_matrix = steel_matrix.rename(index = Data.year)
    goods_matrix = goods_matrix.rename(index = Data.year)
    CO2          = CO2.rename(index = Data.year)
    in_matrix    = in_matrix.rename(index = Data.year)
    ot_matrix    = ot_matrix.rename(index = Data.year)
    st_matrix    = st_matrix.rename(index = Data.year)
    scrap        = scrap.rename(index = Data.year)
    
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
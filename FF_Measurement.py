import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def search_in_array_forward(arr, target):
    min_index = 0
    min_diff = abs(arr[0] - target)
    
    for i, num in enumerate(arr):
        diff = abs(num - target)
        if diff <= min_diff:
            min_diff = diff
            min_index = i
        else:
            break
    return min_index

def search_in_array_reverse(arr, target):
    min_index = len(arr) - 1
    min_diff = abs(arr[-1] - target)
    
    for i in range(len(arr) - 1, -1, -1):
        diff = abs(arr[i] - target)
        if diff <= min_diff:
            min_diff = diff
            min_index = i
        else:
            break
    return min_index

names = pd.read_csv(r'D:\Serial_Links\python\tcq_falling_TSPC_register.csv', nrows=1).columns.tolist()
data = pd.read_csv(r'D:\Serial_Links\python\tcq_falling_TSPC_register.csv')

no_of_corners = len(names)//2
MOS = []
Temp = []
vdd = []

for i in range(0, len(names), 2):
    input_string = names[i]

    mos_start = input_string.find('MOS=') + len('MOS=')
    mos_end = input_string.find(',', mos_start)
    mos_value = input_string[mos_start:mos_end]
    MOS.append(mos_value)

    temp_start = input_string.find('temperature=') + len('temperature=')
    temp_end = input_string.find(',', temp_start)
    temperature_value = input_string[temp_start:temp_end]
    Temp.append(temperature_value)

    vdd_start = input_string.find('VDD=') + len('VDD=')
    vdd_end = input_string.find(')', vdd_start)
    vdd_value = input_string[vdd_start:vdd_end]
    vdd.append(vdd_value)


corner_dict = {
    'process': MOS,
    'temp': Temp,
    'vdd': vdd
}

for i in range(no_of_corners):
    globals()[f"Corner{i}tcq"] = data.iloc[:, 2*i].to_numpy()
    globals()[f"Corner{i}delay"] = data.iloc[:, 2*i + 1].to_numpy()

plt.figure(figsize=(10, 8))

for i in range(no_of_corners):
    x = np.array([float(val) if isinstance(val, str) else val for val in globals()[f"Corner{i}tcq"] if not pd.isna(val) and (isinstance(val, str) and val.strip() != '' or isinstance(val, (int, float)))])
    y = np.array([float(val) if isinstance(val, str) else val for val in globals()[f"Corner{i}delay"] if not pd.isna(val) and (isinstance(val, str) and val.strip() != '' or isinstance(val, (int, float)))])
    f = interp1d(x, y, kind='linear')
    n_of_points = 1000
    new_x = np.linspace(min(x), max(x), n_of_points)
    new_y = f(new_x)

    plt.plot(new_x, new_y, label=f'Corner{i} Interpolated Line')
    
    print(f"Process: {corner_dict['process'][i]}      Temp = {corner_dict['temp'][i]}      VDD = {corner_dict['vdd'][i]}")
    print()
    min_tcq = min(new_y)
    print(f"Minimum TCQ value is {min_tcq * 10**12:.2f} ps")

    target_y_setup = 1.1 * min_tcq
    #target_index = np.abs(new_y - target_y).argmin() 
    target_index_setup= search_in_array_forward(new_y, target_y_setup)
    setup_time = new_x[target_index_setup]
    if target_index_setup < n_of_points/2:
        if setup_time * 10**12 < 15:
            print(f"\033[92mSetup time value is {setup_time * 10**12:.2f} ps\033[0m")
        else:
            print(f"\033[91mSetup time value is {setup_time * 10**12:.2f} ps\033[0m") 

    target_y_hold = 1.05 * min_tcq
    #target_index = np.abs(new_y - target_y).argmin() 
    target_index_hold= search_in_array_reverse(new_y, target_y_hold)
    hold_time = 62.5e-12 - new_x[target_index_hold]
    if target_index_hold > n_of_points/2:
        print(f"Hold time value is {hold_time * 10**12:.2f} ps") 
    else:
        print(f"Couldn't measure hold time")
   
    print(f"_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ")
    
plt.xlabel('Delay values')
plt.ylabel('TCQ values')
plt.title('TCQ vs Delay')
plt.grid(True)
#plt.legend()
plt.show()
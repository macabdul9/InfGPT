import pandas as pd
import os

root = os.path.join(os.getcwd(), 'MMMUGenResults')

agpt = os.path.join(root, 'AnyGPT_gen')
phi2 = os.path.join(root, 'phi-2')

for file_name in os.listdir(agpt):
    file_path1 = os.path.join(agpt, file_name)
    file_path2 = os.path.join(phi2, file_name)
    print(file_path1, file_path2, sep='\n')

    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    print(df1.columns, df2.columns, sep='\n')
    break
    try:
        df2['generated_answer'] = df1['prediction']
    except 
    df2.to_csv(file_path1, index=False)

import pandas as pd
import os

root = os.path.join(os.getcwd(), 'MMMUGenResults')

column_to_delete = "gpt-4o-mini_output"
agpt = os.path.join(root, 'AnyGPT_gen')
phi2 = os.path.join(root, 'phi-2')

print(len(os.listdir(agpt)), len(os.listdir(phi2)))
for file_name in os.listdir(agpt):
    
    file_path_anygpt = os.path.join(agpt, file_name)
    file_path_phi2 = os.path.join(phi2, file_name)
    
    # print(file_path_anygpt, file_path_phi2, sep='\n')

    df_anygpt = pd.read_csv(file_path_anygpt)
    df_phi2 = pd.read_csv(file_path_phi2)
    if "generated_answer" in df_anygpt.columns:
        # print("Already processed")
        continue
    
    # print("Before", df_anygpt.columns, df_phi2.columns, sep='\n')
    # delete  column_to_delete
    if column_to_delete in df_phi2.columns:
        df_phi2.drop(columns=[column_to_delete], inplace=True)
    # then replace the answer column with the one from anygpt
    df_phi2['generated_answer'] = df_anygpt['prediction']
    # if file_name == "Agriculture.csv":
    #     import pdb; pdb.set_trace()
    
    
    
    # print("After", df_anygpt.columns, df_phi2.columns, sep='\n')
    # break
    # save the file
    df_phi2.to_csv(file_path_anygpt, index=False)

import numpy as np
import csv
import argparse
import os
import datetime
from scipy.stats import ranksums
from scipy.stats import ranksums, wilcoxon, ttest_rel, ttest_ind


def signi_test(bl_data, rl_data):
    _, p_contact = ttest_rel(bl_data, rl_data)
    return p_contact


def calculate_spl(path_length_list, success_list, shortest_path_lengt_list ):
    spl = 0
    for i, pl in enumerate(path_length_list):
        spl += success_list[i] * (shortest_path_lengt_list[i] / max(pl, shortest_path_lengt_list[i]))
    return spl / len(path_length_list)

def get_np_function(data):
    return np.mean(data), np.std(data), np.var(data)

def get_metrics(suc_data, dual_suc_data, path_data, col_data, con_data, short_path_data, rew_data = None):
    # 0: success, 1: path_lentgth, 2: collision rate, 3:contact rate, 4: reward, 5: spl
    eval_data = []
    eval_data.append(get_np_function(suc_data))
    eval_data.append(get_np_function(path_data[dual_suc_data == True]))
    eval_data.append(get_np_function(col_data[dual_suc_data == True]))
    eval_data.append(get_np_function(con_data[dual_suc_data == True]))
    if rew_data is not None:
        eval_data.append(get_np_function(rew_data))
    else: eval_data.append([0,0,0])
    spl = calculate_spl(path_data, suc_data, short_path_data)
    eval_data.append([spl, 0, 0])
    return eval_data

def load_data(folder_name, baseline):
    success_rate = np.load(base_folder + "evaluation_folders/" + folder_name + "/success_list.npy")
    collision_rate = np.load(base_folder + "evaluation_folders/" + folder_name + "/collision_list.npy")
    contact_rate = np.load(base_folder + "evaluation_folders/" + folder_name + "/contact_list.npy")
    path_length_list = np.load(base_folder + "evaluation_folders/" + folder_name + "/path_length_list.npy")
    short_lengt_list = np.load(base_folder + "evaluation_folders/" + folder_name + "/shortest_path_length_list.npy")
    reward = None
    if not baseline:
        reward = np.load(base_folder + "evaluation_folders/" + folder_name + "/reward_list.npy")
    return success_rate, path_length_list, collision_rate, contact_rate, short_lengt_list, reward


def write_data(writer, folder, time, data):
    writer.writerow([folder, str(time), str(data[0][0]), str(data[3][0]), str(data[2][0]), str(data[5][0]),
                     str(data[1][0]), str(data[4][0]),str(data[0][1]), str(data[3][1]), str(data[2][1]),
                     str(data[1][1]), str(data[4][1])])

#define parser
parser = argparse.ArgumentParser()
parser.add_argument('--rl', type=str, default=None)
parser.add_argument('--bl', type=str, default=None)
args = parser.parse_args()
base_folder = os.path.dirname(__file__)
rl_folder = args.rl
bl_folder = args.bl

dual_eval = True if args.rl is not None and args.bl is not None else False

if dual_eval:
    suc_data_rl, path_data_rl, col_data_rl, con_data_rl, short_path_data_rl, rew_data_rl = load_data(rl_folder, False)
    suc_data_bl, path_data_bl, col_data_bl, con_data_bl, short_path_data_bl, rew_data_bl = load_data(bl_folder, True)
    dual_suc_data = suc_data_rl & suc_data_bl
    con_data_rl[np.argwhere(np.isnan(con_data_rl))] = 0
    col_data_rl[np.argwhere(np.isnan(col_data_rl))] = 0
    con_data_bl[np.argwhere(np.isnan(con_data_bl))] = 0
    col_data_bl[np.argwhere(np.isnan(col_data_bl))] = 0

    # con_data_rl = np.delete(con_data_rl, np.argwhere(np.isnan(con_data_rl)))
    # col_data_rl = np.delete(col_data_rl, np.argwhere(np.isnan(col_data_rl)))
    # con_data_bl = np.delete(con_data_bl, np.argwhere(np.isnan(con_data_bl)))
    # col_data_bl = np.delete(col_data_bl, np.argwhere(np.isnan(col_data_bl)))
    eval_data_rl = get_metrics(suc_data_rl, dual_suc_data, path_data_rl, col_data_rl, con_data_rl, short_path_data_rl, rew_data_rl)
    eval_data_bl = get_metrics(suc_data_bl, dual_suc_data, path_data_bl, col_data_bl, con_data_bl, short_path_data_bl, rew_data_bl)
    signi_con = signi_test(con_data_bl[dual_suc_data == True], con_data_rl[dual_suc_data == True])
    signi_col = signi_test(col_data_bl[dual_suc_data == True], col_data_rl[dual_suc_data == True])
    signi_path = signi_test(path_data_bl[dual_suc_data == True], path_data_rl[dual_suc_data == True])
    test_col_rl = col_data_rl[dual_suc_data == True]
    test_col_rl = test_col_rl[test_col_rl > 0]
    test_col_bl = col_data_bl[dual_suc_data == True]
    test_col_bl = test_col_bl[test_col_bl > 0]
    print("Sinificant Con: ", signi_con < 0.05, " with ", signi_con)
    print("Sinificant Col: ", signi_col < 0.05, " with ", signi_col)
    print("Sinificant Path: ", signi_path < 0.05, " with ", signi_path)
    print(eval_data_rl[0][0], " vs ", eval_data_bl[0][0])
else:
    if args.rl is not None:
        suc_data_rl, path_data_rl, col_data_rl, con_data_rl, short_path_data_rl, rew_data_rl = load_data(rl_folder, False)
        eval_data_rl = get_metrics(suc_data_rl, suc_data_rl, path_data_rl, col_data_rl, con_data_rl, short_path_data_rl, rew_data_rl)
        print(eval_data_rl[0][0])

    if args.bl is not None:
        suc_data_bl, path_data_bl, col_data_bl, con_data_bl, short_path_data_bl, rew_data_bl = load_data(bl_folder, True)
        eval_data_bl = get_metrics(suc_data_bl, suc_data_bl, path_data_bl, col_data_bl, con_data_bl, short_path_data_bl, rew_data_bl)
        print(eval_data_bl[0][0])

time = datetime.datetime.now()
csv_header = ["name", "time", "mean success rate", "mean contact rate", "mean collision rate", "spl",
              "mean_path_length", "mean reward", "std success rate",  "std contact rate", "std collision rate",
              "std_path_length", "std reward"]
with open(base_folder+'evaluation_file.csv', mode='a') as eval_file:
     evaluation_writer = csv.writer(eval_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
     evaluation_writer.writerow(csv_header)
     if dual_eval:
         print(eval_data_rl[0][0], eval_data_bl[0][0])
         write_data(evaluation_writer, rl_folder, time, eval_data_rl)
         write_data(evaluation_writer, bl_folder, time, eval_data_bl)
     else:
         if args.rl is not None:
             print(eval_data_rl[0][0])
             write_data(evaluation_writer, rl_folder, time, eval_data_rl)
         if args.bl is not None:
             print(eval_data_bl[0][0])
             write_data(evaluation_writer, bl_folder, time, eval_data_bl)

print("Done")

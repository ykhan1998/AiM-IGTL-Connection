import glob
import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import FIG_DIR, RES_DIR, STATUS_2, STATUS_3, STATUS_4
from MRTI.src.mrti_utils import compute_status_map
from utils import compute_result_from_status


def eval_real_ablation_results(ablation_res_map, lesion_mask):
    # ablation_res_map is binary
    print(ablation_res_map.shape)
    print(lesion_mask.shape)
    
    status_map = compute_status_map(lesion_mask, ablation_res_map)
    
    # probe center as ablated
    status_map[49:52, 49:52] = STATUS_4
    ablated_tumor_idx = status_map == STATUS_4
    ablated_healthy_idx = status_map == STATUS_2

    im = plt.imshow(status_map)

    values = [3, 0, 1, 2]

    colors = [im.cmap(im.norm(value)) for value in values]

    # labels = ["Ablated Lesion", "Ablated Healthy", "Unablted Lesion", "Unablated Healthy"]
    # patches = [ mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
    # plt.legend(handles=patches, loc=0)
    plt.axis('off')
    plt.savefig(FIG_DIR + f"/{datetime.now().strftime('%Y%m%d-%H-%M-%S')}.png")

    # pixel size 0.7mm * 0.7mm = 0.49 mm^2
    print(f"Tumor size: {np.count_nonzero(lesion_mask)} mm^2")
    print(f"Ablated tumor: \
            {np.count_nonzero(ablated_tumor_idx) / np.count_nonzero(lesion_mask)*100}%")
    print(f"Ablated healthy: \
            {np.count_nonzero(ablated_healthy_idx) / np.count_nonzero(lesion_mask)*100}%")


def overlay_contours(ablation_res_map, lesion_mask):
    status_map = compute_status_map(lesion_mask, ablation_res_map)

    h, w = 1000, 1000
    
    ablated_exp = np.zeros(status_map.shape)
    ablated_exp[49:52, 49:52] = 1
    ablated_exp[status_map == 3] = 1
    ablated_exp[status_map == 0] = 1
    ablated_exp_up = cv2.resize(ablated_exp, (h, w))
    ablated_exp_contours, _ = cv2.findContours(ablated_exp_up.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    status_map_plan = np.load('/home/yiwei/ConformalAblation/results/icap_plan/2s66/u1e3/ablation_res.npy')

    tumor = np.zeros(status_map_plan.shape)
    tumor[49:52, 49:52] = 1
    tumor[status_map_plan == STATUS_3] = 1
    tumor[status_map_plan == STATUS_4] = 1
    tumor_up = cv2.resize(tumor, (h, w))
    tumor_contours, _ = cv2.findContours(tumor_up.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ablated_plan = np.zeros(status_map_plan.shape)
    ablated_plan[49:52, 49:52] = 1
    ablated_plan[status_map_plan == STATUS_4] = 1
    ablated_plan[status_map_plan == STATUS_2] = 1
    ablated_plan_up = cv2.resize(ablated_plan, (h, w))
    ablated_plan_contours, _ = cv2.findContours(ablated_plan_up.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    canvas = np.ones((h, w, 3)) * 255
    canvas[int(h/2-2):int(h/2+4), int(w/2-4):int(w/2+2)] = (0, 0, 0)
    cv2.drawContours(canvas, ablated_exp_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(canvas, ablated_plan_contours, -1, (255, 0, 0), 2)
    cv2.drawContours(canvas, tumor_contours, -1, (0, 0, 255), 2)

    # draw   at (h/2, w/2)
    cv2.circle(canvas, (int(h/2+1), int(w/2-1)), 125, (255, 0, 255), 2)

    # cv2.imshow("test", canvas) 
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 

    x, y = 300, 300
    crop_canvas = canvas[y:y+400, x:x+400]

    cv2.imwrite(FIG_DIR + f"/{datetime.now().strftime('%Y%m%d-%H-%M-%S')}.png", crop_canvas)


def analyze_plan_effort(plan_dir):
    # get folders under plan_dir
    exp_ls = sorted(glob.glob(plan_dir + "/*"))
    print(exp_ls)
    score_ls = []
    for exp in exp_ls:
        sim_res_map = np.load(exp + "/ablation_res.npy")
        score = eval_result_status(sim_res_map)
        score_ls.append(score)
    trial_ls = np.arange(1, len(exp_ls) + 1)

    # plot score vs trial
    plt.plot(trial_ls, score_ls)
    plt.scatter(trial_ls, score_ls)
    plt.xticks(trial_ls)
    plt.xlim(0.5, len(exp_ls) + 1.5)
    plt.ylim(40, 100)
    plt.xlabel("Trial")
    plt.ylabel("Score")
    plt.savefig(FIG_DIR + "/score_vs_trial.png")

    return score_ls, trial_ls


def eval_human_planning(human_data_folder):
    human_score_ls = {}
    for lesion_exp_folder in glob.glob(human_data_folder + "/*"):
        lesion_id = os.path.basename(lesion_exp_folder)      
        human_score_ls[lesion_id] = []
        for exp_folder in glob.glob(lesion_exp_folder + "/*"):
            sim_res_map = np.load(exp_folder + "/ablation_res.npy")

            total_score, ablated_tumor_score, ablated_healthy_score, avg_radial_diff = compute_result_from_status(sim_res_map)
            human_score_ls[lesion_id].append((total_score, ablated_tumor_score, ablated_healthy_score, avg_radial_diff))

    # Save human score as datafrmae of columns lesion_id, total_score, ablated_tumor_score, ablated_healthy_score
    df = pd.DataFrame([(lesion_id, *scores) for lesion_id, scores_list in human_score_ls.items() for scores in scores_list],
                      columns=['lesion_id', 'total_score', 'ablated_tumor_score', 'ablated_healthy_score', 'avg_radial_diff'])
    
    df.to_csv(RES_DIR + "/human_planning_scores.csv", index=False)

    return human_score_ls


def compare_human_vs_rl(human_score_ls, rl_score_ls):
    lesion_ids = list(human_score_ls.keys())
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(lesion_ids)))
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    best_radial_diffs = []
    
    for i, (lesion_id, color) in enumerate(zip(lesion_ids, colors)):
        scores = human_score_ls[lesion_id]
        total_scores = [score[0] for score in scores]
        radial_diffs = [score[3] for score in scores]
        
        # Find the index of the highest score
        best_score_index = total_scores.index(max(total_scores))
        best_radial_diffs.append(radial_diffs[best_score_index])
        
        # Human scores
        ax.scatter([i] * len(total_scores), total_scores, color=color, alpha=0.6, s=30, label=f'Human - {lesion_id}')
        
        # RL score
        ax.scatter(i, rl_score_ls[i], color=color, marker='x', s=100, label=f'RL - {lesion_id}')
        
        # Add boxplot for human scores
        bp = ax.boxplot([total_scores], positions=[i], widths=0.3, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.3), medianprops=dict(color='black'))
    
    ax.set_xticks(range(len(lesion_ids)))
    ax.set_xlabel('Lesion ID')
    ax.set_ylabel('Score')
    
    # Adjust legend to show only one entry per lesion
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    ax.set_xlim(-1, len(lesion_ids))
    ax.set_ylim(50, 95)
    ax.set_xlabel("Lesion ID")
    ax.set_ylabel("Score")
    ax.set_xticks(range(len(lesion_ids)))
    ax.set_xticklabels(range(1, len(lesion_ids)+1))
    
    plt.savefig(os.path.join(FIG_DIR, "human_vs_rl.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Compute and print the average radial difference of the best trials
    avg_best_radial_diff = sum(best_radial_diffs) / len(best_radial_diffs)
    print(f"Average radial difference of the best trials: {avg_best_radial_diff:.4f}")
    
    return avg_best_radial_diff


if __name__ == "__main__":
    # ablation_res_map = np.load('/mnt/sda1/data/conformal_ablation/20221111/test2-results/ablated_mask.npy')
    # lesion_mask = np.load('/home/yiwei/ConformalAblation/lesion_masks/icap/11-S66.npy')
    # eval_real_ablation_results(ablation_res_map, lesion_mask)

    # overlay_contours(ablation_res_map, lesion_mask)

    # ablation_res = np.load(r'C:\Users\adam\Sound\ConformalAblation\iCAP\data\human_plan\00000-000-S73\best\ablation_res.npy')
    # eval_sim_results(ablation_res)

    # plan_dir = '/home/yiwei/ConformalAblation/icap_plan'
    # lesion_ls = sorted(glob.glob(plan_dir + "/*"))
    # print(lesion_ls)

    # for lesion_dir in lesion_ls:
    #     analyze_plan_effort(lesion_dir)

    human_data_folder = r'/mnt/sda1/data/conformal_ablation/human_plan'
    human_score_ls = eval_human_planning(human_data_folder)
    rl_score_ls = [82.674, 86.475, 77.358, 83.751, 79.623, 78.745, 82.163, 81.996]

    compare_human_vs_rl(human_score_ls, rl_score_ls)    

    # viz_best_human_planning(human_data_folder)

    # evaluate_radial_diff()
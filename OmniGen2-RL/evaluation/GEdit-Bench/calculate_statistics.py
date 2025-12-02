import os
import pandas as pd
from collections import defaultdict
import sys
import numpy as np
import math

GROUPS = [
    "background_change",
    "color_alter",
    "style_change",
    "subject-add",
    "subject-remove",
    "subject-replace",
    "material_alter",
    "motion_change",
    "ps_human",
    "text_change",
    "tone_transfer",
]

GROUPS2 = [
    "background_change",
    "color_alter",
    "material_alter",
    "motion_change",
    "ps_human",
    "style_change",
    "subject-add",
    "subject-remove",
    "subject-replace",
    "text_change",
    "tone_transfer",
]

def analyze_scores(result_dir, language, num_samples):    # 这些 group_scores 字典用于存储每个 group 的最终平均分
    group_scores_semantics = {}
    group_scores_quality = {}
    group_scores_overall = {}
    group_scores_semantics_intersection = {}
    group_scores_quality_intersection = {}
    group_scores_overall_intersection = {}
    
    # 外部循环，处理每一个 group
    for group_name in GROUPS:
        data_point_samples = defaultdict(list)
        
        # 循环读取 num_samples 个评分文件
        for turn in range(num_samples):
            csv_path = os.path.join(result_dir, f"{group_name}_gpt_score{'_sample' + str(turn) if turn > 0 else ''}.csv")
            if not os.path.exists(csv_path):
                print(f"Warning: File not found, skipping: {csv_path}")
                continue

            with open(csv_path, 'r') as f:
                df = pd.read_csv(f)
                
                for _, row in df.iterrows():
                    # 过滤语言
                    if row['instruction_language'] != language:
                        continue
                    
                    # 定义唯一标识符
                    unique_key = os.path.basename(row['source_image']).split('_SRCIMG')[0]
                    
                    # 计算 overall_score
                    semantics_score = row['sementics_score']
                    quality_score = row['quality_score']
                    overall_score = math.sqrt(semantics_score * quality_score)
                    
                    # 将当前样本的分数信息存入字典
                    sample_data = {
                        'semantics_score': semantics_score,
                        'quality_score': quality_score,
                        'overall_score': overall_score,
                        'intersection_exist': row['intersection_exist']
                    }
                    
                    # 按唯一标识符聚合所有样本
                    data_point_samples[unique_key].append(sample_data)

        # --- 核心改动部分：第二阶段 - 筛选与计算 ---
        # 现在 data_point_samples 已经收集了所有测试项的所有样本数据。
        # 我们需要遍历它，为每个测试项找到最佳样本，然后将最佳分数存入最终列表。
        
        best_semantics_scores = []
        best_quality_scores = []
        best_overall_scores = []
            
        for unique_key, samples in data_point_samples.items():
            if not samples:
                continue

            # 从当前测试项的所有样本中，找到 overall_score 最高的那个
            # max() 函数的 key 参数可以让我们指定按字典中的哪个值来比较
            best_sample = max(samples, key=lambda s: s['overall_score'])
            
            # 将这个最佳样本的分数添加到最终列表中
            best_semantics_scores.append(best_sample['semantics_score'])
            best_quality_scores.append(best_sample['quality_score'])
            best_overall_scores.append(best_sample['overall_score'])
            
        
        group_scores_semantics[group_name] = np.mean(best_semantics_scores)
        group_scores_quality[group_name] = np.mean(best_quality_scores)
        group_scores_overall[group_name] = np.mean(best_overall_scores)

    print("\n--- Overall Model Averages ---")

    print("\nSemantics:")
    model_scores = [group_scores_semantics[group] for group in GROUPS]
    model_avg = np.mean(model_scores)
    group_scores_semantics["avg_semantics"] = model_avg

    # print("\nSemantics Valid Num:")
    # model_scores = [group_scores_semantics_valid_num[group] for group in GROUPS]
    # model_avg = np.mean(model_scores)
    # group_scores_semantics_valid_num["avg_semantics_valid_num"] = model_avg

    # print("\nSemantics Intersection:")
    # model_scores = [group_scores_semantics_intersection[group] for group in GROUPS]
    # model_avg = np.mean(model_scores)
    # group_scores_semantics_intersection["avg_semantics"] = model_avg

    # print("\nSemantics Valid Num Intersection:")
    # model_scores = [group_scores_semantics_valid_num_intersection[group] for group in GROUPS]
    # model_avg = np.mean(model_scores)
    # group_scores_semantics_valid_num_intersection["avg_semantics_valid_num"] = model_avg

    print("\nQuality:")
    model_scores = [group_scores_quality[group] for group in GROUPS]
    model_avg = np.mean(model_scores)
    group_scores_quality["avg_quality"] = model_avg

    # print("\nQuality Valid Num:")
    # model_scores = [group_scores_quality_valid_num[group] for group in GROUPS]
    # model_avg = np.mean(model_scores)
    # group_scores_quality_valid_num["avg_quality_valid_num"] = model_avg

    # print("\nQuality Intersection:")
    # model_scores = [group_scores_quality_intersection[group] for group in GROUPS]
    # model_avg = np.mean(model_scores)
    # group_scores_quality_intersection["avg_quality"] = model_avg

    # print("\nQuality Valid Num Intersection:")
    # model_scores = [group_scores_quality_valid_num_intersection[group] for group in GROUPS]
    # model_avg = np.mean(model_scores)
    # group_scores_quality_valid_num_intersection["avg_quality_valid_num"] = model_avg

    print("\nOverall:")
    model_scores = [group_scores_overall[group] for group in GROUPS]
    model_avg = np.mean(model_scores)
    group_scores_overall["avg_overall"] = model_avg

    # print("\nOverall Valid Num:")
    # model_scores = [group_scores_overall_valid_num[group] for group in GROUPS]
    # model_avg = np.mean(model_scores)
    # group_scores_overall_valid_num["avg_overall_valid_num"] = model_avg


    return (
        group_scores_semantics,
        group_scores_quality,
        group_scores_overall,
        # group_scores_semantics_valid_num,
        # group_scores_quality_valid_num,
        # group_scores_overall_valid_num
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="/results/")
    parser.add_argument("--language", type=str, default="en", choices=["en", "cn"])
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--groups", type=str, default="GROUPS", choices=["GROUPS", "GROUPS2"])
    args = parser.parse_args()
    result_dir = args.result_dir

    # result_dir = os.path.join(result_dir, "viescore")

    print("\nOverall:")

    (
        group_scores_semantics,
        group_scores_quality,
        group_scores_overall,
        # group_scores_semantics_valid_num,
        # group_scores_quality_valid_num,
        # group_scores_overall_valid_num
    ) = analyze_scores(result_dir, language=args.language, num_samples=args.num_samples)
    
    if args.groups == "GROUPS":
        groups = GROUPS
    else:
        groups = GROUPS2

    for group_name in groups:
        print(f"{group_name}: {group_scores_semantics[group_name]:.2f}, {group_scores_quality[group_name]:.2f}, {group_scores_overall[group_name]:.2f}")

    print(f"Average: {group_scores_semantics['avg_semantics']:.2f}, {group_scores_quality['avg_quality']:.2f}, {group_scores_overall['avg_overall']:.2f}")

    print("Semantics: " + " & ".join([f"{group_scores_semantics[group_name]:.2f}" for group_name in groups] + [f"{group_scores_semantics['avg_semantics']:.2f}"]))
    print("Quality: " + " & ".join([f"{group_scores_quality[group_name]:.2f}" for group_name in groups] + [f"{group_scores_quality['avg_quality']:.2f}"]))
    print("Overall: " + " & ".join([f"{group_scores_overall[group_name]:.2f}" for group_name in groups] + [f"{group_scores_overall['avg_overall']:.2f}"]))

    # print("\nValid Num:")
    # for group_name in GROUPS:
    #     print(f"{group_name}: {group_scores_semantics_valid_num[group_name]:.2f}, {group_scores_quality_valid_num[group_name]:.2f}, {group_scores_overall_valid_num[group_name]:.2f}")

    # print(f"Average Valid Num: {group_scores_semantics_valid_num['avg_semantics_valid_num']:.2f}, {group_scores_quality_valid_num['avg_quality_valid_num']:.2f}, {group_scores_overall_valid_num['avg_overall_valid_num']:.2f}")

    # print("\nIntersection:")
    # for group_name in GROUPS:
    #     print(f"{group_name}: {group_scores_semantics_intersection[group_name]:.2f}, {group_scores_quality_intersection[group_name]:.2f}, {group_scores_overall_intersection[group_name]:.2f}")

    # print(f"Average Intersection: {group_scores_semantics_intersection['avg_semantics']:.2f}, {group_scores_quality_intersection['avg_quality']:.2f}, {group_scores_overall_intersection['avg_overall']:.2f}")

    # print("\nValid Num Intersection:")
    # for group_name in GROUPS:
    #     print(f"{group_name}: {group_scores_semantics_valid_num_intersection[group_name]:.2f}, {group_scores_quality_valid_num_intersection[group_name]:.2f}, {group_scores_overall_valid_num_intersection[group_name]:.2f}")

    # print(f"Average Valid Num Intersection: {group_scores_semantics_valid_num_intersection['avg_semantics_valid_num']:.2f}, {group_scores_quality_valid_num_intersection['avg_quality_valid_num']:.2f}, {group_scores_overall_valid_num_intersection['avg_overall_valid_num']:.2f}")

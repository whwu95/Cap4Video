import numpy as np
import sys
sys.path.append(".")
from metrics import compute_metrics

def fusion_scores():

    video_matrix = np.load('sim_matrix/msrvtt_video_matrix.npy')
    titles_matrix = np.load('sim_matrix/msrvtt_titles_matrix.npy')

    print("video_matrix sim matrix size: {}, {}".format(video_matrix.shape, video_matrix.shape))
    print("titles_shot_matrix sim matrix size: {}, {}".format(titles_matrix.shape, titles_matrix.shape))
    tv_video_metrics = compute_metrics(video_matrix)
    vt_video_metrics = compute_metrics(video_matrix.T)
    print('\t Length-T: {}, Length-V:{}'.format(len(video_matrix), len(video_matrix[0])))
    print("Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
          format(tv_video_metrics['R1'], tv_video_metrics['R5'], tv_video_metrics['R10'],
                 tv_video_metrics['MR'], tv_video_metrics['MeanR']))
    print("Video-to-Text:")
    print('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
          format(vt_video_metrics['R1'], vt_video_metrics['R5'], vt_video_metrics['R10'],
                 vt_video_metrics['MR'], vt_video_metrics['MeanR']))

    tv_titles_metrics = compute_metrics(titles_matrix)
    vt_titles_metrics = compute_metrics(titles_matrix.T)
    print('\t Length-T: {}, Length-V:{}'.format(len(titles_matrix), len(titles_matrix[0])))
    print("Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
          format(tv_titles_metrics['R1'], tv_titles_metrics['R5'], tv_titles_metrics['R10'],
                 tv_titles_metrics['MR'], tv_titles_metrics['MeanR']))
    print("Video-to-Text:")
    print('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
          format(vt_titles_metrics['R1'], vt_titles_metrics['R5'], vt_titles_metrics['R10'],
                 vt_titles_metrics['MR'], vt_titles_metrics['MeanR']))

    a = 0.73
    b = 0.27
    fusion_sim_matrix = a * video_matrix + b * titles_matrix
    fusion_tv_metrics = compute_metrics(fusion_sim_matrix)
    fusion_vt_metrics = compute_metrics(fusion_sim_matrix.T)
    print('\t Length-T: {}, Length-V:{}'.format(len(fusion_sim_matrix), len(fusion_sim_matrix[0])))
    print("Text-to-Video fusion:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
          format(fusion_tv_metrics['R1'], fusion_tv_metrics['R5'], fusion_tv_metrics['R10'],
                 fusion_tv_metrics['MR'], fusion_tv_metrics['MeanR']))
    print("Video-to-Text fusion :")
    print('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
          format(fusion_vt_metrics['R1'], fusion_vt_metrics['R5'], fusion_vt_metrics['R10'],
                 fusion_vt_metrics['MR'], fusion_vt_metrics['MeanR']))

if __name__ == "__main__":
    fusion_scores()
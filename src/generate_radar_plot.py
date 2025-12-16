import matplotlib.pyplot as plt
import numpy as np

def generate_radar_plot(metrics, save_path="radar_plot.png"):
    tasks = []
    scores = []

    if "code_search" in metrics:
        tasks.append("Code Search")
        scores.append(metrics["code_search"]["accuracy"])

    if "clone_detection" in metrics:
        tasks.append("Clone Detection")
        scores.append(metrics["clone_detection"]["f1"])

    if "code_repair" in metrics:
        tasks.append("Code Repair")
        scores.append(metrics["code_repair"]["pass@1"])

    if "test_generation" in metrics:
        tasks.append("Test Generation")
        scores.append(metrics["test_generation"]["bug_detection@1"])

    angles = np.linspace(0, 2*np.pi, len(scores), endpoint=False)
    scores += scores[:1]
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, scores)
    ax.fill(angles, scores, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, tasks)
    ax.set_ylim(0,1)

    plt.title("Multi-Task Performance Radar")
    plt.savefig(save_path)
    plt.close()

    print(f"Radar plot saved to {save_path}")

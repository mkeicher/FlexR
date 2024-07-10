from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict

def plot_tsne(embeddings, hue_labels, annotation_labels, title=None, return_fig=False, annotation_cluster_threshold = 1, seed=42):

    # Assuming triplet_embeddings, triplets, triplet_attributes, triplet_locations are already defined
    tsne = TSNE(n_components=2, verbose=0, learning_rate='auto', init='pca', random_state=seed)
    tsne_results = tsne.fit_transform(embeddings)
    tsne_df = pd.DataFrame({
        # 'prompt': triplets, 
        'hue': hue_labels,
        'annotation': annotation_labels,
        'tsne_1': tsne_results[:, 0],
        'tsne_2': tsne_results[:, 1],
    })

    # Calculate pairwise distances
    distances = pdist(tsne_results)
    dist_matrix = squareform(distances)

    # Define a distance threshold
    distance_threshold = annotation_cluster_threshold

    # Group nearby points with the same location
    def group_nearby_labels(tsne_df, dist_matrix, threshold):
        groups = defaultdict(list)
        visited = set()

        for i in range(len(tsne_df)):
            if i not in visited:
                group = [i]
                visited.add(i)
                for j in range(i + 1, len(tsne_df)):
                    if tsne_df.loc[i, 'annotation'] == tsne_df.loc[j, 'annotation'] and dist_matrix[i, j] < threshold:
                        group.append(j)
                        visited.add(j)
                groups[tsne_df.loc[i, 'annotation']].append(group)
        return groups

    # Apply grouping
    merged_groups = group_nearby_labels(tsne_df, dist_matrix, distance_threshold)

    fig = plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        x="tsne_1", y="tsne_2",
        hue="hue",
        palette=sns.color_palette("hls", len(tsne_df.hue.unique())),
        data=tsne_df,
        legend="full",
        alpha=.75, s=100
    )
    sns.move_legend(ax, "upper right", bbox_to_anchor=(0, 1))

    texts = []
    for location, groups in merged_groups.items():
        for group in groups:
            if len(group) > 1:
                x = np.mean(tsne_df.loc[group, 'tsne_1'])
                y = np.mean(tsne_df.loc[group, 'tsne_2'])
                texts.append(ax.text(x, y, location, 
                                    horizontalalignment='left', 
                                    size=6, color='black'))
            else:
                idx = group[0]
                texts.append(ax.text(tsne_df.loc[idx, 'tsne_1'], tsne_df.loc[idx, 'tsne_2'], 
                                    location, 
                                    horizontalalignment='left', 
                                    size=6, color='black'))

    # Adjust text to avoid overlaps
    adjust_text(texts) #, arrowprops=dict(arrowstyle='->', color='red'))

    if title is not None:
        plt.title(title)

    if return_fig:
        return fig
    else:
        plt.show()


def plot_tsne_group(embeddings_dict, hue_labels, annotation_labels, title=None, return_fig=False, annotation_cluster_threshold=2, seed=42):
    
    def tsne_transform(embeddings, seed):
        tsne = TSNE(n_components=2, verbose=0, learning_rate='auto', init='pca', random_state=seed)
        return tsne.fit_transform(embeddings)

    def group_nearby_labels(tsne_df, dist_matrix, threshold):
        groups = defaultdict(list)
        visited = set()
        for i in range(len(tsne_df)):
            if i not in visited:
                group = [i]
                visited.add(i)
                for j in range(i + 1, len(tsne_df)):
                    if tsne_df.loc[i, 'annotation'] == tsne_df.loc[j, 'annotation'] and dist_matrix[i, j] < threshold:
                        group.append(j)
                        visited.add(j)
                groups[tsne_df.loc[i, 'annotation']].append(group)
        return groups

    def create_tsne_df(embeddings, hue_labels, annotation_labels, seed):
        tsne_results = tsne_transform(embeddings, seed)
        tsne_df = pd.DataFrame({
            'attributes': hue_labels,
            'annotation': annotation_labels,
            'tsne_1': tsne_results[:, 0],
            'tsne_2': tsne_results[:, 1],
        })
        return tsne_df

    def plot_single_tsne(ax, tsne_df, dist_matrix, distance_threshold, subtitle, legend=False):
        merged_groups = group_nearby_labels(tsne_df, dist_matrix, distance_threshold)
        sns.scatterplot(
            x="tsne_1", y="tsne_2",
            hue="attributes",
            palette=sns.color_palette("hls", len(tsne_df.attributes.unique())),
            data=tsne_df,
            legend=legend,
            alpha=.75, s=100,
            ax=ax
        )
        texts = []
        for location, groups in merged_groups.items():
            for group in groups:
                if len(group) > 1:
                    x = np.mean(tsne_df.loc[group, 'tsne_1'])
                    y = np.mean(tsne_df.loc[group, 'tsne_2'])
                    texts.append(ax.text(x, y, location, 
                                        horizontalalignment='left', 
                                        size=7, color='black'))
                else:
                    idx = group[0]
                    texts.append(ax.text(tsne_df.loc[idx, 'tsne_1'], tsne_df.loc[idx, 'tsne_2'], 
                                        location, 
                                        horizontalalignment='left', 
                                        size=7, color='black'))
        adjust_text(texts, ax=ax)
        ax.set_title(subtitle)
        ax.set_axis_off()

    num_embeddings = len(embeddings_dict)
    fig, axes = plt.subplots(1, num_embeddings, figsize=(10 * num_embeddings, 8))
    
    if title is not None:
        plt.suptitle(title, y=0.95, fontsize=16)

    for i, (ax, (subtitle, embeddings)) in enumerate(zip(axes, embeddings_dict.items())):
        tsne_df = create_tsne_df(embeddings, hue_labels, annotation_labels, seed)
        distances = pdist(tsne_df[['tsne_1', 'tsne_2']])
        dist_matrix = squareform(distances)
        plot_single_tsne(ax, tsne_df, dist_matrix, annotation_cluster_threshold, subtitle, legend=i == 0)

    plt.subplots_adjust(wspace=0.05) 

    if return_fig:
        return fig
    else:
        plt.show()
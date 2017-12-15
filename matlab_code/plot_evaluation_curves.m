
%% Plot our results

labels_path = '/Users/ignaciorlando/Documents/Research/encoder-based-regularization/data/DRIVE/test/labels';
labels_extension = '.gif';

scores_path = '/Users/ignaciorlando/Documents/Research/encoder-based-regularization/results/DRIVE/test/without-augmentation/scores_mat/';
scores_extension = '.mat';

fov_masks_path = '/Users/ignaciorlando/Documents/Research/encoder-based-regularization/data/DRIVE/test/masks';
masks_extension = '.gif';

[auc_ours, auc_pr_ours, tpr, fpr, recall, precision] = get_curves(scores_path, scores_extension, labels_path, labels_extension, fov_masks_path, masks_extension);

figure, plot(recall, precision, 'LineWidth', 2);
hold on

%% Plot other results

labels_path = '/Users/ignaciorlando/Documents/Research/encoder-based-regularization/data/DRIVE/test/labels';
labels_extension = '.gif';

scores_path = '/Users/ignaciorlando/Downloads/DRIU_DRIVE_precomputed';
scores_extension = '.png';

fov_masks_path = '/Users/ignaciorlando/Documents/Research/encoder-based-regularization/data/DRIVE/test/masks';
masks_extension = '.gif';

[auc_driu, auc_pr_driu, tpr, fpr, recall, precision] = get_curves(scores_path, scores_extension, labels_path, labels_extension, fov_masks_path, masks_extension);
plot(recall, precision, 'LineWidth', 2);

%% Add the legend
legend({['Ours. AUC-PR/RE = ', num2str(auc_pr_ours)], ['DRIU. AUC-PR/RE = ', num2str(auc_pr_driu)]}, 'Location', 'southwest');
xlim([0.3 1])
ylim([0.3 1])
xlabel('Recall');
ylabel('Precision');
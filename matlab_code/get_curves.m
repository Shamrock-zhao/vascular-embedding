

function [auc, auc_pr, tpr, fpr, recall, precision] = get_curves(scores_path, score_extension, label_path, labels_extension, fov_masks_path, masks_extension)

    % get score filenames
    scores_files = dir(fullfile(scores_path, strcat('*', score_extension)));
    scores_files = {scores_files.name};
    
    % get label filenames
    label_files = dir(fullfile(label_path, strcat('*', labels_extension)));
    label_files = {label_files.name};
    
    % get fov masks filenames
    fov_masks_files = dir(fullfile(fov_masks_path, strcat('*', masks_extension)));
    fov_masks_files = {fov_masks_files.name};
    
    % control if the number of files is correct
    assert(length(scores_files) == length(label_files), 'We need the same number of scores al label files');
    
    % concatenate all the scores and labels
    scores_ = [];
    labels_ = [];
    for i = 1 : length(scores_files)
        
        % load scores and labels
        if strcmp(score_extension, '.mat')
            load(fullfile(scores_path, scores_files{i}));
        else
            scores = imread(fullfile(scores_path, scores_files{i}));
        end
        current_labels = 2 * (imread(fullfile(label_path, label_files{i})) > 0) - 1;
        % load the mask
        current_fov_mask = imread(fullfile(fov_masks_path, fov_masks_files{i})) > 0;
        % concatenate
        scores_ = cat(1, scores_, scores(current_fov_mask));
        labels_ = cat(1, labels_, current_labels(current_fov_mask));
        
    end
    
    % roc curve
    [tpr, tnr, info] = vl_roc(labels_, scores_);
    fpr = 1 - tnr;
    auc = info.auc;
    % pr/re curve
    [recall, precision, info] = vl_pr(labels_, scores_);
    auc_pr = info.auc;

end
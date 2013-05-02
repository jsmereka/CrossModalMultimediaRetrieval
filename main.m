close all; clear; clc;
addtopath;
dbstop if error;
if(isunix)
    root = '/afs/ece.cmu.edu/project/kumar/work8/smerekaj/wikipedia_dataset/';
else
    root = 'E:/Datasets/Media Retrieval/wikipedia_dataset/';
end

showgraph = 1;

builddata = 1; % if(1) then get data from the files and run feature extraction, otherwise load features from ACM-MM paper

% params for text features
forcerebuildtxt = 0; % if(1) forces the feature extraction to be performed again
stoplistfile = 'stopwordlist.txt';
freqcutoff = 0; % cut off words that are under a certain frequency
remove_nums = 1; % remove numbers from text data
dlda = 2; % 0 = LDA, 1 = DLDA, 2 = NB, 3 = loads ACM paper

% params for img features
forcerebuildimg = 0; % if(1) forces the feature extraction to be performed again
imgsize = [300 300]; % resize loaded images
sec_rows = 7; sec_cols = 7; % for image segmentation using gist descriptors
featnum = 1; % 0 = sift, 1 = gist
modelnum = 1; % Bag of words: 0 = vector quantized counts, 1 = naive bayes, 2 = plsa, Raw features = 4, loads ACM paper = 3


if(builddata)
    fprintf(1,'1) Building Text and Image Training Features\n');
    [T_tr, cat_ids_trn, alpha, beta, delta, txtlabelidx] = build_txt_features(root, stoplistfile, freqcutoff, remove_nums, dlda, forcerebuildtxt);
    
    [I_tr, ~, trnparam1, trnparam2, imglabelidx] = build_img_features(root, imgsize, [sec_rows, sec_cols], featnum, modelnum, forcerebuildimg);
    
    fprintf(1,'\n');
else
    fprintf(1,'1) Loading Text and Image Training Features\n\n');
    % load preset training data
    load(fullfile(root,'raw_features.mat'),'T_tr','I_tr');
    cat_ids_trn = loaddata(root,'trainset_txt_img_cat.list');
end

fprintf(1,'2) Learning projections from training features\n\n');
% CCA
options.PrjX = 1; options.PrjY = 1;
options.RegX = 0; options.RegY = 0;
[W_i, W_t] = CCA(I_tr', T_tr', options);

P_i = (W_i'*I_tr')'; P_t = (W_t'*T_tr')';

% raw_training_images = P_i-repmat(mean(P_i), size(P_i,1), 1);
% raw_training_text = P_t-repmat(mean(P_t), size(P_t,1), 1);
raw_training_images = P_i;
raw_training_text = P_t;

fprintf(1,'3) Learning multinomial regression fit from projections\n\n');
% logistic regression
warning off all;
image_fit = glmnet(raw_training_images, cat_ids_trn, 'multinomial');
text_fit = glmnet(raw_training_text, cat_ids_trn, 'multinomial');
warning on all;

clear P_i P_t I_tr T_tr raw_training_images raw_training_text cat_ids_trn options;

fprintf(1,'\n');

if(builddata)
    fprintf(1,'1) Building Text and Image Testing Features\n');
    % get features to produce T_te, I_te
    [T_te, cat_ids_tst] = apply_txt_features(root, alpha, beta, delta, txtlabelidx, dlda, forcerebuildtxt);
    
    I_te = apply_img_features(root, imgsize, [sec_rows, sec_cols], trnparam1, trnparam2, imglabelidx, featnum, modelnum, forcerebuildimg);
    
    fprintf(1,'\n');
else
    fprintf(1,'1) Loading Text and Image Testing Features\n\n');
    % load preset training data
    load(fullfile(root, 'raw_features.mat'),'T_te','I_te');
    cat_ids_tst = loaddata(root,'testset_txt_img_cat.list');
end

fprintf(1,'2) Applying projections to testing features\n\n');
% CCA
P_i = (W_i'*I_te')'; P_t = (W_t'*T_te')';

% raw_testing_images = P_i-repmat(mean(P_i), size(P_i,1), 1);
% raw_testing_text = P_t-repmat(mean(P_t), size(P_t,1), 1);
raw_testing_images = P_i;
raw_testing_text = P_t;

clear I_te P_i W_i T_te P_t W_t root builddata;

L = length(cat_ids_tst);

fprintf(1,'3) Applying multinomial regression to projected test data\n\n');
% logistic regression
semantic_testing_images = glmnetPredict(image_fit, 'response', raw_testing_images);
semantic_testing_images = semantic_testing_images(:,:,end);
% rank 1-ID for category 
[~, img_testing_idx] = max(semantic_testing_images,[],2);
img_cat_correct = L-sum((img_testing_idx-cat_ids_tst)~=0);
fprintf(1,'Image category acc = %.2f%%\n', (img_cat_correct/L)*100);

semantic_testing_text = glmnetPredict(text_fit, 'response', raw_testing_text);
semantic_testing_text = semantic_testing_text(:,:,end);
% rank 1-ID for category
[~, txt_testing_idx] = max(semantic_testing_text,[],2);
txt_cat_correct = L-sum((txt_testing_idx-cat_ids_tst)~=0);
fprintf(1,'Text category acc  = %.2f%%\n', (txt_cat_correct/L)*100);

clear image_fit text_fit txt_testing_idx img_testing_idx a;


% semantic matching
topics = max(cat_ids_tst);
disttype = '';
fprintf(1,'\n\nRetreiving text from image query and image from text query\n\n');
for i = 2:2
    switch(i)
        case 1
            disttype = 'cosine';
        case 2
            disttype = 'correlation';
        case 3
            disttype = 'chebychev';
        case 4
            disttype = 'euclidean';
        case 5
            disttype = 'cityblock';
        case 6
            disttype = 'hamming';
    end
    % query image to produce rankings of all the text
    [text_idx, img_query_dist] = knnsearch(semantic_testing_text,semantic_testing_images,'Distance',disttype,'k',L);
    img_query_mat = cat_ids_tst(text_idx);
    img_query_auth = (img_query_mat == repmat(cat_ids_tst,[1,L]));
    img_ap = zeros(L,1);
    for j = 1:L
        [recall, precision] = util_calc_precision_recall_curve(img_query_auth(j,:));
        img_ap(j) = util_average_precision(recall',precision');
    end
    fprintf(1,'%s: Image mean ap = %.4f, ACM-MM Paper = 0.277\n', disttype, mean(img_ap));
    
    imgconf = zeros(topics);
    for q = 1:topics
        for j = 1:topics
            ap = zeros(sum(cat_ids_tst == q | cat_ids_tst == j),1);
            for k = 1:sum(cat_ids_tst == q | cat_ids_tst == j)
                [recall, precision] = util_calc_precision_recall_curve(img_query_auth(k,(cat_ids_tst == q | cat_ids_tst == j)));
                ap(k) = util_average_precision(recall',precision');
            end
            ap(isnan(ap) | isinf(ap)) = 0;
            imgconf(q,j) = mean(ap);
        end
    end
    
    
    if(showgraph)
        figure;
        imagesc(imgconf); colorbar; colormap gray; title('Image Query to Retrieve Text');
        figure;
        [recall, precision] = util_calc_precision_recall_curve(img_query_auth(:));
        tmp = randi([1 max(cat_ids_tst)],L,L);
        tmp = (tmp == repmat(cat_ids_tst,[1,L]));
        [ran_recall, ran_precision] = util_calc_precision_recall_curve(tmp(:));
        plot(recall,precision,'-k','LineWidth',2); 
        hold on; plot(ran_recall,ran_precision,'--r','LineWidth',2); 
        xlabel('recall'); ylabel('precision'); title('Image Query to Retrieve Text');
        legend('Model','Random','Location','NorthEast'); box on;
        axis([0 1 0 1]);
    end
    
    % query text to produce rankings of all the images
    [img_idx, txt_query_dist] = knnsearch(semantic_testing_images,semantic_testing_text,'Distance',disttype,'k',L);
    txt_query_mat = cat_ids_tst(img_idx);
    txt_query_auth = (txt_query_mat == repmat(cat_ids_tst,[1,L]));
    txt_ap = zeros(L,1);
    for j = 1:L
        [recall, precision] = util_calc_precision_recall_curve(txt_query_auth(j,:));
        txt_ap(j) = util_average_precision(recall',precision');
    end
    fprintf(1,'%s: Text mean ap  = %.4f, ACM-MM Paper = 0.226\n', disttype, mean(txt_ap));
    
    txtconf = zeros(topics);
    for q = 1:topics
        for j = 1:topics
            ap = zeros(sum(cat_ids_tst == q | cat_ids_tst == j),1);
            for k = 1:sum(cat_ids_tst == q | cat_ids_tst == j)
                [recall, precision] = util_calc_precision_recall_curve(txt_query_auth(k,(cat_ids_tst == q | cat_ids_tst == j)));
                ap(k) = util_average_precision(recall',precision');
            end
            ap(isnan(ap) | isinf(ap)) = 0;
            txtconf(q,j) = mean(ap);
        end
    end
    
    if(showgraph)
        figure;
        imagesc(txtconf); colorbar; colormap gray; title('Text Query to Retrieve Image');
        figure;
        [recall, precision] = util_calc_precision_recall_curve(txt_query_auth(:));
        tmp = randi([1 max(cat_ids_tst)],L,L);
        tmp = (tmp == repmat(cat_ids_tst,[1,L]));
        [ran_recall, ran_precision] = util_calc_precision_recall_curve(tmp(:));
        plot(recall,precision,'-k','LineWidth',2); 
        hold on; plot(ran_recall,ran_precision,'--r','LineWidth',2); 
        xlabel('recall'); ylabel('precision'); title('Text Query to Retrieve Image');
        legend('Model','Random','Location','NorthEast'); box on;
        axis([0 1 0 1]);
    end
    
end



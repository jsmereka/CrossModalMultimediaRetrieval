function [T_te, cat_ids_tst] = apply_txt_features(root, alpha, beta, delta, labelindex, dlda, forcerebuild)

flag = 0;


% get features to produce T_te
if(forcerebuild)
    flag = 1;
else
    % process takes awhile to do...use what i've already got done
    try
        % load preprocessed text infomation
        if(dlda == 1)
            load('Txt_tst_DLDA.mat','T_te','cat_ids_tst');
        elseif(dlda == 0)
            load('Txt_tst_LDA.mat','T_te','cat_ids_tst'); 
        elseif(dlda == 2)
            load('Txt_tst_NB.mat','T_te','cat_ids_tst'); 
        end
        fprintf(1,'\tLoaded testing text features\n');
    catch e
        flag = 1;
    end
end
if(dlda == 3)
    load(fullfile(root,'raw_features.mat'),'T_te');
    cat_ids_tst = loaddata(root,'testset_txt_img_cat.list');
    flag = 0;
end
if(flag)
    try
        % load training data and vocabulary
        load('vocab.mat'); fprintf(1,'\tLoaded vocabulary database\n');
        load('loadeddata_txt_tst.mat','txts','cat_ids_tst'); fprintf(1,'\tLoaded test docs\n');
    catch e
        error('Run training before test to build vocabulary and get data from file');
    end
    
    % format text data
    fprintf('\tFormating input data\n');
    %WordOccurance = format_txt_data(txts,words,cat_ids_tst,freqcutoff,1);
    load('tmpdlda_tst.mat');
    
    if(dlda == 1)
        fprintf(1,'\tUsing D-LDA to derive text features\n');
        
        c = max(cat_ids_tst);   % classes
        
        [M,V] = size(WordOccurance);
        
        % get phi on test documents
        phi = fastLdaEstep(alpha,beta,WordOccurance);
        
        % get features to produce T_te
        rawY = delta' * phi; %(c-1)*M
        rawY = [rawY; zeros(1,M)];  %c*M
        T_te = exp(rawY) .* (ones(c,1) * (1./sum(exp(rawY),1)));
        T_te = T_te';
        
        save('Txt_tst_DLDA.mat','T_te','cat_ids_tst');
    elseif(dlda == 0)
        fprintf(1,'\tUsing LDA to derive text features\n');
        
        N = 700; % number of iterations
        k = size(delta,1); % number of clusters
        
        fprintf(1,'\tGetting text features\n');
        T_te = sBPpredict(WordOccurance', delta, N, alpha, beta, 1, 0);
        
        % reshuffel features accordingly
        T_te(1:k,:) = T_te(labelindex,:);
        
        T_te = T_te';
        
        save('Txt_tst_LDA.mat','T_te','cat_ids_tst');
    elseif(dlda == 2)
        predict = log(alpha) * WordOccurance' + repmat(log(beta),1,size(WordOccurance,1));
        [~,class] = max(predict); class = class';
        T_te = predict';
        %acc = 100 * (sum((class-cat_ids_tst)==0)/length(cat_ids_tst));
        save('Txt_tst_NB.mat','T_te','cat_ids_tst');
    end    
end
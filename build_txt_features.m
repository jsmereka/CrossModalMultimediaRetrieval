function [T_tr, cat_ids_trn, alpha, beta, delta, labelindex] = build_txt_features(root, stoplistfile, freqcutoff, remove_nums, dlda, forcerebuild)

flag = 0;

% get features to produce T_tr
if(forcerebuild)
    flag = 1;
else
    % process takes awhile to do...use what i've already got done
    try
        % load preprocessed text infomation
        if(dlda == 1)
            load('Txt_trn_DLDA.mat','T_tr','cat_ids_trn','alpha','beta','delta','labelindex');
        elseif(dlda == 0)
            load('Txt_trn_LDA.mat','T_tr','cat_ids_trn','alpha','beta','delta','labelindex'); 
        elseif(dlda == 2)
            load('Txt_trn_NB.mat','T_tr','cat_ids_trn','alpha','beta','delta','labelindex'); 
        end
        fprintf(1,'\tLoaded training text features\n');
    catch e
        flag = 1;
    end
end
if(dlda == 3)
    load(fullfile(root,'raw_features.mat'),'T_tr');
    cat_ids_trn = loaddata(root,'trainset_txt_img_cat.list');
    alpha = []; beta = []; delta = []; labelindex = []; flag = 0;
end
if(flag)
    try
        % load training data and vocabulary
        load('vocab.mat'); fprintf(1,'\tLoaded vocabulary database\n');
        load('loadeddata_txt_trn.mat','txts','cat_ids_trn'); fprintf(1,'\tLoaded training docs\n');
    catch e
        % load data from file and build vocabulary
        [cat_ids_tst, ~, txts] = loaddata(root,'testset_txt_img_cat.list',[],0,1);
        txts = filter_stopwords(txts,stoplistfile,remove_nums);
        save('loadeddata_txt_tst.mat','cat_ids_tst','txts');
        tmp = vertcat(txts{1:end});
        clear imgs cat_ids_tst;
        [cat_ids_trn, ~, txts] = loaddata(root,'trainset_txt_img_cat.list',[],0,1);
        txts = filter_stopwords(txts,stoplistfile,remove_nums);
        save('loadeddata_txt_trn.mat','cat_ids_trn','txts');
        tmp2 = vertcat(txts{1:end});
        words = unique(vertcat(tmp,tmp2));
        clear tmp tmp2;
        save('vocab.mat','words'); % vocabulary
    end
    
    % format text data
    if(dlda == 1)
        fprintf(1,'\tUsing D-LDA to derive text features\n');
        % supervised LDA
        fprintf(1,'\tFormating input data for training\n');
        %[WordOccurance, ClassLabel] = format_txt_data(txts,words,cat_ids_trn,freqcutoff,1);
        load('tmpdlda.mat');
        
        c = max(cat_ids_trn);   % classes
        k = c;                  % topics
        
        [M,V] = size(WordOccurance);
        
        %initalpha = rand(k,1); % k*1; Dirichlet Distribution
        initeta = ones(k,c-1); % k*(c-1), regression parameter for c-1 classes
        
        if(k == c)
            initbeta = [ClassLabel, ones(M,1)-sum(ClassLabel,2)]' * WordOccurance;
            initbeta = initbeta ./ (sum(initbeta,2) * ones(1,V)); % k*V; Discrete Distributions for k topics
        else
            initbeta = rand(k,V);
        end
        
        lap = 10^-5; % laplacian smoothing parameter
        
        
        flag = 0; % if flag=1, use the change of perplexity to check the convergence
        % if flag=0, use the change of parameter to check the convergence
        
        Nfolds = 5;
        
        % find a good alpha parameter
        fprintf(1,'\tCross-validation to tune initial alpha parameter\n');
        CVO = cvpartition(length(cat_ids_trn), 'kfold', Nfolds);
        acc = zeros(Nfolds,1);
        for i = 1:Nfolds
            trIdx = CVO.training(i);
            teIdx = CVO.test(i);
            
            initalpha = 10^-i .* 10^3 .* ones(k,1);
            
            [alph,bet,et] = learnFastDlda(WordOccurance(trIdx,:),ClassLabel(trIdx,:),initalpha,initbeta,initeta,lap,flag,0);
            
            phi = fastLdaEstep(alph,bet,WordOccurance(teIdx,:));
            
            rawY = et' * phi; %(c-1)*M
            rawY = [rawY; zeros(1,sum(teIdx))];  %c*M
            theta = exp(rawY) .* (ones(c,1) * (1./sum(exp(rawY),1)));
            
            [~,predY] = max(theta,[],1);
            
            acc(i) = 100*(sum((predY' - cat_ids_trn(teIdx))==0)/length(teIdx));
            fprintf(1,'\tIteration %i: Acc = %f%%\n', i, acc(i));
        end
        labelindex = [];
        lapp = find(acc == max(acc));
        
        [alpha,beta,delta,phi] = learnFastDlda(WordOccurance,ClassLabel,10^-lapp .* 10^3 .* ones(k,1),initbeta,initeta,lap,flag,0);
        save('DLDA_trn_params.mat','alpha','beta','delta','labelindex');
        
        % get features to produce T_trn
        rawY = delta' * phi; %(c-1)*M
        rawY = [rawY; zeros(1,M)];  %c*M
        T_tr = exp(rawY) .* (ones(c,1) * (1./sum(exp(rawY),1)));
        T_tr = T_tr';
        
        save('Txt_trn_DLDA.mat','T_tr','cat_ids_trn','alpha','beta','delta','labelindex');
    elseif(dlda == 0)
        % unsupervised LDA
        fprintf(1,'\tUsing LDA to derive text features\n');
        
        fprintf('\tFormating input data for training\n');
        WordOccurance = format_txt_data(txts,words,cat_ids_trn,freqcutoff,1);
                
        k = max(cat_ids_trn); % topics
        
        % hyperparameters
%         numiter = 10;
%         alph = [0.01:((50/k)-0.01)/numiter:(50/k)];
%         bet = [0.0001:(0.01-0.0001)/(length(alph)-1):0.01];
        bet = [0.5,0.25,0.1,0.01,0.001];
        alph = [0.5,0.25,0.1,0.01,0.001];
        
        N = 700; % number of iterations
        
        WordOccurance = WordOccurance';
        
        Nfolds = length(alph);
        
        fprintf(1,'\tCross-validation to tune hyperparameters\n');
        
        CVO = cvpartition(length(cat_ids_trn), 'kfold', Nfolds);
        acc = zeros(Nfolds,Nfolds); phi = cell(Nfolds,Nfolds); labelidx = phi;
        for j = 1:length(bet)
            trIdx = CVO.training(j);
            teIdx = CVO.test(j);
            for i = 1:length(alph)                
                [phi{j,i},trn_theta] = sBPtrain(WordOccurance(:, trIdx), k, N, alph(i), bet(j), 1, 0);
                theta = sBPpredict(WordOccurance(:, teIdx), phi{j,i}, N, alph(i), bet(j), 1, 0);

                [~,predX] = max(trn_theta,[],1);

                labelidx{j,i} = getnewlabels(cat_ids_trn(trIdx),predX,k);
                testlabel = cat_ids_trn(teIdx);
                newlabels = zeros(size(testlabel));
                for q = 1:k
                    newlabels((testlabel == q)) = labelidx{j,i}(q);
                end

                [~,predY] = max(theta,[],1);

                acc(j,i) = 100*(sum((predY' - newlabels)==0)/length(teIdx));
                fprintf(1,'\tIteration (%i, %i): alpha = %f, beta = %f, Acc = %f%%\n', j, i, alph(i), bet(j), acc(j,i));
            end
        end
        
        [r,c] = ind2sub(size(acc),find(acc == max(max(acc)),1));
        alpha = alph(c); beta = bet(r); 
        
        fprintf(1,'\tGetting text features\n');
        
        [delta,theta] = sBPtrain(WordOccurance, k, N, alpha, beta, 1, 0);
        [~,predX] = max(theta,[],1);
        labelindex = getnewlabels(cat_ids_trn,predX,k);
        % reshuffel features accordingly
        theta(1:k,:) = theta(labelindex,:);
        
        save('LDA_trn_params.mat','alpha','beta','delta','labelindex');
        T_tr = theta';
        
        save('Txt_trn_LDA.mat','T_tr','cat_ids_trn','alpha','beta','delta','labelindex');
    elseif(dlda == 2)
        %WordOccurance = format_txt_data(txts,words,cat_ids_trn,freqcutoff,1);
        load('tmpdlda.mat');
        alpha = 1/length(words);
        k = max(cat_ids_trn); % topics
        % training
        y = zeros(k,1);
        x = zeros(k,length(words))+alpha;
        for j = 1:k
            x(j,:) = x(j,:) + sum(WordOccurance((cat_ids_trn == j),:),1);
            % calculate priors
            x(j,:) = x(j,:) ./ sum(x(j,:));
            y(j) = sum(cat_ids_trn == j);
        end
        y = y ./ sum(y);
        
        % testing
        predict = log(x) * WordOccurance' + repmat(log(y),1,size(WordOccurance,1));
        [~,class] = max(predict); class = class';
        T_tr = predict';
        acc = 100 * (sum((class-cat_ids_trn)==0)/length(cat_ids_trn));
        alpha = x; beta = y; delta = []; labelindex = [];
        save('NB_trn_params.mat','alpha','beta','delta','labelindex');
        
        save('Txt_trn_NB.mat','T_tr','cat_ids_trn','alpha','beta','delta','labelindex');
    end
end
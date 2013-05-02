function [I_tr, cat_ids_trn, trnparam1, trnparam2, labelidx] = build_img_features(root, imgsize, rowcol, featid, modelid, forcerebuild)

flag = 0;
if(isunix)
    parallel = 1;
else
    parallel = 0;
end
reloadimgs = 0; % forcefully reload images from file
getpoints = 0; % get descriptors or load from file
docluster = 0; % perform kmeans or load centers from file
getcounts = 0; % get visual word counts or load from file
trnparam1 = []; trnparam2 = []; labelidx = []; acc = 0;
% get features to produce I_tr
if(forcerebuild)
    flag = 1;
else
    % process takes awhile to do...use what i've already got done
    try
        % load preprocessed image infomation
        if(featid == 0)
            load(['Img_trn_sift' num2str(modelid) '.mat'],'I_tr','cat_ids_trn');
        else
            load(['Img_trn_gist' num2str(modelid) '.mat'],'I_tr','cat_ids_trn');
        end
        switch(modelid)
            case {0,3}
                trnparam1 = []; trnparam2 = []; labelidx = [];
            case 1
                if(featid == 0)
                    load('ImgNB_trn_params_sift.mat','x','y');
                else
                    load('ImgNB_trn_params_gist.mat','x','y');
                end
                trnparam1 = x; trnparam2 = y; labelidx = [];
            case 2
                if(featid == 0)
                    load('ImgPLSA_trn_params_sift.mat','Pw_z','beta','labelidx');
                else
                    load('ImgPLSA_trn_params_gist.mat','Pw_z','beta','labelidx');
                end
                trnparam1 = Pw_z; trnparam2 = beta;
        end
        fprintf(1,'\tLoaded training image features\n');
    catch e
        flag = 1;
    end
end
if(flag)
    if(reloadimgs == 0)
        try
            % load training data
            load('loadeddata_img_trn.mat','imgs','cat_ids_trn'); fprintf(1,'\tLoaded training images\n');
            if((size(imgs{1},1) ~= imgsize(1)) || (size(imgs{1},2) ~= imgsize(2)))
                error('Re-loading images');
            end
            %        load('loadeddata_img_trn.mat','cat_ids_trn'); fprintf(1,'\tLoaded training images\n');
        catch e
            % load data from file and build vocabulary
            [cat_ids_trn, imgs] = loaddata(root,'trainset_txt_img_cat.list',imgsize,1,0);
            save('loadeddata_img_trn.mat','cat_ids_trn','imgs');
            getpoints = 1; docluster = 1; getcounts = 1;
        end
    else
        [cat_ids_trn, imgs] = loaddata(root,'trainset_txt_img_cat.list',imgsize,1,0);
        save('loadeddata_img_trn.mat','cat_ids_trn','imgs');
        getpoints = 1; docluster = 1; getcounts = 1;
    end
    if(modelid < 4)
        % get interest points from each image and perform k-means clustering
        fprintf(1,'\tGetting image interest points\n');
        all_descriptors = []; descriptors = cell(1,length(cat_ids_trn));
        if(getpoints == 0)
            try
                if(featid == 0)
                    load('sift_descriptors.mat');
                else
                    load('gist_descriptors.mat');
                end
            catch e
                getpoints = 1; docluster = 1; getcounts = 1;
            end
        end
        if(getpoints)
            featureparam.orientationsPerScale = [8 8 8 8];
            featureparam.numberBlocks = 4;
            featureparam.fc_prefilt = 4;
            
            if(parallel)
                a = findResource; clustersize = a.ClusterSize;
                matlabpool close force local;
                matlabpool('open', clustersize);
                parfor i = 1:length(imgs)
                    if(featid == 0)
                        if(size(imgs{i},3)>1)
                            imgs{i} = rgb2gray(imgs{i});
                        end
                        [~,descriptors{i}] = vl_sift(im2single(imgs{i}));
                    else
                        sections = extract_secs_large(imgs{i}, rowcol(1), rowcol(2), 1.0, 0, 0);
                        [m, n, q] = size(imgs{i});
                        sz = [floor(m / rowcol(1)), floor(n / rowcol(2))];
                        for k = 1:length(sections)
                            if((size(sections{k},1) ~= sz(1)) || (size(sections{k},2) ~= sz(2)))
                                sections{k} = zero_embed(sections{k}, sz(1), sz(2));
                            end
                            tmp = LMgist(sections{k}, '', featureparam)';
                            tmp(isnan(tmp) | isinf(tmp)) = 0;
                            if(sum(tmp) ~= 0)
                                descriptors{i} = horzcat(descriptors{i}, tmp);
                            end
                        end
                    end
                    fprintf(1,'\tImage %i complete\n',i);
                end
                
                for i = 1:length(descriptors)
                    all_descriptors = horzcat(all_descriptors, descriptors{i});
                end
            else
                for i = 1:length(imgs)
                    im = imgs{i};
                    % if the image is in color, convert to grayscale...
                    if(featid == 0)
                        if(size(im,3)>1)
                            im = rgb2gray(im);
                        end
                        [~,descriptors{i}] = vl_sift(im2single(im));
                    else
                        sections = extract_secs_large(im, rowcol(1), rowcol(2), 1.0, 0, 0);
                        [m, n, q] = size(im);
                        sz = [floor(m / rowcol(1)), floor(n / rowcol(2))];
                        descriptors{i} = [];
                        for k = 1:length(sections)
                            if((size(sections{k},1) ~= sz(1)) || (size(sections{k},2) ~= sz(2)))
                                sections{k} = zero_embed(sections{k}, sz(1), sz(2));
                            end
                            tmp = LMgist(sections{k}, '', featureparam)';
                            tmp(isnan(tmp) | isinf(tmp)) = 0;
                            if(sum(tmp) ~= 0)
                                descriptors{i} = horzcat(descriptors{i},tmp);
                            end
                        end
                    end
                    all_descriptors = horzcat(all_descriptors, descriptors{i});
                end
            end
            if(featid == 0)
                save('sift_descriptors.mat','descriptors');
            else
                save('gist_descriptors.mat','descriptors');
            end
        end
        clear imgs;
        if(featid == 0)
            codebook_size = 128;
        else
            codebook_size = prod(rowcol);
        end
        cluster_options.maxiters = 100;
        cluster_options.verbose = 0;
        
        % call kmeans clustering routine by Mark Everingham
        fprintf(1,'\tPerforming k-means clustering on interest points to develop codebook\n');
        if(docluster == 0)
            try
                if(featid == 0)
                    load('sift_centers.mat');
                else
                    load('gist_centers.mat');
                end
            catch e
                docluster = 1; getcounts = 1;
            end
        end
        if(docluster)
            if(isempty(all_descriptors))
                for i = 1:length(descriptors)
                    all_descriptors = horzcat(all_descriptors, descriptors{i});
                end
            end
            if(isunix)
                [~,centers] = kmeans(double(all_descriptors'), codebook_size);
                centers = centers';
            else
                centers = vgg_kmeans(double(all_descriptors), codebook_size, cluster_options);
            end
            if(featid == 0)
                save('sift_centers.mat','centers');
            else
                save('gist_centers.mat','centers');
            end
        end
        clear all_descriptors;
        
        fprintf(1,'\tGetting visual word counts\n');
        % ex. 1 2 10 = word 2 occurs 10 times in doc 1, i(1) = 2, j(1) = 1, s(1) = 10
        % X(i(k),j(k)) = s(k)
        if(getcounts == 0)
            try
                if(featid == 0)
                    load('sift_counts.mat');
                else
                    load('gist_counts.mat');
                end
            catch e
                getcounts = 1;
            end
        end
        if(getcounts)
            X = zeros(codebook_size,length(cat_ids_trn));
            for i = 1:length(cat_ids_trn)
                distance = pdist2(centers',double(descriptors{i}'));
                
                % Now find the closest center for each point
                [~,descriptor_vq] = min(distance,[],1);
                
                % Now compute histogram over codebook entries for image
                histogram = hist(descriptor_vq,1:codebook_size);
                
                % put into large matrix for training
                X(:,i) = histogram';
            end
            if(featid == 0)
                save('sift_counts.mat','X');
            else
                save('gist_counts.mat','X');
            end
        end
        clear descriptors distance descriptor_vq histogram;
        
        k = max(cat_ids_trn); % number of topics
        
        fprintf(1,'\tLearning visual word relation to topics\n');
        
        
        switch(modelid)
            case 0
                % load preprocessed image infomation
                I_tr = X';
                trnparam1 = []; trnparam2 = []; labelidx = [];
                acc = 0;
            case 1 % Naive Bayes
                alpha = 1/codebook_size;
                X = X';
                % training
                y = zeros(k,1);
                x = zeros(k,codebook_size)+alpha;
                for j = 1:k
                    x(j,:) = x(j,:) + sum(X((cat_ids_trn == j),:),1);
                    % calculate priors
                    x(j,:) = x(j,:) ./ sum(x(j,:));
                    y(j) = sum(cat_ids_trn == j);
                end
                y = y ./ sum(y);
                
                % testing
                predict = log(x) * X' + repmat(log(y),1,size(X,1));
                [~,class] = max(predict); class = class';
                I_tr = predict';
                acc = 100 * (sum((class-cat_ids_trn)==0)/length(cat_ids_trn));
                if(featid == 0)
                    save('ImgNB_trn_params_sift.mat','x','y');
                else
                    save('ImgNB_trn_params_gist.mat','x','y');
                end
                trnparam1 = x; trnparam2 = y; labelidx = [];
            case 2 % PLSA
                Learn.Verbosity = 0;
                Learn.Max_Iterations = 100;
                Learn.heldout = .1; % for tempered EM only, percentage of held out data
                Learn.Min_Likelihood_Change = 1;
                Learn.Folding_Iterations = 20; % for TEM only: number of fiolding
                Learn.TEM = 1; %tempered or not tempered
                
                [Pw_z,predX,Pd,Li,perp,beta] = pLSA(sparse(X),[],k,Learn);
                
                predX = pLSA_EMfold(sparse(X),Pw_z,[],25,beta);
                
                [~,class] = max(predX); class = class';
                labelidx = getnewlabels(cat_ids_trn,class,k);
                newlabels = zeros(size(cat_ids_trn));
                for q = 1:k
                    newlabels((cat_ids_trn == q)) = labelidx(q);
                end
                predX(1:k,:) = predX(labelidx,:);
                I_tr = predX';                
                acc = 100 * (sum((class-newlabels)==0)/length(newlabels));
                if(featid == 0)
                    save('ImgPLSA_trn_params_sift.mat','Pw_z','beta','labelidx');
                else
                    save('ImgPLSA_trn_params_gist.mat','Pw_z','beta','labelidx');
                end
                trnparam1 = Pw_z; trnparam2 = beta;
            case 3
                load(fullfile(root,'raw_features.mat'),'I_tr');
                trnparam1 = []; trnparam2 = []; labelidx = [];
                acc = 0;
        end
    else
        featureparam.orientationsPerScale = [8 8 8 8];
        featureparam.numberBlocks = 4;
        featureparam.fc_prefilt = 4;
        
        if(parallel)
            descriptors = cell(1,length(cat_ids_trn));
            a = findResource; clustersize = a.ClusterSize;
            matlabpool close force local;
            matlabpool('open', clustersize);
            parfor i = 1:length(imgs)
                if(featid == 0)
                    if(size(imgs{i},3)>1)
                        imgs{i} = rgb2gray(imgs{i});
                    end
                    [~,descriptors{i}] = vl_sift(im2single(imgs{i}));
                    descriptors{i} = princomp(double(descriptors{i}'));
                else
                    descriptors{i} = LMgist(imgs{i}, '', featureparam)';
                    descriptors{i}(isnan(descriptors{i}) | isinf(descriptors{i})) = 0;
                end
                fprintf(1,'\tImage %i complete\n',i);
            end
            
            for i = 1:length(descriptors)
                I_tr(i,:) = double(descriptors{i}(:)');
            end
        else
            for i = 1:length(imgs)
                im = imgs{i};
                % if the image is in color, convert to grayscale...
                if(featid == 0)
                    if(size(im,3)>1)
                        im = rgb2gray(im);
                    end
                    [~,descriptors] = vl_sift(im2single(im));
                    descriptors = princomp(double(descriptors'));
                else
                    descriptors = LMgist(imgs{i}, '', featureparam)';
                    descriptors(isnan(descriptors) | isinf(descriptors)) = 0;
                end
                I_tr(i,:) = double(descriptors(:)');
            end
        end
        trnparam1 = []; trnparam2 = []; labelidx = []; acc = 0;
    end
    
    I_tr(isnan(I_tr) | isinf(I_tr)) = 0;
    
    if(featid == 0)
        save(['Img_trn_sift' num2str(modelid) '.mat'],'I_tr','cat_ids_trn');
    else
        save(['Img_trn_gist' num2str(modelid) '.mat'],'I_tr','cat_ids_trn');
    end
    
    fprintf(1,'\tTraining accuracy = %.1f%%\n', acc);
end
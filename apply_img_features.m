function [I_te, cat_ids_tst] = apply_img_features(root, imgsize, rowcol, trnparam1, trnparam2, labelidx, featid, modelid, forcerebuild)

flag = 0;
if(isunix)
    parallel = 1;
else
    parallel = 0;
end
reloadimgs = 0; % forcefully reload images from file
getpoints = 0; % get descriptors or load from file
getcounts = 0; % get visual word counts or load from file


% get features to produce I_te
if(forcerebuild)
    flag = 1;
else
    % process takes awhile to do...use what i've already got done
    try
        % load preprocessed image infomation
        if(featid == 0)
            load(['Img_tst_sift' num2str(modelid) '.mat'],'I_te','cat_ids_tst');
        else
            load(['Img_tst_gist' num2str(modelid) '.mat'],'I_te','cat_ids_tst');
        end
        fprintf(1,'\tLoaded testing image features\n');
    catch e
        flag = 1;
    end
end
if(flag)
    if(reloadimgs == 0)
		try
			% load training data
			load('loadeddata_img_tst.mat','imgs','cat_ids_tst'); fprintf(1,'\tLoaded testing images\n');
			if((size(imgs{1},1) ~= imgsize(1)) || (size(imgs{1},2) ~= imgsize(2)))
				error('Re-loading images');
            end
		catch e
			% load data from file and build vocabulary
			[cat_ids_tst, imgs] = loaddata(root,'testset_txt_img_cat.list',imgsize,1,0);
			save('loadeddata_img_tst.mat','cat_ids_tst','imgs');
            getpoints = 1; getcounts = 1; 
		end
	else
		[cat_ids_tst, imgs] = loaddata(root,'testset_txt_img_cat.list',imgsize,1,0);
		save('loadeddata_img_tst.mat','cat_ids_tst','imgs');
        getpoints = 1; getcounts = 1; 
	end
    
    if(modelid < 4)
        % get interest points from each image and perform k-means clustering
        fprintf(1,'\tGetting image interest points\n');
        if(getpoints == 0)
            try
                if(featid == 0)
                    load('sift_descriptors_tst.mat');
                else
                    load('gist_descriptors_tst.mat');
                end
            catch e
                getpoints = 1; getcounts = 1; 
            end
        end
        if(getpoints)
            getcounts = 1;
            descriptors = cell(1,length(cat_ids_tst));
            featureparam.orientationsPerScale = [8 8 8 8];
            featureparam.numberBlocks = 1;
            featureparam.fc_prefilt = 1;

            if(parallel)
                a = findResource; clustersize = a.ClusterSize;
                matlabpool close force local;
                matlabpool('open', clustersize);
                parfor i = 1:length(imgs)
                    if(featid == 0)
                        if(size(im,3)>1)
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
                end
            end
            if(featid == 0)
                save('sift_descriptors_tst.mat','descriptors');
            else
                save('gist_descriptors_tst.mat','descriptors');
            end	
        end
        clear imgs;

        fprintf(1,'\tGetting visual word counts\n');
        % ex. 1 2 10 = word 2 occurs 10 times in doc 1, i(1) = 2, j(1) = 1, s(1) = 10
        % X(i(k),j(k)) = s(k)
        if(getcounts == 0)
            try
                if(featid == 0)
                    load('sift_counts_tst.mat');
                else
                    load('gist_counts_tst.mat');
                end
            catch e
                getcounts = 1;
            end
        end
        if(getcounts)
            if(featid == 0)
                load('sift_centers.mat');
            else
                load('gist_centers.mat');
            end
            codebook_size = size(centers,2);
            X = zeros(codebook_size,length(cat_ids_tst));
            for i = 1:length(cat_ids_tst)
                distance = pdist2(centers',double(descriptors{i}'));

                % Now find the closest center for each point
                [~,descriptor_vq] = min(distance,[],1);

                % Now compute histogram over codebook entries for image
                histogram = hist(descriptor_vq,1:codebook_size);

                % put into large matrix for training
                X(:,i) = histogram';
            end
            if(featid == 0)
                save('sift_counts_tst.mat','X');
            else
                save('gist_counts_tst.mat','X');
            end	
        end
        clear descriptors distance descriptor_vq histogram;

        k = max(cat_ids_tst); % number of topics

        fprintf(1,'\tApplying visual word relation to topics\n');

        switch(modelid)
            case 0
                % load preprocessed image infomation
                I_te = X';
                acc = 0;
            case 1 % Naive Bayes
                predict = log(trnparam1) * X + repmat(log(trnparam2),1,size(X,2));
                [~,class] = max(predict); class = class';
                I_te = predict';    
                acc = 100 * (sum((class-cat_ids_tst)==0)/length(cat_ids_tst));
            case 2 % PLSA
                predX = pLSA_EMfold(sparse(X),trnparam1,[],25,trnparam2);
                [~,class] = max(predX); class = class';

                newlabels = zeros(size(cat_ids_tst));
                for q = 1:k
                    newlabels((cat_ids_tst == q)) = labelidx(q);
                end

                predX(1:k,:) = predX(labelidx,:);
                I_te = predX';
                acc = 100 * (sum((class-newlabels)==0)/length(newlabels));
            case 3
                load(fullfile(root,'raw_features.mat'),'I_te');
                acc = 0;
        end
    else
        featureparam.orientationsPerScale = [8 8 8 8];
        featureparam.numberBlocks = 4;
        featureparam.fc_prefilt = 4;
        
        if(parallel)
            descriptors = cell(1,length(cat_ids_tst));
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
                I_te(i,:) = double(descriptors{i}(:)');
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
                I_te(i,:) = double(descriptors(:)');
            end
        end
        acc = 0;
    end
    
    I_te(isnan(I_te) | isinf(I_te)) = 0;
    
    if(featid == 0)
        save(['Img_tst_sift' num2str(modelid) '.mat'],'I_te','cat_ids_tst');
    else
        save(['Img_tst_gist' num2str(modelid) '.mat'],'I_te','cat_ids_tst');
    end
    
    %fprintf(1,'\tTesting accuracy = %.1f%%\n', acc);    
end
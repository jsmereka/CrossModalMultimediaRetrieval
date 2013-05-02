function addtopath()

addpath(genpath(fullfile(pwd,'glmnet_matlab'))); % regression
addpath(genpath(fullfile(pwd,'dlda'))); % supervised LDA
addpath(genpath(fullfile(pwd,'TMBP'))); % unsupervised LDA
addpath(genpath(fullfile(pwd,'LSCCA'))); % CCA
addpath(genpath(fullfile(pwd,'bag_of_words'))); % bag of words
addpath(genpath(fullfile(pwd,'gist'))); %
addpath(genpath(fullfile(pwd,'plsa'))); % 

rmpath(fullfile(pwd,'bag_of_words','sift','glnx86'));
rmpath(fullfile(pwd,'bag_of_words','sift','glnxa64'));
rmpath(fullfile(pwd,'bag_of_words','sift','win32'));
rmpath(fullfile(pwd,'bag_of_words','sift','win64'));
addpath(fullfile(pwd,'bag_of_words','sift','src'));
if(isunix)
    switch computer
        case 'GLNX86'
            addpath(fullfile(pwd,'bag_of_words','sift','glnx86'));
        case 'GLNXA64'
            addpath(fullfile(pwd,'bag_of_words','sift','glnxa64'));
        otherwise
            error('computer type not recognized');
    end
else
    switch computer
        case 'PCWIN'
            addpath(fullfile(pwd,'bag_of_words','sift','win32'));
        case 'PCWIN64'
            addpath(fullfile(pwd,'bag_of_words','sift','win64'));
        otherwise
            error('computer type not recognized');
    end
end
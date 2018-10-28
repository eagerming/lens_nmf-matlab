function result = Ranking_evaluate(index_of_groundtruth_in_ordered_recommendation, num_of_groundtruth, groundtruth)
    
    result.map = MAP(index_of_groundtruth_in_ordered_recommendation);
    result.precision = Precision(index_of_groundtruth_in_ordered_recommendation);
    result.recall = Recall(index_of_groundtruth_in_ordered_recommendation, num_of_groundtruth);
    result.NDCG = NDCGatK(index_of_groundtruth_in_ordered_recommendation, groundtruth);
    result.hr = HitRate(index_of_groundtruth_in_ordered_recommendation);
    result.arhr = ARHR(index_of_groundtruth_in_ordered_recommendation);
end




% Note: all K in functions below is represent the min(K, numOf_groundTruth) 
% in higher level calling function.


function hr = HitRate(index)
    K = length(index);
    if sum(index <= K) > 0
        hr = 1;
    else
        hr = 0;
    end
end


function arhr = ARHR(index)
    K = length(index);
    if sum(index <= K)>0
        arhr = 1.0 / min(index);
    else
        arhr = 0;
    end
end

function map = MAP(index)
% Mean average precition

    K = length(index);
    index_hit = index(index <= K);
    
    if ~isempty(index_hit)
        index_hit = sort(index_hit);
        map = sum((1:length(index_hit))' ./ index_hit) / K;
    else
        map = 0;
    end
end



function recall = Recall(index, numOfTruth)
% index: index of groundtruth in ordered(ranked) top K recommendation list
% example:
% idea: index = [2 4 3 1]; numOfTruth = 4 --->   Recall(index) = 1
% real world situation: index = [4 27 11 10];  -----> Recall(index) < 1
    
    K = length(index);
    recall = sum(index <= K) / numOfTruth;

end
 
function prec = Precision(index)
% index: index of groundtruth in ordered(ranked) top K recommendation list
% example:
% idea: index = [2 4 3 1]; --->   Precision(index) = 1
% real world situation: index = [4 27 11 10];  -----> Precision(index) < 1

    K = length(index);
    prec = sum(index <= K) / K;
end



function [NDCGK] = NDCGatK(pos_y, y)
% y: the real world score.
% pos_y: The corresponding position of score y in the recommendation order.

% example:
% y = [3 4 5 6];
% real situation: pos_y = [32 2 11 1];  -----> NDCGatK(pos_y, y) < 1;
% idea: pos_y = [1 2 3 4];  ------> NDCGatK(pos_y, y) = 1


K = length(y);
pos_y = pos_y(:)';
y = y(:)';
sortedRank = sort(y,'descend');

nominator = (2 .^ y - 1)./ log((pos_y + 1));
denominator =  (2 .^ sortedRank - 1) ./ log((1:K) + 1);

NDCGK = sum(nominator) ./ sum(denominator);
end
